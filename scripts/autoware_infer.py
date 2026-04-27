#!/usr/bin/env python3
"""
autoware_infer.py — 4-camera capture + Autoware live perception stack.

Runs on system Python 3.12 (torch 2.10 + CUDA). Sister process to
``ps5_drive.py`` (Python 3.13, no torch). Layout:

    ┌───────────────────────┐   JPEGs/state via /tmp  ┌──────────────────────┐
    │  autoware_infer.py    │ ──────────────────────▶ │  web/app.py (Flask)  │
    │  • opens 4 USB cams   │                         │  MJPEG + /state JSON │
    │  • runs SceneSeg,     │                         └──────────────────────┘
    │    Scene3D, EgoLanes, │
    │    AutoSteer, AutoSpeed       ──────▶ /tmp/autoware_state.json
    │  • renders viz tiles  │                 │
    └───────────────────────┘                 ▼
                                     ┌───────────────────────┐
                                     │  ps5_drive.py         │
                                     │  --autosteer reads    │
                                     │  steer_deg from here  │
                                     └───────────────────────┘

Writes:
  /tmp/cart_frames/{slug}.jpg        latest frame per stream (atomic rename)
    cameras: front_wide, front_narrow, left, right
    viz:     lanes, depth, objects, seg
  /tmp/autoware_state.json           {steer_deg, active_cam, fps, stale, ts}

AutoSteer expects a forward-facing narrow view; ``front_narrow`` is the
inference input. ``front_wide`` is mounted upside down on this cart, so we
180°-flip it before encoding (UI shouldn't have to know about install
orientation).

Usage:
    /usr/bin/python3 scripts/autoware_infer.py
    /usr/bin/python3 scripts/autoware_infer.py --no-infer       # cams only
    /usr/bin/python3 scripts/autoware_infer.py --no-viz         # cams + steer only
    /usr/bin/python3 scripts/autoware_infer.py --jpeg-hz 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

# autoware_vision_pilot is a sibling of PRODUCTION — hardcoded path lets us
# run without installing it, since it has no pyproject and lives outside.
AUTOWARE_ROOT = Path("/home/caddy/autoware_vision_pilot")
sys.path.insert(0, str(AUTOWARE_ROOT))
sys.path.insert(0, str(AUTOWARE_ROOT / "Models"))

FRAMES_DIR_DEFAULT = Path("/tmp/cart_frames")
STATE_FILE_DEFAULT = Path("/tmp/autoware_state.json")

# Order of physical USB cameras after v4l2 auto-discovery. Index `i` in
# the scan maps to SLUGS[i].
# Verified live (2026-04-25): on this cart's current USB topology, the
# discovery order is wide / narrow / left / right — index 1 is the
# narrow-FOV varifocal lens used for inference, index 2 is the left-side
# fisheye. Re-plug USB cables and this order may change; that's the time
# to revisit this list.
SLUGS = ("front_wide", "front_narrow", "left", "right")
INFERENCE_SLUG = "front_narrow"
# Cameras with non-standard install orientation. cv2.flip codes:
#   0  = vertical flip, 1 = horizontal, -1 = both (180° rotation).
CAMERA_ORIENTATION_FIX = {
    "front_wide": -1,   # mounted upside down on the cart
}
# The 4 model-output streams that ride alongside the camera streams in the
# UI. These slugs land in /tmp/cart_frames/ exactly like the cameras and
# Flask serves them through the same /cam/<slug>.* endpoints.
VIZ_SLUGS = ("lanes", "depth", "objects", "seg")
ALL_STREAM_SLUGS = SLUGS + VIZ_SLUGS

CAM_W, CAM_H = 640, 480     # per-camera capture; four 1080p streams saturate USB2
INF_SIZE = (640, 320)       # SceneSeg / Scene3D / EgoLanes / AutoSteer input (w, h)
AUTOSPEED_SIZE = 640        # AutoSpeed input is square, letterboxed
JPEG_QUALITY = 72            # 72 ≈ 35 KB per 640x480 frame
STEER_SMOOTHING = 0.8        # matches run_live.py; first-order EMA on argmax
EGOLANES_THRESH = 0.04
AUTOSPEED_CONF = 0.40
AUTOSPEED_NMS_IOU = 0.45
AUTOWARE_STATE_FRESH_S = 0.3


# ═══════════════════════════════════════════════════════════════════════════
# Camera discovery
# ═══════════════════════════════════════════════════════════════════════════

def discover_v4l2_indices(count: int = 4, max_scan: int = 16) -> list[int]:
    """Return the first ``count`` v4l2 indices that actually produce a frame.

    On the Jetson each physical USB camera exposes multiple ``/dev/video*``
    nodes (raw capture, metadata, compressed); the streaming nodes are at
    indices [0, 2, 6, 10] — not 0..3. Scan up through the double-digits so
    we actually find them all.
    """
    found: list[int] = []
    for idx in range(max_scan):
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        ok, _ = cap.read()
        cap.release()
        if ok:
            found.append(idx)
            if len(found) >= count:
                break
    return found


def open_camera(index: int) -> cv2.VideoCapture | None:
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


class CameraReader(threading.Thread):
    """Background grabber — always keeps the latest frame.

    cv2.VideoCapture buffers up to a few frames internally; reading from
    the main loop at a lower rate than the camera produces would serve
    stale frames. A dedicated thread draining into a single-slot
    last-write-wins fixes that.

    Applies any per-camera orientation fix at grab time so downstream code
    (inference + JPEG writers) sees frames in their "intended" pose.
    """

    def __init__(self, cap: cv2.VideoCapture, slug: str):
        super().__init__(daemon=True, name=f"cam-{slug}")
        self.cap = cap
        self.slug = slug
        self.flip_code = CAMERA_ORIENTATION_FIX.get(slug)
        self.lock = threading.Lock()
        self.frame: np.ndarray | None = None
        self.frame_count = 0
        self.last_ok_s = 0.0
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if ok and frame is not None:
                if self.flip_code is not None:
                    frame = cv2.flip(frame, self.flip_code)
                with self.lock:
                    self.frame = frame
                    self.frame_count += 1
                    self.last_ok_s = time.monotonic()
            else:
                time.sleep(0.01)

    def latest(self) -> np.ndarray | None:
        with self.lock:
            return None if self.frame is None else self.frame

    def stop(self) -> None:
        self._stop.set()
        self.cap.release()


# ═══════════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════════

# Cityscapes-style class colors used by SceneSeg's argmax output. Stored as
# BGR for direct OpenCV use; matches run_live.py's CITYSCAPES_LUT byte-for-byte.
_CITYSCAPES_LUT_NUMPY = np.array([
    [128, 64, 128], [232, 35, 244], [70, 70, 70], [156, 102, 102], [153, 153, 190],
    [153, 153, 153], [30, 170, 250], [0, 220, 220], [35, 142, 107], [152, 251, 152],
    [180, 130, 70], [60, 20, 220], [0, 0, 255], [142, 0, 0], [70, 0, 0],
    [100, 60, 0], [100, 80, 0], [230, 0, 0], [32, 11, 119],
], dtype=np.uint8)
# Three lane-class overlay colors (left lane, right lane, ego lane).
_EGOLANES_COLORS_NUMPY = np.array([
    [255, 153, 0],    # cyan-ish (lane 0)
    [255, 56, 255],   # magenta (lane 1)
    [87, 255, 87],    # green (lane 2)
], dtype=np.float32)


def load_all_models(device, with_autospeed: bool):
    """Load every Autoware perception head we display.

    JIT trace failures silently fall back to eager mode. AutoSpeed's
    dynamic control flow doesn't trace cleanly (run_live.py skips it).
    """
    import torch
    from Models.model_components.scene_seg_network import SceneSegNetwork
    from Models.model_components.scene_3d_network import Scene3DNetwork
    from Models.model_components.ego_lanes_network import EgoLanesNetwork
    from Models.model_components.auto_steer_network import AutoSteerNetwork

    weights = AUTOWARE_ROOT / "weights"

    def _load(net, fname):
        net.load_state_dict(torch.load(weights / fname,
                                        weights_only=True, map_location="cpu"))
        return net.half().to(device).eval()

    models = {}
    models["scene_seg"] = _load(SceneSegNetwork(), "scene_seg.pth")
    # Scene3DNetwork wraps a SceneSeg backbone; constructor takes one.
    models["scene_3d"] = _load(Scene3DNetwork(SceneSegNetwork()), "scene_3d.pth")
    models["ego_lanes"] = _load(EgoLanesNetwork(), "ego_lanes.pth")
    models["auto_steer"] = _load(AutoSteerNetwork(), "auto_steer.pth")

    if with_autospeed:
        # auto_speed.pth is pickled differently — it's a dict with the
        # fully-built model under the 'model' key, not a state_dict.
        asp = torch.load(weights / "auto_speed.pth",
                         map_location="cpu", weights_only=False)["model"]
        models["auto_speed"] = asp.to(device).eval()

    print("[models] JIT trace ...")
    dummy_img = torch.randn(1, 3, INF_SIZE[1], INF_SIZE[0],
                             dtype=torch.float16, device=device)
    dummy_steer = torch.randn(1, 6, INF_SIZE[1] // 4, INF_SIZE[0] // 4,
                               dtype=torch.float16, device=device)
    with torch.no_grad():
        for name in ("scene_seg", "scene_3d", "ego_lanes"):
            try:
                m = torch.jit.freeze(torch.jit.trace(models[name], dummy_img))
                for _ in range(3):
                    m(dummy_img)
                models[name] = m
            except Exception as e:
                print(f"[models] {name} JIT failed: {e}")
        try:
            m = torch.jit.freeze(torch.jit.trace(models["auto_steer"], dummy_steer))
            for _ in range(3):
                m(dummy_steer)
            models["auto_steer"] = m
        except Exception as e:
            print(f"[models] auto_steer JIT failed: {e}")
        torch.cuda.synchronize()

    return models


# ═══════════════════════════════════════════════════════════════════════════
# Inference + visualization
# ═══════════════════════════════════════════════════════════════════════════

class InferencePipeline:
    """Runs the full perception stack on one BGR frame.

    Returns a dict with the predicted steering angle plus visualization
    images (BGR, sized to CAM_W×CAM_H so they slot into the same UI grid
    as the camera streams). All viz rendering happens on GPU where it's
    cheap; only the final ``cv2.cvtColor``-equivalent + resize hop crosses
    back to host memory.
    """

    def __init__(self, device, with_autospeed: bool = True):
        import torch
        import torch.nn.functional as F  # noqa: F401 — used in inference body
        self.torch = torch
        self.device = device
        self.with_autospeed = with_autospeed
        self.models = load_all_models(device, with_autospeed=with_autospeed)

        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float16,
                                  device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float16,
                                 device=device).view(1, 3, 1, 1)
        self.cityscapes_lut = torch.from_numpy(_CITYSCAPES_LUT_NUMPY).to(device)
        # Pre-built viridis 256-entry BGR LUT for depth viz. cmapy isn't
        # installed in this env; build it ourselves with cv2's COLORMAP_VIRIDIS.
        viridis_bgr = cv2.applyColorMap(
            np.arange(256, dtype=np.uint8), cv2.COLORMAP_VIRIDIS,
        ).squeeze()
        self.viridis_lut = torch.from_numpy(viridis_bgr).to(device)
        self.egolanes_colors = torch.from_numpy(_EGOLANES_COLORS_NUMPY).to(device)

        # Temporal AutoSteer state — keep last EgoLanes raw output to feed
        # the 6-channel concat input. None on cold start; first frame uses
        # the same tensor twice so the prediction is still valid.
        self.prev_el = None
        self.smoothed_steer = 0.0

    # ----- preprocess -------------------------------------------------------

    def _preprocess(self, frame_bgr):
        torch = self.torch
        resized = cv2.resize(frame_bgr, INF_SIZE)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        t = (torch.from_numpy(rgb).to(self.device, non_blocking=True)
             .permute(2, 0, 1).unsqueeze(0).half() / 255.0)
        return (t - self.mean) / self.std

    def _autospeed_preprocess(self, inf_tensor):
        """Letterbox the perception-input tensor up to AUTOSPEED_SIZE."""
        torch = self.torch
        from torch.nn import functional as F
        raw = inf_tensor * self.std + self.mean
        _, _, h, w = raw.shape
        tw, th = AUTOSPEED_SIZE, AUTOSPEED_SIZE
        scale = min(tw / w, th / h)
        nh, nw = int(h * scale), int(w * scale)
        py, px = (th - nh) // 2, (tw - nw) // 2
        if scale != 1.0:
            raw = F.interpolate(raw, size=(nh, nw), mode="bilinear", align_corners=False)
        padded = torch.full((1, 3, th, tw), 114.0 / 255.0,
                             dtype=torch.float16, device=self.device)
        padded[:, :, py:py + nh, px:px + nw] = raw
        return padded, scale, px, py

    # ----- viz --------------------------------------------------------------

    def _viz_seg(self, ss_pred):
        cls_map = ss_pred.squeeze(0).argmax(dim=0)
        bgr = self.cityscapes_lut[cls_map].cpu().numpy()
        return cv2.resize(bgr, (CAM_W, CAM_H))

    def _viz_depth(self, s3d_pred):
        torch = self.torch
        p = s3d_pred.squeeze()
        pmin, pmax = p.min(), p.max()
        rng = pmax - pmin
        if rng > 1e-6:
            norm = ((p - pmin) / rng * 255.0).clamp(0, 255).to(torch.uint8)
        else:
            norm = torch.zeros_like(p, dtype=torch.uint8)
        bgr = self.viridis_lut[norm.long()].cpu().numpy()
        return cv2.resize(bgr, (CAM_W, CAM_H))

    def _viz_lanes(self, el_pred, frame_bgr):
        torch = self.torch
        from torch.nn import functional as F
        h, w = frame_bgr.shape[:2]
        pred_sq = el_pred.squeeze(0).float()
        pred_up = F.interpolate(
            pred_sq.unsqueeze(0), size=(h, w), mode="nearest",
        ).squeeze(0)
        pred_sig = torch.sigmoid(pred_up)
        overlay = torch.zeros(h, w, 3, dtype=torch.float32, device=self.device)
        for c in range(3):
            mask = pred_sig[c] > EGOLANES_THRESH
            overlay[mask] = self.egolanes_colors[c]
        frame_gpu = torch.from_numpy(frame_bgr).to(self.device, non_blocking=True)
        blended = (frame_gpu.float() + overlay * 0.5).clamp(0, 255).to(torch.uint8)
        return blended.cpu().numpy()

    def _viz_objects(self, frame_bgr, dets, orig_w, orig_h):
        out = frame_bgr.copy()
        fh, fw = out.shape[:2]
        sx, sy = fw / orig_w, fh / orig_h
        # CIPO-1/2/3 = closest in path / second-closest / other. Three
        # readable colors so multiple boxes don't all blur together.
        colors = {0: (0, 0, 255), 1: (0, 255, 255), 2: (255, 255, 0)}
        labels = {0: "CIPO-1", 1: "CIPO-2", 2: "CIPO-3"}
        for det in dets:
            x1, y1, x2, y2, conf, cls = det
            c = int(cls)
            color = colors.get(c, (255, 255, 255))
            x1i, y1i = int(x1 * sx), int(y1 * sy)
            x2i, y2i = int(x2 * sx), int(y2 * sy)
            cv2.rectangle(out, (x1i, y1i), (x2i, y2i), color, 2)
            cv2.putText(out, f"{labels.get(c, 'obj')} {conf:.0%}",
                        (x1i, max(0, y1i - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        if not dets:
            cv2.putText(out, "no obstacles", (12, fh - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        return out

    # ----- main entry -------------------------------------------------------

    def __call__(self, frame_bgr):
        torch = self.torch
        from torchvision import ops

        tensor = self._preprocess(frame_bgr)

        with torch.no_grad():
            # Concurrent SceneSeg / Scene3D / EgoLanes via CUDA streams —
            # they're independent and the GPU has plenty of SM headroom.
            stream_ss = torch.cuda.Stream()
            stream_3d = torch.cuda.Stream()
            stream_el = torch.cuda.Stream()
            with torch.cuda.stream(stream_ss):
                ss_pred = self.models["scene_seg"](tensor)
            with torch.cuda.stream(stream_3d):
                s3d_pred = self.models["scene_3d"](tensor)
            with torch.cuda.stream(stream_el):
                el_pred = self.models["ego_lanes"](tensor)
            torch.cuda.synchronize()

            # AutoSteer needs two consecutive EgoLanes outputs concatenated
            # along channel dim. First frame: feed the same tensor twice.
            if self.prev_el is None:
                self.prev_el = el_pred
            steer_in = torch.cat((self.prev_el, el_pred), dim=1)
            steer_out = self.models["auto_steer"](steer_in)
            if isinstance(steer_out, tuple):
                steer_out = steer_out[1]
            raw_steer = float(steer_out.squeeze(0).argmax().item() - 30)
            self.prev_el = el_pred

            dets, brake_level = [], 0.0
            if self.with_autospeed and "auto_speed" in self.models:
                speed_t, scale, px, py = self._autospeed_preprocess(tensor)
                raw = self.models["auto_speed"](speed_t)
                preds = raw.squeeze(0).permute(1, 0)
                class_probs = preds[:, 4:].sigmoid()
                scores, class_ids = torch.max(class_probs, dim=1)
                mask = scores > AUTOSPEED_CONF
                if mask.sum().item() > 0:
                    boxes = preds[mask, :4]
                    sc, ci = scores[mask], class_ids[mask]
                    if sc.shape[0] > 50:
                        topk = sc.topk(50)
                        boxes, sc, ci = (boxes[topk.indices], topk.values,
                                          ci[topk.indices])
                    xy = torch.empty_like(boxes)
                    xy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
                    xy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
                    xy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
                    xy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
                    keep = ops.nms(xy.float(), sc.float(), AUTOSPEED_NMS_IOU)
                    xy, sc, ci = xy[keep], sc[keep], ci[keep]
                    # Map letterboxed 640x640 coords back to INF_SIZE.
                    xy[:, [0, 2]] = ((xy[:, [0, 2]] - px) / scale).clamp(0, INF_SIZE[0])
                    xy[:, [1, 3]] = ((xy[:, [1, 3]] - py) / scale).clamp(0, INF_SIZE[1])
                    cx = (xy[:, 0] + xy[:, 2]) / 2.0
                    bw = xy[:, 2] - xy[:, 0]
                    bh = xy[:, 3] - xy[:, 1]
                    center_off = (cx - INF_SIZE[0] / 2.0).abs() / (INF_SIZE[0] / 2.0)
                    lane_weight = (1.0 - center_off * 1.5).clamp(min=0)
                    area_ratio = (bw * bh) / (INF_SIZE[0] * INF_SIZE[1])
                    proximity = (area_ratio * 15.0).clamp(max=1.0)
                    per_det = proximity * lane_weight * sc.float()
                    brake_level = float(min(1.0, per_det.max().item()
                                             if per_det.numel() > 0 else 0.0))
                    combined = torch.cat([xy, sc.unsqueeze(1),
                                           ci.float().unsqueeze(1)], dim=1)
                    dets = combined.cpu().tolist()

        # Build the four UI tiles. Lanes/objects stack onto the camera
        # frame so the operator sees overlay-on-reality; depth/seg show
        # the model output directly so the structure is readable.
        viz_lanes = self._viz_lanes(el_pred, frame_bgr)
        viz_depth = self._viz_depth(s3d_pred)
        viz_seg = self._viz_seg(ss_pred)
        viz_objects = self._viz_objects(frame_bgr, dets,
                                          INF_SIZE[0], INF_SIZE[1])

        self.smoothed_steer = (
            STEER_SMOOTHING * self.smoothed_steer
            + (1 - STEER_SMOOTHING) * raw_steer
        )

        return {
            "steer_raw": raw_steer,
            "steer_smoothed": self.smoothed_steer,
            "object_count": len(dets),
            "brake_level": brake_level,
            "viz_lanes": viz_lanes,
            "viz_depth": viz_depth,
            "viz_seg": viz_seg,
            "viz_objects": viz_objects,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Atomic file writers (rename is atomic on same filesystem)
# ═══════════════════════════════════════════════════════════════════════════

def write_jpeg_atomic(path: Path, frame: np.ndarray) -> None:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        return
    tmp = path.with_suffix(".jpg.tmp")
    tmp.write_bytes(buf.tobytes())
    os.replace(tmp, path)


def write_state_atomic(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(data, f)
    os.replace(tmp, path)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames-dir", default=str(FRAMES_DIR_DEFAULT),
                   help="Directory for per-stream JPEGs (atomic replace).")
    p.add_argument("--state-file", default=str(STATE_FILE_DEFAULT),
                   help="JSON output for predicted steer + active cam.")
    p.add_argument("--no-infer", action="store_true",
                   help="Skip all inference; only publish camera frames.")
    p.add_argument("--no-viz", action="store_true",
                   help="Run AutoSteer for the steering value, but skip "
                        "SceneSeg/Scene3D/AutoSpeed and don't publish viz tiles.")
    p.add_argument("--jpeg-hz", type=float, default=15.0,
                   help="Max per-stream JPEG publish rate (default 15).")
    p.add_argument("--indices", type=int, nargs="*", default=None,
                   help="Override v4l2 indices (e.g. --indices 0 1 2 3).")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    frames_dir = Path(args.frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    state_path = Path(args.state_file)
    state_path.parent.mkdir(parents=True, exist_ok=True)

    print("[cams] discovering v4l2 devices...")
    indices = args.indices or discover_v4l2_indices(count=len(SLUGS))
    print(f"[cams] indices: {indices}")
    if not indices:
        print("ERROR: no cameras found. Try: v4l2-ctl --list-devices")
        return 1

    readers: list[CameraReader] = []
    for idx, slug in zip(indices, SLUGS):
        cap = open_camera(idx)
        if cap is None:
            print(f"[cams] WARN: idx {idx} ({slug}) failed to open")
            continue
        r = CameraReader(cap, slug)
        r.start()
        readers.append(r)
        print(f"[cams] {slug:<12} -> /dev/video{idx}"
              + (f"  (flip={CAMERA_ORIENTATION_FIX[slug]})"
                 if slug in CAMERA_ORIENTATION_FIX else ""))

    if not readers:
        print("ERROR: opened 0 cameras.")
        return 1

    time.sleep(0.4)

    pipe: InferencePipeline | None = None
    active_slug = INFERENCE_SLUG
    if not args.no_infer:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[models] device: {device}")
        if device.type == "cuda":
            print(f"[models] gpu: {torch.cuda.get_device_name(0)}")
        pipe = InferencePipeline(device, with_autospeed=not args.no_viz)
        print(f"[models] loaded ({'with' if not args.no_viz else 'without'} viz).")
        slug_map = {r.slug: r for r in readers}
        if active_slug not in slug_map:
            print(f"[models] WARN: {active_slug} not among cams — disabling inference")
            pipe = None

    jpeg_period = 1.0 / max(args.jpeg_hz, 1.0)
    next_jpeg_t = time.monotonic()

    infer_times: list[float] = []
    last = {
        "steer_raw": 0.0, "steer_smoothed": 0.0,
        "object_count": 0, "brake_level": 0.0,
        "viz_lanes": None, "viz_depth": None,
        "viz_seg": None, "viz_objects": None,
    }

    print("[run] publishing to", frames_dir, "and", state_path)
    try:
        while True:
            loop_t = time.monotonic()

            # ── 1) Inference (when enabled). Runs as fast as the GPU
            # allows so the steering target is fresh; viz JPEGs are
            # written at the JPEG cadence below.
            if pipe is not None:
                slug_map = {r.slug: r for r in readers}
                frame = slug_map[active_slug].latest()
                if frame is not None:
                    t0 = time.monotonic()
                    try:
                        result = pipe(frame)
                        last.update(result)
                    except Exception as e:
                        # One bad frame shouldn't take down the whole
                        # process — log and keep serving the last known
                        # state. Most commonly this is a CUDA OOM blip.
                        print(f"[infer] frame failed: {e}")
                    infer_times.append(time.monotonic() - t0)
                    if len(infer_times) > 20:
                        infer_times.pop(0)

            # ── 2) Publish frames + viz at JPEG cadence. Single budget
            # across all streams keeps disk I/O bounded.
            if loop_t >= next_jpeg_t:
                for r in readers:
                    frame = r.latest()
                    if frame is None:
                        continue
                    write_jpeg_atomic(frames_dir / f"{r.slug}.jpg", frame)
                # Viz tiles: only when we have outputs and viz is on.
                if pipe is not None and not args.no_viz:
                    for slug, key in [
                        ("lanes", "viz_lanes"),
                        ("depth", "viz_depth"),
                        ("seg", "viz_seg"),
                        ("objects", "viz_objects"),
                    ]:
                        viz = last.get(key)
                        if viz is not None:
                            write_jpeg_atomic(frames_dir / f"{slug}.jpg", viz)
                next_jpeg_t = loop_t + jpeg_period

            # ── 3) Publish state.
            mean_dt = sum(infer_times) / len(infer_times) if infer_times else 0.0
            write_state_atomic(state_path, {
                "steer_deg": float(last["steer_smoothed"]),
                "steer_deg_raw": float(last["steer_raw"]),
                "active_cam": active_slug if pipe is not None else None,
                "inference": pipe is not None,
                "viz": (pipe is not None and not args.no_viz),
                "viz_streams": list(VIZ_SLUGS) if (pipe is not None and not args.no_viz) else [],
                "object_count": int(last["object_count"]),
                "fps": (1.0 / mean_dt) if mean_dt > 0 else 0.0,
                "cams": [r.slug for r in readers],
                "ts": time.time(),
            })

            if pipe is None:
                time.sleep(0.02)
    except KeyboardInterrupt:
        print("\n[run] interrupted.")
    finally:
        for r in readers:
            r.stop()
        print("[run] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
