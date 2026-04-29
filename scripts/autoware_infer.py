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
    /usr/bin/python3 scripts/autoware_infer.py --video clip.mp4 # replay a clip
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
#   0  = flip across the x-axis (vertical),
#   1  = flip across the y-axis (horizontal),
#  -1  = both (180° rotation).
CAMERA_ORIENTATION_FIX = {
    "front_wide": 0,    # flip across x-axis — camera mount inverts vertically
}
# The 4 model-output streams that ride alongside the camera streams in the
# UI. These slugs land in /tmp/cart_frames/ exactly like the cameras and
# Flask serves them through the same /cam/<slug>.* endpoints.
VIZ_SLUGS = ("lanes", "depth", "objects", "seg")
# Auxiliary streams not shown as UI tiles but published so the 3D scene
# (web/static/js/scene.js) can texture-load them. ``lanes_solo`` is the
# raw EgoLanes overlay on a black background — the 3D scene additively
# blends it onto the road in front of the cart so the predicted lanes
# appear as a glow under the simulated golf cart.
AUX_SLUGS = ("lanes_solo",)
ALL_STREAM_SLUGS = SLUGS + VIZ_SLUGS + AUX_SLUGS

CAM_W, CAM_H = 640, 480     # per-camera capture; four 1080p streams saturate USB2
INF_SIZE = (640, 320)       # SceneSeg / Scene3D / EgoLanes / AutoSteer input (w, h)
AUTOSPEED_SIZE = 640        # AutoSpeed input is square, letterboxed
JPEG_QUALITY = 72            # 72 ≈ 35 KB per 640x480 frame
STEER_SMOOTHING = 0.8        # matches run_live.py; first-order EMA on argmax
EGOLANES_THRESH = 0.04
AUTOSPEED_CONF = 0.40
AUTOSPEED_NMS_IOU = 0.45
AUTOWARE_STATE_FRESH_S = 0.3

# AutoSteer / EgoLanes were trained on a forward-facing narrow view and
# work best around a ~30° horizontal FOV. The narrow USB camera and the
# replay videos are both wider than that out of the box, so we center-
# crop the front_narrow stream by ``tan(target/2) / tan(source/2)`` and
# resize the crop back up to ``CAM_W × CAM_H`` so downstream code (model
# preprocess, JPEG writers, UI tile) sees a frame that looks like it
# came from a narrower lens. ``--narrow-fov-deg`` opts in; otherwise the
# raw source frame passes through unchanged.
NARROW_SOURCE_FOV_DEG_DEFAULT = 60.0


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


def center_crop_zoom(frame: np.ndarray, ratio: float) -> np.ndarray:
    """Center-crop ``frame`` by ``ratio`` (0,1] then resize back to its
    original WxH so the apparent FOV shrinks but the consumer-side
    resolution is unchanged. ``ratio >= 1`` is a no-op (no crop).
    """
    if ratio >= 1.0 or ratio <= 0.0:
        return frame
    h, w = frame.shape[:2]
    nh = max(2, int(round(h * ratio)))
    nw = max(2, int(round(w * ratio)))
    y0 = (h - nh) // 2
    x0 = (w - nw) // 2
    cropped = frame[y0:y0 + nh, x0:x0 + nw]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


class CameraReader(threading.Thread):
    """Background grabber — always keeps the latest frame.

    cv2.VideoCapture buffers up to a few frames internally; reading from
    the main loop at a lower rate than the camera produces would serve
    stale frames. A dedicated thread draining into a single-slot
    last-write-wins fixes that.

    Applies any per-camera orientation fix at grab time so downstream code
    (inference + JPEG writers) sees frames in their "intended" pose.

    ``crop_ratio`` (default 1.0 = no crop) shrinks the apparent FOV by
    center-cropping then resizing back to the original WxH. Used to
    narrow the front_narrow stream to the ~30° FOV AutoSteer/EgoLanes
    were trained on; the orientation flip happens first so the crop is
    still centered on the cart's forward axis when the camera is
    installed upside-down.
    """

    def __init__(self, cap: cv2.VideoCapture, slug: str,
                 crop_ratio: float = 1.0):
        super().__init__(daemon=True, name=f"cam-{slug}")
        self.cap = cap
        self.slug = slug
        self.flip_code = CAMERA_ORIENTATION_FIX.get(slug)
        self.crop_ratio = crop_ratio
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
                if self.crop_ratio < 1.0:
                    frame = center_crop_zoom(frame, self.crop_ratio)
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
# Video-file source (replay mode)
# ═══════════════════════════════════════════════════════════════════════════

class VideoFileSource(threading.Thread):
    """Reads a video file at its native FPS as a stand-in for a USB camera.

    Same single-slot last-write-wins contract as ``CameraReader`` so the
    rest of the pipeline (inference, JPEG publishing, state writing) is
    completely unaware of where frames came from. Frames are letterboxed
    into ``CAM_W × CAM_H`` so JPEG sizes and UI tile aspect match the
    live-camera path byte-for-byte.

    ``loop=True`` (default) seeks back to frame 0 at EOF so a short clip
    keeps the UI animated indefinitely. With ``loop=False`` the source
    holds the last frame and stops advancing once the clip ends.
    """

    def __init__(self, path: Path, loop: bool = True):
        super().__init__(daemon=True, name=f"video-{path.name}")
        self.path = path
        self.loop = loop
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {path}")
        self.native_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.src_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.src_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.lock = threading.Lock()
        self.frame: np.ndarray | None = None
        self.frame_count = 0
        self.video_pos = 0          # 0-based frame index of latest frame
        self.loops = 0
        self.last_ok_s = 0.0
        self._stop = threading.Event()

    def _letterbox(self, frame: np.ndarray) -> np.ndarray:
        """Resize+pad ``frame`` to ``(CAM_W, CAM_H)`` preserving aspect.

        Black bars top/bottom or left/right as needed. Keeping the aspect
        ratio matters because the inference pipeline assumes a forward-
        facing view and a non-uniform squish would slightly skew lane
        geometry.
        """
        sh, sw = frame.shape[:2]
        if sw == CAM_W and sh == CAM_H:
            return frame
        scale = min(CAM_W / sw, CAM_H / sh)
        nw, nh = max(1, int(sw * scale)), max(1, int(sh * scale))
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        out = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
        py, px = (CAM_H - nh) // 2, (CAM_W - nw) // 2
        out[py:py + nh, px:px + nw] = resized
        return out

    def run(self) -> None:
        period = 1.0 / max(self.native_fps, 1.0)
        next_t = time.monotonic()
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok or frame is None:
                if self.loop:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.loops += 1
                    print(f"[video] looped ({self.loops} restarts) "
                          f"{self.path.name}")
                    next_t = time.monotonic()
                    continue
                # Non-loop EOF: hold the last frame, stop ticking the clock.
                print(f"[video] reached end of {self.path.name} (no loop)")
                break
            framed = self._letterbox(frame)
            with self.lock:
                self.frame = framed
                self.frame_count += 1
                self.video_pos = int(
                    self.cap.get(cv2.CAP_PROP_POS_FRAMES) or 0
                )
                self.last_ok_s = time.monotonic()
            # Pace at native FPS. If we're falling behind (inference slow),
            # don't accumulate "owed" sleeps — just resync from now.
            next_t += period
            sleep_s = next_t - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_t = time.monotonic()

    def latest(self) -> np.ndarray | None:
        with self.lock:
            return None if self.frame is None else self.frame

    def stop(self) -> None:
        self._stop.set()
        self.cap.release()


class VideoSlotReader:
    """Slug-attached view onto a shared ``VideoFileSource``.

    Duck-types ``CameraReader`` so the main loop's ``readers`` list works
    unchanged: inference picks ``slug_map[INFERENCE_SLUG]`` and the JPEG
    publisher iterates ``readers``. All four UI tiles end up showing the
    same video frame — ``front_narrow`` is the one fed into the perception
    stack; the others are cosmetic copies so the grid isn't half-empty.

    ``crop_ratio`` is applied per-slot in ``latest()`` (not on the
    shared source) so the narrow-FOV crop only affects ``front_narrow``
    and the other three slugs keep showing the full frame.
    """

    def __init__(self, source: VideoFileSource, slug: str,
                 crop_ratio: float = 1.0):
        self.source = source
        self.slug = slug
        self.flip_code = None
        self.crop_ratio = crop_ratio
        self.frame_count = 0  # unused by the loop, but kept for parity
        self.last_ok_s = 0.0

    def latest(self) -> np.ndarray | None:
        frame = self.source.latest()
        if frame is None or self.crop_ratio >= 1.0:
            return frame
        return center_crop_zoom(frame, self.crop_ratio)

    def start(self) -> None:
        # The underlying source thread is started once at the top level;
        # individual slot views are passive.
        pass

    def stop(self) -> None:
        # Source is stopped once at the top level.
        pass


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
        """Returns (blended, solo).

        ``blended`` — the camera frame with lane colors layered on top
        (the existing UI ``lanes`` tile).
        ``solo`` — the same lane mask but on a black background. The 3D
        scene additively blends this onto the road plane in front of the
        cart so black pixels contribute nothing and the predicted lanes
        show up as a glow.

        The solo mask is also dilated. Raw EgoLanes lines are ~1-3 pixels
        wide; JPEG quality 72 + 4:2:0 chroma subsampling averages those
        bright pixels into the black neighbors and the result blends to
        nearly invisible on the road. Widening to ~9 pixels lets the
        colors survive compression with full saturation.
        """
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
        # Solo: dilate on CPU so JPEG compression doesn't eat the thin
        # lane lines. Cheap (one ~5 ms cv2.dilate on a 640x480) and only
        # touches the solo branch — the UI ``lanes`` tile keeps its
        # original sharp overlay.
        solo_np = overlay.clamp(0, 255).to(torch.uint8).cpu().numpy()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        solo_np = cv2.dilate(solo_np, kernel, iterations=1)
        return blended.cpu().numpy(), solo_np

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
        viz_lanes, viz_lanes_solo = self._viz_lanes(el_pred, frame_bgr)
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
            "viz_lanes_solo": viz_lanes_solo,
            "viz_depth": viz_depth,
            "viz_seg": viz_seg,
            "viz_objects": viz_objects,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Atomic file writers (rename is atomic on same filesystem)
# ═══════════════════════════════════════════════════════════════════════════

def write_jpeg_atomic(path: Path, frame: np.ndarray,
                       quality: int = JPEG_QUALITY) -> None:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
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
    p.add_argument("--video", default=None,
                   help="Path to a video file to replay as the inference "
                        "input instead of opening live USB cameras. The "
                        "frames are letterboxed to the camera resolution and "
                        "fanned out across all four UI tiles, so downstream "
                        "consumers (autosteer, /state, MJPEG) behave exactly "
                        "as if a real camera produced them.")
    p.add_argument("--no-loop", action="store_true",
                   help="With --video, stop at end-of-file instead of "
                        "seeking back to frame 0.")
    p.add_argument("--narrow-fov-deg", type=float, default=None,
                   help="Center-crop the front_narrow stream so its "
                        "apparent horizontal FOV is this many degrees. "
                        "AutoSteer/EgoLanes prefer ~30°. Default: no crop.")
    p.add_argument("--narrow-source-fov-deg", type=float,
                   default=NARROW_SOURCE_FOV_DEG_DEFAULT,
                   help="Source horizontal FOV (degrees) used to compute "
                        "the crop ratio for --narrow-fov-deg. Set this to "
                        "the actual lens FOV (live) or the recording "
                        f"camera FOV (video). Default: {NARROW_SOURCE_FOV_DEG_DEFAULT}°.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    frames_dir = Path(args.frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    state_path = Path(args.state_file)
    state_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute the narrow-FOV crop ratio once so both the live-camera and
    # video-replay branches can hand it to the front_narrow reader.
    narrow_crop_ratio = 1.0
    if args.narrow_fov_deg is not None:
        import math
        target = max(1.0, min(170.0, args.narrow_fov_deg))
        source = max(1.0, min(170.0, args.narrow_source_fov_deg))
        if target >= source:
            print(f"[fov] WARN: target {target}° >= source {source}° — "
                  "no crop applied")
        else:
            narrow_crop_ratio = (math.tan(math.radians(target / 2.0))
                                  / math.tan(math.radians(source / 2.0)))
            print(f"[fov] front_narrow center-crop {target:.1f}°/"
                  f"{source:.1f}° → ratio {narrow_crop_ratio:.3f}")

    video_source: VideoFileSource | None = None
    readers: list = []  # CameraReader | VideoSlotReader, duck-typed
    if args.video:
        video_path = Path(args.video).expanduser()
        if not video_path.exists():
            print(f"ERROR: --video path does not exist: {video_path}")
            return 1
        print(f"[video] opening {video_path}")
        try:
            video_source = VideoFileSource(video_path, loop=not args.no_loop)
        except RuntimeError as e:
            print(f"ERROR: {e}")
            return 1
        video_source.start()
        duration_s = (video_source.total_frames / video_source.native_fps
                      if video_source.native_fps > 0 else 0.0)
        print(f"[video] source: {video_source.src_w}x{video_source.src_h} "
              f"@ {video_source.native_fps:.2f} fps, "
              f"{video_source.total_frames} frames "
              f"(~{duration_s:.1f}s){'  loop' if not args.no_loop else ''}")
        for slug in SLUGS:
            crop = narrow_crop_ratio if slug == INFERENCE_SLUG else 1.0
            r = VideoSlotReader(video_source, slug, crop_ratio=crop)
            readers.append(r)
            extra = (f"  (narrow-fov crop {crop:.3f})"
                     if crop < 1.0 else "")
            print(f"[video] {slug:<12} -> {video_path.name} (replay){extra}")
        # Wait for the first decoded frame so inference doesn't trip over
        # an empty source on its very first iteration.
        wait_t0 = time.monotonic()
        while video_source.latest() is None:
            if time.monotonic() - wait_t0 > 5.0:
                print("ERROR: video produced no frames within 5s")
                video_source.stop()
                return 1
            time.sleep(0.05)
        print("[video] first frame decoded, ready")
    else:
        print("[cams] discovering v4l2 devices...")
        indices = args.indices or discover_v4l2_indices(count=len(SLUGS))
        print(f"[cams] indices: {indices}")
        if not indices:
            print("ERROR: no cameras found. Try: v4l2-ctl --list-devices")
            return 1

        for idx, slug in zip(indices, SLUGS):
            cap = open_camera(idx)
            if cap is None:
                print(f"[cams] WARN: idx {idx} ({slug}) failed to open")
                continue
            crop = narrow_crop_ratio if slug == INFERENCE_SLUG else 1.0
            r = CameraReader(cap, slug, crop_ratio=crop)
            r.start()
            readers.append(r)
            extras = []
            if slug in CAMERA_ORIENTATION_FIX:
                extras.append(f"flip={CAMERA_ORIENTATION_FIX[slug]}")
            if crop < 1.0:
                extras.append(f"narrow-fov crop {crop:.3f}")
            tail = f"  ({', '.join(extras)})" if extras else ""
            print(f"[cams] {slug:<12} -> /dev/video{idx}{tail}")

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
        "viz_lanes": None, "viz_lanes_solo": None,
        "viz_depth": None, "viz_seg": None, "viz_objects": None,
    }

    print("[run] publishing to", frames_dir, "and", state_path)
    last_video_log_t = 0.0
    try:
        while True:
            loop_t = time.monotonic()

            # Periodic video progress log so the operator can see the clip
            # advancing in /tmp/autoware_infer.log without scraping JPEG
            # mtimes. ~1 Hz is enough.
            if video_source is not None and loop_t - last_video_log_t > 1.0:
                last_video_log_t = loop_t
                pos = video_source.video_pos
                total = max(video_source.total_frames, 1)
                t_s = pos / max(video_source.native_fps, 1.0)
                infer_fps = (1.0 / (sum(infer_times) / len(infer_times))
                             if infer_times else 0.0)
                print(f"[video] frame {pos}/{total} "
                      f"(t={t_s:6.2f}s, loops={video_source.loops}) "
                      f"steer={last['steer_smoothed']:+6.2f}° "
                      f"infer_fps={infer_fps:5.1f}")

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
                # ``lanes_solo`` rides along here too — same publish
                # cadence, same atomic-rename guarantee, but it's an
                # auxiliary stream consumed by scene.js (not a UI tile).
                if pipe is not None and not args.no_viz:
                    # (slug, last-key, jpeg-quality). lanes_solo gets a
                    # higher quality because it's the additive-blend
                    # texture for the 3D scene; default 72 mangles the
                    # thin colored strokes on the otherwise-black field
                    # into near-invisibility on the road.
                    for slug, key, qual in [
                        ("lanes", "viz_lanes", JPEG_QUALITY),
                        ("depth", "viz_depth", JPEG_QUALITY),
                        ("seg", "viz_seg", JPEG_QUALITY),
                        ("objects", "viz_objects", JPEG_QUALITY),
                        ("lanes_solo", "viz_lanes_solo", 92),
                    ]:
                        viz = last.get(key)
                        if viz is not None:
                            write_jpeg_atomic(
                                frames_dir / f"{slug}.jpg", viz, qual,
                            )
                next_jpeg_t = loop_t + jpeg_period

            # ── 3) Publish state.
            mean_dt = sum(infer_times) / len(infer_times) if infer_times else 0.0
            state_payload = {
                "steer_deg": float(last["steer_smoothed"]),
                "steer_deg_raw": float(last["steer_raw"]),
                "active_cam": active_slug if pipe is not None else None,
                "inference": pipe is not None,
                "viz": (pipe is not None and not args.no_viz),
                "viz_streams": list(VIZ_SLUGS) if (pipe is not None and not args.no_viz) else [],
                "object_count": int(last["object_count"]),
                "fps": (1.0 / mean_dt) if mean_dt > 0 else 0.0,
                "cams": [r.slug for r in readers],
                "source": "video" if video_source is not None else "camera",
                "ts": time.time(),
            }
            if video_source is not None:
                state_payload["video"] = {
                    "path": str(video_source.path),
                    "name": video_source.path.name,
                    "frame": video_source.video_pos,
                    "total_frames": video_source.total_frames,
                    "native_fps": round(video_source.native_fps, 2),
                    "loops": video_source.loops,
                    "loop": not args.no_loop,
                }
            write_state_atomic(state_path, state_payload)

            if pipe is None:
                time.sleep(0.02)
    except KeyboardInterrupt:
        print("\n[run] interrupted.")
    finally:
        for r in readers:
            r.stop()
        if video_source is not None:
            video_source.stop()
        print("[run] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
