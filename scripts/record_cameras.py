#!/usr/bin/env python3
"""Simple 4-camera viewer + recorder web UI.

Usage:
    python record_cameras.py
    python record_cameras.py --front-narrow 0 --front-wide 2 --cross-left 6 --cross-right 10

Open http://<jetson-ip>:8080 in a browser.
"""
import argparse
import datetime as dt
import threading
import time
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, render_template_string, request

CAM_NAMES = ["front-narrow", "front-wide", "cross-left", "cross-right"]

# From PRODUCTION/scripts/alpamayo_infer.py: front-narrow is mounted
# upside-down + y-mirrored on this cart, so flip code -1 (180°).
CAMERA_FLIP = {"front-narrow": -1}


class Camera:
    def __init__(self, name: str, device: int, width: int, height: int, fps: int):
        self.name = name
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.flip_code = CAMERA_FLIP.get(name)
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        if not self.cap.isOpened():
            print(f"[WARN] Could not open /dev/video{device} for {name}")
        self.lock = threading.Lock()
        self.frame = None
        self.writer: cv2.VideoWriter | None = None
        self.writer_thread: threading.Thread | None = None
        self.writer_stop = threading.Event()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            if self.flip_code is not None:
                frame = cv2.flip(frame, self.flip_code)
            with self.lock:
                self.frame = frame

    def get_jpeg(self) -> bytes | None:
        with self.lock:
            frame = None if self.frame is None else self.frame.copy()
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return buf.tobytes() if ok else None

    def _write_loop(self):
        # Write at a fixed wall-clock cadence so saved video duration ==
        # real-time duration. If the camera delivers slower than fps,
        # frames get duplicated; if faster, they get dropped. Either way
        # playback matches reality.
        #
        # Critical: when a tick is late we MUST NOT reset the baseline —
        # we let the loop burst-write to catch up so the total frame
        # count over T real seconds is exactly round(T * fps). Otherwise
        # the file ends up shorter than the recording and plays fast.
        period = 1.0 / self.fps
        next_tick = time.monotonic()
        while not self.writer_stop.is_set():
            with self.lock:
                frame = None if self.frame is None else self.frame
            if frame is not None and self.writer is not None:
                self.writer.write(frame)
            next_tick += period
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                self.writer_stop.wait(sleep_for)

    def start_recording(self, out_path: Path):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            str(out_path), fourcc, self.fps, (self.width, self.height)
        )
        self.writer_stop.clear()
        self.writer_thread = threading.Thread(target=self._write_loop, daemon=True)
        self.writer_thread.start()

    def stop_recording(self):
        if self.writer_thread is not None:
            self.writer_stop.set()
            self.writer_thread.join(timeout=2)
            self.writer_thread = None
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def release(self):
        self.running = False
        self.thread.join(timeout=1)
        self.stop_recording()
        self.cap.release()


app = Flask(__name__)
cameras: dict[str, Camera] = {}
state = {"recording": False, "folder": None, "started_at": None}
state_lock = threading.Lock()


PAGE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>Caddy Camera Recorder</title>
<style>
  body { background:#111; color:#eee; font-family: system-ui, sans-serif; margin:0; padding:16px; }
  h1 { margin: 0 0 12px 0; font-size: 20px; }
  .grid { display:grid; grid-template-columns: repeat(4, 1fr); gap:8px; }
  .cell { background:#000; border:1px solid #333; position:relative; }
  .cell img { width:100%; display:block; }
  .label { position:absolute; top:6px; left:8px; background:rgba(0,0,0,0.6); padding:2px 8px; font-size:13px; border-radius:3px; }
  .controls { margin-top:14px; display:flex; align-items:center; gap:12px; }
  button { font-size:18px; padding:10px 24px; border:0; border-radius:4px; cursor:pointer; }
  .rec { background:#c0392b; color:#fff; }
  .stop { background:#444; color:#fff; }
  .status { font-family: monospace; }
  .dot { display:inline-block; width:10px; height:10px; border-radius:50%; background:#666; margin-right:6px; vertical-align:middle; }
  .dot.on { background:#e74c3c; animation: pulse 1s infinite; }
  @keyframes pulse { 50% { opacity:0.3; } }
</style>
</head>
<body>
  <h1>Caddy Camera Recorder</h1>
  <div class="grid">
    {% for name in names %}
    <div class="cell">
      <div class="label">{{ name }}</div>
      <img src="/video_feed/{{ name }}" />
    </div>
    {% endfor %}
  </div>
  <div class="controls">
    <button id="btn" class="rec" onclick="toggle()">Record</button>
    <span class="status"><span id="dot" class="dot"></span><span id="status">idle</span></span>
  </div>
<script>
async function refresh() {
  const r = await fetch('/status'); const j = await r.json();
  const btn = document.getElementById('btn');
  const dot = document.getElementById('dot');
  const status = document.getElementById('status');
  if (j.recording) {
    btn.textContent = 'Stop'; btn.className = 'stop';
    dot.classList.add('on');
    status.textContent = 'REC  ' + j.folder + '   elapsed ' + j.elapsed + 's';
  } else {
    btn.textContent = 'Record'; btn.className = 'rec';
    dot.classList.remove('on');
    status.textContent = j.last_folder ? ('saved: ' + j.last_folder) : 'idle';
  }
}
async function toggle() {
  const r = await fetch('/status'); const j = await r.json();
  await fetch(j.recording ? '/stop' : '/start', { method:'POST' });
  refresh();
}
setInterval(refresh, 500); refresh();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE, names=CAM_NAMES)


@app.route("/video_feed/<name>")
def video_feed(name):
    if name not in cameras:
        return "unknown camera", 404
    cam = cameras[name]

    def gen():
        while True:
            jpeg = cam.get_jpeg()
            if jpeg is None:
                time.sleep(0.05)
                continue
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            )
            time.sleep(1 / 20)

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    with state_lock:
        elapsed = (
            int(time.time() - state["started_at"]) if state["started_at"] else 0
        )
        return jsonify(
            recording=state["recording"],
            folder=state["folder"],
            elapsed=elapsed,
            last_folder=state.get("last_folder"),
        )


@app.route("/start", methods=["POST"])
def start():
    with state_lock:
        if state["recording"]:
            return jsonify(ok=True)
        ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder = Path.cwd() / f"Caddy-Training-Data-{ts}"
        folder.mkdir(parents=True, exist_ok=True)
        for name, cam in cameras.items():
            cam.start_recording(folder / f"{name}.mp4")
        state["recording"] = True
        state["folder"] = str(folder)
        state["started_at"] = time.time()
        print(f"[REC] started -> {folder}")
        return jsonify(ok=True, folder=str(folder))


@app.route("/stop", methods=["POST"])
def stop():
    with state_lock:
        if not state["recording"]:
            return jsonify(ok=True)
        for cam in cameras.values():
            cam.stop_recording()
        folder = state["folder"]
        state["recording"] = False
        state["last_folder"] = folder
        state["folder"] = None
        state["started_at"] = None
        print(f"[REC] stopped -> {folder}")
        return jsonify(ok=True, folder=folder)


def parse_args():
    p = argparse.ArgumentParser()
    # Defaults per PRODUCTION/scripts/alpamayo_infer.py discovery order.
    p.add_argument("--front-narrow", type=int, default=0)
    p.add_argument("--front-wide", type=int, default=6)
    p.add_argument("--cross-left", type=int, default=2)
    p.add_argument("--cross-right", type=int, default=10)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    return p.parse_args()


def main():
    args = parse_args()
    devices = {
        "front-narrow": args.front_narrow,
        "front-wide": args.front_wide,
        "cross-left": args.cross_left,
        "cross-right": args.cross_right,
    }
    for name in CAM_NAMES:
        cameras[name] = Camera(
            name, devices[name], args.width, args.height, args.fps
        )
        print(f"[CAM] {name} -> /dev/video{devices[name]}")
    try:
        app.run(host=args.host, port=args.port, threaded=True, debug=False)
    finally:
        for cam in cameras.values():
            cam.release()


if __name__ == "__main__":
    main()
