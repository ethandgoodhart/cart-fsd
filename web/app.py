import json
import os
import time

from flask import Flask, Response, abort, jsonify, render_template

STATE_FILE = os.environ.get("CART_STATE_FILE", "/tmp/cart_state.json")
AUTOWARE_STATE_FILE = os.environ.get(
    "AUTOWARE_STATE_FILE", "/tmp/autoware_state.json"
)
FRAMES_DIR = os.environ.get("CART_FRAMES_DIR", "/tmp/cart_frames")
STATE_FRESH_S = 1.0
AUTOWARE_FRESH_S = 0.5  # autoware_infer writes at ~15 Hz; >500 ms = stale

# Must match scripts/autoware_infer.py::ALL_STREAM_SLUGS. A request for
# any other slug returns 404 — never stream arbitrary paths off the
# filesystem. Order: 4 raw cameras, then 4 model-output viz tiles.
CAM_SLUGS = (
    "front_wide", "front_narrow", "left", "right",
    "lanes", "depth", "seg", "objects",
)
MJPEG_FPS = 15           # per-client frame rate
MJPEG_STALE_S = 2.0      # stop streaming if frame file hasn't updated

app = Flask(__name__)


def _load_json(path: str, fresh_s: float) -> tuple[dict, bool]:
    """Return (data, stale). Missing/corrupt file → empty dict + stale=True."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}, True
    stale = time.time() - float(data.get("ts", 0)) > fresh_s
    return data, stale


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/state")
def state():
    data, stale = _load_json(STATE_FILE, STATE_FRESH_S)
    if not data:
        data = {
            "mph": 0, "gas": 0, "brake": 0,
            "controller_connected": False,
            "arduino_connected": False,
            "motor_connected": False,
        }
    if stale:
        # ps5_drive.py only writes this file while it's running, so stale
        # state means the driver exited — usually a controller drop. Treat
        # everything as disconnected so the UI dots turn gray.
        data["mph"] = 0
        data["stale"] = True
        data["controller_connected"] = False
        data["arduino_connected"] = False
        data["motor_connected"] = False

    # Merge Autoware state into a sub-key so the UI can render the active
    # camera indicator + predicted angle + viz tiles without racing two
    # fetches. Field names mirror autoware_infer.py's state JSON 1:1
    # (running/stale are server-derived; everything else is pass-through).
    auto, auto_stale = _load_json(AUTOWARE_STATE_FILE, AUTOWARE_FRESH_S)
    data["autoware"] = {
        "running": bool(auto) and not auto_stale,
        "inference": bool(auto.get("inference", False)),
        "viz": bool(auto.get("viz", False)),
        "viz_streams": list(auto.get("viz_streams", [])),
        "object_count": int(auto.get("object_count", 0)),
        "steer_deg": float(auto.get("steer_deg", 0.0)) if auto else 0.0,
        "active_cam": auto.get("active_cam"),
        "fps": float(auto.get("fps", 0.0)) if auto else 0.0,
        "cams": auto.get("cams", []),
        "stale": auto_stale,
    }
    return jsonify(data)


def _frame_path(slug: str) -> str | None:
    if slug not in CAM_SLUGS:
        return None
    return os.path.join(FRAMES_DIR, f"{slug}.jpg")


@app.route("/cam/<slug>.jpg")
def cam_snapshot(slug: str):
    """Single most-recent frame as a plain JPEG. Useful for polling clients
    (e.g. `<img>` with ?t=timestamp). MJPEG endpoint below is preferred for
    continuous playback."""
    path = _frame_path(slug)
    if path is None:
        abort(404)
    try:
        with open(path, "rb") as f:
            buf = f.read()
    except FileNotFoundError:
        abort(404)
    return Response(buf, mimetype="image/jpeg", headers={
        "Cache-Control": "no-store, must-revalidate",
    })


def _mjpeg_stream(path: str):
    """Generator that yields multipart frames at MJPEG_FPS. Reads the file's
    mtime to skip re-sending the same JPEG; exits quietly if the file stops
    updating (autoware_infer died) so the browser reconnects cleanly.
    """
    boundary = b"--caddyframe"
    period = 1.0 / MJPEG_FPS
    last_mtime = 0.0
    last_yield = 0.0
    idle_start: float | None = None
    while True:
        try:
            mtime = os.path.getmtime(path)
        except FileNotFoundError:
            # No frames yet — back off and try again. autoware_infer might
            # still be loading models on startup.
            time.sleep(0.2)
            continue
        now = time.monotonic()
        if mtime != last_mtime and now - last_yield >= period:
            try:
                with open(path, "rb") as f:
                    buf = f.read()
            except FileNotFoundError:
                time.sleep(0.05)
                continue
            last_mtime = mtime
            last_yield = now
            idle_start = None
            yield (boundary + b"\r\n"
                   b"Content-Type: image/jpeg\r\n"
                   b"Content-Length: " + str(len(buf)).encode() + b"\r\n\r\n"
                   + buf + b"\r\n")
        else:
            # Detect the producer having died — don't hold the socket open
            # forever rebroadcasting the last frame.
            if idle_start is None:
                idle_start = now
            elif now - idle_start > MJPEG_STALE_S:
                return
            time.sleep(period / 2)


@app.route("/cam/<slug>.mjpg")
def cam_mjpeg(slug: str):
    path = _frame_path(slug)
    if path is None:
        abort(404)
    return Response(_mjpeg_stream(path),
                     mimetype="multipart/x-mixed-replace; boundary=caddyframe",
                     headers={"Cache-Control": "no-store, must-revalidate"})


if __name__ == "__main__":
    # threaded=True so multiple MJPEG clients (4 cams × N browsers) don't
    # serialize on the dev server. Still a dev server — put it behind nginx
    # for anything other than local use.
    app.run(host="127.0.0.1", port=5050, debug=False, threaded=True)
