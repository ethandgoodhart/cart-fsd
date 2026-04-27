#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

PROFILE="$(mktemp -d)"
STATE_FILE="/tmp/cart_state.json"
AUTOWARE_STATE_FILE="/tmp/autoware_state.json"
FRAMES_DIR="/tmp/cart_frames"

# --mockspeed=N (or --mockspeed N) short-circuits the real PS5/Arduino/ODrive
# stack and runs scripts/mock_state.py instead, so the UI animates (lane
# dashes, wheel, pedals, green dots) with a synthetic MPH. Handy for UI
# work when the hardware isn't plugged in.
# --autosteer turns on the Autoware inference sidecar + ps5_drive --autosteer.
MOCK_SPEED=""
AUTOSTEER=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mockspeed=*) MOCK_SPEED="${1#*=}"; shift ;;
        --mockspeed)   MOCK_SPEED="$2"; shift 2 ;;
        --autosteer)   AUTOSTEER=1; shift ;;
        *) echo "start.sh: unknown arg $1" >&2; exit 2 ;;
    esac
done

cleanup() {
    kill "$SRV" 2>/dev/null || true
    kill "$DRIVE_LOOP" 2>/dev/null || true
    kill "$MOCK_PID" 2>/dev/null || true
    kill "$AUTOWARE_PID" 2>/dev/null || true
    pkill -P "$DRIVE_LOOP" 2>/dev/null || true
    pkill -f "scripts/ps5_drive.py" 2>/dev/null || true
    pkill -f "scripts/mock_state.py" 2>/dev/null || true
    pkill -f "scripts/autoware_infer.py" 2>/dev/null || true
    rm -rf "$PROFILE"
    rm -f "$STATE_FILE" "$STATE_FILE.tmp"
    rm -f "$AUTOWARE_STATE_FILE" "$AUTOWARE_STATE_FILE.tmp"
    rm -rf "$FRAMES_DIR"
}
trap cleanup EXIT

export CART_STATE_FILE="$STATE_FILE"
export AUTOWARE_STATE_FILE
export CART_FRAMES_DIR="$FRAMES_DIR"
mkdir -p "$FRAMES_DIR"

# Autoware inference sidecar — always runs the full perception stack so
# the lane/depth/objects/seg viz tiles populate regardless of whether the
# operator is using --autosteer. The --autosteer flag is solely about
# whether ps5_drive USES Autoware's steering output to command the wheel.
# ./start.sh --no-autoware skips even loading the models if you need the
# UI without spinning up CUDA (development on a non-Jetson host, etc.).
AUTOWARE_ARGS=(--frames-dir "$FRAMES_DIR" --state-file "$AUTOWARE_STATE_FILE")
# autoware_infer.py needs torch — it runs on the system Python 3.12 that
# has the Jetson CUDA wheel, not PRODUCTION's uv-managed 3.13.
/usr/bin/python3 scripts/autoware_infer.py "${AUTOWARE_ARGS[@]}" \
    >>/tmp/autoware_infer.log 2>&1 &
AUTOWARE_PID=$!

# Web UI (Flask) — reads $CART_STATE_FILE for the MPH readout.
(cd web && exec python3 app.py) >/tmp/flask.log 2>&1 &
SRV=$!

if [[ -n "$MOCK_SPEED" ]]; then
    echo "[start] MOCK mode — mph=$MOCK_SPEED (skipping PS5/Arduino/ODrive)" >>/tmp/ps5_drive.log
    python3 scripts/mock_state.py --mph "$MOCK_SPEED" --state-file "$STATE_FILE" \
        >>/tmp/ps5_drive.log 2>&1 &
    MOCK_PID=$!
else
    # PS5 drive loop — ps5_drive.py needs a DualSense already paired before
    # pygame can init its joystick layer. Wait for one to show up in
    # /proc/bus/input/devices, launch the driver, and if the driver exits
    # (BT drop, out of range) loop back and wait for reconnect. State-file
    # staleness tells the web UI when the controller is gone.
    (
        while true; do
            while ! grep -qi -E '(sony|dualsense|wireless controller|ps5)' /proc/bus/input/devices 2>/dev/null; do
                sleep 1
            done
            echo "[start] controller detected — launching ps5_drive" >>/tmp/ps5_drive.log
            PS5_ARGS=(--headless --state-file "$STATE_FILE")
            if [[ -n "$AUTOSTEER" ]]; then
                PS5_ARGS+=(--autosteer --autosteer-state-file "$AUTOWARE_STATE_FILE")
            fi
            uv run python scripts/ps5_drive.py "${PS5_ARGS[@]}" >>/tmp/ps5_drive.log 2>&1 || true
            echo "[start] ps5_drive exited — waiting for controller to reconnect" >>/tmp/ps5_drive.log
            sleep 1
        done
    ) &
    DRIVE_LOOP=$!
fi

for i in {1..30}; do
    curl -s -o /dev/null http://127.0.0.1:5050 && break
    sleep 0.2
done

firefox --no-remote --new-instance --profile "$PROFILE" --kiosk http://127.0.0.1:5050
