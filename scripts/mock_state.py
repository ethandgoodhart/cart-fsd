#!/usr/bin/env python3
"""Write a synthetic cart state file for UI development.

Useful when the Arduino / ODrive / DualSense aren't hooked up but you
still want to see the web UI animate (lane dashes scrolling at the given
MPH, pedal bar lit, wheel turning, connection dots green).

    python3 scripts/mock_state.py --mph 18 --state-file /tmp/cart_state.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path


def write_state_file(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mph", type=float, default=15.0,
                    help="Mock speed in MPH (default 15).")
    ap.add_argument("--state-file", default="/tmp/cart_state.json")
    ap.add_argument("--hz", type=float, default=20.0)
    args = ap.parse_args()

    path = Path(args.state_file)
    period = 1.0 / args.hz
    t0 = time.monotonic()
    mph = max(0.0, args.mph)

    print(f"[mock] writing {path} @ {args.hz:.0f} Hz with mph={mph:.1f}")

    while True:
        t = time.monotonic() - t0
        # Gentle sinusoidal steering + throttle wiggle so the UI breathes.
        steer = math.sin(t * 0.4) * 12.0
        gas_frac = max(0.0, min(1.0, mph / 20.0))
        brake_frac = 0.0
        try:
            write_state_file(path, {
                "mode": "mock",
                "dry_run": True,
                "stick_steering": "mock",
                "lx": math.sin(t * 0.4) * 0.3,
                "l2": 0.0,
                "r2": gas_frac,
                "steer_deg": steer,
                "gas": gas_frac * 0.68,
                "brake": 0.0,
                "gas_cap": 0.45,
                "brake_max": 1.0,
                "gas_frac": gas_frac,
                "brake_frac": brake_frac,
                "mph": mph,
                "controller_connected": True,
                "arduino_connected": True,
                "motor_connected": True,
                "hb_age_s": 0.0,
                "hb_faulted": False,
                "hb_fault_reason": None,
                "ts": time.time(),
            })
        except Exception as e:
            print(f"[mock] write failed: {e!r}")
        time.sleep(period)


if __name__ == "__main__":
    raise SystemExit(main())
