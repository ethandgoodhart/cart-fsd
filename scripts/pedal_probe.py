#!/usr/bin/env python3
"""Pedal probe: drive Mega G/B targets from PS5 triggers and print STAT replies.

One-off diagnostic — no ODrive, no UI. Squeeze R2/L2 and watch whether the
firmware's actual pot readings (g, b) track the targets (tg, tb).
"""
from __future__ import annotations

import os, sys, time, threading
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Keep SDL polling when the pygame window is not focused.
os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")

import pygame
import serial

from limits import BRAKE_POT_MAX, PS5_GAS_LIMIT, effective_gas_cap

AXIS_L2, AXIS_R2 = 4, 5
TRIG_L2_FS, TRIG_R2_FS = 0.91, 1.00

def trig(js, axis, fs):
    if js.get_numaxes() <= axis: return 0.0
    v = max(0.0, min(1.0, (js.get_axis(axis) + 1.0) / 2.0))
    return min(1.0, v / fs) if fs > 0 else 0.0

def reader(ser, stop, state):
    while not stop.is_set():
        try:
            line = ser.readline().decode(errors="replace").strip()
        except Exception:
            break
        if not line: continue
        if line.startswith("STAT"):
            try:
                kv = dict(p.split("=") for p in line.split(",")[1:])
                state["g"]  = float(kv.get("g", 0))
                state["b"]  = float(kv.get("b", 0))
                state["tg"] = float(kv.get("tg", 0))
                state["tb"] = float(kv.get("tb", 0))
                state["fs"] = kv.get("fs", "?")
            except Exception: pass
        else:
            print(f"[mega] {line}")

def main():
    gas_cap = effective_gas_cap(PS5_GAS_LIMIT)
    print(f"[probe] gas_cap={gas_cap:.3f}  brake_cap={BRAKE_POT_MAX:.3f}")
    port = "/dev/cu.usbmodem21401"
    ser = serial.Serial(port, 115200, timeout=0.2)
    print(f"[probe] opened {port}; Mega boot delay...")
    time.sleep(2.2)

    pygame.init(); pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("no controller"); return 1
    js = pygame.joystick.Joystick(0); js.init()
    print(f"[probe] controller: {js.get_name()}")
    pygame.display.set_mode((320, 80))
    pygame.display.set_caption("pedal probe (focus me, Esc to quit)")

    state = {"g":0,"b":0,"tg":0,"tb":0,"fs":"?"}
    stop = threading.Event()
    th = threading.Thread(target=reader, args=(ser, stop, state), daemon=True)
    th.start()

    clock = pygame.time.Clock()
    last_print = 0.0
    running = True
    try:
        while running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT: running = False
                elif e.type == pygame.KEYDOWN and e.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
            l2 = trig(js, AXIS_L2, TRIG_L2_FS)
            r2 = trig(js, AXIS_R2, TRIG_R2_FS)
            gas = r2 * gas_cap
            brake = l2 * BRAKE_POT_MAX
            try:
                ser.write(f"G {gas:.3f}\nB {brake:.3f}\n".encode())
            except Exception as ex:
                print(f"[probe] write fail: {ex}"); break
            now = time.monotonic()
            if now - last_print > 0.1:
                last_print = now
                print(f"R2={r2:.2f} L2={l2:.2f}  ->  tg={state['tg']:.3f} g={state['g']:.3f}  |  tb={state['tb']:.3f} b={state['b']:.3f}  fs={state['fs']}")
            clock.tick(50)
    finally:
        try: ser.write(b"S\n")
        except Exception: pass
        stop.set()
        time.sleep(0.15)
        ser.close()
        pygame.quit()
    return 0

if __name__ == "__main__":
    sys.exit(main())
