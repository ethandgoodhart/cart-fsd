#!/usr/bin/env python3
"""
ps5_controller_test.py — Read + visualize PS5 DualSense inputs (NO cart output).

This script ONLY reads the controller and draws a UI. It does not touch
the ODrive, linear actuators, or any cart control at all. Use it to
confirm the controller is paired and that the axes behave the way we
expect before wiring it to `main.py`.

Inputs displayed:
  - L2 trigger (0.0 .. 1.0)
  - R2 trigger (0.0 .. 1.0)  -> intended throttle source
  - Left stick X/Y           -> intended steering source
  - Left stick magnitude     -> "how fast to turn" (0.0 .. 1.0)
  - Left stick angle

Pair the controller first (macOS: System Settings -> Bluetooth, hold
PS + Create on the controller until the lightbar flashes, pair, then
run this script). USB-C also works.

Usage:
    uv run python scripts/ps5_controller_test.py
    uv run python scripts/ps5_controller_test.py --list  # list detected controllers
"""

import argparse
import math
import os
import sys

# Keep SDL polling the controller when the pygame window is not focused
# (macOS freezes axes on focus loss otherwise).
os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")

import pygame

# SDL game-controller axis indices. Using the Controller API means these
# are consistent across platforms / drivers, so the DualSense maps the
# same way on macOS over USB-C or Bluetooth.
AXIS_LEFT_X = 0
AXIS_LEFT_Y = 1
AXIS_RIGHT_X = 2
AXIS_RIGHT_Y = 3
AXIS_L2 = 4
AXIS_R2 = 5

# Raw axis range is int16. Triggers report 0..32767 (released..pressed);
# sticks report -32768..32767.
AXIS_MAX = 32767.0

# Deadzone for the left stick (fraction of full deflection). Anything
# inside this is treated as zero so resting drift doesn't look like input.
STICK_DEADZONE = 0.08

# Per-trigger calibration: some DualSense units (and some driver stacks on
# macOS) don't quite report full-scale on the triggers. If you squeeze L2
# all the way and it tops out at 0.91, set TRIGGER_MAX_L2 = 0.91 and the
# displayed value will be rescaled so full-squeeze reads 1.00.
TRIGGER_MAX_L2 = 0.91
TRIGGER_MAX_R2 = 1.00

# Window geometry.
WINDOW_W, WINDOW_H = 900, 560
FPS = 60

# Colors.
BG = (18, 20, 26)
PANEL = (30, 33, 42)
TEXT = (230, 232, 238)
MUTED = (130, 135, 150)
ACCENT = (90, 170, 255)
ACCENT_DIM = (45, 85, 130)
GOOD = (120, 220, 140)
WARN = (240, 180, 90)
BAD = (235, 100, 100)


def apply_deadzone(value: float, deadzone: float) -> float:
    """Rescale a stick value so (|v| <= deadzone) -> 0 and the remainder
    is re-expanded to -1..1 for smooth behavior past the deadzone."""
    if abs(value) <= deadzone:
        return 0.0
    sign = 1.0 if value > 0 else -1.0
    return sign * (abs(value) - deadzone) / (1.0 - deadzone)


def list_controllers() -> None:
    pygame.init()
    pygame.joystick.init()
    count = pygame.joystick.get_count()
    if count == 0:
        print("No controllers detected. Pair the PS5 controller first.")
        return
    print(f"{count} controller(s) detected:")
    for i in range(count):
        js = pygame.joystick.Joystick(i)
        js.init()
        print(f"  [{i}] {js.get_name()}  guid={js.get_guid()}  axes={js.get_numaxes()}")


def draw_trigger_bar(surface, font, label, value, x, y, w, h):
    """Vertical-ish horizontal bar for a 0..1 trigger value."""
    pygame.draw.rect(surface, PANEL, (x, y, w, h), border_radius=8)
    fill_w = int(w * max(0.0, min(1.0, value)))
    color = ACCENT if value < 0.95 else GOOD
    pygame.draw.rect(surface, color, (x, y, fill_w, h), border_radius=8)
    pygame.draw.rect(surface, ACCENT_DIM, (x, y, w, h), width=2, border_radius=8)

    label_surf = font.render(label, True, TEXT)
    val_surf = font.render(f"{value:.2f}", True, TEXT)
    surface.blit(label_surf, (x, y - 28))
    surface.blit(val_surf, (x + w - val_surf.get_width(), y - 28))


def draw_stick_pad(surface, font, label, lx, ly, magnitude, angle_deg, cx, cy, radius):
    """2D pad showing stick position + crosshair + magnitude ring."""
    pygame.draw.circle(surface, PANEL, (cx, cy), radius)
    pygame.draw.circle(surface, ACCENT_DIM, (cx, cy), radius, width=2)
    pygame.draw.circle(surface, ACCENT_DIM, (cx, cy), int(radius * STICK_DEADZONE), width=1)
    pygame.draw.line(surface, ACCENT_DIM, (cx - radius, cy), (cx + radius, cy), width=1)
    pygame.draw.line(surface, ACCENT_DIM, (cx, cy - radius), (cx, cy + radius), width=1)

    # Filled magnitude ring.
    mag_r = int(radius * min(1.0, magnitude))
    if mag_r > 2:
        pygame.draw.circle(surface, (40, 70, 100), (cx, cy), mag_r, width=2)

    dot_x = int(cx + lx * radius)
    dot_y = int(cy + ly * radius)
    pygame.draw.line(surface, ACCENT, (cx, cy), (dot_x, dot_y), width=3)
    pygame.draw.circle(surface, ACCENT, (dot_x, dot_y), 10)
    pygame.draw.circle(surface, TEXT, (dot_x, dot_y), 10, width=2)

    label_surf = font.render(label, True, TEXT)
    surface.blit(label_surf, (cx - radius, cy - radius - 28))


def draw_text_block(surface, font_big, font_small, lines, x, y, line_height=28):
    for i, (label, value, color) in enumerate(lines):
        label_surf = font_small.render(label, True, MUTED)
        val_surf = font_big.render(value, True, color)
        surface.blit(label_surf, (x, y + i * line_height))
        surface.blit(val_surf, (x + 180, y + i * line_height - 4))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List controllers and exit")
    parser.add_argument("--index", type=int, default=0, help="Controller index to use (default 0)")
    args = parser.parse_args()

    if args.list:
        list_controllers()
        return

    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        print("ERROR: No controllers detected.")
        print("  - macOS: System Settings -> Bluetooth. Hold PS + Create on the")
        print("    controller until the lightbar double-flashes, then pair.")
        print("  - Or plug in via USB-C.")
        print("Run with --list once paired to confirm.")
        sys.exit(1)

    if args.index >= pygame.joystick.get_count():
        print(f"ERROR: --index {args.index} but only {pygame.joystick.get_count()} controller(s) connected.")
        sys.exit(1)

    js = pygame.joystick.Joystick(args.index)
    js.init()
    name = js.get_name()
    print(f"Using controller [{args.index}]: {name}")
    if "DualSense" not in name and "Wireless Controller" not in name and "PS5" not in name:
        print(f"  Warning: name doesn't look like a DualSense. Axes may be mapped differently.")

    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("PS5 Controller Test (read-only)")
    clock = pygame.time.Clock()
    font_big = pygame.font.SysFont("Menlo", 28, bold=True)
    font_med = pygame.font.SysFont("Menlo", 20)
    font_small = pygame.font.SysFont("Menlo", 16)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False
            elif event.type == pygame.JOYDEVICEREMOVED:
                print("Controller disconnected.")
                running = False

        num_axes = js.get_numaxes()
        # Triggers: raw SDL convention is -1 released .. +1 fully pressed.
        # Normalize to 0..1 so it reads like a throttle.
        l2_raw = js.get_axis(AXIS_L2) if num_axes > AXIS_L2 else -1.0
        r2_raw = js.get_axis(AXIS_R2) if num_axes > AXIS_R2 else -1.0
        l2 = max(0.0, min(1.0, (l2_raw + 1.0) / 2.0))
        r2 = max(0.0, min(1.0, (r2_raw + 1.0) / 2.0))
        # Rescale so each trigger's observed max maps to 1.00.
        l2 = min(1.0, l2 / TRIGGER_MAX_L2) if TRIGGER_MAX_L2 > 0 else 0.0
        r2 = min(1.0, r2 / TRIGGER_MAX_R2) if TRIGGER_MAX_R2 > 0 else 0.0

        lx_raw = js.get_axis(AXIS_LEFT_X) if num_axes > AXIS_LEFT_X else 0.0
        ly_raw = js.get_axis(AXIS_LEFT_Y) if num_axes > AXIS_LEFT_Y else 0.0
        lx = apply_deadzone(lx_raw, STICK_DEADZONE)
        ly = apply_deadzone(ly_raw, STICK_DEADZONE)
        magnitude = min(1.0, math.hypot(lx, ly))
        # Angle: 0° = right, 90° = down (screen convention), so flip Y for a
        # human-readable "0° = up, 90° = right" compass-ish angle.
        angle_deg = math.degrees(math.atan2(lx, -ly)) if magnitude > 0 else 0.0

        screen.fill(BG)

        title = font_big.render("PS5 Controller — read-only (no cart output)", True, TEXT)
        screen.blit(title, (24, 20))
        sub = font_small.render(f"device: {name}", True, MUTED)
        screen.blit(sub, (24, 54))

        # Triggers
        draw_trigger_bar(screen, font_med, "L2 (brake?)", l2, 40, 130, 360, 40)
        draw_trigger_bar(screen, font_med, "R2 (throttle)", r2, 40, 220, 360, 40)

        # Stick pad
        pad_cx, pad_cy, pad_r = 640, 260, 150
        draw_stick_pad(screen, font_med, "Left stick (steering)", lx, ly, magnitude, angle_deg, pad_cx, pad_cy, pad_r)

        # Numeric readouts
        mag_color = GOOD if magnitude > 0.05 else MUTED
        r2_color = GOOD if r2 > 0.05 else MUTED
        l2_color = WARN if l2 > 0.05 else MUTED
        readouts = [
            ("R2 throttle",   f"{r2:5.2f}",                r2_color),
            ("L2 brake",      f"{l2:5.2f}",                l2_color),
            ("stick X",       f"{lx:+.2f}  (raw {lx_raw:+.2f})", TEXT),
            ("stick Y",       f"{ly:+.2f}  (raw {ly_raw:+.2f})", TEXT),
            ("turn magnitude",f"{magnitude:5.2f}",          mag_color),
            ("turn angle",    f"{angle_deg:+6.1f}°",        TEXT),
        ]
        draw_text_block(screen, font_med, font_small, readouts, 40, 320)

        hint = font_small.render("Esc / Q to quit. Values are read-only — not sent to the cart.", True, MUTED)
        screen.blit(hint, (24, WINDOW_H - 28))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
