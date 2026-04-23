#!/usr/bin/env python3
"""
ps5_steer.py — velocity-control the cart's steering wheel from a DualSense.

Left stick X -> motor velocity. How far you push the stick sets the
rotation speed; release the stick and the wheel decelerates back to
zero. This is velocity control, not position follow — contrast with
``ps5_drive.py --mode steering`` which maps stick X directly to a column
angle setpoint.

Ramp behavior mirrors ``main.py``'s trapezoidal trajectory, just in
velocity space: ODrive's ``VEL_RAMP`` input mode drives the actual
velocity toward the target at a fixed acceleration (``--accel``, turns/s²).

Soft-limit behavior: if the steering column is already at (or past)
``STEERING_MAX_DEG`` / ``STEERING_MIN_DEG`` from ``limits.py`` and the
stick asks for more in that direction, the commanded velocity is pinned
to zero in that direction. You can always reverse out of a limit.

Sign convention: positive motor turns = positive column degrees (set by
how the HTD 5M belt is installed — see ``docs/steering.md``). We default
to ``stick_right -> motor_positive``. If on first run the wheel turns
the wrong way, pass ``--invert`` to flip.

Buttons:
    Triangle  — toggle "return to 0°" homing. While homing, stick is
                ignored; homing auto-completes when within 0.5° of zero
                and stopped. Grab the stick or press Triangle again to
                abort. Override button index with ``--home-button``.

Usage:
    uv run python scripts/ps5_steer.py                  # default: 3 turns/s cap
    uv run python scripts/ps5_steer.py --invert         # flip steering sign
    uv run python scripts/ps5_steer.py --max-vel 5.0    # bench-test cap
    uv run python scripts/ps5_steer.py --accel 15       # gentler ramp
    uv run python scripts/ps5_steer.py --home-button 2  # use Square instead of Triangle
    uv run python scripts/ps5_steer.py --dry-run        # no ODrive, just UI

    # If the motor lags commanded velocity through a hard spot (watch
    # the UI's "tracking err" row turn amber/red), raise these:
    uv run python scripts/ps5_steer.py --current-lim 20
    uv run python scripts/ps5_steer.py --current-lim 20 --vel-integrator-gain 0.6
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Keep SDL delivering joystick input when the pygame window is not focused
# (macOS freezes axes on focus loss otherwise). Must be set before ``import
# pygame`` so SDL picks it up at initialization.
os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")

import pygame

from limits import (
    STEERING_MAX_DEG,
    STEERING_MIN_DEG,
    motor_turns_to_steering_deg,
    steering_deg_to_motor_turns,
)

# --- Controller mapping ----------------------------------------------------
AXIS_LEFT_X = 0
STICK_DEADZONE = 0.08

# DualSense face buttons on SDL 2.28+ (macOS / Linux): Cross=0, Circle=1,
# Square=2, Triangle=3. Override with --home-button if your platform maps
# differently.
BUTTON_TRIANGLE = 3

# --- Homing behavior ("return to 0°" when Triangle is pressed) -------------
# Auto-complete once both conditions are met: close enough in angle and
# nearly stopped. Below the angle tol alone is not enough — the motor
# may still be coasting through zero under VEL_RAMP inertia.
HOMING_ANGLE_TOL_DEG = 0.5
HOMING_VEL_TOL = 0.1       # turns/s

# If the user grabs the stick harder than this (raw, pre-deadzone) while
# homing, assume they want manual control back. Higher than the normal
# deadzone so small drift on a released stick doesn't auto-cancel.
HOMING_STICK_CANCEL = 0.30

# --- Steering velocity defaults (motor-side, same units as main.py) --------
# 3.0 turns/s on the motor is 1.0 wheel-turn/s through the 3:1 belt (one
# full steering-wheel revolution per second at full stick) — fast enough
# for lock-to-lock in under a second but not so fast that handling feels
# twitchy. Raise with --max-vel when bench-testing.
DEFAULT_MAX_VEL = 3.0    # turns/s

# VEL_RAMP acceleration. At 25 turns/s^2 the ramp from standstill to
# DEFAULT_MAX_VEL takes ~0.2 s — crisp-feeling without being jerky. This
# is also the *decel* the limit-predictor assumes when it decides how
# soon to start braking before a soft stop.
DEFAULT_ACCEL = 25.0     # turns/s^2

# The ODrive's own ``vel_limit`` refuses to track any command above it.
# Set it a bit above --max-vel so an in-flight command at the cap doesn't
# flap against the limit and raise errors.
VEL_LIMIT_MARGIN = 1.5   # turns/s

# Safety factor on accel when computing "max safe velocity toward a soft
# limit". We command the ODrive's ramp to chase a velocity target that
# can be decelerated in the remaining distance using only 85% of the
# configured accel — the extra 15% is headroom for the lag between the
# target velocity and the actual (measured) velocity in VEL_RAMP mode,
# so we never cross a soft stop even when the ramp hasn't caught up yet.
BRAKE_ACCEL_FRACTION = 0.85

CONTROL_HZ = 100.0

# --- UI --------------------------------------------------------------------
WINDOW_W, WINDOW_H = 640, 380
BG = (18, 20, 26)
TEXT = (230, 232, 238)
MUTED = (130, 135, 150)
ACCENT = (90, 170, 255)
GOOD = (120, 220, 140)
WARN = (240, 180, 90)
BAD = (235, 100, 100)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def apply_deadzone(value: float, deadzone: float) -> float:
    """|v| <= deadzone -> 0, rest re-expanded to -1..1 past the deadzone."""
    if abs(value) <= deadzone:
        return 0.0
    sign = 1.0 if value > 0 else -1.0
    return sign * (abs(value) - deadzone) / (1.0 - deadzone)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def init_controller(index: int):
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("ERROR: no controllers detected. Pair the DualSense first")
        print("  (confirm with: uv run python scripts/ps5_controller_test.py --list).")
        sys.exit(1)
    if index >= pygame.joystick.get_count():
        print(
            f"ERROR: --index {index} but only "
            f"{pygame.joystick.get_count()} controller(s) connected."
        )
        sys.exit(1)
    js = pygame.joystick.Joystick(index)
    js.init()
    name = js.get_name()
    print(f"[ps5] using [{index}]: {name}")
    if not any(tag in name for tag in ("DualSense", "Wireless Controller", "PS5")):
        print("[ps5] WARNING: name doesn't look like a DualSense; axes may differ.")
    return js


# ---------------------------------------------------------------------------
# ODrive wrapper
# ---------------------------------------------------------------------------

class SteeringVel:
    """ODrive S1 in ``VELOCITY_CONTROL`` + ``VEL_RAMP`` input mode.

    Configures vel ramp rate + vel limit, arms, and exposes a simple
    ``set_vel(turns_per_sec)`` that callers can spam at the control
    rate. The ramp planner handles accel/decel on the ODrive side so
    there's no time-stepped math here.
    """

    def __init__(
        self,
        max_vel: float,
        accel: float,
        dry_run: bool = False,
        current_lim: float | None = None,
        current_hard_max: float | None = None,
        vel_gain: float | None = None,
        vel_integrator_gain: float | None = None,
        lift_torque_cap: bool = False,
    ):
        self.max_vel = max_vel
        self.accel = accel
        self.dry_run = dry_run
        self.odrv = None
        self.axis = None
        self.start_pos = 0.0
        self._AxisState = None
        self._InputMode = None
        # Populated by _apply_tuning after connect; read by the UI for live diagnostics.
        self._iq_getter = None
        self._current_soft_max = float("nan")

        if dry_run:
            print("[steering] dry-run: ODrive will NOT be opened.")
            return

        import odrive  # type: ignore
        from odrive.enums import (  # type: ignore
            AxisState, ControlMode, InputMode,
        )

        self._AxisState = AxisState
        self._InputMode = InputMode

        print("[steering] connecting to ODrive...")
        self.odrv = odrive.find_any(timeout=10)
        self.axis = self.odrv.axis0
        print(
            f"[steering] connected: serial {self.odrv.serial_number}, "
            f"bus {self.odrv.vbus_voltage:.1f}V"
        )

        if self.axis.active_errors != 0:
            print(f"[steering] clearing active errors: {self.axis.active_errors}")
            self.odrv.clear_errors()
            time.sleep(0.3)

        # Be in IDLE before rewriting control/input mode. Switching
        # control_mode on an armed axis is defined but has raised errors
        # in the past when going between POSITION and VELOCITY, so be
        # conservative — this is a one-time boot step.
        self.axis.requested_state = AxisState.IDLE
        time.sleep(0.2)

        # Read + optionally override tracking-related tuning. Non-fatal:
        # the object tree differs between firmware generations (legacy
        # ODrive 3.x uses axis.motor.config.current_lim, S1/Pro on 0.6.x
        # uses axis.config.motor.current_soft_max), so we probe for each
        # field and skip what isn't there. The script still runs with
        # whatever the ODrive was calibrated with if we can't override.
        self._apply_tuning(
            current_lim, current_hard_max, vel_gain, vel_integrator_gain,
            lift_torque_cap=lift_torque_cap,
        )

        self.axis.controller.config.control_mode = ControlMode.VELOCITY_CONTROL
        self.axis.controller.config.input_mode = InputMode.VEL_RAMP
        self.axis.controller.config.vel_ramp_rate = accel
        self.axis.controller.config.vel_limit = max_vel + VEL_LIMIT_MARGIN

        # Preset the command to zero BEFORE arming — otherwise whatever
        # leftover input_vel was there from a previous run gets tracked
        # immediately on entering CLOSED_LOOP_CONTROL.
        self.axis.controller.input_vel = 0.0

        self.axis.requested_state = AxisState.CLOSED_LOOP_CONTROL
        time.sleep(0.3)
        if self.axis.current_state != AxisState.CLOSED_LOOP_CONTROL:
            raise RuntimeError(
                f"ODrive failed to enter closed-loop control: "
                f"state={self.axis.current_state}, disarm={self.axis.disarm_reason}"
            )

        self.start_pos = self.axis.pos_estimate
        print(
            f"[steering] VEL_RAMP armed: max_vel={max_vel:.1f} t/s, "
            f"accel={accel:.1f} t/s^2, vel_limit="
            f"{self.axis.controller.config.vel_limit:.1f} t/s"
        )
        print(
            f"[steering] zero reference = {self.start_pos:.4f} turns "
            f"(whatever the wheel looks like right now is 0°)"
        )

    # --- Tuning helpers ---------------------------------------------------

    def _probe_field(self, path: str):
        """Return (getter, setter, label) for ``axis.<path>`` if it exists,
        else None. ``path`` is a dotted attribute chain relative to ``axis``.
        """
        axis = self.axis
        parts = path.split(".")
        try:
            obj = axis
            for p in parts[:-1]:
                obj = getattr(obj, p)
            leaf = parts[-1]
            _ = getattr(obj, leaf)  # read-probe; will raise if missing
            return (
                (lambda obj=obj, leaf=leaf: getattr(obj, leaf)),
                (lambda v, obj=obj, leaf=leaf: setattr(obj, leaf, v)),
                path,
            )
        except Exception:
            return None

    def _resolve_current_fields(self):
        """Return (soft_handle, hard_handle). Either may be None.

        Handles two ODrive firmware generations:

        * 0.6.x (S1, Pro): ``axis.config.motor.current_soft_max`` (soft,
          continuous) plus ``current_hard_max`` (instant trip).
        * 3.x / legacy:    ``axis.motor.config.current_lim`` (single
          field, no separate hard max).
        """
        soft = (self._probe_field("config.motor.current_soft_max")
                or self._probe_field("motor.config.current_lim"))
        hard = self._probe_field("config.motor.current_hard_max")
        return soft, hard

    def _resolve_torque_soft_max(self):
        """``axis.controller.config.torque_soft_max`` if present.

        S1/Pro firmware has a torque limiter independent of current. If
        configured, it caps torque even when current_lim has headroom.
        Worth surfacing because an unexpectedly-set torque_soft_max
        looks exactly like "motor can't keep up" in the UI even though
        there's plenty of electrical authority.
        """
        return self._probe_field("controller.config.torque_soft_max")

    def _resolve_iq_getter(self):
        """``Iq_measured`` — measured quadrature current, proxy for
        actual torque the motor is producing right now.

        Path varies across firmware generations; probe in priority order.
        """
        return (
            self._probe_field("motor.foc.Iq_measured")          # S1 / Pro 0.6.x
            or self._probe_field("motor.current_control.Iq_measured")
            or self._probe_field("motor.Iq_measured")           # legacy 3.x
        )

    def _apply_tuning(
        self,
        current_lim: float | None,
        current_hard_max: float | None,
        vel_gain: float | None,
        vel_integrator_gain: float | None,
        lift_torque_cap: bool = False,
    ) -> None:
        ctrl_cfg = self.axis.controller.config

        # Velocity gains live at the same path on every firmware generation.
        try:
            vg_str = (
                f"vel_gain={ctrl_cfg.vel_gain:.3f}, "
                f"vel_integrator_gain={ctrl_cfg.vel_integrator_gain:.3f}"
            )
        except Exception as e:
            vg_str = f"vel_gain/vel_integrator_gain unreadable ({e!r})"

        soft_handle, hard_handle = self._resolve_current_fields()
        torque_cap_handle = self._resolve_torque_soft_max()

        def _read(handle, label, unit="A"):
            if handle is None:
                return f"{label}=<n/a>"
            try:
                val = handle[0]()
                if val is None:
                    return f"{handle[2]}=<None>"
                if math.isinf(val):
                    return f"{handle[2]}=inf"
                return f"{handle[2]}={val:.2f} {unit}"
            except Exception as e:
                return f"{handle[2]} unreadable ({e!r})"

        print(
            f"[steering] baseline tuning: "
            f"{_read(soft_handle, 'current-soft')}, "
            f"{_read(hard_handle, 'current-hard')}, "
            f"{vg_str}"
        )
        torque_str = _read(torque_cap_handle, "torque_soft_max", unit="Nm")
        print(f"[steering] torque cap: {torque_str}")
        if torque_cap_handle is not None:
            try:
                tval = torque_cap_handle[0]()
                if tval is not None and not math.isinf(tval) and tval < 5.0:
                    print(
                        f"[steering]   ⚠  torque_soft_max={tval:.2f} Nm is finite and "
                        f"low — this likely caps real torque below what current_soft_max "
                        f"({_read(soft_handle, '_')}) would otherwise allow. If the "
                        f"motor can't keep up, this is a prime suspect. Disable with "
                        f"--no-torque-cap or raise it in odrivetool."
                    )
            except Exception:
                pass

        def _set(handle, new_value, what):
            if handle is None:
                print(f"[steering]   → SKIPPED --{what}={new_value}: "
                      f"field not present on this firmware.")
                return
            try:
                handle[1](float(new_value))
                print(f"[steering]   → {handle[2]} overridden to {new_value:.1f} A")
            except Exception as e:
                print(f"[steering]   → FAILED to set {handle[2]}: {e!r}")

        # If the user raised the soft limit above the hard trip without
        # also raising hard_max, the ODrive will throw MOTOR_OVERCURRENT
        # on the first big push. Warn loudly — better than silently
        # letting them chase a phantom bug.
        if (current_lim is not None and hard_handle is not None
                and current_hard_max is None):
            try:
                current_hard_now = hard_handle[0]()
                if current_lim > current_hard_now:
                    print(
                        f"[steering]   ⚠  --current-lim={current_lim:.1f} A exceeds "
                        f"current_hard_max={current_hard_now:.1f} A. The ODrive will "
                        f"throw MOTOR_OVERCURRENT the moment you pull that much "
                        f"current. Pass --current-hard-max too (e.g. "
                        f"--current-hard-max {current_lim * 1.25:.1f})."
                    )
            except Exception:
                pass

        # Apply in the right order: raise hard_max BEFORE soft (so we
        # never have a moment where soft > hard).
        if current_hard_max is not None:
            _set(hard_handle, current_hard_max, "current-hard-max")
        if current_lim is not None:
            _set(soft_handle, current_lim, "current-lim")

        if vel_gain is not None:
            try:
                ctrl_cfg.vel_gain = float(vel_gain)
                print(f"[steering]   → vel_gain overridden to {vel_gain:.3f}")
            except Exception as e:
                print(f"[steering]   → FAILED to set vel_gain: {e!r}")

        if vel_integrator_gain is not None:
            try:
                ctrl_cfg.vel_integrator_gain = float(vel_integrator_gain)
                print(f"[steering]   → vel_integrator_gain overridden to {vel_integrator_gain:.3f}")
            except Exception as e:
                print(f"[steering]   → FAILED to set vel_integrator_gain: {e!r}")

        if lift_torque_cap and torque_cap_handle is not None:
            try:
                torque_cap_handle[1](float("inf"))
                print(f"[steering]   → torque_soft_max lifted to +inf (diagnostic mode).")
            except Exception as e:
                print(f"[steering]   → FAILED to lift torque_soft_max: {e!r}")

        # Stash handles for live UI diagnostics.
        self._iq_getter = self._resolve_iq_getter()
        try:
            self._current_soft_max = (
                float(soft_handle[0]()) if soft_handle is not None else float("nan")
            )
        except Exception:
            self._current_soft_max = float("nan")
        if self._iq_getter is None:
            print("[steering]   (Iq_measured path not found — live current readout disabled)")

    # --- Runtime ----------------------------------------------------------

    def set_vel(self, motor_vel: float) -> None:
        motor_vel = clamp(motor_vel, -self.max_vel, self.max_vel)
        if self.dry_run or self.axis is None:
            return
        self.axis.controller.input_vel = motor_vel

    def iq_measured(self) -> float:
        """Measured quadrature current (A), or NaN if unavailable."""
        if self.dry_run or self._iq_getter is None:
            return float("nan")
        try:
            return float(self._iq_getter[0]())
        except Exception:
            return float("nan")

    @property
    def current_soft_max(self) -> float:
        return self._current_soft_max

    def vbus(self) -> float:
        if self.dry_run or self.odrv is None:
            return float("nan")
        try:
            return float(self.odrv.vbus_voltage)
        except Exception:
            return float("nan")

    def angle_deg(self) -> float:
        """Column angle (deg) relative to whatever position we armed at."""
        if self.dry_run or self.axis is None:
            return 0.0
        return motor_turns_to_steering_deg(self.axis.pos_estimate - self.start_pos)

    def vel_estimate(self) -> float:
        if self.dry_run or self.axis is None:
            return 0.0
        return float(self.axis.vel_estimate)

    def stop(self) -> None:
        """Ramp to zero, revert input mode, and **always** disarm.

        Each step is isolated: a failure early on (e.g. dropped USB
        packet during input_mode write) cannot prevent the IDLE command
        from running. Final step verifies the axis actually reached IDLE
        and prints a clear success/failure line — so the operator knows
        from the console whether the motor is safe.
        """
        if self.dry_run or self.axis is None:
            return

        # 1) Zero the velocity command so VEL_RAMP decelerates us.
        try:
            self.axis.controller.input_vel = 0.0
        except Exception as e:
            print(f"[steering] stop: failed to zero input_vel: {e!r}")

        # 2) Wait briefly for actual velocity to fall. Don't yank
        #    straight to IDLE from full speed — the motor would coast
        #    uncontrolled with zero torque. Cap the wait at 1 s so a
        #    stuck vel_estimate can't block shutdown forever.
        try:
            t0 = time.time()
            while time.time() - t0 < 1.0:
                if abs(self.vel_estimate()) < 0.1:
                    break
                time.sleep(0.03)
        except Exception as e:
            print(f"[steering] stop: decel wait failed: {e!r}")

        # 3) Best-effort revert to PASSTHROUGH input mode (so the next
        #    script starts from a known input mode). Not safety-
        #    critical — if this fails, we still disarm in step 4.
        try:
            if self._InputMode is not None:
                self.axis.controller.config.input_mode = self._InputMode.PASSTHROUGH
        except Exception as e:
            print(f"[steering] stop: input_mode revert failed: {e!r}")

        # 4) DISARM. Safety-critical. If the axis is latched in an
        #    error state it may refuse to IDLE, so clear errors first
        #    and retry. We verify the landed state before returning.
        if self._AxisState is None:
            print("[steering] stop: AxisState enum never loaded — cannot disarm via API.")
            return

        try:
            self.axis.requested_state = self._AxisState.IDLE
        except Exception as e:
            print(f"[steering] stop: initial IDLE request failed: {e!r}")

        # Wait for the axis to actually land in IDLE.
        t0 = time.time()
        while time.time() - t0 < 1.0:
            try:
                if self.axis.current_state == self._AxisState.IDLE:
                    break
            except Exception:
                pass
            time.sleep(0.05)

        # If still not idle, try clearing errors and commanding again.
        try:
            state = self.axis.current_state
        except Exception:
            state = None

        if state != self._AxisState.IDLE:
            print(f"[steering] stop: axis did not IDLE on first try (state={state}); "
                  f"clearing errors and retrying.")
            try:
                if self.odrv is not None:
                    self.odrv.clear_errors()
                time.sleep(0.1)
                self.axis.requested_state = self._AxisState.IDLE
                t0 = time.time()
                while time.time() - t0 < 1.0:
                    if self.axis.current_state == self._AxisState.IDLE:
                        break
                    time.sleep(0.05)
            except Exception as e:
                print(f"[steering] stop: retry disarm failed: {e!r}")

        # Final verdict.
        try:
            final_state = self.axis.current_state
            disarm_reason = self.axis.disarm_reason
        except Exception as e:
            print(f"[steering] stop: could not read final state: {e!r}")
            print("[steering] ⚠  UNABLE TO CONFIRM DISARM — treat motor as LIVE.")
            return

        if final_state == self._AxisState.IDLE:
            print(f"[steering] ✓ disarmed (axis IDLE, disarm_reason={disarm_reason}).")
        else:
            print(f"[steering] ⚠  DISARM FAILED — axis still in state {final_state}. "
                  f"Hit the ODrive e-stop / cut power before touching the cart.")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def draw_ui(screen, font, font_small, state: dict) -> None:
    screen.fill(BG)

    title = font.render("PS5 steer — velocity control", True, ACCENT)
    screen.blit(title, (18, 14))
    sub = font_small.render(
        "DRY RUN — nothing sent to ODrive" if state["dry_run"] else "live ODrive",
        True, WARN if state["dry_run"] else MUTED,
    )
    screen.blit(sub, (18, 48))

    if state.get("homing"):
        banner = font.render("△  HOMING → 0°", True, ACCENT)
        screen.blit(banner, (WINDOW_W - banner.get_width() - 18, 14))

    y = 82
    tgt_color = GOOD if abs(state["target_vel"]) > 0.01 else MUTED
    if state["at_limit"]:
        angle_color = BAD
    elif state["braking"]:
        angle_color = WARN
    else:
        angle_color = TEXT

    if state["at_limit"]:
        tag = "  [AT LIMIT]"
    elif state["braking"]:
        tag = "  [braking]"
    else:
        tag = ""

    # Tracking-error coloring — diagnostic for fluid motion.
    # Only meaningful when the commanded velocity isn't near zero; a
    # stopped motor has "error 0" trivially.
    abs_err = abs(state["tracking_err"])
    if abs(state["target_vel"]) < 0.2:
        err_color = MUTED
        err_tag = ""
    elif abs_err < 0.2:
        err_color = GOOD
        err_tag = "  ok"
    elif abs_err < 0.8:
        err_color = WARN
        err_tag = "  lagging"
    else:
        err_color = BAD
        err_tag = "  MOTOR CAN'T KEEP UP"

    # Iq (actual motor current) — the ground truth for what the motor
    # is really doing. If tracking err is red but Iq isn't anywhere
    # near current_soft_max, the controller is being held back by
    # something OTHER than electrical torque (torque_soft_max, gain
    # saturation, vel_limit — the usual suspects). If Iq IS pegged at
    # current_soft_max, it's real saturation and you need more amps.
    iq = state["iq_measured"]
    iq_cap = state["current_soft_max"]
    if math.isnan(iq):
        iq_str = "<n/a>"
        iq_color = MUTED
    elif math.isnan(iq_cap) or iq_cap <= 0:
        iq_str = f"{iq:+5.2f} A"
        iq_color = TEXT
    else:
        pct = abs(iq) / iq_cap * 100.0
        if pct > 90.0:
            iq_color = BAD
            saturation = "  SATURATED — raise current_soft_max"
        elif pct > 60.0:
            iq_color = WARN
            saturation = "  pushing hard"
        else:
            iq_color = TEXT
            saturation = ""
        iq_str = f"{iq:+5.2f} A   ({pct:4.1f}% of {iq_cap:.1f} A cap){saturation}"

    vbus_str = (f"{state['vbus']:.1f} V" if not math.isnan(state["vbus"]) else "<n/a>")

    lines = [
        ("stick X",       f"{state['lx']:+.2f}"
                          f"  (deadzone {STICK_DEADZONE:.2f})",               TEXT),
        ("target vel",    f"{state['target_vel']:+5.2f} turns/s"
                          f"   (cmd {state['commanded_vel']:+5.2f}, cap ±{state['max_vel']:.1f})",
                                                                              tgt_color),
        ("actual vel",    f"{state['actual_vel']:+5.2f} turns/s",             TEXT),
        ("tracking err",  f"{state['tracking_err']:+5.2f} turns/s{err_tag}",  err_color),
        ("motor Iq",      iq_str,                                             iq_color),
        ("bus voltage",   vbus_str,                                           TEXT),
        ("column angle",  f"{state['angle']:+6.1f}°"
                          f"   soft {STEERING_MIN_DEG:+.0f}..{STEERING_MAX_DEG:+.0f}°"
                          f"{tag}",                                            angle_color),
    ]
    for label, value, color in lines:
        screen.blit(font_small.render(label, True, MUTED), (18, y))
        screen.blit(font.render(value, True, color), (170, y - 4))
        y += 32

    hint = font_small.render(
        "△ (Triangle) → return to 0°     •     Esc / Q / close window → stop + idle.",
        True, MUTED,
    )
    screen.blit(hint, (18, WINDOW_H - 26))
    pygame.display.flip()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--index", type=int, default=0,
        help="Controller index when multiple are paired (default 0).",
    )
    parser.add_argument(
        "--max-vel", type=float, default=DEFAULT_MAX_VEL,
        help=f"Max motor velocity, turns/s (default {DEFAULT_MAX_VEL}).",
    )
    parser.add_argument(
        "--accel", type=float, default=DEFAULT_ACCEL,
        help=f"VEL_RAMP accel, turns/s^2 (default {DEFAULT_ACCEL}).",
    )
    parser.add_argument(
        "--invert", action="store_true",
        help="Flip sign if stick-left turns the wheel right.",
    )
    parser.add_argument(
        "--home-button", type=int, default=BUTTON_TRIANGLE,
        help=(
            f"Button index that triggers 'return to 0°' homing "
            f"(default {BUTTON_TRIANGLE} = Triangle on SDL2 macOS/Linux)."
        ),
    )
    # Tracking tuning (fluid motion through hard spots). Leave unset
    # and the script uses whatever the ODrive was calibrated with.
    parser.add_argument(
        "--current-lim", type=float, default=None,
        help=(
            "Override motor current soft limit (A) — the continuous "
            "allowed current. Raise if the motor is torque-saturated "
            "in high-load regions (watch the UI's tracking err row). "
            "Stock M8325s is usually 10-15 A; 20-25 A is a reasonable "
            "upper bound for short bursts."
        ),
    )
    parser.add_argument(
        "--current-hard-max", type=float, default=None,
        help=(
            "Override motor current HARD trip (A) on S1/Pro firmware. "
            "Must be >= --current-lim or the ODrive will throw "
            "MOTOR_OVERCURRENT. Typically 1.25x the soft limit."
        ),
    )
    parser.add_argument(
        "--vel-gain", type=float, default=None,
        help="Override ODrive velocity P gain. Higher = snappier reaction to tracking dips.",
    )
    parser.add_argument(
        "--vel-integrator-gain", type=float, default=None,
        help="Override ODrive velocity I gain. Higher = faster catch-up through hard spots.",
    )
    parser.add_argument(
        "--lift-torque-cap", action="store_true",
        help=(
            "Set controller.config.torque_soft_max = +inf for this "
            "session. Use when the baseline tuning line shows a finite "
            "torque cap that's secretly limiting the motor even though "
            "current_lim has headroom."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Don't open the ODrive; just read the stick and draw the UI.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sign = -1.0 if args.invert else 1.0
    print(f"[config] max_vel={args.max_vel:.2f} turns/s  accel={args.accel:.2f} turns/s^2  "
          f"sign={sign:+.0f}")

    js = init_controller(args.index)

    # Construct SteeringVel inside the try/finally so that if __init__
    # partially succeeds (connected + armed, then something later
    # raised) we still run stop() on the way out and disarm the axis.
    steering: SteeringVel | None = None
    exit_code = 0
    homing = False
    try:
        steering = SteeringVel(
            args.max_vel,
            args.accel,
            dry_run=args.dry_run,
            current_lim=args.current_lim,
            current_hard_max=args.current_hard_max,
            vel_gain=args.vel_gain,
            vel_integrator_gain=args.vel_integrator_gain,
            lift_torque_cap=args.lift_torque_cap,
        )

        screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption("PS5 steer")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("Menlo", 22, bold=True)
        font_small = pygame.font.SysFont("Menlo", 14)

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.type == pygame.JOYDEVICEREMOVED:
                    print("[ps5] controller disconnected — stopping.")
                    running = False
                elif event.type == pygame.JOYBUTTONDOWN and event.button == args.home_button:
                    homing = not homing
                    print(f"[home] {'ENGAGED — returning to 0°' if homing else 'cancelled'}")

            lx_raw = js.get_axis(AXIS_LEFT_X) if js.get_numaxes() > AXIS_LEFT_X else 0.0
            lx = apply_deadzone(lx_raw, STICK_DEADZONE)

            # Read angle before building target_vel so the homing branch
            # (which targets angle = 0) has it available.
            angle = steering.angle_deg()
            a_brake = args.accel * BRAKE_ACCEL_FRACTION  # turns/s^2

            if homing and abs(lx_raw) > HOMING_STICK_CANCEL:
                print("[home] stick grabbed — aborting.")
                homing = False

            if homing:
                # Return-to-zero is structurally identical to the soft-
                # limit predictor: target angle is 0, remaining distance
                # is |angle|, velocity direction is toward zero. Same
                # sqrt(2·a·d) envelope guarantees smooth decel into the
                # waypoint with no overshoot past 0°.
                err_deg = -angle
                remaining_turns = steering_deg_to_motor_turns(abs(err_deg))
                v_safe = math.sqrt(2.0 * a_brake * remaining_turns)
                direction = 1.0 if err_deg > 0 else -1.0
                target_vel = direction * min(args.max_vel, v_safe)

                vel_actual = steering.vel_estimate()
                if abs(angle) <= HOMING_ANGLE_TOL_DEG and abs(vel_actual) <= HOMING_VEL_TOL:
                    print(f"[home] reached 0° (angle={angle:+.2f}°). Stick control resumed.")
                    target_vel = 0.0
                    homing = False
            else:
                target_vel = sign * lx * args.max_vel

            commanded_vel = target_vel  # pre-soft-limit, for UI comparison

            # Predictive soft-limit braking.
            #
            # A pure "clamp target to zero at the stop" is too late —
            # from 5 turns/s at 10 turns/s^2 decel that would overshoot
            # the stop by ~150° of column travel before the ramp lands.
            # Instead we continuously enforce:
            #
            #     |v_target| <= sqrt(2 * a_brake * remaining_distance)
            #
            # which is the highest velocity from which VEL_RAMP can
            # still stop in the distance left before the limit. As we
            # approach a stop the allowed |v| falls smoothly to zero,
            # so the stick still moves you right up to the edge but
            # can't drive you past it.
            at_limit = False
            if target_vel > 0:
                remaining_deg = STEERING_MAX_DEG - angle
                if remaining_deg <= 0.0:
                    target_vel = 0.0
                    at_limit = True
                else:
                    remaining_turns = steering_deg_to_motor_turns(remaining_deg)
                    v_safe = math.sqrt(2.0 * a_brake * remaining_turns)
                    if target_vel > v_safe:
                        target_vel = v_safe
                        at_limit = v_safe < 0.2 * args.max_vel
            elif target_vel < 0:
                remaining_deg = angle - STEERING_MIN_DEG
                if remaining_deg <= 0.0:
                    target_vel = 0.0
                    at_limit = True
                else:
                    remaining_turns = steering_deg_to_motor_turns(remaining_deg)
                    v_safe = math.sqrt(2.0 * a_brake * remaining_turns)
                    if -target_vel > v_safe:
                        target_vel = -v_safe
                        at_limit = v_safe < 0.2 * args.max_vel

            braking = (abs(target_vel) + 1e-3) < abs(commanded_vel)
            steering.set_vel(target_vel)

            actual_vel = steering.vel_estimate()
            # Tracking error: how far behind the commanded velocity the
            # motor actually is. If this spikes consistently around a
            # particular column angle, that's a mechanical hard spot and
            # you need more torque (current_lim) or faster integral
            # (vel_integrator_gain). If it's uniformly high everywhere,
            # lift current_lim first.
            tracking_err = target_vel - actual_vel

            draw_ui(screen, font, font_small, {
                "dry_run": args.dry_run,
                "lx": lx,
                "commanded_vel": commanded_vel,
                "target_vel": target_vel,
                "max_vel": args.max_vel,
                "actual_vel": actual_vel,
                "tracking_err": tracking_err,
                "iq_measured": steering.iq_measured(),
                "current_soft_max": steering.current_soft_max,
                "vbus": steering.vbus(),
                "angle": angle,
                "at_limit": at_limit,
                "braking": braking,
                "homing": homing,
            })
            clock.tick(CONTROL_HZ)

    except KeyboardInterrupt:
        print("\n[main] KeyboardInterrupt — stopping.")
    except Exception as e:
        print(f"[main] fatal: {e}")
        exit_code = 1
    finally:
        # Safety: disarm no matter how we got here. If the ODrive
        # construction blew up part-way through, steering may still be
        # None — but in that case there's nothing armed to disarm.
        if steering is not None:
            steering.stop()
        else:
            print("[main] no SteeringVel instance — nothing to disarm.")
        pygame.quit()
        print("[main] done.")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
