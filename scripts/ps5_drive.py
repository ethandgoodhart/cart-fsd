#!/usr/bin/env python3
"""
ps5_drive.py — Drive the cart with a PS5 DualSense controller.

Reads the DualSense over pygame and commands both cart subsystems with
authority clamped by ``limits.py``:

  - Left stick X  → steering column (ODrive **POSITION + TRAP_TRAJ**, same
    planner as ``main.py`` sweeps). ``integrated`` slews a held target angle;
    ``absolute`` maps stick position to angle.
  - R2 trigger    → gas pedal target (Arduino Mega over USB serial)
  - L2 trigger    → brake pedal target (Arduino Mega over USB serial)

Gas is always clamped to
``effective_gas_cap(PS5_GAS_LIMIT) = min(GAS_POT_MAX, GLOBAL_SPEED_LIMIT, PS5_GAS_LIMIT)``
so the top-level governor in ``limits.py`` wins even if this script is
asked to push harder.

Modes (``--mode``):

  full      (default) steering + pedals
  steering  ODrive only; pedals are left alone
  pedals    pedals only; ODrive is not even opened

Expected Arduino pedal protocol (USB serial, 115200 baud, newline-terminated):

  ``G <value>``   gas target pot value (0.0 .. GAS_POT_MAX)
  ``B <value>``   brake target pot value (0.0 .. BRAKE_POT_MAX)
  ``S``           stop both actuators immediately

Safety:

  - The pygame window must be focused for input.
  - Esc / Q / close-window → graceful stop of everything that was opened.
  - ``--dry-run`` skips all hardware and just prints what would be sent,
    handy for checking input mapping without the cart present.

Stick steering (``--stick-steering``):

  ``integrated`` — stick slews a **target column angle** (°); the ODrive
  **trap trajectory** tracks it with the same vel/accel caps as calibration
  sweeps (``main.py``). Release the stick and the target (and wheel) **hold**.

  ``absolute`` — stick position maps directly to target angle; center → 0°.

Usage:

    uv run python scripts/ps5_drive.py                              # full, integrated steer
    uv run python scripts/ps5_drive.py --stick-steering absolute    # wheel follows stick
    uv run python scripts/ps5_drive.py --mode steering              # steering only
    uv run python scripts/ps5_drive.py --mode pedals                # pedals only
    uv run python scripts/ps5_drive.py --dry-run                    # no hardware
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

# Run-from-anywhere: make the project root importable so ``limits.py`` at
# the repo root resolves regardless of the current working directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# SDL pauses joystick polling when the pygame window loses focus (macOS in
# particular will freeze axes at their last value if you alt-tab to another
# app). Setting this hint before ``import pygame`` keeps the DualSense live
# regardless of which window is frontmost — necessary for a cart-driving
# tool where you might legitimately tab away to read logs or adjust a
# config, and the pedals must not silently stop responding.
os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")

import pygame

from limits import (
    BRAKE_POT_MAX,
    FSD_GAS_LIMIT,
    GAS_POT_MAX,
    GLOBAL_SPEED_LIMIT,
    PS5_GAS_LIMIT,
    STEERING_MAX_DEG,
    STEERING_MIN_DEG,
    effective_gas_cap,
    motor_turns_to_steering_deg,
    steering_deg_to_motor_turns,
)

# --- Controller mapping (matches scripts/ps5_controller_test.py) -----------

AXIS_LEFT_X = 0
AXIS_L2 = 4
AXIS_R2 = 5

# DualSense face buttons on SDL 2.28+ (macOS / Linux): Cross=0, Circle=1,
# Square=2, Triangle=3. Matches scripts/ps5_steer.py.
BUTTON_TRIANGLE = 3

# Default deadzone (e.g. other scripts). Steering X uses ``STEER_STICK_DEADZONE`` so
# more physical travel maps to rate — a large deadzone + remap makes mid‑stick feel weak.
STICK_DEADZONE = 0.08
STEER_STICK_DEADZONE = 0.055

# Per-trigger calibration — some DualSense units/driver stacks don't quite
# report full-scale on the triggers; rescale so full-squeeze reads 1.0.
TRIGGER_MAX_L2 = 0.91
TRIGGER_MAX_R2 = 1.00

# --- Cart-side mapping ------------------------------------------------------

# Steering authority for PS5 driving. Uses the cart-wide soft limits
# defined in limits.py (``STEERING_MIN/MAX_DEG``) — to widen/narrow the
# human-driving range, edit limits.py so the change is consistent with
# the autonomy stack when it lands.
PS5_STEERING_MAX_DEG = min(STEERING_MAX_DEG, -STEERING_MIN_DEG)

# Trapezoidal planner caps — **match ``main.py``** so PS5 steering uses the same
# consistent ODrive-generated motion as calibration sweeps (motor-side units).
PS5_TRAP_VEL_MAX = 8.0      # motor turns/s
PS5_TRAP_ACCEL_MAX = 15.0  # motor turns/s²
PS5_TRAP_DECEL_MAX = 15.0

# Integrated mode: how fast the **target angle** slews with full stick (°/s at |lx|=1).
# The wheel follows via TRAP_TRAJ; high values let the planner saturate at PS5_TRAP_*.
STEER_INTEGRATE_RATE_DPS = 320.0

# Motor current headroom: scale current_soft_max toward current_hard_max with
# demand ∈ [0,1] from |stick|, |d(lx)/dt|, and |d(ω_cmd)/dt| so fast reversals
# keep torque for acceleration instead of hitting an overly low soft limit.
PS5_STEER_CURRENT_BLEND_MIN = 0.45   # demand=0 → keep at least this fraction of headroom
PS5_STEER_CURRENT_HARD_CAP_A = 80.0  # never raise soft_max above this (motor / fuse sanity)

# Reference scales for mapping stick dynamics into current demand (tunable).
STICK_DLX_REF = 10.0                  # |dlx/dt| ≈ this → full “stick dynamics” contribution
VEL_CMD_ACCEL_REF = 120.0             # motor turns/s² reference for ω_cmd slew

# ODrive axis watchdog: the motor-side equivalent of the Mega's heartbeat.
# The host must call ``axis.watchdog_feed()`` inside this window or the
# ODrive disarms itself. Only enforced once the axis enters closed-loop
# control, so setup-time pauses don't trip it. Matches the Mega's 300 ms.
STEERING_WATCHDOG_S = 0.30

# Main-loop rate for reading the controller and pushing commands.
CONTROL_HZ = 50.0

MODES = ("full", "steering", "pedals")
STICK_STEERING = ("absolute", "integrated")

# --- UI colors --------------------------------------------------------------

WINDOW_W, WINDOW_H = 680, 360
BG = (18, 20, 26)
TEXT = (230, 232, 238)
MUTED = (130, 135, 150)
ACCENT = (90, 170, 255)
GOOD = (120, 220, 140)
WARN = (240, 180, 90)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def apply_deadzone(value: float, deadzone: float) -> float:
    """Rescale a stick value so |v| <= deadzone maps to 0 and the remainder
    is re-expanded to -1..1 for smooth past-the-deadzone behavior."""
    if abs(value) <= deadzone:
        return 0.0
    sign = 1.0 if value > 0 else -1.0
    return sign * (abs(value) - deadzone) / (1.0 - deadzone)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def read_trigger(js, axis: int, scale: float) -> float:
    """Read an SDL trigger axis (-1 released, +1 squeezed) and return 0..1."""
    if js.get_numaxes() <= axis:
        return 0.0
    raw = js.get_axis(axis)
    value = max(0.0, min(1.0, (raw + 1.0) / 2.0))
    return min(1.0, value / scale) if scale > 0 else 0.0


def controller_alive(js) -> tuple[bool, str]:
    """Return ``(True, "")`` while the DualSense is still usable.

    Bluetooth drops (out of range, dead battery, driver glitch) frequently
    *don't* produce a ``JOYDEVICEREMOVED`` event — SDL keeps the Joystick
    object around and quietly freezes the axes at their last value. The
    firmware watchdog will retract the pedals on its own after 300 ms, but
    by then the host is still telling the cart to gas. We actively probe
    every frame so the host stops commanding the moment the link dies.

    Checks, in order of strictness:
      1. ``pygame.joystick.get_count() == 0`` — SDL has fully forgotten it.
      2. ``js.get_numaxes()`` raises — object invalidated mid-session.
      3. ``js.get_init()`` is False — someone released the handle.
    """
    if pygame.joystick.get_count() == 0:
        return False, "no joysticks enumerated"
    try:
        if not js.get_init():
            return False, "joystick handle uninitialized"
        _ = js.get_numaxes()
    except pygame.error as e:
        return False, f"joystick invalidated: {e}"
    return True, ""


def find_arduino_port() -> str | None:
    """Best-effort Mega auto-detect; returns None if nothing obvious found.

    Avoids grabbing the ODrive (VID 0x1209 / 0x0483). Prefers USB-VID
    matches for Arduino / clones / FTDI, then falls back to the
    shortest-named usbmodem-ish device.
    """
    try:
        import serial.tools.list_ports  # type: ignore
    except ImportError:
        return None

    ARDUINO_VIDS = {0x2341, 0x2A03, 0x1A86, 0x0403, 0x10C4}
    ODRIVE_VIDS = {0x1209, 0x0483}

    ports = [
        p for p in serial.tools.list_ports.comports()
        if any(tag in p.device for tag in ("usbmodem", "ttyACM", "ttyUSB", "usbserial"))
    ]
    arduino = [p for p in ports if p.vid in ARDUINO_VIDS]
    if arduino:
        return arduino[0].device
    non_odrive = [p for p in ports if p.vid not in ODRIVE_VIDS]
    if non_odrive:
        return min(non_odrive, key=lambda p: len(p.device)).device
    return None


# ---------------------------------------------------------------------------
# hardware wrappers
# ---------------------------------------------------------------------------

# Host side of the heartbeat contract with firmware/pedal_control.ino.
# Every successful write to the Mega resets its watchdog; if the host
# goes quiet for longer than this, the firmware drops both pedals. We
# mirror that threshold here so the UI can turn red just before the cart
# actually fails safe, and so the main loop aborts steering too (losing
# pedals while still commanding the wheel is worse than stopping both).
HEARTBEAT_TIMEOUT_S = 0.30


class PedalLink:
    """Sends gas/brake target pot values to the Arduino Mega over USB serial
    and watches the link health.

    Gas and brake are clamped against their pot-max ceilings as a
    last-ditch safety net — callers are expected to have already applied
    the per-mode cap from ``limits.effective_gas_cap()``.

    Heartbeat contract:
      - Every successful write timestamps ``self.last_ok_s`` (monotonic).
      - Any OS-level serial failure (cable yanked, port vanished, write
        throws OSError / SerialException) sets ``self.faulted = True`` and
        stops touching the port. The main loop should check ``.faulted``
        and exit — there's no automatic reconnect here by design: if we
        lose pedals, we can't trust the rest of the cart state either.
      - The firmware also runs its own watchdog (see
        ``sketches/pedal_control/pedal_control.ino``). Even if this Python
        process is frozen or killed, the Mega drops both actuators to MIN
        ~300 ms after the last received byte.

    Under ``--dry-run`` the serial port is never opened; all writes are
    no-ops but ``last_ok_s`` still advances so the UI looks normal.
    """

    def __init__(self, port: str | None, baud: int = 115200, dry_run: bool = False):
        self.ser = None
        self.dry_run = dry_run
        self.faulted = False
        self.fault_reason: str | None = None
        self.last_ok_s = time.monotonic()

        # E-stop state mirrored from the Mega. ``estop_active`` flips True
        # the instant we parse ``EVT,ESTOP,1`` from the serial stream. The
        # main loop checks this and bails so steering gets idled — the
        # firmware already forces brake=MAX and gas=MIN on its side, but
        # we also need to stop commanding the ODrive or the motor will
        # fight the brake until the watchdog takes over.
        self.estop_active = False
        self._rx_buf = bytearray()

        if dry_run:
            print("[pedals] dry-run: commands will NOT be sent over serial.")
            return

        import serial  # type: ignore

        resolved = port or find_arduino_port()
        if resolved is None:
            raise RuntimeError(
                "Couldn't auto-detect the Arduino Mega. Pass --arduino-port /dev/tty.X "
                "(run scripts/sensor_test.py --list to see attached ports)."
            )
        print(f"[pedals] opening {resolved} @ {baud} baud")
        self.ser = serial.Serial(resolved, baud, timeout=0.5)
        # Mega resets on port open; give the bootloader a moment before the
        # first write so commands aren't sent into the DTR-reset void.
        time.sleep(2.0)

    @property
    def heartbeat_age_s(self) -> float:
        """Seconds since the last successful write (or link init)."""
        return time.monotonic() - self.last_ok_s

    @property
    def healthy(self) -> bool:
        return not self.faulted and self.heartbeat_age_s < HEARTBEAT_TIMEOUT_S

    def _mark_fault(self, reason: str) -> None:
        if self.faulted:
            return
        self.faulted = True
        self.fault_reason = reason
        print(f"[pedals] FAULT: {reason} — bailing out.")

    def send(self, gas: float, brake: float) -> None:
        """Push a gas+brake target pair. Records heartbeat on success."""
        gas = clamp(gas, 0.0, GAS_POT_MAX)
        brake = clamp(brake, 0.0, BRAKE_POT_MAX)

        if self.dry_run:
            self.last_ok_s = time.monotonic()
            return
        if self.ser is None or self.faulted:
            return

        payload = f"G {gas:.3f}\nB {brake:.3f}\n".encode("ascii")
        try:
            self.ser.write(payload)
            self.last_ok_s = time.monotonic()
        except Exception as e:  # SerialException, OSError, etc — all fatal
            self._mark_fault(f"serial write failed: {e!r}")

    def poll(self) -> None:
        """Drain any pending bytes from the Mega and handle event lines.

        The firmware emits ``EVT,ESTOP,1`` on press and ``EVT,ESTOP,0`` on
        release, plus ``STAT,...`` at 10 Hz with a trailing ``es=0|1``
        field. We only act on the EVT lines here — STAT is ignored by the
        host (the firmware is authoritative on pedal targets). Called
        every frame from the main loop; non-blocking.
        """
        if self.dry_run or self.ser is None or self.faulted:
            return
        try:
            pending = self.ser.in_waiting
        except Exception as e:
            self._mark_fault(f"in_waiting failed: {e!r}")
            return
        if pending <= 0:
            return
        try:
            self._rx_buf.extend(self.ser.read(pending))
        except Exception as e:
            self._mark_fault(f"serial read failed: {e!r}")
            return
        while b"\n" in self._rx_buf:
            line, _, rest = self._rx_buf.partition(b"\n")
            self._rx_buf = bytearray(rest)
            text = line.decode("ascii", errors="replace").strip()
            if not text:
                continue
            if text.startswith("EVT,ESTOP,"):
                active = text.endswith(",1")
                if active != self.estop_active:
                    self.estop_active = active
                    state = "ENGAGED" if active else "released"
                    print(f"[pedals] E-STOP {state} (from Mega: {text})")
            elif text.startswith("ERR,") or text.startswith("INFO,"):
                # Surface firmware-side warnings/info; STAT is too noisy.
                print(f"[mega] {text}")

    def stop(self) -> None:
        """Best-effort 'release both pedals NOW' on the way out.

        Repeated writes because the first one sometimes gets swallowed if
        the port is already half-gone (USB yank race). If this fails the
        firmware's own heartbeat watchdog will do the same thing ~300 ms
        after the last byte it saw anyway.
        """
        if self.dry_run or self.ser is None:
            print("[pedals] STOP")
            return
        for _ in range(3):
            try:
                self.ser.write(b"S\n")
                time.sleep(0.02)
            except Exception:
                break

    def close(self) -> None:
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass


def _odrive_cfg(obj: object) -> object:
    """Return ``obj.config`` when it exists; otherwise the flat ``obj`` (libodrive style).

    Never fall back to ``obj`` when ``config`` is missing on **motor** — motor
    fields like ``current_soft_max`` only exist on ``motor.config`` in legacy
    layouts, not on ``motor`` itself.
    """
    if hasattr(obj, "config"):
        return obj.config
    return obj


def _motor_amp_limits(axis) -> tuple[float | None, float | None]:
    """``(current_soft_max, current_hard_max)`` in A, or ``(None, None)`` if not on this API."""
    m = axis.motor
    if hasattr(m, "config"):
        c = m.config
        if hasattr(c, "current_soft_max") and hasattr(c, "current_hard_max"):
            return float(c.current_soft_max), float(c.current_hard_max)
    return None, None


def _set_motor_current_soft_max(axis, value: float) -> None:
    m = axis.motor
    if hasattr(m, "config") and hasattr(m.config, "current_soft_max"):
        m.config.current_soft_max = value


class KeepAlive(threading.Thread):
    """Dedicated 50 Hz heartbeat thread for the Mega and the ODrive watchdog.

    macOS AppNap throttles the pygame main loop to ~1 Hz whenever the
    pygame window loses focus, which is slower than either the Mega's
    300 ms serial heartbeat timeout or the ODrive's 300 ms axis watchdog
    timeout. Both would then trip — the pedals would retract, the ODrive
    would IDLE — just because the operator alt-tabbed to Settings.

    This thread ticks at a fixed rate independent of the render loop.
    Each tick it:
      - writes a single ``H\\n`` byte to the Mega (the firmware treats
        any byte as a heartbeat, so the watchdog stays fed without
        changing the last-commanded G/B targets), and
      - calls ``axis.watchdog_feed()`` on the ODrive so the axis stays
        armed at the last commanded ``input_pos``.

    Commands still flow through the main loop (``pedals.send``,
    ``steering.command_deg``). While backgrounded, those update at the
    throttled rate, which is fine: the firmware holds the last target
    until the host resumes. This thread just makes sure the firmware
    doesn't give up on us in the meantime.

    Thread safety: the lock owned here is also acquired by main-thread
    hardware calls (see ``hw_lock`` in main()). Never touch the
    hardware without it.
    """

    def __init__(
        self,
        pedals: "PedalLink | None",
        steering: "SteeringLink | None",
        lock: threading.Lock,
        hz: float = 50.0,
    ):
        super().__init__(daemon=True, name="cart-keepalive")
        self.pedals = pedals
        self.steering = steering
        self.lock = lock
        self.period = 1.0 / hz
        self.stop_event = threading.Event()

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        # time.monotonic-scheduled loop: drift-free even if a tick runs long.
        next_t = time.monotonic() + self.period
        while not self.stop_event.is_set():
            now = time.monotonic()
            if now < next_t:
                # Wake on stop_event OR when the tick is due — whichever first.
                self.stop_event.wait(timeout=next_t - now)
                if self.stop_event.is_set():
                    break
            next_t += self.period
            # Resync if we fell far behind (e.g. process was suspended).
            overshoot = time.monotonic() - next_t
            if overshoot > 5 * self.period:
                next_t = time.monotonic() + self.period

            with self.lock:
                self._tick()

    def _tick(self) -> None:
        # Mega heartbeat — a bare ``H`` ping, no target change. Protected
        # against a faulted/closed port: if writes are failing the main
        # loop already saw the fault and is exiting.
        p = self.pedals
        if p is not None and p.ser is not None and not p.faulted and not p.dry_run:
            try:
                p.ser.write(b"H\n")
                p.last_ok_s = time.monotonic()
            except Exception:
                # Let PedalLink.send / main loop surface the fault on its
                # next iteration — don't stomp state from here.
                pass

        # ODrive axis watchdog feed. Only once the watchdog has been
        # armed (which happens right before the main loop starts).
        s = self.steering
        if (
            s is not None
            and not s.dry_run
            and s._watchdog_enabled
            and s.axis is not None
        ):
            try:
                s.axis.watchdog_feed()
            except Exception:
                pass


class SteeringLink:
    """ODrive S1 wrapper for PS5 steering.

    Both ``integrated`` and ``absolute`` use **POSITION_CONTROL** and
    **TRAP_TRAJ** with the same ``vel_limit`` / ``accel_limit`` / ``decel_limit``
    as ``main.py`` calibration sweeps so motion is generated consistently on
    the ODrive, not by host-side velocity filtering.
    """

    def __init__(self, dry_run: bool = False, stick_steering: str = "integrated"):
        self.dry_run = dry_run
        self.stick_steering = stick_steering
        self.odrv = None
        self.axis = None
        self.start_pos = 0.0
        self._AxisState = None
        self._InputMode = None
        self._saved_current_soft = None
        self._saved_current_hard = None
        self._prev_target_motor = 0.0
        self._prev_vel_cmd_motor = 0.0
        self._watchdog_enabled = False

        if dry_run:
            print("[steering] dry-run: ODrive will NOT be opened.")
            return

        import odrive  # type: ignore
        from odrive.enums import AxisState, ControlMode, InputMode  # type: ignore

        self._AxisState = AxisState
        self._InputMode = InputMode

        print("[steering] connecting to ODrive...")
        self.odrv = odrive.find_any(timeout=10)
        self.axis = self.odrv.axis0
        print(
            f"[steering] connected: serial {self.odrv.serial_number}, "
            f"bus {self.odrv.vbus_voltage:.1f}V"
        )

        # Disable any residual watchdog BEFORE clear_errors / setup. If a
        # previous session died without running stop() (e.g. USB got yanked
        # mid-drive), ``enable_watchdog`` is still True on the ODrive. The
        # setup pauses below would then consume the 300 ms window and the
        # watchdog would re-trip before we entered CLOSED_LOOP_CONTROL,
        # leaving the axis stuck IDLE with disarm_reason=WATCHDOG.
        self._axis_cfg = _odrive_cfg(self.axis)
        self._watchdog_available = (
            hasattr(self._axis_cfg, "enable_watchdog")
            and hasattr(self.axis, "watchdog_feed")
        )
        if self._watchdog_available:
            try:
                self.axis.watchdog_feed()        # pet in case it's about to trip
                self._axis_cfg.enable_watchdog = False
            except Exception as e:
                print(f"[steering] couldn't quiet residual watchdog: {e!r}")

        if self.axis.active_errors != 0:
            print(f"[steering] clearing active errors: {self.axis.active_errors}")
            self.odrv.clear_errors()
            time.sleep(0.3)

        self._controller_cfg = _odrive_cfg(self.axis.controller)
        self._trap_traj_cfg = _odrive_cfg(self.axis.trap_traj)

        self._saved_current_soft, self._saved_current_hard = _motor_amp_limits(self.axis)
        if self._saved_current_soft is not None and self._saved_current_hard is not None:
            print(
                f"[steering] motor current_soft_max={self._saved_current_soft:.2f}A  "
                f"current_hard_max={self._saved_current_hard:.2f}A "
                f"(PS5 scales soft toward hard under demand)"
            )
        else:
            print(
                "[steering] motor amp limits not on ``motor.config`` (ODrive 0.6+ layout) — "
                "skipping dynamic current_soft_max scaling"
            )

        mode_label = "integrated (trap target slew)" if stick_steering == "integrated" else "absolute"
        print(f"[steering] mode: POSITION + TRAP_TRAJ — {mode_label}")
        print(
            f"[steering] trap_traj vel/accel/decel = "
            f"{PS5_TRAP_VEL_MAX}/{PS5_TRAP_ACCEL_MAX}/{PS5_TRAP_DECEL_MAX} "
            f"(same as main.py sweeps)"
        )
        self._controller_cfg.control_mode = ControlMode.POSITION_CONTROL
        self._controller_cfg.input_mode = InputMode.TRAP_TRAJ
        self._trap_traj_cfg.vel_limit = PS5_TRAP_VEL_MAX
        self._trap_traj_cfg.accel_limit = PS5_TRAP_ACCEL_MAX
        self._trap_traj_cfg.decel_limit = PS5_TRAP_DECEL_MAX

        self.axis.requested_state = AxisState.CLOSED_LOOP_CONTROL
        time.sleep(0.35)
        if self.axis.current_state != AxisState.CLOSED_LOOP_CONTROL:
            raise RuntimeError(
                f"ODrive failed to enter closed-loop control: "
                f"state={self.axis.current_state}, disarm={self.axis.disarm_reason}"
            )

        self.start_pos = self.axis.pos_estimate
        print(
            f"[steering] zero reference = {self.start_pos:.4f} turns "
            f"(whatever the wheel looks like right now is 0°)"
        )

        # Set the watchdog timeout now; enable is deferred to arm_watchdog()
        # because pygame.display.set_mode() and SysFont on macOS can burn
        # >300 ms, which would trip the watchdog before we ever commanded
        # a position. main() calls arm_watchdog() once the control loop is
        # about to spin.
        if self._watchdog_available:
            try:
                if hasattr(self._axis_cfg, "watchdog_timeout"):
                    self._axis_cfg.watchdog_timeout = STEERING_WATCHDOG_S
                print(
                    f"[steering] axis watchdog configured "
                    f"(timeout={STEERING_WATCHDOG_S * 1000:.0f} ms, "
                    f"enabled later by arm_watchdog())"
                )
            except Exception as e:
                self._watchdog_available = False
                print(f"[steering] watchdog config failed: {e!r}")
        else:
            print("[steering] axis watchdog not available on this firmware — "
                  "USB-drop safety limited to Mega heartbeat.")

    def column_deg_estimate(self) -> float:
        """Column angle (deg) relative to startup, from encoder."""
        if self.dry_run or self.axis is None:
            return 0.0
        return motor_turns_to_steering_deg(self.axis.pos_estimate - self.start_pos)

    def _apply_dynamic_current(self, demand: float) -> None:
        """Raise current_soft_max toward hard_max when demand is high (0..1)."""
        if self.dry_run or self.axis is None:
            return
        if self._saved_current_soft is None or self._saved_current_hard is None:
            return
        demand = clamp(float(demand), 0.0, 1.0)
        soft0 = self._saved_current_soft
        hard = min(self._saved_current_hard, PS5_STEER_CURRENT_HARD_CAP_A)
        if hard <= soft0 * 1.01:
            return
        span = hard - soft0
        head = PS5_STEER_CURRENT_BLEND_MIN + (1.0 - PS5_STEER_CURRENT_BLEND_MIN) * demand
        target = soft0 + span * head
        target = min(hard, max(soft0, target))
        try:
            _set_motor_current_soft_max(self.axis, target)
        except Exception:
            pass

    def _current_demand(
        self,
        lx: float,
        dlx_dt: float,
        vel_cmd_motor: float,
        prev_vel_cmd_motor: float,
        dt_s: float,
    ) -> float:
        a = abs(lx)
        b = min(1.0, abs(dlx_dt) / STICK_DLX_REF)
        c = 0.0
        if dt_s > 1e-6:
            c = min(1.0, abs(vel_cmd_motor - prev_vel_cmd_motor) / dt_s / VEL_CMD_ACCEL_REF)
        return max(a, 0.55 * b + 0.55 * c)

    def command_deg(
        self,
        column_deg: float,
        lx: float = 0.0,
        dlx_dt: float = 0.0,
        dt_s: float = 1.0 / CONTROL_HZ,
    ) -> None:
        """Send column angle (deg) vs startup as a trap-traj ``input_pos`` setpoint."""
        column_deg = clamp(column_deg, -PS5_STEERING_MAX_DEG, PS5_STEERING_MAX_DEG)
        if self.dry_run or self.axis is None:
            return

        target_motor = steering_deg_to_motor_turns(column_deg)
        vel_cmd_motor = (target_motor - self._prev_target_motor) / max(dt_s, 1e-6)
        self._prev_target_motor = target_motor

        self.axis.controller.input_pos = self.start_pos + target_motor

        # Pet the axis watchdog each frame — otherwise the ODrive idles
        # itself ~STEERING_WATCHDOG_S from now. That's the whole point of
        # the watchdog, so we only feed while actively driving.
        if self._watchdog_enabled:
            try:
                self.axis.watchdog_feed()
            except Exception:
                pass

        dem = self._current_demand(
            lx, dlx_dt, vel_cmd_motor, self._prev_vel_cmd_motor, dt_s,
        )
        self._prev_vel_cmd_motor = vel_cmd_motor
        self._apply_dynamic_current(dem)

    def arm_watchdog(self) -> None:
        """Enable the axis watchdog. Must be called immediately before the
        main control loop starts feeding, so the ~300 ms window isn't
        consumed by host-side setup (pygame display init, SysFont, etc.).
        """
        if self.dry_run or self.axis is None or not self._watchdog_available:
            return
        try:
            self.axis.watchdog_feed()  # prime
            self._axis_cfg.enable_watchdog = True
            self._watchdog_enabled = True
            print(
                f"[steering] axis watchdog ARMED @ "
                f"{STEERING_WATCHDOG_S * 1000:.0f} ms "
                f"(ODrive will IDLE if host goes silent)"
            )
        except Exception as e:
            print(f"[steering] watchdog arm failed: {e!r}")

    def rebase_zero(self) -> float:
        """Make the wheel's current encoder position the new 0°.

        Triangle during drive uses this when the operator has rotated the
        wheel mechanically (e.g. jogged it by hand, recovered from a trip)
        and wants to re-define "straight" without restarting the script.

        Also commands ``input_pos`` to the new zero and resets the internal
        target memory so the trap-traj planner doesn't keep slewing toward
        the old reference.
        """
        if self.dry_run or self.axis is None:
            self.start_pos = 0.0
            self._prev_target_motor = 0.0
            self._prev_vel_cmd_motor = 0.0
            return 0.0
        self.start_pos = float(self.axis.pos_estimate)
        self._prev_target_motor = 0.0
        self._prev_vel_cmd_motor = 0.0
        try:
            self.axis.controller.input_pos = self.start_pos
            if self._watchdog_enabled:
                self.axis.watchdog_feed()
        except Exception as e:
            print(f"[steering] rebase_zero: controller write failed: {e!r}")
        return self.start_pos

    def stop(self) -> None:
        if self.dry_run or self.axis is None:
            return
        # Disable the watchdog first so our own IDLE transition isn't a
        # race against the watchdog's auto-idle. No-op if it wasn't armed.
        if self._watchdog_enabled:
            try:
                self._axis_cfg.enable_watchdog = False
            except Exception:
                pass
            self._watchdog_enabled = False
        try:
            self.axis.controller.input_pos = self.start_pos
            time.sleep(0.25)
            if self._saved_current_soft is not None:
                _set_motor_current_soft_max(self.axis, self._saved_current_soft)
            self._controller_cfg.input_mode = self._InputMode.PASSTHROUGH
            self.axis.requested_state = self._AxisState.IDLE
        except Exception as e:
            print(f"[steering] stop failed: {e}")


# ---------------------------------------------------------------------------
# controller + UI
# ---------------------------------------------------------------------------

def init_controller(index: int):
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("ERROR: no controllers detected. Pair the DualSense first,")
        print("  then confirm with: uv run python scripts/ps5_controller_test.py --list")
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
        print("[ps5] WARNING: name doesn't look like a DualSense; axes may be mapped differently.")
    return js


def draw_ui(screen, font, font_small, state: dict) -> None:
    screen.fill(BG)

    title = font.render(f"PS5 drive — mode: {state['mode'].upper()}", True, ACCENT)
    screen.blit(title, (18, 14))
    sub = font_small.render(
        f"{'DRY RUN — nothing sent to hardware' if state['dry_run'] else 'live hardware'}",
        True, WARN if state['dry_run'] else MUTED,
    )
    screen.blit(sub, (18, 48))

    y = 82
    steer_mode = state.get("stick_steering", "integrated")
    steer_hint = "trap slew" if steer_mode == "integrated" else "absolute"
    lines = [
        ("steer",
         f"{state['steer_deg']:+6.1f}°   (stick X {state['lx']:+.2f}, {steer_hint}, "
         f"cap ±{PS5_STEERING_MAX_DEG:.0f}°)",
         TEXT if state['control_steering'] else MUTED),
        ("gas target",
         f"{state['gas']:.3f}    (R2 {state['r2']:.2f}, cap {state['gas_cap']:.3f})",
         GOOD if state['gas'] > 0.01 and state['control_pedals'] else MUTED),
        ("brake target",
         f"{state['brake']:.3f}    (L2 {state['l2']:.2f}, cap {BRAKE_POT_MAX:.3f})",
         WARN if state['brake'] > 0.01 and state['control_pedals'] else MUTED),
    ]
    for label, value, color in lines:
        screen.blit(font_small.render(label, True, MUTED), (18, y))
        screen.blit(font.render(value, True, color), (160, y - 4))
        y += 32

    y += 12
    caps = [
        f"GAS_POT_MAX         = {GAS_POT_MAX:.3f}   (hardware ceiling)",
        f"GLOBAL_SPEED_LIMIT  = {GLOBAL_SPEED_LIMIT:.3f}   (top-level, applies to every mode)",
        f"PS5_GAS_LIMIT       = {PS5_GAS_LIMIT:.3f}",
        f"FSD_GAS_LIMIT       = {FSD_GAS_LIMIT:.3f}   (reference — autonomy only)",
        f"effective PS5 cap   = {state['gas_cap']:.3f}   (min of the three above)",
    ]
    for c in caps:
        screen.blit(font_small.render(c, True, MUTED), (18, y))
        y += 18

    # Heartbeat indicator — only meaningful when pedals are being driven.
    if state["control_pedals"]:
        hb_age = state["hb_age_s"]
        if state["hb_faulted"]:
            hb_color = (235, 100, 100)
            hb_text = f"PEDAL LINK FAULTED — {state['hb_fault_reason']}"
        elif hb_age > HEARTBEAT_TIMEOUT_S:
            hb_color = (235, 100, 100)
            hb_text = f"HEARTBEAT STALE  {hb_age * 1000:.0f} ms  (firmware failsafe imminent)"
        elif hb_age > HEARTBEAT_TIMEOUT_S * 0.5:
            hb_color = WARN
            hb_text = f"heartbeat slow   {hb_age * 1000:.0f} ms"
        else:
            hb_color = GOOD
            hb_text = f"heartbeat OK     {hb_age * 1000:.0f} ms"
        screen.blit(font_small.render(hb_text, True, hb_color), (18, y + 6))

    hint = font_small.render(
        "△ Triangle → rezero steering     •     Esc / Q / close window → graceful stop.",
        True, MUTED,
    )
    screen.blit(hint, (18, WINDOW_H - 26))
    pygame.display.flip()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=MODES, default="full",
        help="Which subsystems to actually drive (default: full).",
    )
    parser.add_argument(
        "--index", type=int, default=0,
        help="Controller index when multiple are paired (default 0).",
    )
    parser.add_argument(
        "--arduino-port", default=None,
        help="Serial device for the Mega (e.g. /dev/tty.usbmodem1401). "
             "Auto-detects when omitted.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Don't open the ODrive or serial port — just read inputs and draw the UI.",
    )
    parser.add_argument(
        "--stick-steering", choices=STICK_STEERING, default="integrated",
        help="integrated = trap-traj target slew (hold at release); absolute = stick maps to angle.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    control_steering = args.mode in ("full", "steering")
    control_pedals = args.mode in ("full", "pedals")

    gas_cap = effective_gas_cap(PS5_GAS_LIMIT)
    print(
        f"[limits] effective PS5 gas cap = {gas_cap:.3f}  "
        f"(min of GAS_POT_MAX={GAS_POT_MAX}, "
        f"GLOBAL_SPEED_LIMIT={GLOBAL_SPEED_LIMIT}, "
        f"PS5_GAS_LIMIT={PS5_GAS_LIMIT})"
    )
    print(f"[mode] {args.mode}: steering={control_steering}, pedals={control_pedals}")
    print(f"[stick-steering] {args.stick_steering}")

    js = init_controller(args.index)

    pedals: PedalLink | None = None
    steering: SteeringLink | None = None
    exit_code = 0

    try:
        if control_pedals:
            pedals = PedalLink(args.arduino_port, dry_run=args.dry_run)
        if control_steering:
            steering = SteeringLink(
                dry_run=args.dry_run, stick_steering=args.stick_steering,
            )

        screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption(
            f"PS5 drive — {args.mode} ({args.stick_steering} steer)"
        )
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("Menlo", 22, bold=True)
        font_small = pygame.font.SysFont("Menlo", 14)

        steer_target_deg = 0.0
        if (
            args.stick_steering == "integrated"
            and control_steering
            and steering is not None
            and not steering.dry_run
        ):
            # Match trap target to the column angle at launch so we do not
            # command a jump toward 0° while the wheel is already offset.
            steer_target_deg = steering.column_deg_estimate()

        prev_lx = 0.0
        last_mono = time.monotonic()
        running = True

        # Arm the ODrive watchdog *now*, immediately before the feed loop
        # starts — all the heavyweight pygame/font setup above is already
        # done, so the first watchdog_feed() inside command_deg() lands
        # well within STEERING_WATCHDOG_S of this enable.
        if steering is not None:
            steering.arm_watchdog()

        # Hardware lock shared with the KeepAlive thread. Every main-loop
        # HW call (``pedals.send/poll``, ``steering.command_deg``, Triangle
        # rebase) acquires this, so the daemon's 50 Hz heartbeats can't
        # interleave with target writes and confuse the firmware/ODrive.
        hw_lock = threading.Lock()
        keepalive = KeepAlive(pedals, steering, hw_lock)
        keepalive.start()

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.type == pygame.JOYDEVICEREMOVED:
                    print("[ps5] controller disconnected — stopping.")
                    running = False
                elif event.type == pygame.JOYBUTTONDOWN and event.button == BUTTON_TRIANGLE:
                    # Triangle → re-zero the steering reference to whatever
                    # the wheel looks like right now. Integrated mode's
                    # target angle is also reset so the trap planner doesn't
                    # yank the wheel back to the old zero.
                    if control_steering and steering is not None:
                        with hw_lock:
                            new_zero = steering.rebase_zero()
                        steer_target_deg = 0.0
                        print(
                            f"[steering] △ Triangle: zero rebased to "
                            f"{new_zero:.4f} turns (wheel is now 0°)"
                        )

            # Drain any telemetry/events from the Mega (EVT,ESTOP,...).
            # Done before the liveness check so an e-stop press is handled
            # even on the frame the controller also drops.
            if pedals is not None:
                with hw_lock:
                    pedals.poll()
                if pedals.estop_active:
                    print("[main] E-STOP engaged on Mega — idling steering.")
                    running = False
                    break

            # Proactive liveness check: SDL sometimes misses silent BT drops.
            # Break out before we read axes so the finally-block gets to send
            # S to the Mega; don't wait for the firmware heartbeat to expire.
            alive, reason = controller_alive(js)
            if not alive:
                print(f"[ps5] controller lost ({reason}) — retracting pedals.")
                running = False
                break

            lx_raw = js.get_axis(AXIS_LEFT_X) if js.get_numaxes() > AXIS_LEFT_X else 0.0
            lx = apply_deadzone(lx_raw, STEER_STICK_DEADZONE)
            l2 = read_trigger(js, AXIS_L2, TRIGGER_MAX_L2)
            r2 = read_trigger(js, AXIS_R2, TRIGGER_MAX_R2)

            now = time.monotonic()
            dt_s = now - last_mono
            last_mono = now
            if dt_s <= 0.0 or dt_s > 0.25:
                dt_s = 1.0 / CONTROL_HZ

            raw_dlx = (lx - prev_lx) / dt_s if dt_s > 0 else 0.0
            dlx_dt = clamp(raw_dlx, -28.0, 28.0)

            if args.stick_steering == "integrated":
                steer_target_deg += lx * STEER_INTEGRATE_RATE_DPS * dt_s
                steer_target_deg = clamp(
                    steer_target_deg,
                    -PS5_STEERING_MAX_DEG,
                    PS5_STEERING_MAX_DEG,
                )
                if (
                    control_steering
                    and steering is not None
                    and not steering.dry_run
                ):
                    with hw_lock:
                        steer_deg = steering.column_deg_estimate()
                else:
                    steer_deg = steer_target_deg
            else:
                steer_deg = lx * PS5_STEERING_MAX_DEG

            prev_lx = lx

            gas = r2 * gas_cap
            brake = l2 * BRAKE_POT_MAX

            # Drive pedals first so the heartbeat ticks before we commit
            # to moving the steering wheel this frame.
            if control_pedals and pedals is not None:
                with hw_lock:
                    pedals.send(gas, brake)
                if pedals.faulted:
                    # Pedal link is gone — the Mega will retract both
                    # pedals on its own watchdog in ~300 ms, but we must
                    # also stop asking the ODrive to steer an uncommanded
                    # cart. Bail out of the loop; the finally-block idles
                    # the ODrive and sends a final S (best effort).
                    print("[main] pedal link faulted — aborting control loop.")
                    running = False

            if running and control_steering and steering is not None:
                cmd_deg = (
                    steer_target_deg
                    if args.stick_steering == "integrated"
                    else steer_deg
                )
                with hw_lock:
                    steering.command_deg(cmd_deg, lx=lx, dlx_dt=dlx_dt, dt_s=dt_s)

            draw_ui(screen, font, font_small, {
                "mode": args.mode,
                "dry_run": args.dry_run,
                "control_steering": control_steering,
                "control_pedals": control_pedals,
                "stick_steering": args.stick_steering,
                "lx": lx, "l2": l2, "r2": r2,
                "steer_deg": steer_deg,
                "gas": gas, "brake": brake,
                "gas_cap": gas_cap,
                "hb_age_s": pedals.heartbeat_age_s if pedals is not None else 0.0,
                "hb_faulted": pedals.faulted if pedals is not None else False,
                "hb_fault_reason": pedals.fault_reason if pedals is not None else None,
            })
            clock.tick(CONTROL_HZ)

    except KeyboardInterrupt:
        print("\n[main] KeyboardInterrupt — stopping.")
    except Exception as e:
        print(f"[main] fatal: {e}")
        exit_code = 1
    finally:
        # Stop the keep-alive thread FIRST so it can't race the final
        # stop() calls below. ``keepalive`` only exists once we reached
        # the arm_watchdog step — earlier failures skip straight to
        # hardware teardown.
        try:
            keepalive.stop()
            keepalive.join(timeout=1.0)
        except NameError:
            pass
        if pedals is not None:
            pedals.stop()
            pedals.close()
        if steering is not None:
            steering.stop()
        pygame.quit()
        print("[main] done.")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
