# /// script
# requires-python = ">=3.10"
# dependencies = ["rerun-sdk>=0.42", "pyserial>=3.5", "numpy"]
# ///
"""Stream ICM-20948 (+ AK09916 magnetometer) data from the QT Py ESP32-S3 and visualize it in Rerun.

The firmware prints lines of the form
    D,<micros>,ax,ay,az,gx,gy,gz[,mx,my,mz]      (g, deg/s, uT)
Orientation is estimated with a Mahony filter: gyro, plus gravity from the accelerometer for
tilt, plus the magnetometer for heading (world +X = magnetic north, +Z = up). A virtual
"racket" arm is attached to the sensor and the path of its tip is drawn as a fading trail.

Startup:
  1. Magnetometer calibration (only if there is no saved calibration, or with --mag-cal):
     rotate the IMU slowly through every orientation (figure-8s, roll it over) for ~20 s.
     The result is saved next to this script in mag_cal.json. Recalibrate whenever the
     sensor's surroundings change (e.g. after mounting it on the racket).
  2. Gyro calibration: keep the IMU still for ~1 s.

    python viz/imu_trace.py                 # auto-detects the QT Py
    python viz/imu_trace.py --mag-cal       # force a new magnetometer calibration
    python viz/imu_trace.py --no-mag        # gyro + accel only (heading drifts)
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import serial

BG = np.array([13, 16, 20], dtype=np.float32)          # approx. Rerun 3D background
SLOW = np.array([40, 160, 255], dtype=np.float32)      # trail colour at low tip speed
FAST = np.array([255, 210, 40], dtype=np.float32)      # ...and at high tip speed
MAG_CAL_PATH = Path(__file__).with_name("mag_cal.json")

AXES = {
    "x": [1, 0, 0], "-x": [-1, 0, 0],
    "y": [0, 1, 0], "-y": [0, -1, 0],
    "z": [0, 0, 1], "-z": [0, 0, -1],
}


# ---------------------------------------------------------------- quaternion math (w, x, y, z)

def q_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def q_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def q_from_accel(a: np.ndarray) -> np.ndarray:
    """Body->world rotation that maps the measured 'up' (specific force at rest) onto world +Z."""
    u = a / np.linalg.norm(a)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(u, z)
    s = np.linalg.norm(axis)
    c = float(np.dot(u, z))
    if s < 1e-8:
        return np.array([1.0, 0, 0, 0]) if c > 0 else np.array([0.0, 1, 0, 0])
    angle = np.arctan2(s, c)
    return np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis / s])


def mag_heading_error(q: np.ndarray, mag: np.ndarray) -> float:
    """Angle (rad) of the horizontal magnetic field in the estimated world frame; 0 when aligned with +X."""
    h = q_to_mat(q) @ mag
    return float(np.arctan2(h[1], h[0]))


def q_align_heading(q: np.ndarray, mag: np.ndarray) -> np.ndarray:
    """Rotate q about world Z so that magnetic north lies along world +X."""
    psi = mag_heading_error(q, mag)
    qz = np.array([np.cos(-psi / 2), 0.0, 0.0, np.sin(-psi / 2)])
    return q_mul(qz, q)


class Mahony:
    def __init__(self, kp: float = 1.5, ki: float = 0.02, kp_mag: float = 0.5):
        self.kp, self.ki, self.kp_mag = kp, ki, kp_mag
        self.q = np.array([1.0, 0, 0, 0])
        self.integral = np.zeros(3)

    def update(self, gyro_rad: np.ndarray, accel_g: np.ndarray, dt: float,
               mag: np.ndarray | None = None) -> None:
        w, x, y, z = self.q
        omega = gyro_rad.copy()

        # Tilt: gravity ("up") direction predicted in the body frame vs. the accelerometer.
        # Only trusted when the accelerometer is mostly measuring gravity (not mid-swing).
        v = np.array([2 * (x * z - w * y), 2 * (w * x + y * z), w * w - x * x - y * y + z * z])
        a_norm = np.linalg.norm(accel_g)
        if abs(a_norm - 1.0) < 0.15:
            e = np.cross(accel_g / a_norm, v)
            self.integral += self.ki * e * dt
            omega += self.kp * e + self.integral

        # Heading: rotate about world Z only, so magnetic disturbances can't corrupt tilt.
        if mag is not None:
            psi = mag_heading_error(self.q, mag)
            omega += self.kp_mag * (q_to_mat(self.q).T @ np.array([0.0, 0.0, -psi]))

        self.q = self.q + 0.5 * q_mul(self.q, np.concatenate([[0.0], omega])) * dt
        self.q /= np.linalg.norm(self.q)


# ---------------------------------------------------------------- magnetometer calibration

class MagCal:
    """Hard-iron offset + diagonal soft-iron scale:  m_cal = (m_raw - offset) * scale."""

    def __init__(self, offset=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), field_ut: float = 50.0):
        self.offset = np.asarray(offset, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)
        self.field_ut = float(field_ut)

    def apply(self, m: np.ndarray) -> np.ndarray:
        return (m - self.offset) * self.scale

    @classmethod
    def fit(cls, pts: np.ndarray) -> "MagCal":
        # Least-squares sphere fit for the offset: |m|^2 = 2 m.c + (r^2 - |c|^2)
        A = np.hstack([2 * pts, np.ones((len(pts), 1))])
        sol, *_ = np.linalg.lstsq(A, (pts ** 2).sum(axis=1), rcond=None)
        offset = sol[:3]
        # Per-axis radii from robust extents around that centre -> diagonal scale
        centred = pts - offset
        radii = (np.percentile(centred, 98, axis=0) - np.percentile(centred, 2, axis=0)) / 2
        field = float(radii.mean())
        return cls(offset, field / radii, field)

    @classmethod
    def load(cls, path: Path) -> "MagCal | None":
        try:
            d = json.loads(path.read_text())
            return cls(d["offset_ut"], d["scale"], d["field_ut"])
        except (OSError, KeyError, ValueError):
            return None

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({
            "offset_ut": self.offset.round(3).tolist(),
            "scale": self.scale.round(4).tolist(),
            "field_ut": round(self.field_ut, 2),
        }, indent=2) + "\n")


# ---------------------------------------------------------------- serial

def find_port() -> str:
    for pattern in ("/dev/serial/by-id/*QT_Py_ESP32-S3*", "/dev/ttyACM*", "/dev/ttyUSB*"):
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[0]
    sys.exit("No serial port found; pass --port")


LEGACY_RE = re.compile(
    r"A\[g\]\s+(\S+)\s+(\S+)\s+(\S+)\s+\|\s+G\[dps\]\s+(\S+)\s+(\S+)\s+(\S+)")
_legacy_warned = False


def parse(line: str):
    """Returns (t_us, accel_g, gyro_dps, mag_ut_or_None) or None.

    Accepts the firmware's 'D,...' lines (with or without magnetometer fields) and, as a
    fallback, the older human-readable 'A[g] ... | G[dps] ...' lines (timestamped on the host).
    """
    global _legacy_warned
    if not line.startswith("D,"):
        m = LEGACY_RE.search(line)
        if not m:
            return None
        if not _legacy_warned:
            _legacy_warned = True
            print("Note: board is running old firmware (A[g]/G[dps] text, no magnetometer).\n"
                  "      Reflash firmware/firmware.ino for 200 Hz, swing-proof ranges and the AK09916.")
        try:
            vals = np.array([float(g) for g in m.groups()])
        except ValueError:
            return None
        return int(time.monotonic() * 1e6) & 0xFFFFFFFF, vals[:3], vals[3:], None

    parts = line.split(",")
    if len(parts) not in (8, 11):
        return None
    try:
        t_us = int(parts[1])
        vals = np.array([float(p) for p in parts[2:]])
    except ValueError:
        return None
    mag = vals[6:9] if len(vals) == 9 else None
    return t_us, vals[:3], vals[3:6], mag


class SampleStream:
    """Reads samples from the serial port with a wrap-safe device clock (seconds)."""

    def __init__(self, ser: serial.Serial):
        self.ser = ser
        self.last_us = None
        self.t = 0.0

    def __iter__(self):
        while True:
            raw = self.ser.readline()
            if not raw:
                continue
            line = raw.decode(errors="replace").strip()
            sample = parse(line)
            if sample is None:
                if line:
                    print(f"[device] {line}")
                continue
            t_us, acc, gyr, mag = sample
            if self.last_us is None:
                self.last_us = t_us
            dt = ((t_us - self.last_us) & 0xFFFFFFFF) * 1e-6
            self.last_us = t_us
            self.t += dt
            yield self.t, dt, acc, gyr, mag


# ---------------------------------------------------------------- rerun

def blueprint() -> rrb.Blueprint:
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(name="IMU motion", origin="world"),
            rrb.Vertical(
                rrb.TimeSeriesView(name="Accel [g]", origin="plots/accel"),
                rrb.TimeSeriesView(name="Gyro [deg/s]", origin="plots/gyro"),
                rrb.TimeSeriesView(name="Mag (calibrated) [uT]", origin="plots/mag"),
                rrb.TimeSeriesView(name="Heading [deg from mag. north]", origin="plots/heading"),
                rrb.TimeSeriesView(name="Tip speed [arm/s]", origin="plots/tip_speed"),
            ),
            column_shares=[3, 2],
        ),
        collapse_panels=True,
    )


def log_static_scene(arm: np.ndarray, arm_len: float) -> None:
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    # World frame: +X = magnetic north, +Y = west, +Z = up
    rr.log("world/origin", rr.Arrows3D(vectors=np.eye(3) * 0.25,
                                       colors=[[180, 60, 60], [60, 180, 60], [60, 60, 180]],
                                       labels=["N", "W", "Up"]), static=True)
    # Sensor board (Adafruit ICM-20948 breakout is ~25.4 x 17.8 mm) + its axes + the virtual arm
    rr.log("world/imu/board", rr.Boxes3D(half_sizes=[0.127, 0.089, 0.015], colors=[[30, 110, 60]],
                                         fill_mode="solid"), static=True)
    rr.log("world/imu/axes", rr.Arrows3D(vectors=np.eye(3) * 0.2,
                                         colors=[[255, 60, 60], [60, 255, 60], [60, 120, 255]]), static=True)
    rr.log("world/imu/arm", rr.LineStrips3D([[[0, 0, 0], (arm * arm_len).tolist()]],
                                            radii=0.008, colors=[[200, 200, 200]]), static=True)
    rr.log("plots/accel", rr.SeriesLines(names=["x", "y", "z"]), static=True)
    rr.log("plots/gyro", rr.SeriesLines(names=["x", "y", "z"]), static=True)
    rr.log("plots/mag", rr.SeriesLines(names=["x", "y", "z"]), static=True)


def log_trail(trail: deque, now: float, trail_s: float, max_speed: float) -> None:
    if len(trail) < 2:
        return
    t = np.fromiter((p[0] for p in trail), dtype=np.float64, count=len(trail))
    pos = np.array([p[1] for p in trail], dtype=np.float32)
    spd = np.fromiter((p[2] for p in trail), dtype=np.float32, count=len(trail))

    fade = np.clip(1.0 - (now - t) / trail_s, 0.0, 1.0)[:-1, None] ** 1.5   # 1 = newest, 0 = oldest
    heat = np.clip(spd / max_speed, 0.0, 1.0)[:-1, None]
    base = SLOW + (FAST - SLOW) * heat
    rgb = BG + (base - BG) * fade
    alpha = 255 * fade
    colors = np.hstack([rgb, alpha]).astype(np.uint8)

    segments = np.stack([pos[:-1], pos[1:]], axis=1)
    radii = (0.003 + 0.012 * fade[:, 0]).astype(np.float32)
    rr.log("world/trail", rr.LineStrips3D(segments, radii=radii, colors=colors))
    rr.log("world/tip", rr.Points3D(pos[-1:], radii=0.03, colors=[[255, 255, 255]]))


# ---------------------------------------------------------------- startup phases

def calibrate_mag(samples, seconds: float) -> MagCal:
    print(f"\nMagnetometer calibration: slowly rotate the IMU through every orientation "
          f"(figure-8s, roll it over) for {seconds:.0f} s ...")
    pts: list[np.ndarray] = []
    last = None
    t_start = None
    next_report = 0.0
    for t, _dt, _acc, _gyr, mag in samples:
        if mag is None:
            continue
        if t_start is None:
            t_start = t
        if last is None or not np.array_equal(mag, last):   # firmware repeats mag between updates
            pts.append(mag)
            last = mag
        elapsed = t - t_start
        if elapsed >= next_report:
            next_report += 1.0
            arr = np.array(pts)
            span = arr.max(axis=0) - arr.min(axis=0)
            print(f"  {elapsed:4.0f} s  samples {len(pts):5d}  axis span [uT] {np.round(span, 1)}")
            rr.set_time("device_time", duration=t)
            rr.log("world/mag_cal", rr.Points3D(arr / 50.0, radii=0.01, colors=[[255, 120, 200]]))
        if elapsed >= seconds:
            arr = np.array(pts)
            span = arr.max(axis=0) - arr.min(axis=0)
            if span.min() < 40.0:      # Earth's field is ~25-65 uT, so each axis should swing ~2x that
                print(f"  Not enough rotation coverage (min axis span {span.min():.1f} uT, want >= 40). "
                      f"Keep rotating ...")
                seconds += 10.0
                continue
            cal = MagCal.fit(arr)
            rr.log("world/mag_cal", rr.Clear(recursive=True))
            return cal
    raise RuntimeError("serial stream ended")


def calibrate_gyro(samples, seconds: float):
    print(f"\nKeep the IMU still for {seconds:.1f} s to calibrate the gyro ...")
    while True:
        acc_l, gyr_l, mag_l = [], [], []
        t_start = None
        for t, _dt, acc, gyr, mag in samples:
            if t_start is None:
                t_start = t
            acc_l.append(acc)
            gyr_l.append(gyr)
            if mag is not None:
                mag_l.append(mag)
            if t - t_start >= seconds:
                break
        g = np.array(gyr_l)
        if g.std(axis=0).max() > 3.0:
            print("Too much motion during calibration, retrying ...")
            continue
        mag_mean = np.array(mag_l).mean(axis=0) if mag_l else None
        return g.mean(axis=0), np.array(acc_l).mean(axis=0), mag_mean


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", help="serial port (default: auto-detect)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--trail", type=float, default=2.0, help="seconds of trail to show")
    ap.add_argument("--arm-axis", choices=AXES, default="x", help="sensor axis the virtual arm points along")
    ap.add_argument("--arm-length", type=float, default=1.0)
    ap.add_argument("--calib", type=float, default=1.0, help="seconds of stillness for gyro bias calibration")
    ap.add_argument("--mag-cal", action="store_true", help="force a new magnetometer calibration")
    ap.add_argument("--mag-cal-time", type=float, default=20.0, help="seconds of magnetometer calibration")
    ap.add_argument("--no-mag", action="store_true", help="ignore the magnetometer")
    ap.add_argument("--fps", type=float, default=60.0, help="3D trail refresh rate")
    args = ap.parse_args()

    port = args.port or find_port()
    arm = np.array(AXES[args.arm_axis], dtype=np.float64)

    rr.init("qtennis_imu")
    rr.spawn(default_blueprint=blueprint())
    log_static_scene(arm, args.arm_length)

    print(f"Opening {port} ...")
    ser = serial.Serial(port, args.baud, timeout=0.5)
    samples = iter(SampleStream(ser))

    try:
        # Find out whether the firmware sends magnetometer data
        _t, _dt, _acc, _gyr, mag0 = next(samples)
        use_mag = mag0 is not None and not args.no_mag
        if mag0 is None and not args.no_mag:
            print("No magnetometer data in the stream: running gyro + accel only (heading will drift).")

        cal = None
        if use_mag:
            cal = None if args.mag_cal else MagCal.load(MAG_CAL_PATH)
            if cal is None:
                cal = calibrate_mag(samples, args.mag_cal_time)
                cal.save(MAG_CAL_PATH)
                print(f"Saved {MAG_CAL_PATH}")
            else:
                print(f"Loaded magnetometer calibration from {MAG_CAL_PATH} (--mag-cal to redo)")
            print(f"  offset [uT] {np.round(cal.offset, 2)}  scale {np.round(cal.scale, 3)}  "
                  f"field {cal.field_ut:.1f} uT")

        bias, acc_mean, mag_mean = calibrate_gyro(samples, args.calib)
        filt = Mahony()
        filt.q = q_from_accel(acc_mean)
        if use_mag and mag_mean is not None:
            filt.q = q_align_heading(filt.q, cal.apply(mag_mean))
        print(f"Gyro bias [deg/s]: {np.round(bias, 3)}  -> streaming. Ctrl-C to quit.")

        trail: deque = deque()
        prev_tip = None
        last_render = 0.0
        max_speed = 25.0 * args.arm_length  # arm-lengths/s mapped to the "hot" colour
        mag_rejected = 0
        last_reject_report = 0.0

        for t, dt, acc, gyr_dps, mag_raw in samples:
            gyr = gyr_dps - bias

            mag = None
            if use_mag and mag_raw is not None:
                mag = cal.apply(mag_raw)
                # Skip samples near magnets / steel, where the field strength is clearly wrong
                if abs(np.linalg.norm(mag) / cal.field_ut - 1.0) > 0.25:
                    mag_rejected += 1
                    mag_for_filter = None
                else:
                    mag_for_filter = mag
            else:
                mag_for_filter = None

            filt.update(np.radians(gyr), acc, min(dt, 0.05), mag_for_filter)

            rot = q_to_mat(filt.q)
            tip = rot @ arm * args.arm_length
            speed = 0.0 if prev_tip is None or dt <= 0 else float(np.linalg.norm(tip - prev_tip) / dt)
            prev_tip = tip

            rr.set_time("device_time", duration=t)
            w, x, y, z = filt.q
            rr.log("world/imu", rr.Transform3D(quaternion=rr.Quaternion(xyzw=[x, y, z, w])))
            rr.log("plots/accel", rr.Scalars(acc))
            rr.log("plots/gyro", rr.Scalars(gyr))
            rr.log("plots/tip_speed", rr.Scalars(speed))
            # Heading of the arm, measured from magnetic north (world +X) towards west (+Y)
            rr.log("plots/heading", rr.Scalars(np.degrees(np.arctan2(tip[1], tip[0]))))
            if mag is not None:
                rr.log("plots/mag", rr.Scalars(mag))

            trail.append((t, tip, speed))
            while trail and t - trail[0][0] > args.trail:
                trail.popleft()

            now = time.monotonic()
            if now - last_render >= 1.0 / args.fps:
                last_render = now
                log_trail(trail, t, args.trail, max_speed)
                if mag is not None:
                    # Measured field in the world frame: should hold steady, pointing north and dipping
                    rr.log("world/mag_field", rr.Arrows3D(vectors=[(rot @ mag / cal.field_ut * 0.6).tolist()],
                                                          colors=[[255, 120, 200]],
                                                          labels=["B"]))
                if mag_rejected and now - last_reject_report > 2.0:
                    print(f"  (ignored {mag_rejected} disturbed magnetometer samples)")
                    mag_rejected = 0
                    last_reject_report = now
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()


if __name__ == "__main__":
    main()
