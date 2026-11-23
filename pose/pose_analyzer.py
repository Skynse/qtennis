# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["mediapipe>=0.10.14", "opencv-python", "numpy", "matplotlib"]
# ///
"""Quick MediaPipe pose analyzer.

Runs MediaPipe Pose Landmarker (heavy model) over a video and writes, next to --out:
  <name>_pose.mp4     annotated video (skeleton + live joint angles)
  <name>_pose.csv     per-frame landmarks (normalized image coords + visibility) and joint angles
  <name>_angles.png   joint angles over time
and prints a summary (detection rate, angle ranges, step cadence estimate).

    uv run pose/pose_analyzer.py VIDEO [--out DIR] [--width 1920]
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

import cv2
import matplotlib
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task")
MODEL_PATH = Path(__file__).parent / "models" / "pose_landmarker_heavy.task"

# MediaPipe Pose landmark indices
L = dict(nose=0, l_shoulder=11, r_shoulder=12, l_elbow=13, r_elbow=14, l_wrist=15, r_wrist=16,
         l_hip=23, r_hip=24, l_knee=25, r_knee=26, l_ankle=27, r_ankle=28,
         l_heel=29, r_heel=30, l_foot=31, r_foot=32)
NAMES = ["nose", "l_eye_in", "l_eye", "l_eye_out", "r_eye_in", "r_eye", "r_eye_out", "l_ear", "r_ear",
         "mouth_l", "mouth_r", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
         "l_pinky", "r_pinky", "l_index", "r_index", "l_thumb", "r_thumb", "l_hip", "r_hip",
         "l_knee", "r_knee", "l_ankle", "r_ankle", "l_heel", "r_heel", "l_foot", "r_foot"]

# angle name -> (a, vertex, b)
ANGLES = {
    "l_elbow": ("l_shoulder", "l_elbow", "l_wrist"),
    "r_elbow": ("r_shoulder", "r_elbow", "r_wrist"),
    "l_shoulder": ("l_hip", "l_shoulder", "l_elbow"),
    "r_shoulder": ("r_hip", "r_shoulder", "r_elbow"),
    "l_hip": ("l_shoulder", "l_hip", "l_knee"),
    "r_hip": ("r_shoulder", "r_hip", "r_knee"),
    "l_knee": ("l_hip", "l_knee", "l_ankle"),
    "r_knee": ("r_hip", "r_knee", "r_ankle"),
}

CONNECTIONS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
               (23, 25), (25, 27), (27, 29), (29, 31), (27, 31), (24, 26), (26, 28), (28, 30),
               (30, 32), (28, 32), (0, 11), (0, 12)]
LEFT_COLOR, RIGHT_COLOR, MID_COLOR = (40, 170, 255), (255, 120, 40), (220, 220, 220)   # BGR: orange, blue


def angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle ABC in degrees (2D, image plane, aspect-corrected pixel coords)."""
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n < 1e-9:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0))))


def side_color(i: int, j: int | None = None) -> tuple[int, int, int]:
    idx = [i] if j is None else [i, j]
    if all(k >= 11 and k % 2 == 1 for k in idx):
        return LEFT_COLOR
    if all(k >= 11 and k % 2 == 0 for k in idx):
        return RIGHT_COLOR
    return MID_COLOR


def draw(frame: np.ndarray, px: np.ndarray, vis: np.ndarray, angles: dict[str, float], min_vis: float) -> None:
    h = frame.shape[0]
    th = max(2, h // 300)
    for i, j in CONNECTIONS:
        if vis[i] > min_vis and vis[j] > min_vis:
            cv2.line(frame, tuple(px[i]), tuple(px[j]), side_color(i, j), th, cv2.LINE_AA)
    for i in range(11, 33):
        if vis[i] > min_vis:
            cv2.circle(frame, tuple(px[i]), th + 2, side_color(i), -1, cv2.LINE_AA)
    fs = h / 1400
    for name, (_, v, _) in ANGLES.items():
        k = L[v]
        if vis[k] > min_vis and not np.isnan(angles[name]):
            org = (int(px[k][0]) + 8, int(px[k][1]) - 8)
            txt = f"{angles[name]:.0f}"
            cv2.putText(frame, txt, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th + 2, cv2.LINE_AA)
            cv2.putText(frame, txt, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th - 1 or 1, cv2.LINE_AA)


def step_cadence(t: np.ndarray, ankle_dx: np.ndarray) -> tuple[int, float]:
    """Steps = sign changes of (left ankle x - right ankle x), i.e. the feet passing each other
    (side view). Returns (step count, steps per minute)."""
    ok = ~np.isnan(ankle_dx)
    if ok.sum() < 10:
        return 0, float("nan")
    tt, d = t[ok], ankle_dx[ok]
    d = np.convolve(d, np.ones(3) / 3, mode="same")          # light smoothing
    thresh = 0.15 * np.nanpercentile(np.abs(d), 90)           # hysteresis against jitter
    state, crossings = 0, []
    for ti, di in zip(tt, d):
        s = 1 if di > thresh else -1 if di < -thresh else 0
        if s and s != state:
            if state:
                crossings.append(ti)
            state = s
    if len(crossings) < 2:
        return len(crossings), float("nan")
    return len(crossings), 60.0 * (len(crossings) - 1) / (crossings[-1] - crossings[0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--out", help="output directory (default: next to the video)")
    ap.add_argument("--width", type=int, default=1920, help="width of the annotated output video")
    ap.add_argument("--min-vis", type=float, default=0.5, help="min landmark visibility to draw/use")
    args = ap.parse_args()

    src = Path(args.video.removeprefix("file://")).expanduser()
    out_dir = Path(args.out) if args.out else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = src.stem

    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading model to {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ow = min(args.width, W)
    oh = int(round(H * ow / W / 2) * 2)
    print(f"{src.name}: {W}x{H} @ {fps:.2f} fps, {n_frames} frames -> annotating at {ow}x{oh}")

    landmarker = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ))

    video_path = out_dir / f"{stem}_pose.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh))
    csv_path = out_dir / f"{stem}_pose.csv"
    csv_f = open(csv_path, "w", newline="")
    cw = csv.writer(csv_f)
    cw.writerow(["frame", "t", "detected"] + [f"{n}_{c}" for n in NAMES for c in ("x", "y", "z", "vis")]
                + [f"angle_{a}" for a in ANGLES])

    ts, ang_hist, ankle_dx = [], {a: [] for a in ANGLES}, []
    detected = 0
    t0 = time.time()
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        small = cv2.resize(frame, (ow, oh), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                          int(round(t * 1000)))

        angles = {a: float("nan") for a in ANGLES}
        row = [i, f"{t:.3f}", 0]
        if res.pose_landmarks:
            detected += 1
            lm = res.pose_landmarks[0]
            xyz = np.array([[p.x, p.y, p.z] for p in lm])
            vis = np.array([p.visibility for p in lm])
            px = np.round(xyz[:, :2] * [ow, oh]).astype(int)
            pxf = xyz[:, :2] * [ow, oh]                          # aspect-correct for angles
            for name, (a, v, b) in ANGLES.items():
                if min(vis[L[a]], vis[L[v]], vis[L[b]]) > args.min_vis:
                    angles[name] = angle_deg(pxf[L[a]], pxf[L[v]], pxf[L[b]])
            draw(small, px, vis, angles, args.min_vis)
            row[2] = 1
            row += [f"{val:.5f}" for p in zip(xyz, vis) for val in (*p[0], p[1])]
            ankle_dx.append(xyz[L["l_ankle"], 0] - xyz[L["r_ankle"], 0])
        else:
            row += [""] * (len(NAMES) * 4)
            ankle_dx.append(float("nan"))
        row += ["" if np.isnan(angles[a]) else f"{angles[a]:.2f}" for a in ANGLES]
        cw.writerow(row)

        ts.append(t)
        for a in ANGLES:
            ang_hist[a].append(angles[a])

        cv2.putText(small, f"t={t:5.2f}s  frame {i}", (12, oh - 14), cv2.FONT_HERSHEY_SIMPLEX,
                    oh / 1400, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(small)
        i += 1
        if i % 50 == 0:
            print(f"  {i}/{n_frames} frames  ({i / (time.time() - t0):.1f} fps)")

    cap.release()
    writer.release()
    csv_f.close()
    landmarker.close()

    ts_a = np.array(ts)
    plot_path = out_dir / f"{stem}_angles.png"
    fig, axs = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for ax, joint in zip(axs.flat, ["knee", "hip", "elbow", "shoulder"]):
        ax.plot(ts_a, ang_hist[f"l_{joint}"], label="left", color="#ffaa28")
        ax.plot(ts_a, ang_hist[f"r_{joint}"], label="right", color="#2878ff")
        ax.set_title(f"{joint} angle [deg]")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")
    for ax in axs[-1]:
        ax.set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=110)

    # OpenCV writes MPEG-4 Part 2; re-encode to H.264 so it plays in browsers/most players
    if shutil.which("ffmpeg"):
        tmp = video_path.with_suffix(".tmp.mp4")
        r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video_path), "-c:v", "libx264",
                            "-crf", "20", "-preset", "fast", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                            str(tmp)])
        if r.returncode == 0:
            tmp.replace(video_path)

    steps, cadence = step_cadence(ts_a, np.array(ankle_dx))
    print(f"\nProcessed {i} frames in {time.time() - t0:.1f} s; pose detected in {detected}/{i} "
          f"({100 * detected / max(i, 1):.0f}%)")
    print(f"{'angle':<12}{'min':>7}{'mean':>7}{'max':>7}   (deg, image plane)")
    for a in ANGLES:
        v = np.array(ang_hist[a])
        if np.isfinite(v).any():
            print(f"{a:<12}{np.nanmin(v):7.0f}{np.nanmean(v):7.0f}{np.nanmax(v):7.0f}")
    print(f"Steps (feet passing, side view): {steps}   cadence ~ {cadence:.0f} steps/min")
    print(f"\nWrote:\n  {video_path}\n  {csv_path}\n  {plot_path}")


if __name__ == "__main__":
    main()
