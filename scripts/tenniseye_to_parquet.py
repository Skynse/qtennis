"""Convert the TennisEye dataset (new_data_all_7.mat) to Parquet.

Source: https://github.com/hongyang-zhao/TennisEye  (Zhao et al., IPSN 2019)
License: non-commercial research use only, NO redistribution -- keep the .mat and the
generated parquet files out of any public repo (data/ is gitignored).

Dataset layout (per the TennisEye README / evaluation code):
  all_data{player}{shot}        100 x 6 double: ax ay az [m/s^2, +-16 g], gx gy gz [deg/s, +-2000]
                                100 Hz racket-handle IMU, one 1 s window per shot
  all_data_label{player}{shot}  [stroke_type, out_speed, out_spin, in_speed, in_spin]
                                stroke_type: 0 serve, 1 forehand, 2 backhand, 3 forehand volley,
                                4 backhand volley; speeds in mph (code divides by 2.24 for m/s),
                                spin in rpm (PlaySight ground truth); 0 = not recorded

Outputs (in the same folder as the .mat by default):
  tenniseye_shots.parquet    one row per shot: ids + labels
  tenniseye_samples.parquet  one row per IMU sample (long format), joinable on shot_id

    python scripts/tenniseye_to_parquet.py [path/to/new_data_all_7.mat] [--out DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

DEFAULT_MAT = Path(__file__).resolve().parent.parent / "data" / "tenniseye" / "new_data_all_7.mat"
STROKES = {0: "serve", 1: "forehand", 2: "backhand", 3: "forehand_volley", 4: "backhand_volley"}
CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz"]
FS = 100.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mat", nargs="?", default=str(DEFAULT_MAT))
    ap.add_argument("--out", help="output directory (default: next to the .mat)")
    args = ap.parse_args()

    mat_path = Path(args.mat)
    out = Path(args.out) if args.out else mat_path.parent
    out.mkdir(parents=True, exist_ok=True)

    m = sio.loadmat(mat_path)
    data, labels = m["all_data"], m["all_data_label"]

    shots, samples = [], []
    shot_id = 0
    for p in range(data.shape[0]):
        d, lab = data[p, 0], labels[p, 0]
        assert d.shape[0] == lab.shape[0], f"player {p + 1}: data/label count mismatch"
        for i in range(d.shape[0]):
            x = np.asarray(d[i, 0], dtype=np.float64)
            y = np.ravel(lab[i, 0]).astype(np.int64)
            assert x.shape[1] == 6 and y.shape == (5,), (p, i, x.shape, y.shape)
            shots.append(dict(
                shot_id=shot_id, player_id=p + 1, player_shot_idx=i + 1,
                stroke_type=int(y[0]), stroke=STROKES[int(y[0])],
                out_speed_mph=int(y[1]), out_spin_rpm=int(y[2]),
                in_speed_mph=int(y[3]), in_spin_rpm=int(y[4]),
                n_samples=len(x),
            ))
            n = len(x)
            samples.append(pd.DataFrame({
                "shot_id": np.full(n, shot_id, dtype=np.int32),
                "sample": np.arange(n, dtype=np.int16),
                "t_s": np.arange(n) / FS,
                **{c: x[:, k] for k, c in enumerate(CHANNELS)},
            }))
            shot_id += 1

    shots_df = pd.DataFrame(shots)
    shots_df["stroke"] = shots_df["stroke"].astype("category")
    samples_df = pd.concat(samples, ignore_index=True)

    shots_path = out / "tenniseye_shots.parquet"
    samples_path = out / "tenniseye_samples.parquet"
    shots_df.to_parquet(shots_path, index=False)
    samples_df.to_parquet(samples_path, index=False)

    print(f"{len(shots_df)} shots from {shots_df.player_id.nunique()} players, {len(samples_df)} samples")
    print(shots_df.groupby("stroke", observed=True).size().to_string())
    print(f"Wrote {shots_path}\n      {samples_path}")


if __name__ == "__main__":
    main()
