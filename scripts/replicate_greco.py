"""Fit the Greco+26 mixture model to the data, optionally with a host bootstrap.
python scripts/replicate_greco.py data/planets.csv cumulative.csv [--config greco refined ...] [--boot 100]"""
import argparse, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from radvalley.data import greco_sample
from radvalley.mixture import fit
from radvalley.configs import CONFIGS

ap = argparse.ArgumentParser()
ap.add_argument("planets"); ap.add_argument("koi")
ap.add_argument("--config", nargs="+", default=list(CONFIGS)); ap.add_argument("--boot", type=int, default=0)
ap.add_argument("--seed", type=int, default=4)
a = ap.parse_args()
d = greco_sample(pd.read_csv(a.planets), a.koi)
print(f"{len(d)} planets; Greco+26 (581 planets): f = 0.43 / 0.75 / 0.67, C = 0.64")
q = lambda x: f"{np.median(x):+.3f} [{np.percentile(x, 16):+.3f}, {np.percentile(x, 84):+.3f}]"
for name in a.config:
    cfg = CONFIGS[name]
    th, r = fit(d, cfg)
    f = r["fbar"]
    msg = (f"{name:14s} f(<1, 1-2, 2-4 Gyr) = {np.round(f, 3)}; Δf = f(2-4) - f(<1) = {f[2] - f[0]:+.3f}; C = {f[0] / f[2]:.2f}; "
           f"beta_P {r['bP']:+.2f}, gamma {r['gam']:+.2f}")
    if r["peak_offsets"] is not None:
        o = r["peak_offsets"]
        msg += f"\n{'':14s} peak shifts vs 2-4 Gyr [dex]: SE <1 {o[0, 0]:+.3f}, 1-2 {o[1, 0]:+.3f}; SN <1 {o[0, 1]:+.3f}, 1-2 {o[1, 1]:+.3f}"
    print(msg, flush=True)
    if a.boot:
        kic = d.kic.values; uh = np.unique(kic); idx = {h: np.where(kic == h)[0] for h in uh}
        rng = np.random.default_rng(a.seed); F, O = [], []
        for _ in range(a.boot):
            p = np.concatenate([idx[h] for h in rng.choice(uh, len(uh))])
            rr = fit(d.iloc[p].reset_index(drop=True), cfg, th0=th)[1]
            F.append(rr["fbar"]); O.append(rr["peak_offsets"])
        F = np.array(F); dF = F[:, 2] - F[:, 0]
        msg = f"{'':14s} bootstrap ({a.boot} host resamples, median [16, 84]): Δf {q(dF)}, P(Δf > 0) = {np.mean(dF > 0):.2f}; C {q(F[:, 0] / F[:, 2])}"
        if O[0] is not None:
            O = np.array(O)
            msg += f"; SE shift <1 {q(O[:, 0, 0])}; SN shift <1 {q(O[:, 0, 1])}"
        print(msg, flush=True)
