"""Analyse mock catalogues from the physical models with the same estimator as the data.
python scripts/model_mocks.py data/planets.csv cumulative.csv params/cp_standard.json [...] [--free SE SN] [--n 16]"""
import argparse, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from radvalley.data import greco_sample
from radvalley.hosts import load_params
from radvalley.mixture import fit
from radvalley.mocks import model_mocks, CFG

ap = argparse.ArgumentParser()
ap.add_argument("planets"); ap.add_argument("koi"); ap.add_argument("params", nargs="+")
ap.add_argument("--free", nargs="*", default=None, help="free peak means: SE, SN or both")
ap.add_argument("--n", type=int, default=16); ap.add_argument("--physics", default="standard")
ap.add_argument("--structure", default="gs19"); ap.add_argument("--quiet", action="store_true")
a = ap.parse_args()
planets = pd.read_csv(a.planets)
d = greco_sample(planets, a.koi)
cfg = dict(CFG, free_peaks=tuple(a.free)) if a.free else CFG
_, r = fit(d, cfg); f = r["fbar"]
msg = f"DATA ({len(d)} planets): f = {np.round(f, 3)}; Δf = {f[2] - f[0]:+.3f}"
if a.free:
    o = r["peak_offsets"]; msg += f"; SE shift <1 Gyr {o[0, 0]:+.3f}, SN shift <1 Gyr {o[0, 1]:+.3f}"
print(msg, flush=True)
pm = lambda x: f"{x.mean():+.3f} ± {x.std():.3f}"
for path in a.params:
    model, th = load_params(path)
    F, O = model_mocks(planets, d, model, th, a.n, free_peaks=a.free, physics=a.physics, structure=a.structure, verbose=not a.quiet)
    msg = f"{os.path.basename(path):22s} mock f = {np.round(F.mean(0), 3)}; Δf = {pm(F[:, 2] - F[:, 0])}"
    if a.free:
        msg += f"; SE shift <1 Gyr {pm(O[:, 0, 0])}; SN shift <1 Gyr {pm(O[:, 0, 1])}"
    print(msg, flush=True)
