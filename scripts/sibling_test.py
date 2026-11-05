"""Is the confirmed-sibling term biased by stellar noise? No-age-trend mocks (true C = 1).
python scripts/sibling_test.py data/planets.csv cumulative.csv [--n 20]"""
import argparse, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from radvalley.data import greco_sample
from radvalley.mocks import sibling_mocks

ap = argparse.ArgumentParser()
ap.add_argument("planets"); ap.add_argument("koi"); ap.add_argument("--n", type=int, default=20)
a = ap.parse_args()
d = greco_sample(pd.read_csv(a.planets), a.koi)
yng, old = d.age_use < 1, d.age_use >= 2
print(f"{len(d)} planets; confirmed-sibling fraction: < 1 Gyr {d.g[yng].mean():.2f}, 2-4 Gyr {d.g[old].mean():.2f}")
Cs, Ct = sibling_mocks(d, a.n)
print(f"no-trend mocks ({a.n}): C with detected siblings = {Cs.mean():.3f} ± {Cs.std() / np.sqrt(a.n):.3f}; "
      f"with true multiplicity = {Ct.mean():.3f} ± {Ct.std() / np.sqrt(a.n):.3f}  (no bias -> 1)")
