"""Build the planet table from the three public inputs.
python scripts/build_sample.py --bouma apjad855ft1_mrt.txt --berger ajab8a33t2_mrt.txt --koi cumulative.csv --out data/planets.csv"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from radvalley.data import build_planets, greco_sample

ap = argparse.ArgumentParser()
ap.add_argument("--bouma", required=True); ap.add_argument("--berger", required=True); ap.add_argument("--koi", required=True)
ap.add_argument("--out", default="data/planets.csv"); ap.add_argument("--rmin", type=float, default=0.5)
a = ap.parse_args()
d = build_planets(a.bouma, a.berger, a.koi, rmin=a.rmin)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
d.to_csv(a.out, index=False)
g = greco_sample(d, a.koi)
print(f"{len(d)} planets ({a.rmin}-4 R_earth, P < 100 d, reliable Bouma+24 ages) -> {a.out}")
print(f"Greco sample (confirmed, 0.1-4 Gyr, sigma_R/R <= 0.25): {len(g)} planets; per age bin "
      f"{[int(((g.age_use >= lo) & (g.age_use < hi)).sum()) for lo, hi in [(0.1, 1), (1, 2), (2, 4)]]}")
