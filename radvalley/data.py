"""Build the planet table: Bouma+24 rotation ages + Berger+20 stellar properties + Kepler DR25 KOI table.

Inputs (see README for download links):
  bouma   Bouma et al. 2024, ApJ 976, 234, Table 1 (machine-readable, apjad855ft1_mrt.txt)
  berger  Berger et al. 2020, AJ 159, 280, Table 2 (machine-readable, ajab8a33t2_mrt.txt)
  koi     NASA Exoplanet Archive cumulative KOI table, CSV with all columns
"""
import numpy as np
import pandas as pd

# Bouma+24 Table 1 byte ranges (1-based, inclusive), from the MRT header
BOUMA_COLS = [("kic", 1, 8), ("koi", 10, 18), ("kepler", 20, 33), ("disp", 35, 43), ("teff", 45, 48), ("e_teff", 50, 52),
              ("prot", 74, 78), ("e_prot", 80, 84), ("r_prot", 86, 104), ("ew_li", 106, 109),
              ("tgyro", 115, 118), ("E_tgyro", 120, 122), ("e_tgyro", 124, 126), ("tli", 138, 141),
              ("R", 153, 159), ("r_R", 161, 179), ("P", 181, 187), ("starflag", 189, 192), ("planetflag", 194, 194),
              ("hires", 196, 196), ("consist", 198, 202), ("f_tgyro", 204, 209), ("f_tli", 211, 215)]

# Berger+20 Table 2 (0-based, end-exclusive slices)
BERGER_COLS = [("kic", 0, 8), ("mstar_b", 9, 14), ("teff_b", 28, 35), ("logg_b", 51, 57), ("feh_b", 71, 77),
               ("rstar_b", 91, 98), ("logL_b", 137, 143), ("gof_b", 208, 214)]

R_SUN_IN_R_EARTH = 109.076


def read_koi(path, cols):
    """Cumulative KOI CSV; skips the '#' header block the Exoplanet Archive prepends (if present)."""
    with open(path) as f:
        skip = 0
        while f.readline().startswith("#"):
            skip += 1
    return pd.read_csv(path, skiprows=skip, usecols=cols)


def _mrt_rows(path):
    lines = open(path).read().splitlines()
    last = max(i for i, l in enumerate(lines) if l.startswith("-----"))
    return [l for l in lines[last + 1:] if l.strip()]


def load_bouma(path):
    """Reliable Bouma+24 KOIs (StarFlag bits 0-10 clear, PlanetFlag = 0), confirmed or candidate.
    age = tgyro in Gyr; ages given only as lower limits (all '> 4 Gyr') get age = NaN, is_limit = True."""
    rows = [[l.ljust(215)[a - 1:b].strip() for _, a, b in BOUMA_COLS] for l in _mrt_rows(path)]
    d = pd.DataFrame(rows, columns=[c for c, _, _ in BOUMA_COLS])
    for c in ("kic", "teff", "prot", "tgyro", "E_tgyro", "e_tgyro", "R", "P", "starflag", "planetflag", "ew_li", "tli"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["f_tgyro"] = d.f_tgyro.str.replace(">", "").str.strip()
    d["gyro_limit"] = pd.to_numeric(d.f_tgyro, errors="coerce")          # lower limit [Myr] where present
    d["is_limit"] = d.gyro_limit.notna()
    sf = d.starflag.fillna(0).astype(int)
    d = d[((sf & 0b11111111111) == 0) & (d.planetflag == 0)]
    d = d[d.disp.isin(["CONFIRMED", "CANDIDATE"])].copy()
    d["age"] = np.where(d.is_limit, np.nan, d.tgyro / 1000.0)
    d["age_lo_limit"] = d.gyro_limit / 1000.0
    return d.reset_index(drop=True)


def load_berger(path):
    rows = [[l[a:b].strip() for _, a, b in BERGER_COLS] for l in _mrt_rows(path)]
    b = pd.DataFrame(rows, columns=[c for c, _, _ in BERGER_COLS]).apply(pd.to_numeric, errors="coerce")
    return b.drop_duplicates("kic")


def build_planets(bouma_path, berger_path, koi_csv, rmin=0.5, rmax=4.0, pmax=100.0):
    """Planet table: Berger+20 M*, R*, L*, [Fe/H]; R_p = koi_ror x R*(Berger); Kepler SNR and noise.
    age_use = rotation age [Gyr], or the 4 Gyr lower limit for limit-only stars."""
    d = load_bouma(bouma_path)
    b = load_berger(berger_path)
    k = read_koi(koi_csv, ["kepoi_name", "koi_ror", "koi_model_snr", "koi_depth", "koi_num_transits", "koi_kepmag"])
    d = d.merge(b, on="kic", how="left").merge(k, left_on="koi", right_on="kepoi_name", how="left")
    d = d[d.mstar_b.notna() & d.koi_ror.notna()].copy()
    d["R_bouma"] = d.R
    d["mstar"], d["rstar"], d["feh"] = d.mstar_b, d.rstar_b, d.feh_b
    d["R"] = d.koi_ror * d.rstar * R_SUN_IN_R_EARTH
    d["a_au"] = (d.mstar * (d.P / 365.25) ** 2) ** (1 / 3)
    d["S"] = 10 ** d.logL_b / d.a_au ** 2
    ntr = np.where(d.koi_num_transits > 0, d.koi_num_transits, 1460.0 / d.P)
    d["snr"] = d.koi_model_snr
    d["noise_ppm"] = d.koi_depth * np.sqrt(ntr) / d.koi_model_snr
    d = d[(d.R > rmin) & (d.R < rmax) & (d.P < pmax) & (d.age.notna() | d.is_limit)].copy()
    d["age_use"] = np.where(d.is_limit, d.age_lo_limit, d.age)
    return d.reset_index(drop=True)


def greco_sample(planets, koi_csv, rmin=0.5, amin=0.1, amax=4.0, max_frac_err=0.25):
    """Greco+26-style sample: confirmed planets, rmin <= R, amin <= age < amax, with
      mes        koi_max_mult_ev (measured MES)
      g          1 if another KOI on the same star is CONFIRMED (Greco's sibling flag)
      lognoise_h log10 of the host's median per-transit noise (used by the sibling test)
      sx         log-radius error: koi_ror fractional error (+) 4 % R* error, in dex
      age_lo_dex, age_hi_dex   asymmetric log-age errors from Bouma+24
    Planets with fractional radius error > max_frac_err are removed (Greco criterion c)."""
    k = read_koi(koi_csv, ["kepid", "kepoi_name", "koi_disposition", "koi_max_mult_ev",
                                                   "koi_depth", "koi_model_snr", "koi_num_transits", "koi_period",
                                                   "koi_ror", "koi_ror_err1", "koi_ror_err2"])
    k["noise"] = k.koi_depth * np.sqrt(np.where(k.koi_num_transits > 0, k.koi_num_transits, 1460 / k.koi_period)) / k.koi_model_snr
    d = planets.merge(k[["kepoi_name", "koi_disposition", "koi_max_mult_ev"]], on="kepoi_name", how="left")
    d["mes"] = d.koi_max_mult_ev
    d = d[np.isfinite(d.mes) & (d.mes > 0) & (d.koi_disposition == "CONFIRMED") & (d.R >= rmin)
          & (d.age_use >= amin) & (d.age_use < amax)].reset_index(drop=True)

    conf = k[k.koi_disposition == "CONFIRMED"]
    sib = conf.groupby("kepid").kepoi_name.apply(set)
    d["g"] = [int(any(o != koi for o in sib.get(kic, set()))) for kic, koi in zip(d.kic, d.koi)]
    host_noise = k.groupby("kepid").noise.median()
    d["lognoise_h"] = np.log10(d.kic.map(host_noise))

    ke = k.set_index("kepoi_name")
    fr = 0.5 * (d.kepoi_name.map(ke.koi_ror_err1) - d.kepoi_name.map(ke.koi_ror_err2)) / d.kepoi_name.map(ke.koi_ror)
    fr = np.hypot(fr.fillna(0.07).clip(0.005, 0.25), 0.04)
    d["sx"] = fr / np.log(10)
    d = d[fr <= max_frac_err].copy()
    t = d.tgyro.values                                                    # Myr
    d["age_hi_dex"] = np.log10((t + d.E_tgyro.values) / t)
    d["age_lo_dex"] = np.log10(t / np.clip(t - d.e_tgyro.values, 30, None))
    d.loc[~np.isfinite(d.age_hi_dex), "age_hi_dex"] = 0.05
    d.loc[~np.isfinite(d.age_lo_dex), "age_lo_dex"] = 0.05
    return d.reset_index(drop=True)
