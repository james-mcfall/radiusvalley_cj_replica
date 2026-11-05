"""Greco et al. (2026, arXiv:2611.02099) two-population mixture model, Sect. 4.1 and App. C.

Two Gaussians in log R (super-Earths, sub-Neptunes). Super-Earth weight per planet:
    logit w_i = logit f(age bin) + beta_P (log P_i - log 8.7) + gamma g_i [+ extra standardised covariates]
Selection = detection (gamma CDF, ~50 % at MES 7.1) x confirmation Phi((xi - 1.056 + 0.066 g) / 0.057),
xi = log10 MES, with MES scaled as R^2 from each planet's measured MES (Greco Eqs. C.3, C.6).
Each planet's likelihood is normalised by its own selection-weighted mixture integral (Greco Eqs. 8-11).
Reported weights are standardised: the mean of w_i over the sample at fixed age bin (Greco Eq. 6).

cfg keys
  rmin            lower radius limit of the model [R_earth] (upper limit 4)
  confirm         include the confirmation stage in the selection
  covariates      include the period and sibling terms
  bins            list of (lo, hi) age bins in Gyr; index 2 is the reference bin for free peaks
  per_planet_err  use each planet's sx instead of 0.02 dex
  exact_conv      convolve the radius error with the selection exactly (otherwise S(x_obs) approximation)
  age_marg        marginalise over each planet's age error (Greco Eq. 11)
  ratio_of_sums   alternative age marginalisation (age posterior conditioned on detection)
  free_peaks      ("SE",), ("SN",) or ("SE", "SN"): per-bin offsets of the peak means (point ages only)
  extra_cov       list of "logM", "feh", "logS" added to logit w (standardised)
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, gamma as gam

PREF = np.log10(8.7)


MES_OFFSET = 0.0   # koi_max_mult_ev is already a measured MES, so Greco's -0.059 dex (expected -> measured) is not applied


def selection(mes_obs, R_obs, xg, confirm, g):
    mes = mes_obs[:, None] * (10 ** xg[None, :] / R_obs[:, None]) ** 2
    det = 0.97 * gam.cdf(np.clip(mes, 0, None), a=29.4, scale=0.284 * 7.1 / 8.3)        # ~50 % at MES 7.1
    if not confirm:
        return det
    xi = np.log10(np.clip(mes, 1e-3, None)) - MES_OFFSET
    conf = norm.cdf((xi - 1.056 + 0.066 * g[:, None]) / 0.057)
    return det * conf


def fit(d, cfg, th0=None):
    xg = np.linspace(np.log10(cfg["rmin"]), np.log10(4.0), 300); dx = xg[1] - xg[0]
    x = np.log10(d.R.values)
    sx = d.sx.values if (cfg.get("per_planet_err") and "sx" in d) else np.full(len(d), 0.02)
    S = selection(d.mes.values, d.R.values, xg, cfg["confirm"], d.g.values)
    Sx = np.array([np.interp(a, xg, s) for a, s in zip(x, S)])
    bins = cfg["bins"]; nb = len(bins); b = np.full(len(d), -1)
    for k, (lo, hi) in enumerate(bins):
        b[(d.age_use >= lo) & (d.age_use < hi)] = k
    # age-uncertainty marginalisation (Greco Eq. 11): probability of each planet's age lying in each bin
    if cfg.get("age_marg") and "age_lo_dex" in d:
        rngA = np.random.default_rng(99); nd = 200
        z = rngA.standard_normal((len(d), nd))
        la = np.log10(d.age_use.values)[:, None] + np.where(z > 0, d.age_hi_dex.values[:, None], d.age_lo_dex.values[:, None]) * z
        ta = 10 ** la
        pk = np.stack([((ta >= lo) & (ta < hi)).mean(1) for lo, hi in bins], 1)
    else:
        pk = np.eye(nb)[np.clip(b, 0, None)] * (b >= 0)[:, None]
    keep = (b >= 0) if not cfg.get("age_marg") else (pk.sum(1) > 0.5)
    pk = pk[keep] / pk[keep].sum(1, keepdims=True)
    x, sx, S, Sx, b = x[keep], sx[keep], S[keep], Sx[keep], b[keep]
    lp = np.log10(d.P.values[keep]) - PREF; g = d.g.values[keep]
    cov = cfg["covariates"]
    def colval(name):
        v = {"logM": np.log10(d.mstar.values), "feh": d.feh.values, "logS": np.log10(d.S.values)}.get(name, None)
        return (v if v is not None else d[name].values)[keep]
    EX = np.array([colval(c) for c in cfg.get("extra_cov", [])]) if cfg.get("extra_cov") else np.zeros((0, int(keep.sum())))
    EX = (EX - EX.mean(1, keepdims=True)) / np.maximum(EX.std(1, keepdims=True), 1e-9)
    nex = EX.shape[0]
    Kx = norm.pdf(x[:, None], xg[None, :], sx[:, None]) if cfg.get("exact_conv") else None

    nfree = (nb - 1) * len(cfg.get("free_peaks", ()))       # extra parameters: per-bin offsets of the chosen peak means
    off0 = 4 + nb + (2 if cov else 0) + nex

    def exterm(t):
        return (t[4 + nb + (2 if cov else 0): off0] @ EX) if nex else 0.0

    def unpack(t):
        return t[0], t[1], np.exp(t[2]), np.exp(t[3]), t[4:4 + nb], (t[4 + nb] if cov else 0.0), (t[5 + nb] if cov else 0.0)

    def peak_offsets(t):
        """(nb, 2) offsets of [SE, SN] log-radius means per age bin; the reference bin (index 2, 2-4 Gyr) has none."""
        o = np.zeros((nb, 2)); j = off0
        for which in cfg.get("free_peaks", ()):
            col = 0 if which == "SE" else 1
            for k in range(nb):
                if k != 2:
                    o[k, col] = t[j]; j += 1
        return o

    def nll(t):
        m1, m2, s1, s2, lf, bp, gm = unpack(t)
        if not (-0.2 < m1 < m2 < 0.6) or s1 > 0.4 or s2 > 0.4:
            return 1e10
        if nfree:                                      # free-peak model: point ages only, evaluate per planet's bin
            o = peak_offsets(t); M1, M2 = m1 + o[b, 0], m2 + o[b, 1]
            if np.any(M1 >= M2) or np.abs(o).max() > 0.2:
                return 1e10
            if cfg.get("exact_conv"):
                pa = (Kx * norm.pdf(xg[None, :], M1[:, None], s1) * S).sum(1) * dx
                pb = (Kx * norm.pdf(xg[None, :], M2[:, None], s2) * S).sum(1) * dx
            else:
                pa = norm.pdf(x, M1, np.hypot(s1, sx)) * Sx; pb = norm.pdf(x, M2, np.hypot(s2, sx)) * Sx
            ZA = (norm.pdf(xg[None, :], M1[:, None], s1) * S).sum(1) * dx; ZB = (norm.pdf(xg[None, :], M2[:, None], s2) * S).sum(1) * dx
            w = 1 / (1 + np.exp(-(lf[b] + bp * lp + gm * g + exterm(t))))
            return -np.sum(np.log((w * pa + (1 - w) * pb) / (w * ZA + (1 - w) * ZB) + 1e-300))
        if cfg.get("exact_conv"):     # numerator: int N(x_obs | x', sx) N(x' | mu, s) S(x') dx'  (Greco Eq. 8)
            pa = (Kx * norm.pdf(xg, m1, s1) * S).sum(1) * dx
            pb = (Kx * norm.pdf(xg, m2, s2) * S).sum(1) * dx
        else:
            pa = norm.pdf(x, m1, np.hypot(s1, sx)) * Sx
            pb = norm.pdf(x, m2, np.hypot(s2, sx)) * Sx
        ZA, ZB = (norm.pdf(xg, m1, s1) * S).sum(1) * dx, (norm.pdf(xg, m2, s2) * S).sum(1) * dx
        num, den, L = 0.0, 0.0, 0.0
        for k in range(nb):                                    # mixture over the age bins each planet may fall in
            w = 1 / (1 + np.exp(-(lf[k] + bp * lp + gm * g + exterm(t))))
            nk, dk = w * pa + (1 - w) * pb, w * ZA + (1 - w) * ZB
            L = L + pk[:, k] * nk / dk                         # Greco Eq. 11: each age draw normalised by its own Z
            num, den = num + pk[:, k] * nk, den + pk[:, k] * dk
        if cfg.get("ratio_of_sums"):                           # alternative: age posterior conditioned on detection
            L = num / den
        return -np.sum(np.log(L + 1e-300))

    t0 = th0 if th0 is not None else np.r_[np.log10(1.35), np.log10(2.4), np.log(0.08), np.log(0.10), np.zeros(nb), ([-2.0, 0.0] if cov else []), np.zeros(nex), np.zeros(nfree)]
    r = minimize(nll, t0, method="Nelder-Mead", options={"maxiter": 20000, "xatol": 1e-5, "fatol": 1e-5})
    r = minimize(nll, r.x, method="Nelder-Mead", options={"maxiter": 20000, "xatol": 1e-6, "fatol": 1e-6})
    m1, m2, s1, s2, lf, bp, gm = unpack(r.x)
    ex = exterm(r.x)
    fbar = np.array([np.mean(1 / (1 + np.exp(-(l + bp * lp + gm * g + ex)))) for l in lf])   # standardised (Greco Eq. 6)
    fraw = 1 / (1 + np.exp(-lf))
    return r.x, dict(n=keep.sum(), nbin=[int((b == k).sum()) for k in range(nb)], fbar=fbar, fraw=fraw, bP=bp, gam=gm,
                     peak_offsets=(peak_offsets(r.x) if nfree else None),
                     mu=(10 ** m1, 10 ** m2), sig=(s1, s2), nll=r.fun)
