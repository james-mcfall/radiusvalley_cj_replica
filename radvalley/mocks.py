"""Mock catalogues analysed with the same mixture estimator as the data.

model_mocks   planets drawn from a physical model (radvalley.hosts) around the real hosts, at their real ages
sibling_mocks planets drawn from the fitted mixture with NO age trend, to test the sibling term for bias
"""
import numpy as np

from . import mixture as M
from .hosts import HostSim

CFG = dict(rmin=0.5, confirm=True, covariates=True, bins=[(0.1, 1), (1, 2), (2, 4)])
XG = np.linspace(np.log10(0.5), np.log10(4.0), 300)


def model_mocks(planets, d, model, th, n_mock=16, seed=8, free_peaks=None, physics="standard",
                structure="gs19", K=60, verbose=True):
    """For each mock: a fresh model realisation (new random numbers, seed 1000 + i); for every real planet one model
    planet around the same host is drawn with probability proportional to the real selection function at its TRUE
    radius; then 4.5 % radius noise; the MES is rescaled with the mock radius. Hosts where no model planet is
    detectable are dropped. Returns (fbar per mock, peak offsets per mock)."""
    S = M.selection(d.mes.values, d.R.values, XG, True, d.g.values)
    rng = np.random.default_rng(seed)
    cfg = dict(CFG, free_peaks=tuple(free_peaks)) if free_peaks else CFG
    F, O = [], []
    for it in range(n_mock):
        H = HostSim(planets, model, K=K, seed=1000 + it, physics=physics, structure=structure)
        pos = {k: i for i, k in enumerate(H.df.koi.values)}
        idx = np.array([pos[k] for k in d.koi.values])
        RR = np.nan_to_num(H.simulate(th, noise=False).reshape(-1, K)[idx], nan=0.0)
        zr = H.z_r.reshape(-1, K)[idx]
        W = np.array([np.interp(np.log10(np.clip(r, 1e-3, None)), XG, s, left=0, right=0) for r, s in zip(RR, S)])
        W *= (RR > 0.5) & (RR < 4)
        ok = W.sum(1) > 0
        if verbose and it == 0 and (~ok).any():
            print(f"  {(~ok).sum()} hosts have no detectable model planet and are dropped", flush=True)
        W, RR, zr = W[ok], RR[ok], zr[ok]
        cum = np.cumsum(W, 1)
        u = rng.random(len(W)) * cum[:, -1]
        pick = np.minimum((cum < u[:, None]).sum(1), K - 1)
        j = np.arange(len(W))
        Rs = RR[j, pick] * (1 + 0.045 * zr[j, pick])
        dk = d[ok].reset_index(drop=True)
        _, r = M.fit(dk.assign(R=Rs, mes=dk.mes.values * (Rs / dk.R.values) ** 2), cfg)
        F.append(r["fbar"])
        O.append(r["peak_offsets"] if free_peaks else np.zeros((3, 2)))
        if verbose:
            print(f"  mock {it + 1}/{n_mock}: f = {np.round(r['fbar'], 3)}", flush=True)
    return np.array(F), np.array(O)


def fit_sibling(d, gw, th0=None):
    """Greco fit with gw in the weight but the real sibling flag d.g in the confirmation selection."""
    g_sel = d.g.values.copy()
    orig = M.selection
    M.selection = lambda mes, R, xg, conf, g: orig(mes, R, xg, conf, g_sel)
    try:
        return M.fit(d.assign(g=gw), CFG, th0)
    finally:
        M.selection = orig


def sibling_mocks(d, n_mock=20, seed=10):
    """No-age-trend mocks: true multiplicity m lowers the SE weight (fitted gamma); a sibling is seen with a
    probability that falls with host noise, as observed. Returns C = f(<1)/f(2-4) per mock for the fit with the
    detected sibling flag and with the true multiplicity. True C = 1."""
    th_g, r_g = fit_sibling(d, d.g.values)
    rng = np.random.default_rng(seed)
    z = ((d.lognoise_h - d.lognoise_h.mean()) / d.lognoise_h.std()).values
    p_m, a, b = 0.62, 0.92, 0.12            # true sibling fraction; P(seen | m) = a - b z (matches observed fractions)
    m1, m2, s1, s2 = th_g[0], th_g[1], np.exp(th_g[2]), np.exp(th_g[3])
    lf0 = np.log(0.6 / 0.4)
    lp = np.log10(d.P.values) - M.PREF
    C_seen, C_true = [], []
    for _ in range(n_mock):
        m = rng.random(len(d)) < p_m
        seen = m & (rng.random(len(d)) < np.clip(a - b * z, 0.05, 1))
        w = 1 / (1 + np.exp(-(lf0 + r_g["bP"] * lp + r_g["gam"] * m)))
        S = M.selection(d.mes.values, d.R.values, XG, True, seen.astype(int))
        R = np.empty(len(d))
        for i in range(len(d)):                 # rejection sampling against each planet's own selection
            while True:
                x = rng.normal(m1, s1) if rng.random() < w[i] else rng.normal(m2, s2)
                if XG[0] <= x <= XG[-1] and rng.random() < np.interp(x, XG, S[i]):
                    break
            R[i] = 10 ** (x + 0.02 * rng.standard_normal())
        e = d.assign(R=R, g=seen.astype(int), mes=d.mes.values * (R / d.R.values) ** 2)
        _, r = fit_sibling(e, e.g.values, th_g)
        C_seen.append(r["fbar"][0] / r["fbar"][2])
        _, r = fit_sibling(e, m.astype(int), th_g)
        C_true.append(r["fbar"][0] / r["fbar"][2])
    return np.array(C_seen), np.array(C_true)
