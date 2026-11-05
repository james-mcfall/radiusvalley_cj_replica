"""Simulate K model planets around every real host (its M*, L*, [Fe/H], orbital period and rotation age).

Core mass:      log10 Mc = logm_mu + logm_sigma z_m + gamma_m log10(M*/Msun) + delta_mz [Fe/H],  Mc clipped to 0.3-60 M_earth
Envelope:       f0 = f0_norm Mc^0.5 10^(f0_scatter_dex z_f + delta_fz [Fe/H]),  clipped to 1e-5-0.5
Core density:   rho x 10^(rho_scatter_dex z_rho);  XUV activity (PE only): 10^(lx_scatter_dex z_lx)
Each planet is evolved with radvalley.physics.evolve and read off at the age-grid point nearest its host's age.
Measurement noise: R x (1 + 0.045 z_r).  All z are standard normals fixed by `seed` (common random numbers).
"""
import json
from dataclasses import replace

import numpy as np

from .physics import Params, Population, evolve, MYR

GRID = np.round(np.logspace(np.log10(0.1), np.log10(14.0), 32), 4)    # model snapshot ages [Gyr]
L_EVOL_ETA = 0.14                                                    # solar main-sequence brightening exponent


def load_params(path):
    j = json.load(open(path))
    return j["model"], j["params"]


def base_params(model, physics="standard", structure="gs19"):
    """physics: "standard" | "brightening" (main-sequence L(t)) | "pe_upgrade" (recombination-limited EUV, PE only)"""
    skw = {"gs19": {}, "lf14": dict(structure="lf14"), "lf14_noatm": dict(structure="lf14", lf_ratm_h=0.0)}[structure]
    p = Params(photosphere=True, **skw)
    if physics == "brightening":
        p = replace(p, l_evol_eta=L_EVOL_ETA)
    if physics == "pe_upgrade" and model == "pe":
        p = replace(p, rr_limit=True)
    return p


class HostSim:
    def __init__(self, planets, model, K=60, seed=0, physics="standard", structure="gs19"):
        """planets: the full table from data.build_planets (hosts with SNR > 0 are used, in table order)."""
        d = planets[np.isfinite(planets.snr) & (planets.snr > 0)].reset_index(drop=True).copy()
        d["feh"] = d.feh.fillna(0.0)
        d["age"] = d.age_use.clip(0.1, 14.0)
        self.df, self.model, self.K, self.n = d, model, K, len(d) * K
        self.p0 = base_params(model, physics, structure)
        rng = np.random.default_rng(seed)
        self.host = np.repeat(np.arange(len(d)), K)
        self.z_m, self.z_f, self.z_rho, self.z_lx, self.z_r = (rng.standard_normal(self.n) for _ in range(5))
        self.age_idx = np.abs(np.log10(GRID)[None, :] - np.log10(d.age.values)[:, None]).argmin(1)[self.host]
        self.P, self.ms, self.feh = d.P.values[self.host], d.mstar.values[self.host], d.feh.values[self.host]
        self.lbol = (10 ** d.logL_b.values)[self.host]

    def params(self, th):
        p = replace(self.p0, rho_core_star=th["rho"])
        if "log_kappa_fac" in th:
            p = replace(p, kappa0=0.1 * 10 ** th["log_kappa_fac"])
        if "t_sat_myr" in th:
            p = replace(p, t_sat0=th["t_sat_myr"] * MYR)
        if "xuv_a0" in th:
            p = replace(p, a0=th["xuv_a0"])
        if "log_f_euv" in th:
            p = replace(p, f_euv=10 ** th["log_f_euv"])
        return p

    def simulate(self, th, noise=True):
        """Radii [R_earth], shape (n_hosts * K,), host-major."""
        logm = th["logm_mu"] + th["logm_sigma"] * self.z_m + th.get("gamma_m", 0.0) * np.log10(self.ms) \
            + th.get("delta_mz", 0.0) * self.feh
        mc = 10 ** np.clip(logm, np.log10(0.3), np.log10(60.0))
        f0 = th["f0_norm"] * mc ** 0.5 * 10 ** (th["f0_scatter_dex"] * self.z_f + th.get("delta_fz", 0.0) * self.feh)
        lx = 0.0 if self.model == "cp" else th.get("lx_scatter_dex", 0.0)
        pop = Population(P=self.P, mc=mc, f0=np.clip(f0, 1e-5, 0.5), mstar=self.ms,
                         rho_fac=10 ** (th["rho_scatter_dex"] * self.z_rho), lx_fac=10 ** (lx * self.z_lx),
                         lbol=self.lbol, age_now=self.df.age.values[self.host])
        out = evolve(pop, self.model, snap_gyr=GRID, p=self.params(th))
        R = out["R"][self.age_idx, np.arange(self.n)]
        return R * (1 + 0.045 * self.z_r) if noise else R
