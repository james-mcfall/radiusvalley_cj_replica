"""
radvalley.physics
=================
Vectorised thermal + atmospheric-mass-loss evolution of close-in
super-Earths / sub-Neptunes under three mass-loss prescriptions:

  "cp"   : core-powered mass loss only        (Gupta & Schlichting 2019, MNRAS 487, 24)
  "pe"   : XUV photoevaporation only          (energy-limited; Owen & Wu 2017;
                                               stellar XUV history of Rogers et al. 2021)
  "both" : both operating; instantaneous rate = max(PE, CP)

All three share ONE planet structure model (G&S19 eqs 1-5), ONE initial
condition and ONE population, so differences between runs come only from
the escape physics.

Units: cgs throughout. Times in seconds internally.
"""
from dataclasses import dataclass, field
import numpy as np

# ---------------------------------------------------------------- constants
G = 6.674e-8
KB = 1.380649e-16
AMU = 1.66054e-24
SIGMA_SB = 5.6704e-5
M_EARTH = 5.972e27
R_EARTH = 6.371e8
M_SUN = 1.989e33
L_SUN = 3.828e33
R_SUN = 6.957e10
AU = 1.496e13
YR = 3.156e7
MYR = 1e6 * YR
GYR = 1e9 * YR
DAY = 86400.0


@dataclass
class Params:
    # atmosphere / core microphysics (G&S19 Table 1)
    mu: float = 2.0 * AMU          # H2 envelope
    mu_c: float = 56.0 * AMU       # core
    gamma: float = 7.0 / 5.0
    gamma_c: float = 4.0 / 3.0
    rho_core_star: float = 5.5     # g/cc, Earth-like (Mc = Rc^4 (rho*/rho_E)^(4/3))
    # opacity at rcb (Freedman et al. 2008 fit used by G&S19)
    kappa0: float = 0.1            # cm^2/g at rho = 1e-3 g/cc
    kappa_exp: float = 0.6
    # photoevaporation (Rogers et al. 2021 eqs 10-13)
    eta0: float = 0.1
    eta_slope: float = 2.0         # eta = eta0 (v_esc/15 km/s)^-eta_slope
    lx_sat: float = 10 ** -3.5     # (L_XUV/L_bol)_sat
    lx_mstar_exp: float = -0.5     # (M*/Msun)^this
    t_sat0: float = 100 * MYR      # t_sat = t_sat0 (M*/Msun)^tsat_mstar_exp
    tsat_mstar_exp: float = -1.0
    a0: float = 0.5                # decline (t/t_sat)^(-1-a0)
    # PE upgrade (both off by default = original energy-limited model)
    f_euv: float = 0.0             # separate EUV channel: L_EUV = f_euv L_X,sat (L_X/L_X,sat)^euv_kw_exp
    euv_kw_exp: float = 0.6        #   King & Wheatley (2021): L_EUV/L_X ∝ F_X^-0.4 -> EUV decays more slowly than X-rays
    rr_limit: bool = False         # EUV flow recombination-limited at high flux (Murray-Clay+09), harmonic bridge
    # star
    zeta: float = 4.5              # L ∝ M*^zeta (Rogers+21, Gupta & Schlichting 20)
    # numerics
    t_start: float = 3 * MYR       # disc dispersal
    f_strip: float = 1e-4          # Matm/Mc below which the planet is a bare core
    step_frac: float = 0.01        # dt = step_frac * min(t_cool, t_loss)  (G&S19)
    xuv_step_frac: float = 0.05    # also dt <= this * t  (resolves XUV decline)
    dR_floor: float = 1e-3         # dR/Rc floor (ultra-thin regime not modelled)
    # optional isothermal radiative layer above the rcb (Owen & Wu 2017 style).
    # If True, the PE absorbing radius and the reported (transit) radius are the
    # radius where pressure falls to p_phot; G&S19 instead take R_p = R_rcb.
    combine: str = "max"           # mode "both": "max", "sum", or "os24" (regime switch on R_XUV vs sonic radius,
                                   #   after Owen & Schlichting 2024)
    pe_radius: str = "auto"        # radius in the PE rate: "auto" (= photosphere setting), "xuv" (tau_XUV = 1 surface)
    sigma_xuv: float = 2e-18       # cm^2, EUV cross-section of atomic H (O&S24)
    cooling_factor: float = 1.0    # diagnostic only: scales the cooling luminosity
    photosphere: bool = False
    # transit / XUV radius model: "gs19" (rcb from the G&S19 energy budget, optional 20 mbar layer)
    # or "lf14" (Lopez & Fortney 2014 fit to full interior models; CP internals stay G&S19)
    structure: str = "gs19"
    lf_age_exp: float = -0.18      # LF14 enhanced opacity; -0.11 for solar metallicity
    lf_ratm_h: float = 9.0         # LF14 radiative layer = this many scale heights (LF14 eq. 2); 0 = none
    l_evol_eta: float = 0.0        # main-sequence brightening: L(t) = L_now (t / age_now)^eta  (Sun ~0.14)
    p_phot: float = 2e4            # dyn/cm^2 = 20 mbar (typical transit pressure)


# ---------------------------------------------------------------- star & orbit
def stellar_lbol(mstar, p: Params):
    return L_SUN * mstar ** p.zeta


def stellar_radius(mstar):
    return R_SUN * mstar ** 0.8


def semimajor_axis(P_days, mstar):
    P = P_days * DAY
    return (G * mstar * M_SUN * P ** 2 / (4 * np.pi ** 2)) ** (1.0 / 3.0)


def t_eq(lbol, a):
    """Zero-albedo, full-redistribution equilibrium temperature."""
    return (lbol / (16 * np.pi * SIGMA_SB * a ** 2)) ** 0.25


def l_xuv(t, lbol, mstar, p: Params):
    tsat = p.t_sat0 * mstar ** p.tsat_mstar_exp
    ratio = p.lx_sat * mstar ** p.lx_mstar_exp
    decline = np.where(t < tsat, 1.0, (np.maximum(t, 1.0) / tsat) ** (-1.0 - p.a0))
    return ratio * decline * lbol


def l_euv(lx, lbol, mstar, p: Params):
    """Long-lived EUV (King & Wheatley 2021): ratio to the saturated X-ray level falls only as (L_X/L_X,sat)^0.6."""
    lx_sat = p.lx_sat * mstar ** p.lx_mstar_exp * lbol
    return p.f_euv * lx_sat * (np.maximum(lx, 1e-300) / lx_sat) ** p.euv_kw_exp


K_B, M_H = 1.380649e-16, 1.6735575e-24
ALPHA_B, H_NU0, T_EUV = 2.6e-13, 20 * 1.602e-12, 1.0e4


def mdot_rr(mc, matm, rp, a, leuv):
    """Recombination-limited EUV outflow (Murray-Clay, Chiang & Murray 2009, eqs 19-24): ionised 1e4 K wind,
    base density from ionisation-recombination balance, launched through the sonic point."""
    mp = mc + matm
    cs = np.sqrt(2 * K_B * T_EUV / M_H)                  # ionised H (mu ~ 0.5)
    H = K_B * T_EUV * rp ** 2 / (M_H * G * mp)           # MC09 base scale height
    feuv = leuv / (4 * np.pi * a ** 2)
    n_base = np.sqrt(feuv / (H_NU0 * ALPHA_B * H))
    rs = np.maximum(G * mp / (2 * cs ** 2), rp)
    expo = np.clip(G * mp / (cs ** 2 * rp) * (rp / rs - 1.0), -700, 0)
    return 4 * np.pi * rs ** 2 * cs * M_H * n_base * np.exp(expo)


# ---------------------------------------------------------------- planet structure
def core_radius(mc, p: Params):
    """G&S19: Mc/ME = (Rc/RE)^4 (rho*/rhoE)^(4/3)."""
    return R_EARTH * (mc / M_EARTH) ** 0.25 * (p.rho_core_star / 5.5) ** (-1.0 / 3.0)


def energy_coeff(matm, mc, p: Params):
    """E_cool = g dR * [A Matm + B Mc]  (G&S19 eq 4). Returns the bracket."""
    A = p.gamma / (2 * p.gamma - 1)
    B = (1.0 / p.gamma) * (p.gamma - 1) / (p.gamma_c - 1) * (p.mu / p.mu_c)
    return A * matm + B * mc


def structure(matm, ecool, mc, rc, teq, p: Params):
    """Given envelope mass and cooling energy, return the envelope structure."""
    g = G * mc / rc ** 2
    dR = ecool / (g * energy_coeff(matm, mc, p))
    dR = np.maximum(dR, p.dR_floor * rc)
    rrcb = rc + dR
    gm1 = p.gamma - 1
    rbp = gm1 / p.gamma * G * mc * p.mu / (KB * teq)            # modified Bondi radius (eq 2)
    rho = matm / (gm1 / p.gamma * 4 * np.pi * rc ** 2 * dR
                  * (rbp * dR / rc ** 2) ** (1.0 / gm1))        # eq 1 inverted
    kappa = p.kappa0 * (rho / 1e-3) ** p.kappa_exp
    lum = p.cooling_factor * 64 * np.pi / 3 * SIGMA_SB * teq ** 4 * rbp / (kappa * rho)   # eq 5
    return dict(g=g, dR=dR, rrcb=rrcb, rho=rho, lum=lum)


def lf14_radius(matm, mc, teq, S_earth, t, rc, age_exp=-0.18, ratm_h=9.0):
    """Lopez & Fortney (2014) eq. 3 + eq. 2: R_p = R_core + R_env + R_atm (cgs).
    R_env = 2.06 RE (Mp/ME)^-0.21 (f/5%)^0.59 (F/FE)^0.044 (age/5 Gyr)^exp ; age floored at 0.1 Gyr.
    R_atm = 9 kT/(g mu) at the top of the envelope (20 mbar transit radius above ~100 bar)."""
    mp = mc + matm
    f = np.maximum(matm / mp, 0.0)
    age = np.maximum(t / GYR, 0.1)
    renv = 2.06 * R_EARTH * (mp / M_EARTH) ** -0.21 * (f / 0.05) ** 0.59 * \
        np.maximum(S_earth, 0.1) ** 0.044 * (age / 5.0) ** age_exp
    r = rc + renv
    g = G * mp / r ** 2
    ratm = ratm_h * KB * teq / (g * 2.35 * AMU)
    return r + np.where(f > 0, ratm, 0.0)



def photo_radius(matm, mc, teq, s, p: Params):
    """Radius of the p_phot surface of an isothermal (T = Teq) layer above the rcb,
    from exact hydrostatic balance with r-dependent gravity:
        1/R_ph = 1/R_rcb - (cs^2 / G Mp) ln(rho_rcb / rho_ph).
    Capped at the sonic radius. Returns R_rcb if photosphere is disabled."""
    if not p.photosphere:
        return s["rrcb"]
    mp = mc + matm
    cs2 = KB * teq / p.mu
    rho_ph = p.p_phot / cs2
    lnr = np.log(np.maximum(s["rho"] / rho_ph, 1.0))
    inv = 1.0 / s["rrcb"] - cs2 / (G * mp) * lnr
    rs = G * mp / (2 * cs2)
    return np.where(inv > 1.0 / rs, 1.0 / np.maximum(inv, 1e-300), rs)


def xuv_radius(matm, mc, teq, s, p: Params, n_iter=4):
    """tau_XUV = 1 surface in the isothermal (T = Teq), bolometrically heated hydrostatic
    layer above the rcb:  n_H(R) * H(R) * sigma_XUV = 1,  H = cs^2 r^2 / (G Mp).
    Returns (R_XUV, R_sonic). R_XUV is NOT capped: R_XUV >= R_sonic means the XUV is absorbed
    in an already-supersonic core-powered flow (O&S24 core-powered regime)."""
    mp = mc + matm
    cs2 = KB * teq / p.mu
    gm_c = G * mp / cs2
    rs = gm_c / 2.0
    m_h = 1.6726e-24
    r = s["rrcb"] * 1.5
    for _ in range(n_iter):
        H = r ** 2 / gm_c
        rho_x = m_h / (p.sigma_xuv * H)
        lnr = np.log(np.maximum(s["rho"] / rho_x, 1.0))
        inv = 1.0 / s["rrcb"] - lnr / gm_c
        r = np.where(inv > 1e-30, 1.0 / np.maximum(inv, 1e-30), 1e30)
        r = np.minimum(r, 1e3 * rs)
    return r, rs


def mdot_cp(matm, mc, rc, teq, s, p: Params):
    """Core-powered: min(energy-limited, Bondi/Parker-limited)  (G&S19 eqs 6, 8)."""
    mp = mc + matm
    cs2 = KB * teq / p.mu
    rs = G * mp / (2 * cs2)
    expo = np.minimum(G * mp / (cs2 * s["rrcb"]), 700.0)
    mB = 4 * np.pi * rs ** 2 * np.sqrt(cs2) * s["rho"] * np.exp(-expo)
    mE = s["lum"] / (s["g"] * rc)
    return np.minimum(mB, mE)


def mdot_pe(matm, mc, a, lx, s, p: Params, rp=None):
    """Energy-limited XUV photoevaporation (Rogers+21 eqs 10-11)."""
    mp = mc + matm
    rp = s["rrcb"] if rp is None else rp
    vesc = np.sqrt(2 * G * mp / rp)
    eta = p.eta0 * (vesc / 1.5e6) ** (-p.eta_slope)
    return eta * np.pi * rp ** 3 * lx / (4 * np.pi * a ** 2 * G * mp)


# ---------------------------------------------------------------- population
def sample_periods(n, rng, pmin=1.0, pmax=100.0, pbreak=8.0):
    """dN/dlogP ∝ P^2 (P<8 d), const (P>8 d)."""
    x0, xb, x1 = np.log(pmin), np.log(pbreak), np.log(pmax)
    w1 = (np.exp(2 * xb) - np.exp(2 * x0)) / 2.0
    w2 = np.exp(2 * xb) * (x1 - xb)
    u = rng.random(n)
    first = u < w1 / (w1 + w2)
    out = np.empty(n)
    u1 = rng.random(first.sum())
    out[first] = 0.5 * np.log(np.exp(2 * x0) + u1 * (np.exp(2 * xb) - np.exp(2 * x0)))
    out[~first] = rng.uniform(xb, x1, (~first).sum())
    return np.exp(out)


def sample_core_masses_lognormal(n, rng, p: Params, mu_log10=0.6, sig_log10=0.3, rcmin=0.6, rcmax=4.0):
    """Log-normal core-mass function (flexible alternative to the Rayleigh + tail form),
    truncated to core radii rcmin-rcmax RE."""
    mmin = rcmin ** 4 * (p.rho_core_star / 5.5) ** (4 / 3)
    mmax = rcmax ** 4 * (p.rho_core_star / 5.5) ** (4 / 3)
    out = np.empty(0)
    while out.size < n:
        x = 10 ** (mu_log10 + sig_log10 * rng.standard_normal(2 * n))
        out = np.concatenate([out, x[(x >= mmin) & (x <= mmax)]])
    return out[:n]


def sample_core_masses(n, rng, p: Params, sigma=2.7, mbreak=5.0, rcmin=0.7, rcmax=4.0):
    """dN/dMc ∝ Mc exp(-Mc^2/2σ^2) below 5 ME, ∝ Mc^-2 above (continuous at break).
    Range set by core radii 0.7-4 RE (G&S19 Table 1)."""
    mmin = (rcmin) ** 4 * (p.rho_core_star / 5.5) ** (4 / 3)
    mmax = (rcmax) ** 4 * (p.rho_core_star / 5.5) ** (4 / 3)
    grid = np.logspace(np.log10(mmin), np.log10(mmax), 20000)
    ray = grid * np.exp(-grid ** 2 / (2 * sigma ** 2))
    tailnorm = mbreak * np.exp(-mbreak ** 2 / (2 * sigma ** 2)) * mbreak ** 2
    pdf = np.where(grid < mbreak, ray, tailnorm * grid ** -2.0)
    cdf = np.concatenate([[0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(grid))])
    cdf /= cdf[-1]
    return np.interp(rng.random(n), cdf, grid)


@dataclass
class Population:
    P: np.ndarray          # days
    mc: np.ndarray         # Earth masses
    f0: np.ndarray         # initial Matm/Mc
    mstar: np.ndarray      # solar masses
    rho_fac: np.ndarray = None   # per-planet core-density multiplier (intrinsic composition scatter)
    lx_fac: np.ndarray = None    # per-star XUV multiplier (activity scatter at fixed mass and age)
    lbol: np.ndarray = None      # optional per-star bolometric luminosity [L_sun] (overrides M*^zeta)
    age_now: np.ndarray = None   # optional present-day age [Gyr] at which lbol applies (for l_evol_eta)

    @property
    def n(self):
        return self.P.size


def make_population(n, seed=0, mstar=1.0, mstar_range=None, p: Params = None,
                    f0_norm=0.05, f0_exp=0.5, f0_scatter_dex=0.0, sigma_m=2.7,
                    rho_scatter_dex=0.0, lx_scatter_dex=0.0, mass_dist="rayleigh",
                    logm_mu=0.6, logm_sigma=0.3):
    p = p or Params()
    rng = np.random.default_rng(seed)
    P = sample_periods(n, rng)
    if mass_dist == "lognormal":
        mc = sample_core_masses_lognormal(n, rng, p, mu_log10=logm_mu, sig_log10=logm_sigma)
    else:
        mc = sample_core_masses(n, rng, p, sigma=sigma_m)
    f0 = f0_norm * mc ** f0_exp * 10 ** (f0_scatter_dex * rng.standard_normal(n))
    if mstar_range is not None:
        ms = np.exp(rng.uniform(np.log(mstar_range[0]), np.log(mstar_range[1]), n))
    else:
        ms = np.full(n, float(mstar))
    rho_fac = 10 ** (rho_scatter_dex * rng.standard_normal(n))
    lx_fac = 10 ** (lx_scatter_dex * rng.standard_normal(n))
    return Population(P=P, mc=mc, f0=f0, mstar=ms, rho_fac=rho_fac, lx_fac=lx_fac)


# ---------------------------------------------------------------- evolution
def evolve(pop: Population, mode: str, snap_gyr=(3.0,), p: Params = None,
           max_iter=200000, verbose=False):
    """Evolve every planet from t_start to each snapshot age.

    Returns dict of arrays with shape (n_snap, n):
      R    : transit radius proxy [RE] (R_rcb, or R_core if stripped)
      f    : Matm/Mc
      stripped : bool
    plus per-planet bookkeeping:
      t_strip   : age at stripping [Gyr] (inf if never)
      lost_pe, lost_cp : envelope mass removed while each process set the rate [g]
      final_mech: 1 = PE set the rate on the step that stripped the planet, 2 = CP, 0 = not stripped
    """
    assert mode in ("cp", "pe", "both")
    p = p or Params()
    n = pop.n
    snaps = np.asarray(snap_gyr, float) * GYR
    ns = snaps.size

    mc = pop.mc * M_EARTH
    rho_fac = np.ones(n) if pop.rho_fac is None else pop.rho_fac
    lx_fac = np.ones(n) if pop.lx_fac is None else pop.lx_fac
    rc = core_radius(mc, p) * rho_fac ** (-1.0 / 3.0)
    ms = pop.mstar
    lbol = stellar_lbol(ms, p) if pop.lbol is None else pop.lbol * L_SUN
    a = semimajor_axis(pop.P, ms)
    teq = t_eq(lbol, a)
    Searth = lbol / (4 * np.pi * a ** 2) / (L_SUN / (4 * np.pi * AU ** 2))
    evolve_L = p.l_evol_eta != 0.0
    if evolve_L:
        lbol0, teq0, S0 = lbol.copy(), teq.copy(), Searth.copy()
        age_now = (np.full(n, 5.0) if pop.age_now is None else np.asarray(pop.age_now, float)) * GYR

    matm = pop.f0 * mc
    g = G * mc / rc ** 2
    ecool = g * rc * energy_coeff(matm, mc, p)       # initial dR = Rc (G&S19)
    t = np.full(n, p.t_start)
    k = np.zeros(n, int)                              # next snapshot index
    stripped = np.zeros(n, bool)
    t_strip = np.full(n, np.inf)
    lost_pe = np.zeros(n)
    lost_cp = np.zeros(n)
    final_mech = np.zeros(n, np.int8)

    R_out = np.zeros((ns, n))
    f_out = np.zeros((ns, n))
    s_out = np.zeros((ns, n), bool)

    active = np.arange(n)
    it = 0
    while active.size and it < max_iter:
        it += 1
        i = active
        if evolve_L:
            fac = (np.maximum(t[i], 0.1 * GYR) / age_now[i]) ** p.l_evol_eta
            lbol[i], teq[i], Searth[i] = lbol0[i] * fac, teq0[i] * fac ** 0.25, S0[i] * fac
        s = structure(matm[i], ecool[i], mc[i], rc[i], teq[i], p)
        if mode in ("cp", "both"):
            m_cp = mdot_cp(matm[i], mc[i], rc[i], teq[i], s, p)
        else:
            m_cp = np.zeros(i.size)
        use_os = (mode == "both" and p.combine == "os24")
        if mode in ("pe", "both"):
            lx = l_xuv(t[i], lbol[i], ms[i], p) * lx_fac[i]
            if p.pe_radius == "xuv" or use_os:
                rx, rsn = xuv_radius(matm[i], mc[i], teq[i], s, p)
                m_pe = mdot_pe(matm[i], mc[i], a[i], lx, s, p, rp=np.minimum(rx, rsn))
            else:
                if p.structure == "lf14":
                    rph = lf14_radius(matm[i], mc[i], teq[i], Searth[i], t[i], rc[i], p.lf_age_exp, p.lf_ratm_h)
                else:
                    rph = photo_radius(matm[i], mc[i], teq[i], s, p)
                m_pe = mdot_pe(matm[i], mc[i], a[i], lx, s, p, rp=rph)
            if p.f_euv > 0 or p.rr_limit:
                rp_e = rph if not (p.pe_radius == "xuv" or use_os) else np.minimum(rx, rsn)
                if p.f_euv > 0:
                    le = l_euv(lx, lbol[i], ms[i], p)
                    m_el_euv = mdot_pe(matm[i], mc[i], a[i], le, s, p, rp=rp_e)
                else:     # rr_limit only: treat the whole XUV budget as EUV
                    le, m_el_euv, m_pe = lx, m_pe, 0.0 * m_pe
                if p.rr_limit:
                    m_rr = mdot_rr(mc[i], matm[i], rp_e, a[i], le)
                    m_el_euv = 1.0 / (1.0 / np.maximum(m_el_euv, 1e-250) + 1.0 / np.maximum(m_rr, 1e-250))
                m_pe = m_pe + m_el_euv
        else:
            m_pe = np.zeros(i.size)
        if use_os:
            # XUV absorbed outside the sonic point -> core-powered flow; otherwise the
            # stronger of the XUV-driven and core-powered rates (heating limit, O&S24 eq 18)
            pe_regime = rx < rsn
            mdot = np.where(pe_regime, np.maximum(m_pe, m_cp), m_cp)
            pe_wins = pe_regime & (m_pe >= m_cp)
        else:
            mdot = m_cp + m_pe if p.combine == "sum" else np.maximum(m_cp, m_pe)
            pe_wins = m_pe >= m_cp

        tcool = ecool[i] / s["lum"]
        tloss = matm[i] / np.maximum(mdot, 1e-250)
        dt = p.step_frac * np.minimum(tcool, tloss)
        dt = np.minimum(dt, p.xuv_step_frac * t[i])
        to_snap = snaps[k[i]] - t[i]
        hit = dt >= to_snap
        dt = np.where(hit, to_snap, dt)

        dm = mdot * dt
        lost_pe[i] += np.where(pe_wins, dm, 0.0)
        lost_cp[i] += np.where(pe_wins, 0.0, dm)
        matm[i] -= dm
        ecool[i] -= s["lum"] * dt
        t[i] += dt
        t[i[hit]] = snaps[k[i[hit]]]           # remove round-off

        # stripping
        newly = (matm[i] / mc[i] < p.f_strip) & ~stripped[i]
        js = i[newly]
        stripped[js] = True
        t_strip[js] = t[js] / GYR
        final_mech[js] = np.where(pe_wins[newly], 1, 2)

        # record snapshots: planets that hit a snapshot this step
        jh = i[hit & ~newly]
        if jh.size:
            sr = structure(matm[jh], ecool[jh], mc[jh], rc[jh], teq[jh], p)
            if p.structure == "lf14":
                R_out[k[jh], jh] = lf14_radius(matm[jh], mc[jh], teq[jh], Searth[jh], t[jh], rc[jh], p.lf_age_exp, p.lf_ratm_h) / R_EARTH
            else:
                R_out[k[jh], jh] = photo_radius(matm[jh], mc[jh], teq[jh], sr, p) / R_EARTH
            f_out[k[jh], jh] = matm[jh] / mc[jh]
            k[jh] += 1
        # stripped planets: fill all remaining snapshots with the bare core
        for j in js:
            R_out[k[j]:, j] = rc[j] / R_EARTH
            f_out[k[j]:, j] = 0.0
            s_out[k[j]:, j] = True
            k[j] = ns

        active = active[k[active] < ns]
        if verbose and it % 2000 == 0:
            print(f"  iter {it}: {active.size} active, median t = "
                  f"{np.median(t[active]) / GYR if active.size else 0:.3g} Gyr")

    if active.size:
        raise RuntimeError(f"{active.size} planets did not finish in {max_iter} iterations")

    return dict(R=R_out, f=f_out, stripped=s_out, t_strip=t_strip,
                lost_pe=lost_pe, lost_cp=lost_cp, final_mech=final_mech,
                Rc=rc / R_EARTH, teq=teq, a_au=a / AU, S=lbol / (4 * np.pi * a ** 2)
                / (L_SUN / (4 * np.pi * AU ** 2)), snap_gyr=np.asarray(snap_gyr, float),
                iterations=it)
