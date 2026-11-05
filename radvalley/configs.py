"""Named mixture-model set-ups used in the scripts and RESULTS.md."""
BINS = [(0.1, 1), (1, 2), (2, 4)]
GRECO = dict(rmin=0.5, confirm=True, covariates=True, bins=BINS)                     # Greco+26 set-up, point ages
CONFIGS = {
    "greco": GRECO,
    "refined": dict(GRECO, per_planet_err=True, age_marg=True),                       # + radius errors, age marginalisation
    "host_controls": dict(GRECO, extra_cov=["logM", "feh", "logS"]),                  # + host mass, metallicity, insolation
    "free_peaks": dict(GRECO, free_peaks=("SE", "SN"), exact_conv=True, per_planet_err=True),
    "free_SN": dict(GRECO, free_peaks=("SN",)),
}
