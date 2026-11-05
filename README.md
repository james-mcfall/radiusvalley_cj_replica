# Radius valley: replicating the Greco et al. (2026) age trend

Do young stars host fewer stripped super-Earths than older stars? If so, the trend tells us how quickly
sub-Neptunes lose their atmospheres. That loss is driven by core-powered mass loss (CP) or by photoevaporation (PE).

This repository does three things:

1. **Replicates** the two-population mixture model of Greco et al. (2026, arXiv:2611.02099). The input is an
   independent sample: Bouma et al. (2024) gyrochronology ages, Berger et al. (2020) stellar properties and
   Kepler DR25 radii.
2. **Stress-tests** the result:
   - per-planet radius errors and age marginalisation;
   - host mass, metallicity and insolation controls;
   - free peak positions;
   - a no-trend sibling-flag test;
   - host-level bootstraps.
3. **Compares** the result with CP and PE evolution models. Each model is run around every real host at its real
   age, put through the same selection function, and analysed with the same estimator as the data.

The results are summarised in [RESULTS.md](RESULTS.md).

## Install

```bash
pip install -r requirements.txt        # numpy, scipy, pandas; Python >= 3.10
```

## Data (not included; all public)

Put these three files in `data/raw/`:

| File | Source |
|---|---|
| `apjad855ft1_mrt.txt` | Bouma et al. 2024, ApJ 976, 234, Table 1 (machine-readable table on the journal page) |
| `ajab8a33t2_mrt.txt` | Berger et al. 2020, AJ 159, 280, Table 2 (machine-readable table on the journal page) |
| `cumulative.csv` | NASA Exoplanet Archive, *Kepler Objects of Interest (cumulative)*: download the table as CSV with all columns |

The KOI table changes over time as dispositions are updated. Planet counts can differ slightly from RESULTS.md if
it was downloaded on a different date.

## Run

```bash
RAW=data/raw; KOI=$RAW/cumulative.csv
python scripts/build_sample.py --bouma $RAW/apjad855ft1_mrt.txt --berger $RAW/ajab8a33t2_mrt.txt --koi $KOI   # -> data/planets.csv
python scripts/replicate_greco.py data/planets.csv $KOI                       # data fits, all set-ups  (~2 min)
python scripts/replicate_greco.py data/planets.csv $KOI --config greco --boot 60
python scripts/sibling_test.py   data/planets.csv $KOI                       # ~10 min
python scripts/model_mocks.py    data/planets.csv $KOI params/*.json --n 16  # ~1 h per model
python scripts/model_mocks.py    data/planets.csv $KOI params/*.json --n 16 --free SE SN
```

Alternatively, `scripts/run_all.sh` runs everything and writes logs to `results/`.

## Code

| Module | Contents |
|---|---|
| `radvalley/data.py` | Parses the Bouma and Berger machine-readable tables, cross-matches them with the KOI table and applies the Greco sample cuts. Also computes per-planet radius and age errors. |
| `radvalley/mixture.py` | The Greco+26 mixture model: selection function (detection × confirmation), likelihood and standardised weights, plus the optional extensions listed in its docstring. |
| `radvalley/configs.py` | The named set-ups used in RESULTS.md. |
| `radvalley/physics.py` | Planet thermal evolution and mass loss. CP follows Gupta & Schlichting 2019. PE is energy-limited with the XUV history of Rogers et al. 2021. Includes an optional EUV channel and recombination limit, and an optional Lopez & Fortney 2014 radius model. |
| `radvalley/hosts.py` | Simulates K model planets around each real host at its age. |
| `radvalley/mocks.py` | Two sets of mock catalogues: model mocks (physics → selection → noise → same estimator as the data) and no-trend sibling mocks. |
| `params/*.json` | The fitted model parameters (see below). |

**Where the model parameters come from.** `params/cp_standard.json` and `params/pe_standard.json` come from an
earlier calibration in a separate working repository (not included here). That calibration fitted each model to the
CKS-Gaia radius distribution, conditional on each host, plus Greco's published young/old contrast.
`cp_slow_cooling.json` is `cp_standard` with the envelope opacity raised by a factor 10^0.5. The mocks only read
these parameters, so the calibration code is not needed to reproduce RESULTS.md.

## How this was made

The code and analysis were written by Claude (Anthropic) working with James McFall. Gemini reviewed them over several
rounds. Every reviewer point and its outcome is logged in the development notes, and the bugs it found are listed in
RESULTS.md. Treat this as an exploratory analysis, not a refereed result.
