# Results

**Status: draft.** The final external review (Gemini Argon, round 5) is still pending.

The sample is 488 confirmed Kepler planets with 0.5 ≤ R < 4 R⊕, P < 100 d and rotation ages 0.1–4 Gyr. The
planets per age bin are 54 / 113 / 321. Each line below gives the script that produced it. Logs are in `results/`.

## 1. The Greco+26 super-Earth age trend replicates, but more weakly

f is the standardised super-Earth fraction in each age bin (< 1, 1–2, 2–4 Gyr). Δf = f(2–4) − f(< 1), and
C = f(< 1) / f(2–4).

| Set-up | f(< 1) / f(1–2) / f(2–4) | Δf | C | Bootstrap Δf (60 host resamples, median [16, 84]) | P(Δf > 0) |
|---|---|---|---|---|---|
| Greco+26 (581 planets, their ages) | 0.43 / 0.75 / 0.67 | +0.24 ± 0.10 | 0.64 | | |
| This work, Greco set-up | 0.56 / 0.72 / 0.70 | +0.133 | 0.81 | +0.111 [+0.054, +0.221] | 0.95 |
| + radius errors, age marginalisation | 0.52 / 0.75 / 0.70 | +0.174 | 0.75 | +0.153 [+0.061, +0.262] | 0.93 |
| + host M*, [Fe/H], S in the weight | 0.55 / 0.73 / 0.70 | +0.150 | 0.79 | | |

Scripts: `replicate_greco.py` (data fits, the bootstrap).

- **There are fewer super-Earths around young hosts, at about 2σ.** The deficit is about half Greco's size.
- **Controlling for host mass, metallicity and insolation does not remove it.** With log S included, β_P becomes
  meaningless (+6.4) because log S and log P are nearly collinear. Δf is unaffected.
- **The sibling term is not biased.** In no-trend mocks where siblings are harder to detect around noisy young
  stars, the fit returns C = 0.99 ± 0.05 (true value 1). Script: `sibling_test.py`.

## 2. With free peak positions the trend is a size shift, not a fraction change

This set-up lets the super-Earth (SE) and sub-Neptune (SN) peak positions differ between age bins. It uses exact
convolution and per-planet errors.

- **The fraction trend disappears:** Δf = +0.058; bootstrap +0.051 [−0.038, +0.145], P(Δf > 0) = 0.63.
- **The SE peak shifts:** −0.143 dex below 1 Gyr (bootstrap −0.167 [−0.200, −0.095]) and −0.101 dex at 1–2 Gyr.
- **The SN peak does not move below 1 Gyr:** −0.000 [−0.017, +0.020] dex.

**The SE shift is not reliable at this size.**
- The bootstrap runs into the estimator's ±0.2 dex limit.
- A direct regression of log R on youth (< 2 Gyr) for R < 1.7 R⊕ gives a much smaller shift: −0.039 ± 0.013 dex.
  The regression holds detectability (MES/R²), period, host mass and [Fe/H] fixed, and uses host-clustered errors.
- The raw median SE radius is 1.09 / 1.16 / 1.35 R⊕ across the three age bins.
- **Conclusion:** young super-Earths appear about 9 % smaller (3σ). The fitted Gaussian peak exaggerates this.

Whether the young deficit is fewer super-Earths or smaller super-Earths cannot be separated with 54 young planets.

## 3. Comparison with mass-loss models

Mocks are drawn from the models around the real hosts and analysed with the same estimator. There are 16 per
model. Script: `model_mocks.py`.

| | Data | CP (core-powered) | PE (photoevaporation) | CP, slow cooling |
|---|---|---|---|---|
| Δf, fixed peaks | +0.133 | +0.124 ± 0.076 | +0.076 ± 0.088 | +0.259 ± 0.082 |
| Δf, free peaks | +0.038 | +0.068 ± 0.078 | +0.049 ± 0.088 | +0.235 ± 0.070 |
| SE shift < 1 Gyr, free peaks (dex) | −0.167 | +0.008 ± 0.027 | −0.008 ± 0.020 | +0.030 ± 0.030 |
| SN shift < 1 Gyr, free peaks (dex) | −0.003 | +0.080 ± 0.036 | +0.040 ± 0.016 | +0.146 ± 0.043 |
| mean f (all bins) | 0.56–0.70 | 0.44–0.56 | 0.47–0.55 | 0.24–0.50 |

The free-peak mocks use the S(x_obs) approximation, so the data column there uses the same estimator.

- **The fixed-peak trend does not discriminate between CP and PE.** Both reproduce it within 1σ.
- **Slow-cooling CP overshoots** the trend (≈ 2.8σ with free peaks) and the young sub-Neptune sizes (≈ 3.4σ).
- **Young sub-Neptunes are not inflated.** Both standard models predict inflation: CP at +0.08 ± 0.04 (≈ 2σ above the
  data) and PE at +0.04 ± 0.02 (≈ 2σ). This is the strongest data–model tension found.
- **Neither model makes young super-Earths smaller.**
- **Both models make too few super-Earths overall.** Their parameters were calibrated on 1–4 R⊕ CKS planets.

## 4. Corrections made during this work

- **Major (found while building this repo):** the planet table had been cut at R > 1 R⊕, while the mixture model
  and the mocks use 0.5–4 R⊕. This removed 41 planets and mismatched data and mocks.
  - The earlier results changed: C 0.67 → 0.81, Δf +0.160 → +0.133.
  - So did the CP vs PE conclusion: "CP preferred" no longer holds.
- **Mock MES not rescaled with the mock radius** (Gemini Argon, round 4): fixed.
- **Mocks reused one model realisation** (Argon, round 2): each mock is now a fresh realisation.
- **Model age grid started at 0.3 Gyr** (Argon, round 3): it now runs 0.1–14 Gyr.
- **Free peaks lacked the exact convolution** (Argon, round 2): added.
- **Double −0.059 dex MES offset** (Gemini Flash, round 1): removed.
- **The bootstrap grouped planets by KOI prefix** instead of by star (Flash, round 2): it now groups by KIC.

## 5. Open issues

- The young-SE size shift could be an astrophysical signal or a radius systematic for young or active stars.
  The test that would separate these is to re-derive R* for young hosts independently.
- The model parameters were not recalibrated on the 0.5–4 R⊕ range.
- There are only 54 young planets. Every young-bin result is limited by that number.
