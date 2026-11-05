#!/usr/bin/env bash
# Reproduce every number in RESULTS.md. Raw inputs in data/raw/ (see README). ~2-4 h on 2 cores.
set -euo pipefail
cd "$(dirname "$0")/.."
RAW=data/raw; KOI=$RAW/cumulative.csv; P=data/planets.csv
mkdir -p results
python scripts/build_sample.py --bouma $RAW/apjad855ft1_mrt.txt --berger $RAW/ajab8a33t2_mrt.txt --koi $KOI --out $P | tee results/sample.log
python scripts/replicate_greco.py $P $KOI --config greco refined host_controls free_peaks | tee results/data_fits.log
python scripts/replicate_greco.py $P $KOI --config greco free_peaks --boot 60 | tee results/bootstrap.log
python scripts/sibling_test.py $P $KOI --n 20 | tee results/sibling.log
python scripts/model_mocks.py $P $KOI params/cp_standard.json params/pe_standard.json params/cp_slow_cooling.json --n 16 --quiet | tee results/mocks_fixed.log
python scripts/model_mocks.py $P $KOI params/cp_standard.json params/pe_standard.json params/cp_slow_cooling.json --free SE SN --n 16 --quiet | tee results/mocks_free.log
