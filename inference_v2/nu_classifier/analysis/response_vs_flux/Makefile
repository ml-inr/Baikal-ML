PY := /home/albert/miniconda3/envs/baikal25/bin/python

.PHONY: all smoke verify notebooks catalogue test clean-data
# The full chain.  Stage 30 takes ~45 min on 16 workers and stage 40 ~35 min on a
# GPU; everything else is seconds.
all: data/00_exp_parts.parquet data/10_run_noise.parquet \
     data/20_achievable_bound.parquet data/31_matched_cells.parquet \
     data/41_matched_calibration.parquet

# Every stage on a small subset, minutes rather than hours.  Run this first to
# check the chain end to end before spending the real time.
smoke:
	$(PY) stages/00_index.py --smoke
	$(PY) stages/10_run_noise.py --smoke
	$(PY) stages/20_flux_reweight.py --smoke
	$(PY) stages/30_split_response.py --smoke
	$(PY) stages/31_response_compare.py
	$(PY) stages/40_time_jitter.py --smoke
	$(PY) stages/41_jitter_validate.py --smoke

# what is in data/, in words -- the sidecars carry provenance but not purpose
catalogue:
	$(PY) src/catalogue.py

verify:
	$(PY) -c "import sys; sys.path.insert(0,'src'); import provenance; \
	from pathlib import Path; p=provenance.verify(Path('data'), Path('config.yaml')); \
	print('\n'.join(p) if p else 'chain clean')"

notebooks:
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

data/00_exp_parts.parquet: stages/00_index.py config.yaml
	$(PY) stages/00_index.py

data/10_run_noise.parquet: stages/10_run_noise.py config.yaml data/00_exp_parts.parquet
	$(PY) stages/10_run_noise.py

data/20_achievable_bound.parquet: stages/20_flux_reweight.py config.yaml
	$(PY) stages/20_flux_reweight.py

data/30_loo_residuals.parquet: stages/30_split_response.py config.yaml src/tracks.py
	$(PY) stages/30_split_response.py

data/31_matched_cells.parquet: stages/31_response_compare.py data/30_loo_residuals.parquet
	$(PY) stages/31_response_compare.py

data/40_jitter_summary.parquet: stages/40_time_jitter.py config.yaml
	$(PY) stages/40_time_jitter.py

data/41_matched_calibration.parquet: stages/41_jitter_validate.py config.yaml \
                                     data/30_loo_residuals.parquet \
                                     data/40_jitter_summary.parquet
	$(PY) stages/41_jitter_validate.py

test:
	$(PY) tests/test_tracks.py

clean-data:
	rm -f data/*.parquet data/*.meta.json
