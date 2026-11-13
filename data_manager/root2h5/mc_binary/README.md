# mc_binary

Reader for the upstream MC binary files:

- `read_wout.py` — the generator `.dat` and the `mcread` `.wout`, documented in
  [doc/mc_binary_formats.md](../../../doc/mc_binary_formats.md).

Used by `data_manager/energy_truth/eos_production_check.ipynb` (source-file checks) and by the
format documentation.

The former extraction/targets pipeline that lived in this directory
(`extract_interactions.py`, `targets.py`, `targets_vec.py`, `validate_targets.py`, …) was
superseded by `data_manager/energy_truth/` and is kept in
`archive_tracked/data_manager/root2h5/energy_truth/`.
