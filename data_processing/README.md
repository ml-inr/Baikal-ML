# Data processing

## Description
Routines used for processing initial `.root` files to `.h5`. The steps are as follows:

1. Use `root2h5.py` for converting `.root` to `.h5` format. Configuration is set in `root2h5_config.yaml`. Processes one *particle* per run. See *h5_raw_format.md* for details on output format.
2. Filter data using `filter_data.py` or multiprocessing variant. Here you can choose to take signal hits only (in various ways), set limits on OM registered charges and many other things. 
3. Calculate normalization coefficients with `get_normalization.py`.
4. Normilize file using `make_normilized.py`.

## Be aware that:
- In `root2h5_config.yaml`, the channel path must always be the last one in the configuration file.
- Currently `take_single_cluster` should always be set to `True`. We do not have convention on how to store and process multicluster data.
- Filtering is not tested on muons individual properties.