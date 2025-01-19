# Data Processing Module

This module provides a pipeline for processing data from `.root` files, converting them into structured formats for machine learning tasks. The process efficiently handles large datasets by reading, processing, and batching data. Below is an overview of the pipeline and usage examples.

# **INFO**
in purpose of Neural Network training mostly `MCMuNuSepBatchGenerator` class with its configs `MCMuNuSepBatchGeneratorConfig` is needed. Usage exaple is at step 5.

# Pipeline Overview
1. ## **Raw `.root` Files**:
    - The process starts with `.root` files containing raw data.

2. ## **RootFileReader** (`data/root_extractor/main.py`):
    - Converts `.root` files into Polars DataFrames. It maps paths inside `.root` files to a DataFrame schema.
    - **Usage Example**:
        ```Python
        from data.root_extractor.main import RootFileReader
        from data.root_extractor.internal_root_paths import MCRootPaths, ExpRootPaths
        # MC file reading
        path = "/net/62/home3/ivkhar/Baikal/data/initial_data/MC_2020/muatm/root/all/1005.root"
        start = 0
        stop = None
        with RootFileReader(path, internal_root_paths=MCRootPaths()) as rr:
            pulses = rr.read_pulses_as_df(start, stop)
            muons = rr.read_muons_as_df(start, stop)
            events = rr.read_events_as_df(start, stop)
            coords = rr.read_OM_coords(start, stop)
        ```

3. ## **process_data** (`data/processor/data_utils.py`):
    - Provides filtering, joining coordinates to pulses and other enrichment for given DataFrames.
    - **Usage Example** (assuming we have dataframes `pulses`, `events`, `coords` and `muons`):
        ```python
        from data.processor.data_utils import process_data
        # set up config
        proc_cfg = dict(
            has_signal_flg = True,
            only_signal = True,  # Whether to filter only signal hits
            min_sig_hits = 8,  # Minimum number of signal hits per cluster to be kept
            min_sig_strings = 2,  # Minimum number of unique signal strings in a cluster
            min_Q = 0,  # Minimum pulse amplitude threshold
            center_times = True,  # Whether to center the event times
            relative_coords = True, # Whether to add coordinates relatively to the clusters centers
            to_calculate_tres = False, # Whether to calculate tres
            t_threshold = 1e5,  # Maximum time threshold for pulse filtering
            same_coordinates = True, # whether coords are same for each event or not
        )
        # process data
        new_pulses, new_events, new_muons = process_data(pulses
            , events
            , coords
            , muons
            , **proc_cfg)
        ```
    - Special setting dataclass `ProcessorConfig` is specified in `data/settings_scheme.py`.

4. ## **ChunksFromPaths** (`data/data_loader.py`):
    - Generates chunks of processed data, collecting information from list of `.root` files.
    - **Usage Example**:
        ```python
        from data.settings_scheme import ProcessorConfig
        from data.data_loader import ChunksFromPaths

        # MC DATA
        # insert your paths
        paths = [
            "/net/62/home3/ivkhar/Baikal/data/initial_data/MC_2020/muatm/root/all/10052.root",
            "/net/62/home3/ivkhar/Baikal/data/initial_data/MC_2020/muatm/root/all/10053.root",
            "/net/62/home3/ivkhar/Baikal/data/initial_data/MC_2020/muatm/root/all/10054.root",
            "/net/62/home3/ivkhar/Baikal/data/initial_data/MC_2020/nuatm/root/all/1048.root",
            "/net/62/home3/ivkhar/Baikal/data/initial_data/MC_2020/nuatm/root/all/1049.root",
            "/net/62/home3/ivkhar/Baikal/data/initial_data/MC_2020/nue2_100pev/root/all/1007.root",
        ]
        chunks = ChunksFromPaths(
            paths,
            is_mc_data=True,
            processor_cfg=ProcessorConfig(
                has_signal_flg=True
                , only_signal=False
                , min_signal_hits=0
                , min_signal_strings=0
                , same_coordinates=True),
            lookforward=float("inf"), # How many events to read from each .root file at once
            events_per_chunk=25_000, # How many events to collect in one chunk
            shuffle_paths=True, # Whether to shuffle paths' lists before generating data.
        )
        # load 1 chunk
        counter = 0
        for pulses, events, muons in chunks:
            counter += 1
            # print(chunks._current_path_idx)
            print(f"#{counter}: {events[['ev_id','cluster_id']].height} events was read")
            break
        ```
    - Special setting dataclass `ChunksFromPathsConfig` is specified in `data/settings_scheme.py`.

5. ## **MCMuNuSepBatchGenerator** (`data/batch_generators.py`):
    - Uses `ChunksFromPaths` to collect large chunks, and iterate over them, returnin batches with torch Tensors. Supports data normalization and augmentation.
    - Returns inputs and targets (labels) for training as a torch.Tensor type.
    - **Usage Example** is quite massive to give info about all the settings fields:
        ```python
        import torch
        from data.settings_scheme import ProcessorConfig
        from data.settings_scheme import (
            ChunksFromPathsConfig,
            MCMuNuSepBatchGeneratorConfig
        )
        from data.batch_generators import MCMuNuSepBatchGenerator, ExpBatchGenerator

        # Setting up configuration of your pipeline
        # It may be configured using dataclasses to be able to log settings well!
        batches_cfg = MCMuNuSepBatchGeneratorConfig(
            chunk_generator_cfg=ChunksFromPathsConfig(
                is_mc_data=True,
                processor_cfg=ProcessorConfig(
                    has_signal_flg=True, # Whether to filter only signal hits. Doesn't work when `has_signal_flg` is False
                    only_signal=False, # Whether to filter only signal hits. Doesn't work when `has_signal_flg` is False
                    min_sig_hits=0, # Minimum number of signal hits per cluster to be kept
                    min_sig_strings=0, # Minimum number of unique signal strings in a cluster
                    min_Q=0.0, # Minimum pulse amplitude threshold
                    center_times=True,# Whether to center the events' times
                    relative_coords=True, # Whether to add coordinates relatively to the clusters centers
                    to_calculate_tres=False, # Whether to calculate tres
                    same_coordinates=True, # If coords in initial data are same for each event or not. When True, accelerates processing.
                ),
                shuffle_paths=False,
                lookforward=50_000, # How many event to read from root file at one iteration while generating chunks
                prefix=None,
            ),
            mu_events_per_chunk=512 * 100, # How many EAS events to collect per chunk
            nu_events_per_chunk=512 * 100, # How many neutrino events to collect per chunk
            batch_size=512,
            features_to_take=["PulsesTime", "PulsesAmpl", "Xrel", "Yrel", "Zrel"], # Which features to use as input data
            do_norm=True,
            # parameters to use as (mean, std) while norming. The same length and order as `features_to_take`
            norm_params=[
                [0., 1374.0],
                [1.4, 23.1],
                [0.0, 60.0],
                [0.0, 60.0],
                [0.0, 260.0],
            ],
            do_augment=True,
            # stds in gauss noise to add in data. The same length and order as `features_to_take`
            augment_params=[
                5.0,  # ns
                0.1,  # ev
                2.0,  # m
                2.0,  # m
                5.0,  # m
            ],
            shuffle=True, # Whether to shuffle events in chunk before generating batches.
        )

        # Creating batch generator for your root_paths
        device = torch.device("cuda:0")
        mu_paths = [
            "/net/62/home3/ivkhar/Baikal/data/initial_data/MC_2020/muatm/root/all/10052.root",
            "/net/62/home3/ivkhar/Baikal/data/initial_data/MC_2020/muatm/root/all/10053.root",
            "/net/62/home3/ivkhar/Baikal/data/initial_data/MC_2020/muatm/root/all/10054.root",
        ]
        nu_paths = [
            "/net/62/home3/ivkhar/Baikal/data/initial_data/MC_2020/nuatm/root/all/1048.root",
            "/net/62/home3/ivkhar/Baikal/data/initial_data/MC_2020/nuatm/root/all/1049.root",
            "/net/62/home3/ivkhar/Baikal/data/initial_data/MC_2020/nue2_100pev/root/all/1007.root",
        ]
        batches = MCMuNuSepBatchGenerator(
            mu_paths, nu_paths, device=device, **batches_cfg.to_shallow_dict()
        )

        # Load 200 batches
        counter = 0
        for data, labels in batches:
            if counter > 200:
                break
            counter += 1
            print(f"#{counter}: {data.shape=}, {labels.shape=}")
        ```
    - Special setting dataclass `MCMuNuSepBatchGeneratorConfig` is specified in `data/settings_scheme.py`.

# Settings

The behavior of the pipeline can be customized using configuration classes found in `data/settings_scheme.py`. Users can define settings for normalization, augmentation, and processing as per their requirements.

To store config to .yaml file and to load it back as dataclass, use `data/settings_manager.py` script.

To save created Config:
```Python
from data.settings_manager import save_data_cfg

# Assuming you have batches_cfg: MCMuNuSepBatchGeneratorConfig
save_data_cfg(cfg=batches_cfg, path='./batches_cfg.yaml')
```

To load stored config and create BatchGenerator, use:

```Python
from data.settings_scheme import MCMuNuSepBatchGeneratorConfig
from data.settings_manager import load_batchgen_cfg

# Assuming you saved configuration as yaml at './batches_cfg.yaml'
batches_cfg = load_batchgen_cfg(path='./batches_cfg.yaml', DataClas=MCMuNuSepBatchGeneratorConfig)
```

```Python
import config_manager as cfgm
from batch_generators import MCMuNuSepBatchGenerator

path_to_cfg = f"./configurations/{name_of_dataset}/cfg.yaml"
cfg = cfgm.load_batchgen_cfg(path_to_cfg)

batches = MCMuNuSepBatchGenerator(mu_train_paths, nu_train_paths, device, **cfg.as_shallow_dict())
for batch in batches:
    data, labels = batch
    print(data.shape, labels.shape)
    break
```