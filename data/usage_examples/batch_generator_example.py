import os
import sys
# Get the current working directory
cwd = os.getcwd()
# Add it to sys.path only if not already included
if cwd not in sys.path:
    sys.path.append(cwd)

import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

import torch

from data.settings_scheme import ProcessorConfig
from data.settings_scheme import (
    ChunksFromPathsConfig,
    MCMuNuSepBatchGeneratorConfig,
    ExpBatchGeneratorConfig,
)
from data.batch_generators import MCMuNuSepBatchGenerator, ExpBatchGenerator

# Setting up configuration of your pipeline to be able to log it well
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
    
