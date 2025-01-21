import sys

PROJECT_PATH = "../"
sys.path.append(f"{PROJECT_PATH}")

import logging

logging.basicConfig(level=logging.INFO)

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
    processor_cfg=ProcessorConfig(True, False, 0, 0, same_coordinates=True).to_dict(),
    lookforward=float("inf"),
    events_per_chunk=25_000,
    shuffle_paths=True,
)
counter = 0
for pulses, events, muons in chunks:
    counter += 1
    # print(chunks._current_path_idx)
    print(f"#{counter}: {events[['ev_id','cluster_id']].height} events was read")
    break
print(f"THE CHUNK OF MC DATA WAS GENERATED AS 3 DATAFRAMES:")
print(f"\tPulses:", pulses, "\n")
print(f"\tEvents:", events, "\n")
print(f"\tMuons:", muons, "\n")

# EXP DATA
# insert your paths
paths = [
    "/net/62/home3/ivkhar/Baikal/data/initial_data/exp_data/2019_cl3_i0120_bevent.root"
]
chunks = ChunksFromPaths(
    paths,
    is_mc_data=False,
    processor_cfg=ProcessorConfig(False, False, 0, 0, same_coordinates=False).to_dict(),
    lookforward=100_000,
    events_per_chunk=100_000,
    shuffle_paths=False,
)

counter = 0
for pulses, events, muons in chunks:
    counter += 1
    # print(chunks._current_path_idx)
    print(f"#{counter}: {events[['ev_id','cluster_id']].height} events was read")
    break
print(f"THE CHUNK OF EXP DATA WAS GENERATED AS 2 DATAFRAMES:")
print(f"\tPulses:", pulses, "\n")
print(f"\tEvents:", events, "\n")
