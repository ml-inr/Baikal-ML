import sys

PROJECT_PATH = "../"
sys.path.append(f"{PROJECT_PATH}")

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
print(f"Data from Monte-Carlo file are extracted as 4 Polars DataFrames:")
print(f"\tPulses:", pulses, "\n")
print(f"\tEvents:", events, "\n")
print(f"\tMuons:", muons, "\n")
print(f"\tCoords:", coords, "\n")

# Experimental file reading
path = (
    "/net/62/home3/ivkhar/Baikal/data/initial_data/exp_data/2019_cl3_i0120_bevent.root"
)
start = 0
stop = 30_000  # since file is huge, stopper is needed
with RootFileReader(path, internal_root_paths=ExpRootPaths()) as rr:
    pulses = rr.read_pulses_as_df(start, stop)
    events = rr.read_events_as_df(start, stop)
    coords = rr.read_OM_coords(start, stop)
    muons = None  # muons are not availiabe at experiment directly
print(f"Data from experimental file are extracted as 3 Polars DataFrames:")
print(f"\tPulses:", pulses, "\n")
print(f"\tEvents:", events, "\n")
print(f"\tCoords:", coords, "\n")
print(f"\tInfo about muons is not availiabe from experiment directly")
