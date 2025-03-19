from dataclasses import dataclass
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import uproot as ur

from data.root_manager.root_reader import RootReader


# @dataclass
# class AugmentParams: ...


@dataclass
class FilterParams:
    only_signal: bool = False
    min_hits: int = 5
    min_strings: int = 2
    min_Q: float = 0
    t_threshold: float = 1e5


@dataclass
class ProcessorSettings:
    center_times: bool = True
    split_multi: bool = True

    filter_cfg: FilterParams = FilterParams()

    add_augment: bool = False
    # aug_params: AugmentParams = AugmentParams()


class Processor:
    def __init__(
        self, paths_to_roots: list[str], config: ProcessorSettings = ProcessorSettings()
    ):
        self.cfg = config
        self.paths = paths_to_roots
        self.prefixes = self._extract_prefixes(paths_to_roots)
        self.OM_coords = self.read_OM_coords(paths_to_roots[0]).set_index(
            ["PulsesChID"]
        )
        self.dfs = None

    @staticmethod
    def _extract_prefixes(paths: list[str]) -> list[str]:
        prefixes = [""] * len(paths)
        for _, path in enumerate(paths):
            particle_type = path.split("/")[-4]
            assert particle_type in [
                "mu",
                "muatm",
                "nu",
                "nuatm",
                "nue2",
                "nue2_100pev",
                "nu2",
                "nu2_100pev"
            ]
            file_name = path.split("/")[-1].split(".")[0]
            prefixes[_] = f"{particle_type}_{file_name}_"
        return prefixes

    @staticmethod
    def read_OM_coords(path):
        with ur.open(path) as rf:
            rr = RootReader(rf)
            OM_coords = rr.read_OM_coords()
        return OM_coords

    @staticmethod
    def read_file(path: str, prefix: str) -> tuple[pd.DataFrame]:
        with ur.open(path) as rf:
            rr = RootReader(rf, prefix=prefix)
            results = (
                rr.read_events_as_df().set_index(["ev_id"]),
                rr.read_pulses_as_df().set_index(["ev_id", "cluster_id"]),
                rr.read_ind_mu_as_df().set_index(["ev_id", "mu_local_id"]),
                rr.read_OM_coords().set_index(["PulsesChID"]),
            )
        return results

    def load_chunk_as_dfs(
        self, chunk_of_paths: list[str], chunk_of_prefixes: list[str]
    ) -> tuple[pd.DataFrame]:
        results = Parallel(n_jobs=-1)(
            delayed(self.read_file)(path, prefix)
            for (path, prefix) in zip(chunk_of_paths, chunk_of_prefixes)
        )
        events, pulses, muons, new_OM_coords = zip(*results)
        for df in new_OM_coords:
            assert self.OM_coords.equals(df)
        events, pulses, muons = pd.concat(events), pd.concat(pulses), pd.concat(muons)
        # pulses = pulses.merge(self.OM_coords[['Xrel', 'Yrel', 'Zrel']], on='PulsesChID')
        return events, pulses, muons

    def enrich_and_filter(
        self,
        events_df: pd.DataFrame,
        pulses_df: pd.DataFrame,
        muons_df: pd.DataFrame,
        filter_settings: FilterParams = FilterParams(),
    ) -> tuple[pd.DataFrame]:

        fil_cfg = filter_settings
        if filter_settings.only_signal:
            pulses_df = pulses_df[pulses_df["is_signal"]]

        pulses_df = pulses_df[
            (pulses_df["PulsesTime"] <= fil_cfg.t_threshold)
            & (pulses_df["PulsesAmpl"] >= fil_cfg.min_Q)
        ]

        signal_hits_per_cluster = (
            pulses_df[["is_signal"]]
            .groupby(["ev_id", "cluster_id"])
            .sum()
            .rename(columns={"is_signal": "signal_hits_num"})
        )
        string_nums = (
            pulses_df[pulses_df["is_signal"]][["string_id"]]
            .groupby(["ev_id", "cluster_id"])
            .nunique()
            .rename(columns={"string_id": "unique_signal_strings_num"})
        )

        events_enrich = events_df.join(signal_hits_per_cluster, how="left").join(
            string_nums, how="left"
        )
        events_filter = events_enrich[
            (events_enrich["signal_hits_num"] >= fil_cfg.min_hits)
            & (events_enrich["unique_signal_strings_num"] >= fil_cfg.min_strings)
        ]

        bad_idxs = signal_hits_per_cluster.index[
            signal_hits_per_cluster["signal_hits_num"] < fil_cfg.min_hits
        ].union(
            string_nums.index[
                string_nums["unique_signal_strings_num"] < fil_cfg.min_strings
            ]
        )
        pulses_filter = pulses_df.drop(bad_idxs)
        # add coords to each hit
        pulses_filter = (
            pulses_filter.reset_index()
            .merge(
                self.OM_coords[["Xrel", "Yrel", "Zrel"]],
                on="PulsesChID"
            )
            .set_index(["ev_id", "cluster_id"])
        )

        muons_filter = muons_df.loc[events_filter.reset_index()["ev_id"]]

        return events_filter, pulses_filter, muons_filter

    # def augment(self) -> tuple[pd.DataFrame]: ...

    def save_chunk_to_h5(self, chunk, path_to_h5: str, mode: str = "a"):
        pass

    def store_all_to_h5(self, path_to_h5): ...

    # def generate_batch(self) -> tuple[np.ndarray]: #shapes = [(batch_size, max_N_hits, feat_num), (N_events, labels_num)]
    #     ...
    #     return data, labels
