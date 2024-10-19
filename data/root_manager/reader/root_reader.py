from typing import Optional

import numpy as np

# import pandas as pd
import polars as pl
import uproot as ur
import awkward as ak
from functools import lru_cache

try:
    from data.root_manager.reader.root_paths import RootPaths, BaseFeaturePaths
    from data.root_manager.reader.polars_schema import DataSchema
    from data.root_manager.constants import Constants as Cnst
except:
    from root_manager.reader.root_paths import RootPaths, BaseFeaturePaths
    from root_manager.reader.polars_schema import DataSchema
    from root_manager.constants import Constants as Cnst

class BadFile(Exception):
    pass


class RootReader:

    def _check_file(self) -> bool:
        try:
            return self.ev_num > 1 or (
                self.ev_num == 1
                and self.rf[self.paths.ev_paths.PulsesN].array(library="np")[0] != 0
            )
        except Exception:
            return False

    def get_start_index(self):
        test = self.rf["Events/BEvent./BEvent.fPulseN"].array(
            library="np", entry_start=0, entry_stop=1
        )[0]
        return 1 if test == 0 else 0

    def __init__(
        self,
        root_file: ur.reading.ReadOnlyDirectory,
        paths: RootPaths = RootPaths(),
        prefix: str = "",
    ):
        self.rf = root_file
        self.paths = paths
        self.ev_paths = self.paths.ev_paths.to_dict()
        self.pulses_paths = self.paths.pulses_paths.to_dict()
        self.ind_mu_paths = self.paths.ind_mu_paths.to_dict()
        self.all_data_paths = {
            **self.paths.ev_paths.to_dict(),
            **self.paths.pulses_paths.to_dict(),
            **self.paths.ind_mu_paths.to_dict(),
        }

        self.prefix = prefix
        self.ev_num = self.rf[self.paths.ev_paths.PulsesN].num_entries
        if not self._check_file():
            raise BadFile("no valid events in file")
        self.ev_ids = pl.Series(
            name="ev_id", values=[f"{self.prefix}{i}" for i in range(self.ev_num)]
        )
        self.start = self.get_start_index()
        self.df_schema = DataSchema().to_dict()

    @lru_cache(maxsize=None)
    def _get_event_arrays(self):
        header = "Events"
        names = self.ev_paths.keys()
        root_paths = [p[7:] for p in self.ev_paths.values()]
        arrs = [
            list(a)
            for a in self.rf[f"{header}"]
            .arrays(root_paths, library="np", entry_start=self.start)
            .values()
        ]
        return names, arrs

    @lru_cache(maxsize=None)
    def _get_pulses_arrays(self):
        header = "Events"
        names = self.pulses_paths.keys()
        root_paths = [p[7:] for p in self.pulses_paths.values()]
        arrs = [
            list(a)
            for a in self.rf[f"{header}"]
            .arrays(root_paths, library="np", entry_start=self.start)
            .values()
        ]
        return names, arrs

    @lru_cache(maxsize=None)
    def _get_muons_arrays(self):
        header = "Events"
        names = self.ind_mu_paths.keys()
        root_paths = [p[7:] for p in self.ind_mu_paths.values()]
        arrs = [
            list(a)
            for a in self.rf[f"{header}"]
            .arrays(root_paths, library="np", entry_start=self.start)
            .values()
        ]
        return names, arrs

    @lru_cache(maxsize=None)
    def _get_all_arrays(self):
        header = "Events"
        names = list(self.all_data_paths.keys())
        root_paths = [p[7:] for p in self.all_data_paths.values()]
        arrs = [
            list(a)
            for a in self.rf[f"{header}"]
            .arrays(root_paths, library="np", entry_start=self.start)
            .values()
        ]
        return names, arrs

    def read_events_as_df(self) -> pl.DataFrame:
        names, arrs = self._get_event_arrays()
        df = pl.DataFrame(arrs, schema=names)
        df = df.cast({k: v for k, v in self.df_schema.items() if k in names})
        df = df.with_columns(self.ev_ids)
        return df

    def read_pulses_as_df(self) -> pl.DataFrame:
        names, arrs = self._get_pulses_arrays()
        df = pl.DataFrame(arrs, schema=names)
        df = df.cast({k: v for k, v in self.df_schema.items() if k in names})
        df = df.with_columns(self.ev_ids)
        return df

    def read_muons_as_df(self) -> pl.DataFrame:
        names, arrs = self._get_muons_arrays()
        df = pl.DataFrame(arrs, schema=names)
        df = df.cast({k: v for k, v in self.df_schema.items() if k in names})
        df = df.with_columns(self.ev_ids)
        return df

    def read_all_data_as_df(self) -> pl.DataFrame:
        names, arrs = self._get_all_arrays()
        df = pl.DataFrame(arrs, schema=names)
        df = df.cast({k: v for k, v in self.df_schema.items() if k in names})
        df = df.with_columns(self.ev_ids)
        return df

    def read_OM_coords(self):
        coords_array = np.array(ak.unzip(self.rf[self.paths.geom_path].array()))[
            :, 0, :
        ]
        df = pl.DataFrame(coords_array.T, schema=["X", "Y", "Z"])
        df = df.with_columns(
            pl.Series(name="PulsesChID", values=range(df.shape[0]), dtype=pl.Int16)
        )
        df = df.with_columns(
            (pl.col("PulsesChID") // Cnst.CHANNEL_DIVISOR)
            .alias("cluster_id")
            .cast(pl.Int8)
        )
        df = df.with_columns(
            (pl.col("PulsesChID") // Cnst.STRING_DIVISOR).alias("string_id")
        )
        #   Add info about clusters centers
        cl_centers = (
            df[["X", "Y", "Z", "cluster_id"]]
            .group_by(["cluster_id"])
            .mean()
            .rename({"X": "Xc", "Y": "Yc", "Z": "Zc"})
        )
        df = df.join(cl_centers, on="cluster_id")
        #   Add coords, relatively to the centers
        expressions = []
        for coord in ["X", "Y", "Z"]:
            expressions.append(
                (pl.col(coord) - pl.col(f"{coord}c")).alias(f"{coord}rel")
            )
        df = df.with_columns(*expressions)
        return df


if __name__ == "__main__":
    """
    Usage example
    """
