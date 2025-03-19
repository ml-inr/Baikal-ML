from typing import Optional

import numpy as np

# import pandas as pd
import polars as pl
import uproot as ur
import awkward as ak
from functools import lru_cache

try:
    from data.root_manager.reader.root_paths import (
        RootPaths,
        PulsesFeaturesPaths,
        BaseFeaturePaths,
    )
    from data.root_manager.reader.polars_schema import DataSchema
    from data.root_manager.constants import Constants as Cnst
except:
    from root_manager.reader.root_paths import (
        RootPaths,
        PulsesFeaturesPaths,
        BaseFeaturePaths,
    )
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
        test = self.rf[f"{self.data_header}/{self.paths.ev_paths.PulsesN}"].array(
            library="np", entry_start=0, entry_stop=1
        )[0]
        return 1 if test == 0 else 0

    def __init__(
        self,
        root_file: ur.reading.ReadOnlyDirectory,
        paths: RootPaths = RootPaths(),
        prefix: str = "",
        data_header: str = "Events",
        coords_header: str = "ArrayConfig",
        start: int = 0,
        stop: int = None,
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
        self.data_header = data_header
        self.coords_header = coords_header
        
        self.start = max(start, self.get_start_index())
        if stop is not None:
            self.stop = stop
            self.ev_num = self.stop-self.start
        else:
            self.ev_num = self.rf[self.data_header][self.paths.ev_paths.PulsesN].num_entries
            self.stop = start+self.ev_num
        
        if not self._check_file():
            raise BadFile("no valid events in file")
        self.ev_ids = pl.Series(
            name="ev_id", values=[f"{self.prefix}{i}" for i in range(start, start+self.ev_num)]
        )
        self.df_schema = DataSchema().to_dict()

        
    @lru_cache(maxsize=None)
    def _get_event_arrays(self):
        names = self.ev_paths.keys()
        arrs = [
            list(a)
            for a in self.rf[f"{self.data_header}"]
            .arrays(
                self.ev_paths.values(),
                library="np",
                entry_start=self.start,
                entry_stop=self.stop,
            )
            .values()
        ]
        return names, arrs

    @lru_cache(maxsize=None)
    def _get_pulses_arrays(self):
        names = self.pulses_paths.keys()
        arrs = [
            list(a)
            for a in self.rf[f"{self.data_header}"]
            .arrays(
                self.pulses_paths.values(),
                library="np",
                entry_start=self.start,
                entry_stop=self.stop,
            )
            .values()
        ]
        return names, arrs

    @lru_cache(maxsize=None)
    def _get_muons_arrays(self):
        names = self.ind_mu_paths.keys()
        arrs = [
            list(a)
            for a in self.rf[f"{self.data_header}"]
            .arrays(
                self.ind_mu_paths.values(),
                library="np",
                entry_start=self.start,
                entry_stop=self.stop,
            )
            .values()
        ]
        return names, arrs
    
    @lru_cache(maxsize=None)
    def _get_coords_arrays(self):
        names = ["X", "Y", "Z"]
        coords = self.rf[self.coords_header][self.paths.geom_path].array(
            entry_start=self.start, entry_stop=self.stop
        ) 
        coords = np.array(ak.unzip(coords)) # array of shape (3, stop-start, 288*num_of_clusters)
        if self.coords_header == "ArrayConfig":
            assert coords.shape[1] == 1
            coords = coords.repeat(self.ev_num, axis=1)
        arrs = [list(values) for values in [*coords]]
        num_of_channels = coords.shape[2]
        return names, arrs, num_of_channels

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

    def read_OM_coords(self) -> pl.DataFrame:
        names, arrs, num_of_channels = self._get_coords_arrays()
        df = pl.DataFrame(
            arrs,
            schema={name: pl.List(pl.Float32) for name in names}
        )
        df = df.with_columns(self.ev_ids)

        ch_ids = np.arange(num_of_channels)
        df = df.with_columns(
            pl.lit(list(ch_ids))
            .cast(pl.Array(pl.Int32, num_of_channels))
            .alias("PulsesChID")
        )
        df = df.with_columns(
            pl.lit(list(ch_ids // Cnst.CHANNEL_DIVISOR))
            .cast(pl.Array(pl.Int8, num_of_channels))
            .alias("cluster_id")
        )
        df = df.with_columns(
            pl.lit(list(ch_ids // Cnst.STRING_DIVISOR))
            .cast(pl.Array(pl.Int8, num_of_channels))
            .alias("string_id")
        )

        df = df.explode(["X", "Y", "Z", "PulsesChID", "cluster_id", "string_id"])

        return df


if __name__ == "__main__":
    """
    Usage example
    """
