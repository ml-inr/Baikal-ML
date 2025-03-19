from typing import Optional
import h5py as h5
import numpy as np

# import pandas as pd
import polars as pl
import uproot as ur
import awkward as ak
from functools import lru_cache

try:
    from data.h5_manager.reader.h5_internal_paths import H5Paths
    from data.h5_manager.reader.polars_schema import DataSchema
    from data.h5_manager.constants import Constants as Cnst
except:
    from h5_manager.reader.h5_internal_paths import H5Paths
    from h5_manager.reader.polars_schema import DataSchema
    from h5_manager.constants import Constants as Cnst


class BadFile(Exception):
    pass


class H5Reader:

    def __init__(
        self,
        h5_file: h5.File,
        particle_type: str = "muatm",
        filenum: int = 1000,
        paths: H5Paths = H5Paths()
    ):
        self.hf = h5_file
        self.particle_type = particle_type
        self.filenum = filenum
        self.ev_paths = paths.ev_paths.to_dict_with_vars(particle_type, filenum)
        self.pulses_paths = paths.pulses_paths.to_dict_with_vars(particle_type, filenum)
        self.ind_mu_paths = paths.ind_mu_paths.to_dict_with_vars(particle_type, filenum)
        self.geom_paths = paths.geom_path.to_dict_with_vars(particle_type, filenum)
        self.df_schema = DataSchema().to_dict()


    @lru_cache(maxsize=None)
    def _get_event_arrays(self):
        arrs = [
            self.hf[path.split('.')[0]][:, int(path.split('.')[1])] 
            if '.' in path else self.hf[path][:] 
            for path in self.ev_paths.values()
        ]
        return self.ev_paths.keys(), arrs

    @lru_cache(maxsize=None)
    def _get_pulses_arrays(self):
        ev_starts = self.hf[self.pulses_paths['EventStarts']][:]
        ev_ids = self.hf[self.pulses_paths['ev_id']][:]
        names, arrs = ['ev_id'], [np.repeat(ev_ids, np.diff(ev_starts))]
        pulses_dict = {k: path for k, path in self.pulses_paths.items() if k not in ['EventStarts','ev_id']}
        # collect arrays
        for name, path in pulses_dict.items():
            names.append(name)
            arrs.append(
                    self.hf[path.split('.')[0]][:, int(path.split('.')[1])] if '.' in path else self.hf[path][:],
                )
        return names, arrs

    @lru_cache(maxsize=None)
    def _get_muons_arrays(self):
        ev_starts = self.hf[self.ind_mu_paths['EventStarts']][:]
        ev_ids = self.hf[self.ind_mu_paths['ev_id']][:]
        names, arrs = ['ev_id'], [np.repeat(ev_ids, np.diff(ev_starts))]
        names.append('mu_local_id'), arrs.append(np.concatenate([np.arange(count) for count in np.diff(ev_starts)]))
        muons_dict = {k: path for k, path in self.ind_mu_paths.items() if k not in ['EventStarts','ev_id']}
        # collect arrays
        for name, path in muons_dict.items():
            names.append(name)
            arrs.append(
                    self.hf[path.split('.')[0]][:, int(path.split('.')[1])] if '.' in path else self.hf[path][:],
                )
        return names, arrs
    
    @lru_cache(maxsize=None)
    def _get_geom_arrays(self) -> tuple[list, list]:
        arrs = [
            self.hf[path.split('.')[0]][:, int(path.split('.')[1])] 
            if '.' in path else self.hf[path][:] 
            for path in self.geom_paths.values()
        ]
        
        return list(self.geom_paths.keys()), arrs

    def read_events_as_df(self) -> pl.DataFrame:
        names, arrs = self._get_event_arrays()
        df = pl.DataFrame(arrs, schema=names)
        df = df.cast({k: v for k, v in self.df_schema.items() if k in names})
        return df.unique()

    def read_pulses_as_df(self) -> pl.DataFrame:
        names, arrs = self._get_pulses_arrays()
        df = pl.DataFrame(arrs, schema=names)
        df = df.cast({k: v for k, v in self.df_schema.items() if k in names})
        return df.unique()

    def read_muons_as_df(self) -> pl.DataFrame:
        names, arrs = self._get_muons_arrays()
        df = pl.DataFrame(arrs, schema=names)
        df = df.cast({k: v for k, v in self.df_schema.items() if k in names})
        return df.unique()
    
    def read_cluster_centers(self):
        names, arrs = self._get_geom_arrays()
        names.append('cluster_id'), arrs.append(list(range(len(arrs[0]))))
        df = pl.DataFrame(arrs, schema=names)
        return df.unique()


if __name__ == "__main__":
    """
    Usage example
    """
