import os
import logging
from typing import Optional, Dict, Tuple

import numpy as np
import polars as pl
import uproot as ur
import awkward as ak

from .internal_root_paths import BaseRootPaths
from .polars_schema import DataSchema
from ..constants import Constants as Cnst

# Custom exception for invalid files
class BadFile(Exception):
    pass

class RootFileReader:
    """
    A class to read and process ROOT files using uproot, polars, and awkward.
    """

    def __init__(
        self, file_path: str, internal_root_paths: Optional[BaseRootPaths] = None, prefix: str = ""
    ):
        """
        Initialize the RootFileReader.
        
        :param file_path: Path to the ROOT file.
        :param internal_root_paths: Paths to ROOT file structures (default: BaseRootPaths).
        :param prefix: Prefix for event IDs.
        """
        self.validate_file_path(file_path)
        self.file_path = file_path
        self.rf = None
        self.paths = internal_root_paths or BaseRootPaths()
        self.prefix = prefix

        self.ev_paths = self.paths.ev_paths.to_dict()
        self.pulses_paths = self.paths.pulses_paths.to_dict()
        self.ind_mu_paths = self.paths.ind_mu_paths.to_dict()

        self.data_header = self.paths.data_header
        self.coords_header = self.paths.coords_header

        self.ev_num = None
        self.ev_ids = None
        self.df_schema = DataSchema().to_dict()

        logging.debug("RootFileReader initialized with file_path=%s, prefix=%s", file_path, prefix)

    @staticmethod
    def validate_file_path(file_path: str):
        """Validate that the file exists and is a valid ROOT file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        if not file_path.endswith(".root"):
            raise ValueError("Invalid file format. Expected a ROOT file.")

    def __enter__(self):
        """Open the ROOT file when entering the context."""
        self.rf = ur.open(self.file_path)
        self.ev_num = self.rf[self.data_header][self.ev_paths["PulsesN"]].num_entries
        if not self._check_file():
            raise BadFile("No valid events in file")
        self.ev_ids = pl.Series(
            name="ev_id", values=[f"{self.prefix}{i}" for i in range(self.ev_num)]
        )
        logging.debug("Successfully opened the ROOT file with %d events.", self.ev_num)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Close the ROOT file safely when exiting the context."""
        if self.rf:
            self.rf.close()
            logging.debug("Closed the ROOT file.")
        return False

    def _check_file(self) -> bool:
        """Check if the ROOT file contains valid events."""
        try:
            valid = self.ev_num > 1 or (
                self.ev_num == 1
                and self.rf[self.paths.ev_paths.PulsesN].array(library="np")[0] != 0
            )
            logging.debug("File check passed: %s", valid)
            return valid
        except Exception as e:
            logging.error("Error while checking file: %s", e)
            return False

    def _handle_header_logic(self, header_type: str, start: int, stop: int):
        """Handle logic for different headers (ArrayConfig vs. Events)."""
        if header_type == "ArrayConfig":
            entry_start = 0
            entry_stop = None
        elif header_type == "Events":
            entry_start = start
            entry_stop = stop
        else:
            raise ValueError(f"Unknown header type: {header_type}")
        logging.debug("Header logic determined: entry_start=%s, entry_stop=%s", entry_start, entry_stop)
        return entry_start, entry_stop

    def _get_data_as_polars(self, paths: Dict[str, str], start: int = 0, stop: int = None) -> pl.DataFrame:
        """Generic method to retrieve arrays from ROOT file and convert to polars dataframe. 
        The fastest way to extract data from root to any dataframe: internal uproot option.
        Pandas -> Polars transformation takes almost 0 time.
        It much faster than transforming to Python lists and then to Polars."""
        logging.debug("Fetching data as Polars DataFrame with start=%d, stop=%s", start, stop)
        names = paths.keys()
        df = self.rf[self.data_header].arrays(paths.values(), library="pd", entry_start=start, entry_stop=stop)
        for col in df.columns:
            df[col] = df[col].to_list()
        df = df.rename(columns={path: name for name, path in paths.items()})
        df = pl.from_pandas(df, schema_overrides={k: v for k, v in self.df_schema.items() if k in names})
        df = df.with_columns(self.ev_ids[start:stop])
        logging.debug("Successfully created Polars DataFrame with columns: %s", df.columns)
        return df

    def read_OM_coords(self, start: int = 0, stop: int = None) -> pl.DataFrame:
        """Read optical module coordinates as a Polars DataFrame."""
        if not self.rf:
            raise RuntimeError("File is not open. Use this class as a context manager.")

        names = ["X", "Y", "Z"]
        entry_start, entry_stop = self._handle_header_logic(self.coords_header, start, stop)

        logging.debug("Reading OM coordinates with entry_start=%s, entry_stop=%s", entry_start, entry_stop)
        coords = self.rf[self.coords_header].arrays([self.paths.geom_path],
            entry_start=entry_start, entry_stop=entry_stop, library="ak")
        coords = np.array(ak.unzip(coords))[0]  # Shape: (num_events, 288*num_clusters)
        num_of_channels = coords.shape[1]

        df = pl.from_numpy(coords, schema={name: self.df_schema[name] for name in names})
        ch_ids = np.arange(num_of_channels)
        df = df.with_columns(
            pl.lit(ch_ids.tolist()).cast(pl.Array(pl.Int16, num_of_channels)).alias("PulsesChID"),
            pl.lit((ch_ids // Cnst.CHANNEL_DIVISOR).tolist()).cast(pl.Array(pl.Int8, num_of_channels)).alias("cluster_id"),
            pl.lit((ch_ids // Cnst.STRING_DIVISOR).tolist()).cast(pl.Array(pl.Int16, num_of_channels)).alias("string_id"),
        )

        # Add event IDs
        if self.coords_header == "ArrayConfig":
            assert coords.shape[0] == 1, f"{coords.shape=}"
            df = df.with_columns(self.ev_ids[start:stop].implode()).explode(["ev_id"])
        elif self.coords_header == "Events":
            df = df.with_columns(self.ev_ids[start:stop])
        else:
            raise ValueError("Unknown header for reading coordinates.")

        logging.debug("Successfully read OM coordinates into a DataFrame with %d rows.", len(df))
        return df.explode([col for col in df.columns if col != "ev_id"])

    def read_events_as_df(self, start: int = 0, stop: int = None) -> pl.DataFrame:
        """Read events as a Polars DataFrame."""
        if not self.rf:
            raise RuntimeError("File is not open. Use this class as a context manager.")
        df = self._get_data_as_polars(self.ev_paths, start, stop)
        logging.debug("Read %d events as DataFrame.", len(df))
        return df

    def read_pulses_as_df(self, start: int = 0, stop: int = None) -> pl.DataFrame:
        """Read pulses as a Polars DataFrame."""
        if not self.rf:
            raise RuntimeError("File is not open. Use this class as a context manager.")
        df = self._get_data_as_polars(self.pulses_paths, start, stop)
        df = df.explode([column for column in df.columns if column != "ev_id"])
        logging.debug("Read pulses as DataFrame with %d rows after exploding.", len(df))
        return df

    def read_muons_as_df(self, start: int = 0, stop: int = None) -> pl.DataFrame:
        """Read muons as a Polars DataFrame."""
        if not self.rf:
            raise RuntimeError("File is not open. Use this class as a context manager.")
        df = self._get_data_as_polars(self.ind_mu_paths, start, stop)
        df = df.with_columns(
            pl.int_ranges(0, pl.col(df.drop('ev_id').columns[0]).list.len(), dtype=pl.Int16).alias("mu_local_id")
        )
        df = df.explode([column for column in df.columns if column != "ev_id"])
        logging.debug("Read muons as DataFrame with %d rows after exploding.", len(df))
        return df
