from dataclasses import dataclass
from typing import Tuple, List
import polars as pl
import numpy as np
from joblib import Parallel, delayed
import uproot as ur

from data.root_manager.root_reader import RootReader
from data.root_manager.constants import Constants as Cnst


# Parameters for filtering the data, allowing flexible configurations
@dataclass
class FilterParams:
    only_signal: bool = False  # Whether to filter only signal events
    min_hits: int = 5  # Minimum number of hits per cluster to be kept
    min_strings: int = 2  # Minimum number of unique strings in a cluster
    min_Q: float = 0  # Minimum pulse amplitude threshold
    t_threshold: float = 1e5  # Maximum time threshold for pulse filtering


# Settings for processing, including options for augmentation and filtering
@dataclass
class ProcessorSettings:
    center_times: bool = True  # Whether to center the event times
    split_multi: bool = True  # Whether to split multi-cluster events
    filter_cfg: FilterParams = FilterParams()  # Configuration for filtering


class Processor:
    def __init__(
        self, paths_to_roots: List[str], config: ProcessorSettings = ProcessorSettings()
    ):
        """
        Initializes the Processor class.
        Args:
            paths_to_roots (List[str]): List of paths to root files.
            config (ProcessorSettings): Settings for processing.
        """
        self.cfg = config
        self.paths = paths_to_roots
        self.prefixes = self._extract_prefixes(paths_to_roots)
        self.OM_coords = self.read_OM_coords(paths_to_roots[0])

    @staticmethod
    def _extract_prefixes(paths: List[str]) -> List[str]:
        """
        Extracts prefixes based on file paths.
        Args:
            paths (List[str]): List of paths to root files.
        Returns:
            List[str]: List of extracted prefixes.
        """
        prefixes = []
        for path in paths:
            particle_type = path.split("/")[-4]
            assert particle_type in [
                "mu",
                "muatm",
                "nu",
                "nuatm",
                "nue2",
                "nue2_100pev",
                "nu2",
                "nu2_100pev",
            ]
            file_name = path.split("/")[-1].split(".")[0]
            prefixes.append(f"{particle_type}_{file_name}_")
        return prefixes

    @staticmethod
    def read_OM_coords(path: str) -> pl.DataFrame:
        """
        Reads OM coordinates from a root file.
        Args:
            path (str): Path to the root file.
        Returns:
            pl.DataFrame: OM coordinates as a DataFrame.
        """
        with ur.open(path) as rf:
            rr = RootReader(rf)
            OM_coords = rr.read_OM_coords()
        return OM_coords

    @staticmethod
    def read_file(
        path: str, prefix: str
    ) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        Reads data from a root file and returns relevant DataFrames.
        Args:
            path (str): Path to the root file.
            prefix (str): Prefix for the file.
        Returns:
            Tuple[pl.DataFrame]: Tuple of DataFrames for events, pulses, muons, and OM coordinates.
        """
        with ur.open(path) as rf:
            rr = RootReader(rf, prefix=prefix)
            results = (
                rr.read_events_as_df(),
                rr.read_pulses_as_df(),
                rr.read_muons_as_df(),
                rr.read_OM_coords(),
            )
        return results

    def read_chunk_as_dfs(
        self, chunk_of_paths: List[str], chunk_of_prefixes: List[str]
    ) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        Loads a chunk of root files and returns concatenated DataFrames.
        Args:
            chunk_of_paths (List[str]): List of file paths.
            chunk_of_prefixes (List[str]): Corresponding prefixes.
        Returns:
            Tuple[pl.DataFrame]: Concatenated DataFrames for events, pulses, and muons.
        """
        results = Parallel(n_jobs=-1)(
            delayed(self.read_file)(path, prefix)
            for path, prefix in zip(chunk_of_paths, chunk_of_prefixes)
        )
        events, pulses, muons, new_OM_coords = zip(*results)
        for df in new_OM_coords:
            assert self.OM_coords.equals(df)  # Ensure OM coordinates match across files
        events, pulses, muons = pl.concat(events), pl.concat(pulses), pl.concat(muons)
        return events, pulses, muons

    @staticmethod
    def get_mu_direction(theta: pl.Expr, phi: pl.Expr):
        theta_rad, phi_rad = theta / 180 * np.pi, phi / 180 * np.pi
        dx_mu, dy_mu, dz_mu = (
            theta_rad.sin() * phi_rad.cos(),
            theta_rad.sin() * phi_rad.sin(),
            theta_rad.cos(),
        )
        dir_norm_mu = (dx_mu**2 + dy_mu**2 + dz_mu**2).sqrt()
        return dx_mu, dy_mu, dz_mu, dir_norm_mu

    @staticmethod
    def get_target_vec_direction(X: pl.Expr, Y, Z, X_mu, Y_mu, Z_mu):
        dx, dy, dz = X - X_mu, Y - Y_mu, Z - Z_mu
        taget_distance = (dx**2 + dy**2 + dz**2).sqrt()
        return dx, dy, dz, taget_distance

    @staticmethod
    def get_t_res(
        target_dir: tuple[pl.Expr, pl.Expr, pl.Expr, pl.Expr],
        mu_dir: tuple[pl.Expr, pl.Expr, pl.Expr, pl.Expr],
        mus_delay: pl.Expr,
        t_detected: pl.Expr,
    ):
        dx, dy, dz, targ_dist = target_dir
        dx_mu, dy_mu, dz_mu, dir_norm_mu = mu_dir
        cosAlpha = (dx * dx_mu + dy * dy_mu + dz * dz_mu) / (
            1e-9 + targ_dist * dir_norm_mu
        )
        sinAlpha = (1 - cosAlpha**2).sqrt()

        dMuon = targ_dist * (cosAlpha - sinAlpha / Cnst.TAN_C)
        tMuon = 1e9 * dMuon / Cnst.C_PART
        dLight = targ_dist * sinAlpha / Cnst.SIN_C
        tLight = 1e9 * dLight / Cnst.C_LIGHT

        t_exp = tMuon + tLight + mus_delay
        t_res_all = t_exp - t_detected

        return t_res_all

    def enrich_and_filter(
        self, events_df: pl.DataFrame, pulses_df: pl.DataFrame, muons_df: pl.DataFrame
    ) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        Filters and enriches the data based on filtering settings.
        Args:
            events_df (pl.DataFrame): DataFrame for events.
            pulses_df (pl.DataFrame): DataFrame for pulses.
            muons_df (pl.DataFrame): DataFrame for muons.
        Returns:
            Tuple[pl.DataFrame]: Filtered DataFrames for events, pulses, and muons.
        """
        filt_cfg = self.cfg.filter_cfg

        # Flatten pulses
        df_pulses_flat = pulses_df.explode(
            [c for c in pulses_df.columns if c not in ["ev_id"]]
        )
        # Enrich with Muon_id, SignalFlg, ClusterID and StringID
        df_pulses_flat = df_pulses_flat.with_columns(
            is_signal=(pl.col("PulsesFlg") != 0),
            cluster_id=(pl.col("PulsesChID") // 288).cast(pl.Int8),
            string_id=(pl.col("PulsesChID") // 36),
            mu_local_id=(pl.col("PulsesFlg") % 1_000_000 - 1).cast(pl.Int16),
        )
        # Filter pulses by 'is_signal' and 'Q' if needed
        if filt_cfg.only_signal:
            df_pulses_flat = df_pulses_flat.filter(pl.col("is_signal"))
        if (minQ := filt_cfg.min_Q) > 0:
            df_pulses_flat = df_pulses_flat.filter(pl.col("PulsesAmpl") >= minQ)
        # Filter bugged big Times
        df_pulses_flat = df_pulses_flat.filter(
            pl.col("PulsesTime") <= filt_cfg.t_threshold
        )
        # Center times
        if self.cfg.center_times:
            df_pulses_flat = df_pulses_flat.with_columns(
                PulsesTime=pl.col("PulsesTime") - pl.col("PulsesTime").mean()
            )
        # Join with OM coordinates
        df_pulses_flat = df_pulses_flat.join(
            self.OM_coords[["PulsesChID", "X", "Y", "Z", "Xrel", "Yrel", "Zrel"]],
            on="PulsesChID",
            how="left",
        )

        # Calculate t_res
        mu_local_ids = events_df["ev_id", "ResponseMuN"].with_columns(
            mu_local_id=pl.int_ranges(end=pl.col("ResponseMuN"), dtype=pl.Int16)
        )
        muons_df = muons_df.join(mu_local_ids[["ev_id", "mu_local_id"]], on="ev_id")
        muons_flat = muons_df.explode(
            [c for c in muons_df.columns if c not in ["ev_id"]]
        )

        pulses_for_tres = df_pulses_flat.filter(pl.col("is_signal"))[
            ["ev_id", "mu_local_id", "PulsesChID", "PulsesTime", "X", "Y", "Z"]
        ]
        muons_for_tres = muons_flat[
            [c for c in muons_flat.columns if c not in ["RespMuEn"]]
        ]
        df_for_tres = pulses_for_tres.join(
            muons_for_tres, on=["ev_id", "mu_local_id"], how="left"
        )
        df_for_tres = df_for_tres.with_columns(
            t_res=self.get_t_res(
                self.get_target_vec_direction(
                    pl.col("X"),
                    pl.col("Y"),
                    pl.col("Z"),
                    pl.col("RespMuTrackX"),
                    pl.col("RespMuTrackY"),
                    pl.col("RespMuTrackZ"),
                ),
                self.get_mu_direction(pl.col("RespMuTheta"), pl.col("RespMuPhi")),
                pl.col("RespMuDelay"),
                pl.col("PulsesTime"),
            )
        )

        # Add t_res to pulses dataframe
        df_pulses_flat = df_pulses_flat.join(
            df_for_tres[["ev_id", "mu_local_id", "PulsesChID", "PulsesTime", "t_res"]],
            on=["ev_id", "mu_local_id", "PulsesChID", "PulsesTime"],
            how="left",
            suffix="for_mu"
        )

        # Sort pulses by ids and time
        df_pulses_flat = df_pulses_flat.sort(["ev_id", "cluster_id", "PulsesTime"])

        # Group by ev_id and cluster_id
        query = [
            pl.col(c_name)
            for c_name in df_pulses_flat.columns
            if c_name not in ["ev_id", "cluster_id", "string_id", "t_res"]
        ] + [
            pl.col("is_signal")
                .alias("num_signal_hits")
                .sum()
                .cast(pl.Int16),  # calc number of signal hits
            pl.col("string_id")
                .filter(pl.col("is_signal"))
                .n_unique()
                .cast(pl.Int8)
                .alias("num_signal_strings"),  # calc number of unique signal strings
            pl.col("t_res")
                .filter(pl.col("is_signal"))
                .min()
                .alias("t_res"),
        ]
        grouped_pulses = df_pulses_flat.group_by(
            ["ev_id", "cluster_id"], maintain_order=True
        ).agg(query)

        # Apply the filtering by hits and strings nums
        grouped_pulses = grouped_pulses.filter(
            (pl.col("num_signal_hits") > filt_cfg.min_hits)
            & (pl.col("num_signal_strings") > filt_cfg.min_strings)
        )

        final_df = events_df.join(grouped_pulses, on="ev_id", how="inner").join(
            muons_df, on="ev_id", how="inner", suffix="_for_mu"
        )

        return final_df
