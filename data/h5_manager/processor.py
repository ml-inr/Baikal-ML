from dataclasses import dataclass
from typing import Tuple, List
import polars as pl
import numpy as np
from joblib import Parallel, delayed
import h5py as h5

try:
    from data.h5_manager.reader.h5_reader import H5Reader
    from data.h5_manager.constants import Constants as Cnst
    from data.h5_manager.settings import ProcessorConfig
except ImportError:
    from h5_manager.reader.h5_reader import H5Reader
    from h5_manager.constants import Constants as Cnst
    from h5_manager.settings import ProcessorConfig


class Processor:
    def __init__(self, paths_to_roots: List[str], config: ProcessorConfig = ProcessorConfig()):
        """
        Initializes the Processor class, setting up paths and configuration for processing ROOT files.

        Args:
            paths_to_roots (List[str]): List of paths to root files.
            config (ProcessorSettings): Settings for processing data (filtering and augmentation options).
        """
        self.cfg = config
        self.signatures = self._extract_signatures(paths_to_roots)
        self.filter_koef = "Not estimated untill processing"

    @staticmethod
    def _extract_signatures(paths: List[str]) -> List[str]:
        """
        Extracts list[(ParticleType, FileNum)] based on file paths, used to differentiate between file types (mu, nu, etc.).

        Args:
            paths (List[str]): List of paths to root files.

        Returns:
            list[str, int]: List of extracted signatures for each file.
        """
        signatures = []
        for path in paths:
            # Extracts the particle type from the path
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
            if ('nue2' in particle_type) or ('nu2' in particle_type):
                particle_type = 'nue2'
            file_num = int(path.split("/")[-1].split(".")[0])
            signatures.append((particle_type, file_num))
        return signatures

    @staticmethod
    def read_cluster_centers(path: str) -> pl.DataFrame:
        """
        Reads OM (Optical Module) coordinates from a ROOT file using the RootReader.

        Args:
            path (str): Path to the root file.

        Returns:
            pl.DataFrame: A Polars DataFrame containing the OM coordinates.
        """
        with h5.File(path) as hf:
            rr = H5Reader(hf)
            OM_coords = rr.read_cluster_centers()
        return OM_coords

    @staticmethod
    def read_file(path: str, prefix: str) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        Reads data from a ROOT file and returns DataFrames for events, pulses, muons, and OM coordinates.

        Args:
            path (str): Path to the root file.
            prefix (str): Prefix used to identify different particle types in the data.

        Returns:
            Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]: DataFrames for events, pulses, muons, and OM coordinates.
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

    def _read_root2df(self) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        Loads a chunk of root files and returns concatenated DataFrames for events, pulses, and muons.

        Returns:
            Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]: Concatenated DataFrames for events, pulses, and muons.
        """
        read_paths = self.paths
        prefixes = self._extract_prefixes(self.paths)

        # Parallel processing to read multiple files at once
        results = Parallel(n_jobs=-2)(
            delayed(self.read_file)(path, prefix) for path, prefix in zip(read_paths, prefixes)
        )
        events, pulses, muons, new_OM_coords = zip(*results)

        # Ensure OM coordinates are the same across files
        for df in new_OM_coords:
            assert self.OM_coords.equals(df)

        # Concatenate the DataFrames
        events, pulses, muons = pl.concat(events), pl.concat(pulses), pl.concat(muons)
        return events, pulses, muons

    @staticmethod
    def get_mu_direction(theta: pl.Expr, phi: pl.Expr) -> Tuple[pl.Expr, pl.Expr, pl.Expr, pl.Expr]:
        """
        Calculates the muon direction vector (dx, dy, dz) and its norm from spherical coordinates (theta, phi).

        Args:
            theta (pl.Expr): Theta angle in degrees.
            phi (pl.Expr): Phi angle in degrees.

        Returns:
            Tuple[pl.Expr, pl.Expr, pl.Expr, pl.Expr]: Direction vectors (dx, dy, dz) and their norm.
        """
        # Convert degrees to radians
        theta_rad, phi_rad = theta / 180 * np.pi, phi / 180 * np.pi
        dx_mu, dy_mu, dz_mu = (
            theta_rad.sin() * phi_rad.cos(),
            theta_rad.sin() * phi_rad.sin(),
            theta_rad.cos(),
        )
        dir_norm_mu = (dx_mu**2 + dy_mu**2 + dz_mu**2).sqrt()
        return dx_mu, dy_mu, dz_mu, dir_norm_mu

    @staticmethod
    def get_target_vec_direction(
        X: pl.Expr, Y: pl.Expr, Z: pl.Expr, X_mu: pl.Expr, Y_mu: pl.Expr, Z_mu: pl.Expr
    ) -> Tuple[pl.Expr, pl.Expr, pl.Expr, pl.Expr]:
        """
        Computes the vector from the target coordinates to the muon direction.

        Args:
            X (pl.Expr): X coordinate of the target.
            Y (pl.Expr): Y coordinate of the target.
            Z (pl.Expr): Z coordinate of the target.
            X_mu (pl.Expr): X coordinate of the muon.
            Y_mu (pl.Expr): Y coordinate of the muon.
            Z_mu (pl.Expr): Z coordinate of the muon.

        Returns:
            Tuple[pl.Expr, pl.Expr, pl.Expr, pl.Expr]: The target direction vectors and the distance between the target and the muon.
        """
        dx, dy, dz = X - X_mu, Y - Y_mu, Z - Z_mu
        target_distance = (dx**2 + dy**2 + dz**2).sqrt()
        return dx, dy, dz, target_distance

    @staticmethod
    def get_t_res(
        target_dir: Tuple[pl.Expr, pl.Expr, pl.Expr, pl.Expr],
        mu_dir: Tuple[pl.Expr, pl.Expr, pl.Expr, pl.Expr],
        mus_delay: pl.Expr,
        t_detected: pl.Expr,
    ) -> pl.Expr:
        """
        Computes the time residual between expected and detected times for pulses.

        Args:
            target_dir (tuple): Direction vectors and distance of the target (dx, dy, dz, distance).
            mu_dir (tuple): Direction vectors and norm of the muon (dx_mu, dy_mu, dz_mu, norm).
            mus_delay (pl.Expr): Muon delay.
            t_detected (pl.Expr): Detected pulse time.

        Returns:
            pl.Expr: Time residuals for the pulses.
        """
        dx, dy, dz, target_dist = target_dir
        dx_mu, dy_mu, dz_mu, dir_norm_mu = mu_dir

        # Calculate angles and corresponding distances
        cosAlpha = (dx * dx_mu + dy * dy_mu + dz * dz_mu) / (1e-9 + target_dist * dir_norm_mu)
        sinAlpha = (1 - cosAlpha**2).sqrt()

        dMuon = target_dist * (cosAlpha - sinAlpha / Cnst.TAN_C)
        tMuon = 1e9 * dMuon / Cnst.C_PART
        dLight = target_dist * sinAlpha / Cnst.SIN_C
        tLight = 1e9 * dLight / Cnst.C_LIGHT

        t_exp = tMuon + tLight + mus_delay
        t_res_all = t_exp - t_detected

        return t_res_all

    def calc_tres(self, muons_df: pl.DataFrame, df_pulses_flat: pl.DataFrame) -> pl.DataFrame:
        # Prepare DataFrames for t_res calculation
        muons_flat = muons_df.explode([c for c in muons_df.columns if c not in ["ev_id"]])
        pulses_for_tres = df_pulses_flat.filter(pl.col("is_signal"))[
            ["ev_id", "mu_local_id", "PulsesChID", "PulsesTime", "X", "Y", "Z"]
        ]
        muons_for_tres = muons_flat[[c for c in muons_flat.columns if c not in ["RespMuEn"]]]
        df_for_tres = pulses_for_tres.join(muons_for_tres, on=["ev_id", "mu_local_id"], how="left")
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
        #  Минимальный tres среди мюонов, которые дали сигнал в данный хит
        df_for_tres = df_for_tres.group_by(["ev_id", "PulsesChID", "PulsesTime"], maintain_order=True).agg(
            pl.col("t_res")
            .filter(pl.col("t_res").abs() == pl.col("t_res").abs().min())
            .first()  # Select t_res with min absolute value
        )

        # Add t_res to the pulses DataFrame
        df_pulses_flat = df_pulses_flat.join(
            df_for_tres[["ev_id", "PulsesChID", "PulsesTime", "t_res"]],
            on=["ev_id", "PulsesChID", "PulsesTime"],
            how="left",
            suffix="for_mu",
        )

        return df_pulses_flat

    def process(self) -> pl.DataFrame:
        """
        Executes data pipeline: Extract, Transform, Filter.
        Enriches pulses with additional information and filters the data based on the filtering configuration.

        Returns:
            pl.DataFrame: filtered DataFrame.
        """

        events_df, pulses_df, muons_df = self._read_root2df()
        # Enrich muons with local ids
        mu_local_ids = events_df["ev_id", "ResponseMuN"].with_columns(
            mu_local_id=pl.int_ranges(end=pl.col("ResponseMuN"), dtype=pl.Int16)
        )
        muons_df = muons_df.join(mu_local_ids[["ev_id", "mu_local_id"]], on="ev_id")

        filt_cfg = self.cfg.filter_cfg

        # Flatten pulses DataFrame
        df_pulses_flat = pulses_df.explode([c for c in pulses_df.columns if c not in ["ev_id"]])

        # Enrich pulses with new columns: signal flags, cluster and string IDs
        df_pulses_flat = df_pulses_flat.with_columns(
            is_signal=(pl.col("PulsesFlg") != 0),
            cluster_id=(pl.col("PulsesChID") // 288).cast(pl.Int8),
            string_id=(pl.col("PulsesChID") // 36),
            mu_local_id=(pl.col("PulsesFlg") % 1_000_000 - 1).cast(pl.Int16),
        )

        # Apply filtering based on the configuration
        if filt_cfg.only_signal:
            df_pulses_flat = df_pulses_flat.filter(pl.col("is_signal"))
        if (minQ := filt_cfg.min_Q) > 0:
            df_pulses_flat = df_pulses_flat.filter(pl.col("PulsesAmpl") >= minQ)

        # Filter out pulses with bugged large times
        df_pulses_flat = df_pulses_flat.filter(pl.col("PulsesTime") <= filt_cfg.t_threshold)

        # Join OM coordinates to pulses DataFrame
        df_pulses_flat = df_pulses_flat.join(
            self.OM_coords[["PulsesChID", "X", "Y", "Z", "Xrel", "Yrel", "Zrel"]],
            on="PulsesChID",
            how="left",
        )

        # Calculate t_res if configured
        if self.cfg.calc_tres:
            df_pulses_flat = self.calc_tres(muons_df, df_pulses_flat)

        # Sort pulses by event ID, cluster ID, and pulse time
        df_pulses_flat = df_pulses_flat.sort(["ev_id", "cluster_id", "PulsesTime"])

        # Group by event ID and cluster ID, performing calculations on grouped data
        query = [
            pl.col(c_name) for c_name in df_pulses_flat.columns if c_name not in ["ev_id", "cluster_id", "string_id"]
        ] + [
            pl.col("is_signal").alias("num_signal_hits").sum().cast(pl.Int16),  # Calculate number of signal hits
            pl.col("string_id")
            .filter(pl.col("is_signal"))
            .n_unique()
            .cast(pl.Int8)
            .alias("num_signal_strings"),  # Calculate unique signal strings
        ]
        grouped_pulses = df_pulses_flat.group_by(["ev_id", "cluster_id"], maintain_order=True).agg(query)

        # Apply filtering based on hit and string counts
        grouped_pulses = grouped_pulses.filter(
            (pl.col("num_signal_hits") >= filt_cfg.min_hits) & (pl.col("num_signal_strings") >= filt_cfg.min_strings)
        )

        # Center times if configured
        if self.cfg.center_times:
            grouped_pulses = grouped_pulses.with_columns(
                PulsesTime=pl.col("PulsesTime")
                - pl.col("PulsesTime").list.mean().repeat_by(pl.col("PulsesTime").list.len())
            )

        # Join filtered pulses and muons with events DataFrame
        final_df = events_df.join(grouped_pulses, on="ev_id", how="inner").join(
            muons_df, on="ev_id", how="inner", suffix="_for_mu"
        )

        final_df = final_df.with_columns(nu_induced=pl.col("ev_id").str.starts_with("nu"))
        final_df = final_df.with_columns(enough_info=(pl.col("num_signal_hits")>=5) & (pl.col("num_signal_strings")>=2))

        self.filter_koef = final_df.shape[0] / events_df.shape[0]

        return final_df
