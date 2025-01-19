import polars as pl
import typing as tp
import logging
from datetime import datetime

try:
    from ..constants import Constants as Cnst
    from .tres_utils import calculate_tres
except ImportError:
    import sys
    PROJECT_PATH = "/home/albert/Baikal-ML/"
    sys.path.append(f"{PROJECT_PATH}")
    from data.constants import Constants as Cnst
    from data.processor.tres_utils import calculate_tres

# Utility function to validate DataFrame columns
def validate_dataframe(df: pl.DataFrame, required_columns: list[str]):
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

def enrich_pulses(df_pulses: pl.DataFrame, has_signal_flg: bool) -> pl.DataFrame:
    """Enrich pulses with new columns such as cluster ID, string ID, and signal flags."""
    pl_expression = [
        (pl.col("PulsesChID") // Cnst.CHANNEL_DIVISOR).cast(pl.Int8).alias("cluster_id"),
        (pl.col("PulsesChID") // Cnst.STRING_DIVISOR).alias("string_id")
    ]
    if has_signal_flg:
        assert "PulsesFlg" in df_pulses.columns
        pl_expression += [
            (pl.col("PulsesFlg") != 0).alias("is_signal"),
            (pl.col("PulsesFlg") % 1_000_000 - 1).cast(pl.Int16).alias("mu_local_id")
        ]
    return df_pulses.with_columns(pl_expression)

def filter_pulses(
    df_pulses: pl.DataFrame,
    only_signal: bool,
    has_signal_flg: bool,
    min_Q: float,
    t_threshold: float
) -> pl.DataFrame:
    """Filter pulses based on configuration parameters."""
    validate_dataframe(df_pulses, ["PulsesAmpl", "PulsesTime"])
    if only_signal and has_signal_flg:
        logging.debug("Filtering only signal pulses")
        df_pulses = df_pulses.filter(pl.col("is_signal"))
    if min_Q > 0:
        logging.debug(f"Filtering pulses with amplitude >= {min_Q}")
        df_pulses = df_pulses.filter(pl.col("PulsesAmpl") >= min_Q)
    return df_pulses.filter(pl.col("PulsesTime") <= t_threshold)

def calculate_relative_coords(df_coords: pl.DataFrame) -> pl.DataFrame:
    """Calculate relative coordinates based on cluster centers."""
    validate_dataframe(df_coords, ["X", "Y", "Z", "ev_id", "cluster_id"])
    logging.debug("Calculating relative coordinates based on cluster centers")
    cl_centers = (
        df_coords[["X", "Y", "Z", "ev_id", "cluster_id"]]
        .group_by(["ev_id", "cluster_id"])
        .mean()
        .rename({"X": "Xc", "Y": "Yc", "Z": "Zc"})
    )
    return df_coords.join(cl_centers, on=["ev_id", "cluster_id"], how="left").with_columns([
        (pl.col("X") - pl.col("Xc")).alias("Xrel"),
        (pl.col("Y") - pl.col("Yc")).alias("Yrel"),
        (pl.col("Z") - pl.col("Zc")).alias("Zrel"),
    ])

def aggregate_clusters(
    df_pulses: pl.DataFrame,
    has_signal_flg: bool,
    min_sig_hits: int,
    min_sig_strings: int
) -> pl.DataFrame:
    """Aggregate clusters and apply filtering based on signal hits and strings."""
    logging.debug("Aggregating clusters and applying thresholds")
    if has_signal_flg:
        validate_dataframe(df_pulses, ["is_signal", "string_id"])
        pl_expression = [
            pl.col("is_signal").sum().cast(pl.Int16).alias("num_signal_hits"),
            pl.col("string_id").filter(pl.col("is_signal")).n_unique().cast(pl.Int8).alias("num_signal_strings"),
        ]
        pulses_agg_info = df_pulses.group_by(["ev_id", "cluster_id"]).agg(pl_expression)
        return pulses_agg_info.filter(
            (pl.col("num_signal_hits") >= min_sig_hits) &
            (pl.col("num_signal_strings") >= min_sig_strings)
        )
    return df_pulses[["ev_id", "cluster_id"]].group_by(["ev_id", "cluster_id"]).agg()

def process_data(
    df_pulses: pl.DataFrame,
    df_events: pl.DataFrame,
    df_coords: pl.DataFrame,
    df_muons: tp.Optional[pl.DataFrame],
    has_signal_flg: bool = True,
    only_signal: bool = True,
    min_sig_hits: int = 5,
    min_sig_strings: int = 2,
    min_Q: float = 0,
    center_times: bool = True,
    relative_coords: bool = True,
    to_calculate_tres: bool = False,
    t_threshold: float = 1e5,
    same_coordinates: bool = False # whether coords are same for each event or not
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Process data by enriching, filtering, and aggregating based on configurations."""
    logging.debug("Starting process_data function")

    # Enrich pulses
    logging.debug("Enriching pulses")
    df_pulses = enrich_pulses(df_pulses, has_signal_flg)

    # Filter pulses
    logging.debug("Filtering pulses")
    df_pulses = filter_pulses(df_pulses, only_signal, has_signal_flg, min_Q, t_threshold)

    # Join coordinates to pulses
    if same_coordinates:
        columns_from_coords = ["X", "Y", "Z", "PulsesChID"]
        to_join_on = ["PulsesChID"]
        logging.debug("Retrieving fixed coordinates")
        df_coords = df_coords.filter(pl.col("ev_id")==pl.col("ev_id").first())
    else:
        columns_from_coords = ["X", "Y", "Z", "PulsesChID", "ev_id"]
        to_join_on = ["PulsesChID", "ev_id"]
    # Calculate relative coordinates
    if relative_coords:
        logging.debug("Calculating relative coordinates")
        df_coords = calculate_relative_coords(df_coords)
        columns_from_coords += ["Xrel", "Yrel", "Zrel"]
    logging.debug("Joining coordinates to pulses")
    df_pulses = df_pulses.join(
        df_coords[columns_from_coords],
        on=to_join_on,
        how="left"
    )

    # Aggregate clusters
    logging.debug("Aggregating clusters")
    pulses_agg_info = aggregate_clusters(df_pulses, has_signal_flg, min_sig_hits, min_sig_strings)
    df_pulses = df_pulses.join(pulses_agg_info, on=["ev_id", "cluster_id"], how="inner")

    # Join aggregated info to events
    logging.debug("Joining aggregated info to events")
    df_events = df_events.join(
        pulses_agg_info.group_by(["ev_id"]).agg([
            pl.col(name) for name in pulses_agg_info.columns if name != "ev_id"
        ]),
        on=["ev_id"],
        how="inner"
    )

    # Calculate t_res if configured
    if to_calculate_tres:
        if df_muons is None or not has_signal_flg:
            logging.warning("Unable to calculate 'tres' on provided data!")
        else:
            logging.debug("Calculating t_res")
            df_with_tres = calculate_tres(df_muons, df_pulses)
            df_pulses = df_pulses.join(
                df_with_tres,
                on=["ev_id", "PulsesChID", "PulsesTime"],
                how="left",
                suffix="for_mu"
            )

    # Center times if configured
    if center_times:
        logging.debug("Centering pulse times")
        time_centers = (
            df_pulses[["PulsesTime", "ev_id", "cluster_id"]]
            .group_by(["ev_id", "cluster_id"])
            .mean()
            .rename({"PulsesTime": "Tc"})
        )
        df_pulses = df_pulses.join(time_centers, on=["ev_id", "cluster_id"]).with_columns(
            (pl.col("PulsesTime") - pl.col("Tc")).cast(pl.Float32).alias("PulsesTime")
        ).drop("Tc")

    # Filter events based on pulses and muons
    logging.debug("Filtering events")
    events_to_take = df_events[["ev_id"]].join(df_pulses[["ev_id"]].unique(), on="ev_id", how="inner")
    if has_signal_flg and df_muons is not None:
        events_to_take = events_to_take.join(df_muons[["ev_id"]].unique(), on="ev_id", how="inner")
    df_events = df_events.join(events_to_take, on=["ev_id"], how="inner").explode("cluster_id")

    logging.debug("Transformation complete")
    return df_pulses, df_events, df_muons


# Test
if __name__=="__main__":
    import sys
    PROJECT_PATH = "../"
    sys.path.append(f"{PROJECT_PATH}")

    from data.root_extractor.main import RootFileReader
    from data.root_extractor.internal_root_paths import MCRootPaths, ExpRootPaths

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)

    path = "/net/62/home3/ivkhar/Baikal/data/initial_data/MC_2020/muatm/root/all/1005.root"
    start = 0
    stop = None
    
    with RootFileReader(path, internal_root_paths = MCRootPaths()) as rr:
        pulses = rr.read_pulses_as_df(start, stop)
        muons = rr.read_muons_as_df(start, stop)
        events = rr.read_events_as_df(start, stop)
        coords = rr.read_OM_coords(start, stop)
        
    proc_cfg = dict(
        has_signal_flg = True,
        only_signal = True,  # Whether to filter only signal hits
        min_sig_hits = 8,  # Minimum number of signal hits per cluster to be kept
        min_sig_strings = 2,  # Minimum number of unique signal strings in a cluster
        min_Q = 0,  # Minimum pulse amplitude threshold
        center_times = True,  # Whether to center the event times
        relative_coords = True, # Whether to add coordinates relatively to the clusters centers
        to_calculate_tres = False, # Whether to calculate tres
        t_threshold = 1e5,  # Maximum time threshold for pulse filtering
        fixed_coordinates = True
    )
    new_pulses, new_events, new_muons = process_data(pulses, events, coords, muons, **proc_cfg)
    print(new_events.head())