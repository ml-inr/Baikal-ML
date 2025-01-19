import polars as pl
import typing as tp

from data.constants import Constants as Cnst
from .geometry_utils import get_target_vec_direction, get_mu_direction


def _get_t_res(
    target_dir: tp.Tuple[pl.Expr, pl.Expr, pl.Expr, pl.Expr],
    mu_dir: tp.Tuple[pl.Expr, pl.Expr, pl.Expr, pl.Expr],
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
    cosAlpha = (dx * dx_mu + dy * dy_mu + dz * dz_mu) / (
        1e-9 + target_dist * dir_norm_mu
    )
    sinAlpha = (1 - cosAlpha**2).sqrt()

    dMuon = target_dist * (cosAlpha - sinAlpha / Cnst.TAN_C)
    tMuon = 1e9 * dMuon / Cnst.C_PART
    dLight = target_dist * sinAlpha / Cnst.SIN_C
    tLight = 1e9 * dLight / Cnst.C_LIGHT

    t_exp = tMuon + tLight + mus_delay
    t_res_all = t_exp - t_detected

    return t_res_all

def calculate_tres(muons_df: pl.DataFrame, df_pulses_flat: pl.DataFrame) -> pl.DataFrame:
    # Prepare DataFrames for t_res calculation
    pulses_for_tres = df_pulses_flat.filter(pl.col("is_signal"))[
        ["ev_id", "mu_local_id", "PulsesChID", "PulsesTime", "X", "Y", "Z"]
    ]
    muons_for_tres = muons_df[
        [c for c in muons_df.columns if c not in ["RespMuEn"]]
    ]
    df_for_tres = pulses_for_tres.join(
        muons_for_tres, on=["ev_id", "mu_local_id"], how="left"
    )
    df_for_tres = df_for_tres.with_columns(
        t_res=_get_t_res(
            get_target_vec_direction(
                pl.col("X"),
                pl.col("Y"),
                pl.col("Z"),
                pl.col("RespMuTrackX"),
                pl.col("RespMuTrackY"),
                pl.col("RespMuTrackZ"),
            ),
            get_mu_direction(pl.col("RespMuTheta"), pl.col("RespMuPhi")),
            pl.col("RespMuDelay"),
            pl.col("PulsesTime"),
        )
    )
    #  Минимальный tres среди мюонов, которые дали сигнал в данный хит
    df_for_tres = df_for_tres.group_by(
        ["ev_id", "PulsesChID", "PulsesTime"], maintain_order=True
    ).agg(
        pl.col("t_res")
        .filter(pl.col("t_res").abs() == pl.col("t_res").abs().min())
        .first()  # Select t_res with min absolute value
    )

    return df_for_tres[["ev_id", "PulsesChID", "PulsesTime", "t_res"]]
