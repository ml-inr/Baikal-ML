import polars as pl
import numpy as np
import typing as tp


def get_mu_direction(
    theta: pl.Expr, phi: pl.Expr
) -> tp.Tuple[pl.Expr, pl.Expr, pl.Expr, pl.Expr]:
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

def get_target_vec_direction(
    X: pl.Expr, Y: pl.Expr, Z: pl.Expr, X_mu: pl.Expr, Y_mu: pl.Expr, Z_mu: pl.Expr
) -> tp.Tuple[pl.Expr, pl.Expr, pl.Expr, pl.Expr]:
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