"""Distance from each event to the Monte Carlo manifolds, in the classifier's own embedding.

Three distances, kept separate on purpose: to the whole MC reference, to the neutrino part
of it, and to the muon part. Conflating the first two has already produced a wrong statement
in the paper, so they never share a column here.

The reference is drawn only from parts marked `reference` in `splits`, so no event is ever
compared against a manifold built from its own part, and never from events the model trained
on. Class proportions follow `reference_class_ratio` in the config.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import duckdb
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from inference_v2.nu_classifier.analysis.exp_excess.build_splits import load_config

logger = logging.getLogger(__name__)

NEIGHBOURS = 20

DISTANCE_TABLE = """
CREATE TABLE IF NOT EXISTS distances (
    event_fk       BIGINT PRIMARY KEY,
    distance_to_mc DOUBLE,
    distance_to_nu DOUBLE,
    distance_to_mu DOUBLE
)
"""


def load_embeddings(connection: duckdb.DuckDBPyConnection, where: str,
                    limit: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    limit_clause = f"LIMIT {limit}" if limit else ""
    rows = connection.execute(f"""
        SELECT e.event_fk, e.embedding
        FROM embeddings e JOIN predictions p USING (event_fk) JOIN splits s USING (event_fk)
        WHERE {where}
        {limit_clause}
    """).fetchall()
    if not rows:
        return np.empty(0, dtype=np.int64), np.empty((0, 128), dtype=np.float32)
    event_fks = np.array([row[0] for row in rows], dtype=np.int64)
    vectors = np.asarray([row[1] for row in rows], dtype=np.float32)
    return event_fks, vectors


def mean_distance_to_nearest(query: torch.Tensor, reference: torch.Tensor,
                             neighbours: int, chunk: int = 4096) -> np.ndarray:
    """Mean distance to the k nearest reference points, computed in chunks on the device."""
    out = torch.empty(query.shape[0], device=query.device)
    for start in range(0, query.shape[0], chunk):
        block = query[start:start + chunk]
        distances = torch.cdist(block, reference)
        nearest = distances.topk(min(neighbours, reference.shape[0]),
                                 largest=False).values
        out[start:start + chunk] = nearest.mean(dim=1)
    return out.cpu().numpy()


def build_reference_set(connection: duckdb.DuckDBPyConnection, config: dict,
                        selection: str) -> dict[str, np.ndarray]:
    """Reference embeddings per class, in the configured proportions."""
    ratio = config["reference_class_ratio"]
    available = {}
    for data_class in ratio:
        _, vectors = load_embeddings(
            connection, f"{selection} AND s.role = 'reference' "
                        f"AND s.data_class = '{data_class}'")
        available[data_class] = vectors
        logger.info(f"  reference pool {data_class}: {len(vectors):,}")

    units = min(len(available[c]) / ratio[c] for c in ratio)
    generator = np.random.default_rng(config["random_seed"])
    chosen = {}
    for data_class, share in ratio.items():
        wanted = int(units * share)
        pool = available[data_class]
        index = generator.choice(len(pool), size=wanted, replace=False)
        chosen[data_class] = pool[index]
        logger.info(f"  reference used {data_class}: {wanted:,}")
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--convergence", action="store_true",
                        help="report how distances move with reference size, then stop")
    arguments = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])

    config = load_config(Path(arguments.config))
    cuts = config["event_selection"]
    selection = (f"p.n_sn_hits >= {cuts['min_sig_hits']} "
                 f"AND p.n_sn_strings >= {cuts['min_sig_strings']} "
                 f"AND NOT s.used_for_labels")

    mc_path = str(PROJECT_ROOT / config["predictions"]["mc"])
    connection = duckdb.connect(mc_path, read_only=True)

    reference = build_reference_set(connection, config, selection)
    device = torch.device(arguments.device)
    neutrino = torch.tensor(np.vstack([reference["nuatm_2020"], reference["nue2_2020"]]),
                            device=device)
    muon = torch.tensor(reference["muatm_2020"], device=device)
    everything = torch.cat([neutrino, muon])
    logger.info(f"  reference: {everything.shape[0]:,} events "
                f"({neutrino.shape[0]:,} nu, {muon.shape[0]:,} mu)")

    if arguments.convergence:
        query_fks, query_vectors = load_embeddings(
            connection, f"{selection} AND s.data_class='muatm_2020' AND p.score > 0.8")
        query = torch.tensor(query_vectors, device=device)
        logger.info(f"  convergence probe on {len(query_fks):,} high-score muons")
        for fraction in [0.05, 0.1, 0.25, 0.5, 1.0]:
            size = max(int(everything.shape[0] * fraction), NEIGHBOURS + 1)
            subset = everything[torch.randperm(everything.shape[0], device=device)[:size]]
            distance = mean_distance_to_nearest(query, subset, NEIGHBOURS)
            logger.info(f"    {fraction:>5.0%} of reference ({size:>8,}): "
                        f"median distance {np.median(distance):.4f}")
        return

    logger.info("  writing distances for high-score events and a control sample")
    connection.close()
    for source in ["mc", "exp"]:
        path = str(PROJECT_ROOT / config["predictions"][source])
        write = duckdb.connect(path)
        write.execute(DISTANCE_TABLE)
        local_selection = selection if source == "mc" else (
            f"p.n_sn_hits >= {cuts['min_sig_hits']} "
            f"AND p.n_sn_strings >= {cuts['min_sig_strings']} AND NOT s.excluded")
        for label, where in [("high score", f"{local_selection} AND p.score > 0.8"),
                             ("control", f"{local_selection} AND p.score <= 0.8")]:
            limit = None if label == "high score" else 200_000
            fks, vectors = load_embeddings(write, where, limit)
            if not len(fks):
                continue
            query = torch.tensor(vectors, device=device)
            frame = {
                "event_fk": fks,
                "distance_to_mc": mean_distance_to_nearest(query, everything, NEIGHBOURS),
                "distance_to_nu": mean_distance_to_nearest(query, neutrino, NEIGHBOURS),
                "distance_to_mu": mean_distance_to_nearest(query, muon, NEIGHBOURS),
            }
            import pandas as pd
            write.register("new_distances", pd.DataFrame(frame))
            write.execute("INSERT OR IGNORE INTO distances SELECT * FROM new_distances")
            write.unregister("new_distances")
            logger.info(f"  {source} {label}: {len(fks):,} events")
        write.close()


if __name__ == "__main__":
    main()
