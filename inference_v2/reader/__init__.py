"""Reading Baikal sources and model predictions, without materialising anything.

    from inference_v2 import reader

    reader.checkpoints()                # which scored checkpoints exist
    r = reader.open("exp_reco",
                    checkpoint="260816_2250_..._FIXED_sn256@best_da_model")

    ev = r.events(where=[reader.H8S3])                    # no hit is read
    for chunk in r.stream(where=[reader.H8S3], budget_mb=512):
        chunk.events, chunk.hits, chunk.parts, chunk.meta

Design in one paragraph: iteration follows **parts**, the structural unit of a
source, because a part is one detector run or one MC file with its own `ev_starts`
and reconstruction.  Rows come from a single streaming DuckDB query ordered by
`event_fk`, which already groups them by part -- `event_fk` is contiguous inside a
part in all four sources, verified on all 47,521 of them.  Hits are read as merged
runs of wanted rows -- maximal groups separated by less than one gzip chunk (72,461
values), since a chunk cannot be read in part.  Nothing is written to disk.

Nothing here reads the `splits` table, and nothing depends on it existing.
"""
from . import parts, predicates, query, registry, schema, spec, stream, training
from .predicates import (
    H8S3, NOT_DA_TARGET, NOT_EXCLUDED, NOT_TRAINED, NOT_TRAINED_EXACT, NotTrained,
    Sql,
)
from .registry import Checkpoint, checkpoints, open_checkpoint, unresolved
from .schema import (
    EXP_RECO_COLUMNS, HIT_VARS, MC_RECO_COLUMNS, PRIME_PRTY_COLUMNS, SN_THRESHOLD,
    STRING_DIVISOR, check_against_converter_config, signal_mask,
)
from .spec import SOURCES, describe
from .stream import Chunk, Progress, Reader


def open(source: str, checkpoint, **kwargs) -> Reader:
    """A reader for one source scored by one checkpoint.

    `checkpoint` names a directory under `preds/`, e.g.
    `260816_2250_..._FIXED_sn256@best_da_model`.  It is deliberately not called
    `run`: in this project a *run* is a detector run, and it is a column of the
    tables this reader returns.
    """
    return Reader(source, checkpoint, **kwargs)


def sources():
    """Every known source, described with the evidence behind its exclusions."""
    return "\n\n".join(describe(name) for name in SOURCES)


__all__ = [
    "open", "sources", "checkpoints", "unresolved", "open_checkpoint",
    "Reader", "Chunk", "Progress", "Checkpoint",
    "H8S3", "NOT_TRAINED", "NOT_TRAINED_EXACT", "NOT_DA_TARGET", "NOT_EXCLUDED",
    "NotTrained", "Sql",
    "SOURCES", "describe",
    "HIT_VARS", "PRIME_PRTY_COLUMNS", "STRING_DIVISOR", "SN_THRESHOLD",
    "signal_mask", "EXP_RECO_COLUMNS", "MC_RECO_COLUMNS",
    "check_against_converter_config",
    "schema", "spec", "registry", "training", "predicates", "query", "parts",
    "stream",
]
