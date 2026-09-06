"""Re-running the sig-noise filter on a drawn sample, with the hits modified.

The point is augmentation: take the hits of a `Sample`, change something about
them -- jitter the times, scale the charges -- and see what the filter, and
downstream the classifier, does differently.

One measurement governs how such a comparison has to be set up.  Re-predicting
a sample **without touching anything** does not reproduce the probabilities
stored in the `*_probs_*.h5` file:

    batch 256:  max |dp| 0.126, median 4.5e-5, 0.21% of hits cross the 0.8
                threshold, and only 85.5% of events keep the same n_sn_hits

The stored values were computed with each part's own neighbouring events filling
the batch; a sample has different neighbours, and the batch composition is part
of the filter's answer (doc/sig_noise_batch_size.md).  So the control for an
augmented run is **the same events re-predicted unaugmented in the same call
pattern**, never the stored `prob` column.  Comparing against the stored column
would report a 15% change in `n_sn_hits` before any noise was added.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from inference_v2.shared.h5_hits import HIT_VARS, STRING_DIVISOR, signal_mask

from . import paths
from .sampling import Sample

#: The batch size the stored probabilities were produced at.  Keep it unless the
#: batch size is itself what is being studied: it selects hits, it is not a
#: speed setting.
SN_BATCH = paths.SN_BATCH

_MODEL_CACHE: dict[str, tuple] = {}


def load_sn(device: str = "cpu"):
    """The canonical sig-noise model, loaded once per device.

    `device` is taken literally.  Note that PyTorch orders CUDA devices by speed
    unless told otherwise, so `cuda:1` need not be the card `nvidia-smi` calls 1;
    set ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` **before the first torch import** in the
    kernel if that matters (doc: cuda_device_ordering).
    """
    if device not in _MODEL_CACHE:
        from inference_v2.shared.model_utils import load_sn_model
        _MODEL_CACHE[device] = load_sn_model(device=device)
    return _MODEL_CACHE[device]


def features(sample: Sample, events: pd.Index | np.ndarray | None = None
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(data, ev_starts, rows)` for the sig-noise model, from a sample's hits.

    `data` is (n_hits, 5) float32 in `HIT_VARS` order -- exactly what the
    converter wrote and what the pipeline feeds the model, un-normalised;
    `predict` normalises on the device the way the pipeline does.

    `events` optionally selects a subset by position in `sample.events`; `rows`
    is the positions actually used, so the returned per-event results can be put
    back where they belong.
    """
    if sample.hits.empty:
        raise ValueError("this sample was drawn with keep_hits=False; "
                         "redraw with hits to re-predict on them")
    hits = sample.hits
    if not hits["event"].is_monotonic_increasing:
        raise AssertionError("hits are not ordered by event; `ev_starts` would "
                             "not describe them")
    rows = (np.arange(len(sample.events)) if events is None
            else np.asarray(events, dtype=np.int64))
    if events is not None:
        keep = np.isin(hits["event"].to_numpy(), rows)
        hits = hits[keep]
        # Renumber so the boundaries below are contiguous.
        order = {row: i for i, row in enumerate(rows)}
        event_of_hit = np.array([order[e] for e in hits["event"].to_numpy()])
    else:
        event_of_hit = hits["event"].to_numpy()

    counts = np.bincount(event_of_hit, minlength=len(rows)).astype(np.int64)
    ev_starts = np.concatenate([[0], np.cumsum(counts)])
    data = np.ascontiguousarray(hits[list(HIT_VARS)].to_numpy(np.float32))
    if ev_starts[-1] != len(data):
        raise AssertionError("event boundaries do not cover the hits")
    return data, ev_starts, rows


def predict_sn_probs(data: np.ndarray, ev_starts: np.ndarray, *, device: str = "cpu",
            batch_size: int = SN_BATCH, desc: str = "") -> np.ndarray:
    """Per-hit signal probability, in the pipeline's own call pattern.

    `normalize=True` matches `predict_mc_h5.py` / `predict_exp_h5.py`: raw
    features go in and are normalised on the device.
    """
    import torch
    from gplotnikov_sig_noise_models.k_nsol_labelneq0_da_hs128_k0p0001 \
        .sig_noise_model_v3 import predict_flat

    model, _config, resolved = load_sn(device)
    return predict_flat(model=model, data=data, ev_starts=ev_starts,
                        batch_size=batch_size, device=torch.device(resolved),
                        normalize=True, desc=desc)


def summarise(probs: np.ndarray, ev_starts: np.ndarray, channels: np.ndarray,
              threshold: float = 0.8) -> pd.DataFrame:
    """Per-event signal hit and string counts, the way the pipeline counts them.

    Same convention as everywhere else: strictly greater, in float32, and a
    string is `channel // 36`.
    """
    mask = signal_mask(probs, threshold)
    starts = np.asarray(ev_starts, dtype=np.intp)
    n_hits = np.add.reduceat(mask.astype(np.int64), starts[:-1])
    n_hits[np.diff(starts) == 0] = 0

    event_of_hit = np.repeat(np.arange(len(starts) - 1), np.diff(starts))
    keyed = np.unique(event_of_hit[mask] * (STRING_DIVISOR * 1000)
                      + (np.asarray(channels)[mask] // STRING_DIVISOR))
    events = keyed // (STRING_DIVISOR * 1000)
    n_strings = np.bincount(events, minlength=len(starts) - 1)
    return pd.DataFrame({"n_sn_hits": n_hits,
                         "n_sn_strings": n_strings[:len(n_hits)],
                         "prob_mean": np.add.reduceat(probs, starts[:-1])
                         / np.maximum(np.diff(starts), 1)})


def compare(sample: Sample, augment: Callable[[np.ndarray, np.random.Generator],
                                              np.ndarray],
            *, events=None, device: str = "cpu", batch_size: int = SN_BATCH,
            threshold: float = 0.8, seed: int = 0) -> pd.DataFrame:
    """Re-predict a sample unaugmented and augmented, and put the two side by side.

    `augment(data, rng) -> data` receives a **copy** of the (n_hits, 5) array in
    `HIT_VARS` order and returns the modified one, e.g.

        def jitter(data, rng):
            data[:, 1] += rng.normal(0.0, 30.0, len(data)).astype(np.float32)
            return data

    Both runs go through the same call with the same batch composition, so the
    only difference between the columns is the augmentation.  The stored `prob`
    is deliberately not used as the baseline -- see the module docstring.
    """
    data, ev_starts, rows = features(sample, events)
    channels = (sample.hits["channel"].to_numpy() if events is None else
                sample.hits[np.isin(sample.hits["event"].to_numpy(), rows)]
                ["channel"].to_numpy())

    # base = predict_sn_probs(data, ev_starts, device=device, batch_size=batch_size,
    #                desc="baseline")
    changed = augment(data.copy(), np.random.default_rng(seed))
    if changed.shape != data.shape or changed.dtype != np.float32:
        raise ValueError("augment must return a float32 array of the same shape")
    after = predict_sn_probs(changed, ev_starts, device=device, batch_size=batch_size,
                    desc="augmented")
    return after

    frame = summarise(base, ev_starts, channels, threshold).add_suffix("_base")
    frame = frame.join(
        summarise(after, ev_starts, channels, threshold).add_suffix("_aug"))
    frame.insert(0, "event_row", rows)
    frame["stored_n_sn_hits"] = sample.events["db_n_sn_hits"].to_numpy()[rows]
    return frame
