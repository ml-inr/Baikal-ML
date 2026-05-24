"""Shared model definition and inference utilities for signal/noise prediction.

v3: Uses model_simplified.py (encoder + main_head only).
    Domain-adaptation weights in the checkpoint are ignored (strict=False).
    importlib machinery removed — model_simplified.py has no relative imports.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import yaml
from tqdm import tqdm


# ── Defaults ───────────────────────────────────────────────────────────

MODEL_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = MODEL_DIR / "best_mc_2020.ckpt"
DEFAULT_CONFIG = MODEL_DIR / "train_config_mc_2020.yaml"

# Normalization params the model was trained with
MODEL_NORM_MEAN = np.array(
    [1.2946190e+00, 1.9616836e-08, 6.4279658e-01, 2.4755356e-01, 3.0921354e+01],
    dtype=np.float32,
)
MODEL_NORM_STD = np.array(
    [3.6713488e+00, 1.3863461e+03, 4.0100330e+01, 3.9045639e+01, 1.5460611e+02],
    dtype=np.float32,
)

# Default: ~2M hits * 5 floats * 4 bytes ≈ 40 MB per GPU chunk
DEFAULT_MAX_GPU_HITS = 20_000_000


# ── Import model_simplified.py (no relative imports) ───────────────────

def _load_simplified_module():
    key = "_snm_simplified"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, MODEL_DIR / "model_simplified.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return sys.modules[key]


SigNoiseModel = _load_simplified_module().SigNoiseModel


# ── Device helper ──────────────────────────────────────────────────────

def get_device(device: str = "auto") -> torch.device:
    """Resolve device string to torch.device. 'auto' picks cuda if available."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


# ── Model loader ───────────────────────────────────────────────────────

def load_model(
    checkpoint_path: str = str(DEFAULT_CHECKPOINT),
    config_path: str = str(DEFAULT_CONFIG),
    device: str = "auto",
) -> tuple:
    """Load SigNoiseModel from checkpoint.  DA weights are ignored (strict=False).

    Returns:
        (model, config, device)
    """
    dev = get_device(device)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    params = config["model"]["params"]
    model = SigNoiseModel(**params).to(dev)

    state_dict = torch.load(checkpoint_path, map_location=dev, weights_only=True)
    missing, _ = model.load_state_dict(state_dict, strict=False)
    if missing:
       raise RuntimeError(f"Missing keys in checkpoint: {missing}")

    model.eval()
    return model, config, dev


# ── Inference ──────────────────────────────────────────────────────────

def _collate_on_device(
    chunk: torch.Tensor,
    ev_starts: np.ndarray,
    batch_start: int,
    batch_end: int,
    chunk_offset: int,
) -> tuple:
    """Build padded batch from a GPU chunk using pad_sequence + vectorized mask."""
    starts = ev_starts[batch_start: batch_end] - chunk_offset
    ends = ev_starts[batch_start + 1: batch_end + 1] - chunk_offset
    lengths = torch.tensor(ends - starts, dtype=torch.long, device=chunk.device)

    sequences = [chunk[s:e] for s, e in zip(starts, ends)]
    x = rnn_utils.pad_sequence(sequences, batch_first=True)  # (B, max_len, 5)
    mask = torch.arange(x.size(1), device=chunk.device).unsqueeze(0) < lengths.unsqueeze(1)

    return x, mask, lengths


def predict_flat(
    model: nn.Module,
    data,
    ev_starts: np.ndarray,
    batch_size: int,
    device: torch.device,
    normalize: bool = False,
    desc: str = "",
    max_gpu_hits: int = DEFAULT_MAX_GPU_HITS,
) -> np.ndarray:
    """Run inference and return flat 1D array of per-hit sig_prob.

    Data is loaded to the device in chunks of up to max_gpu_hits hits,
    normalized on-device if needed, then batched via pad_sequence.

    Args:
        model: Loaded TransformerEncoder.
        data: (n_hits, 5) array or h5py dataset.
        ev_starts: (n_events+1,) CSR-style event boundaries.
        batch_size: Events per batch.
        device: torch.device to run inference on.
        normalize: If True, normalize raw data with MODEL_NORM_MEAN/STD.
        desc: tqdm description string (events count appended automatically).
        max_gpu_hits: Max hits to load onto device at once.

    Returns:
        1D array of float32 sig_prob, length = ev_starts[-1] - ev_starts[0].
    """
    num_events = len(ev_starts) - 1
    total_hits = int(ev_starts[-1] - ev_starts[0])
    sig_prob = np.empty(total_hits, dtype=np.float32)
    hit_offset = 0

    # Normalization tensors on device (created once)
    if normalize:
        norm_mean = torch.tensor(MODEL_NORM_MEAN, device=device)
        norm_std = torch.tensor(MODEL_NORM_STD, device=device)

    pbar_desc = f"{desc} ({num_events} ev)" if desc else None
    pbar = tqdm(total=num_events, desc=pbar_desc, disable=not desc)

    # Process events in GPU-sized chunks
    ev_cursor = 0
    while ev_cursor < num_events:
        # Determine chunk boundary: fit up to max_gpu_hits
        chunk_ev_end = ev_cursor
        chunk_hit_start = int(ev_starts[ev_cursor])
        while chunk_ev_end < num_events:
            next_hits = int(ev_starts[chunk_ev_end + 1]) - chunk_hit_start
            if next_hits > max_gpu_hits and chunk_ev_end > ev_cursor:
                break
            chunk_ev_end += 1
            if next_hits >= max_gpu_hits:
                break

        chunk_hit_end = int(ev_starts[chunk_ev_end])

        # Load chunk to device in one transfer
        chunk_np = data[chunk_hit_start:chunk_hit_end]
        if not isinstance(chunk_np, np.ndarray):
            chunk_np = np.array(chunk_np)
        chunk = torch.tensor(chunk_np, dtype=torch.float32, device=device)

        if normalize:
            chunk = (chunk - norm_mean) / norm_std

        # Process batches within this chunk
        with torch.no_grad():
            for b_start in range(ev_cursor, chunk_ev_end, batch_size):
                b_end = min(b_start + batch_size, chunk_ev_end)

                x, mask, lengths = _collate_on_device(
                    chunk, ev_starts, b_start, b_end, chunk_hit_start,
                )

                output = model(x, mask)          # (B, L, 2)
                probs = torch.sigmoid(output[:, :, 1])

                # Extract valid hits per event
                probs_np = probs.cpu().numpy()
                lengths_np = lengths.cpu().numpy()
                for i in range(b_end - b_start):
                    ln = int(lengths_np[i])
                    sig_prob[hit_offset: hit_offset + ln] = probs_np[i, :ln]
                    hit_offset += ln

                pbar.update(b_end - b_start)

        del chunk
        ev_cursor = chunk_ev_end

    pbar.close()
    assert hit_offset == total_hits, f"Hit count mismatch: {hit_offset} vs {total_hits}"
    return sig_prob
