"""
Prefilter Dataset — Lazy-loading dataset for neutrino prefilter classification.

Uses Parquet event catalogs for indexing and reads hit data from HDF5 on-the-fly,
avoiding loading all hits into memory at init time.

Features:
- Catalog-based event index: n_signal_hits, labels, etc. from Parquet metadata
- Lazy HDF5 reading: direct slice per event (no full-part caching)
- Soft labels: neutrino label = min(1.0, n_signal_hits / h_saturate)
- Class balancing using catalog metadata (no data loading needed)
- Memory-efficient random sampling across all parts via Polars lazy scan
- Multi-worker support: each worker opens its own HDF5 handle
- Compatible with existing collate/trainer API

Classification:
- Muon events: always label 0.0
- Neutrino events (soft='linear'): label = min(1.0, n_signal_hits / h_saturate)
- Neutrino events (soft='sigmoid'): label = sigmoid(k * (n - h_saturate/2))
- Neutrino events (soft='hard'): label = 1.0 if n_signal_hits >= h_min else 0.0
"""

import logging
import math
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

import h5py
import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class EventRecord(NamedTuple):
    """Single event's metadata from catalog."""
    particle_type: str
    h5_part_key: str       # "part_42" for MC, "part_s2020_c01_r0027" for exp
    hit_start_idx: int
    hit_end_idx: int
    n_hits: int
    n_signal_hits: int     # 0 for exp data
    label: float           # soft label (0.0–1.0)
    event_id: str


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PrefilterDataset(Dataset):
    """
    Lazy-loading dataset for neutrino prefilter classification.

    Uses Parquet catalogs for event indexing and reads hits from HDF5
    on-the-fly with an LRU part cache.

    Args:
        h5_path: Path to HDF5 data file.
        catalog_dir: Directory containing Parquet catalog files.
        particle_types: Particle types to load
            (e.g. ['muatm_2020', 'nue2_2020']).
        neutrino_types: Which particle types are neutrinos (positive class).
        events_per_particle: Max events per particle type.
        h_min: Signal hit threshold for hard-label mode.
        max_hits: Truncate events to this many hits.
        soft_label_config: Dict with keys 'mode' ('hard'|'linear'),
            and for 'linear': 'h_saturate' (int).
        balance_classes: If True, balance signal/background counts.
        shuffle_events: Shuffle the event index after building.
        device: Device for tensors returned by collate.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        h5_path: Union[str, Path],
        catalog_dir: Union[str, Path],
        particle_types: List[str],
        neutrino_types: List[str],
        events_per_particle: Dict[str, int],
        h_min: int = 5,
        max_hits: int = 500,
        soft_label_config: Optional[Dict[str, Any]] = None,
        balance_classes: bool = False,
        shuffle_events: bool = True,
        device: str = "cpu",
        seed: Optional[int] = None,
    ):
        self.h5_path = Path(h5_path)
        self.catalog_dir = Path(catalog_dir)
        self.particle_types = particle_types
        self.neutrino_types = neutrino_types
        self.events_per_particle = events_per_particle
        self.h_min = h_min
        self.max_hits = max_hits
        self.soft_label_config = soft_label_config or {"mode": "hard"}
        self.device = device
        self.seed = seed

        if not self.h5_path.exists():
            raise FileNotFoundError(
                f"HDF5 file not found: {self.h5_path}"
            )

        self.rng = np.random.RandomState(seed)
        self.torch_rng = torch.Generator()
        if seed is not None:
            self.torch_rng.manual_seed(seed)

        # Lazy HDF5 handle (opened on first read, re-opened per worker)
        self._h5_file: Optional[h5py.File] = None

        # Build event index from catalogs
        self._event_index = self._build_event_index()

        if balance_classes:
            self._event_index = self._balance_index(
                self._event_index
            )

        if shuffle_events:
            self.rng.shuffle(self._event_index)

        self._calculate_stats()
        self._log_summary()

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def _build_event_index(self) -> List[EventRecord]:
        """Load Parquet catalogs, sample, compute labels."""
        import time

        logger.info(
            f"Building event index for {len(self.particle_types)} "
            f"particle type(s)..."
        )
        all_records: List[EventRecord] = []

        for pt in self.particle_types:
            n_requested = self.events_per_particle.get(pt, 0)
            if n_requested <= 0:
                continue

            is_neutrino = pt in self.neutrino_types
            catalog_path = self._find_catalog(pt)
            if catalog_path is None:
                logger.warning(
                    f"No catalog for '{pt}' in {self.catalog_dir}"
                )
                continue

            logger.info(
                f"Loading {pt}: requesting {n_requested:,} events "
                f"from {catalog_path.name}..."
            )
            t0 = time.time()
            records = self._load_particle_index(
                catalog_path, pt, is_neutrino, n_requested
            )
            elapsed = time.time() - t0
            all_records.extend(records)
            n_sig = sum(1 for r in records if r.label >= 0.5)
            logger.info(
                f"  → {len(records):,} events indexed in {elapsed:.1f}s "
                f"(signal={n_sig:,}, bg={len(records) - n_sig:,})"
            )

        if not all_records:
            raise ValueError(
                f"No events for particle types {self.particle_types}"
            )
        return all_records

    def _find_catalog(self, particle_type: str) -> Optional[Path]:
        """Locate the Parquet catalog file for a particle type."""
        p = self.catalog_dir / f"{particle_type}.parquet"
        if p.exists():
            return p
        p = self.catalog_dir / "exp.parquet"
        if particle_type == "exp" and p.exists():
            return p
        return None

    def _load_particle_index(
        self,
        catalog_path: Path,
        particle_type: str,
        is_neutrino: bool,
        n_requested: int,
    ) -> List[EventRecord]:
        """Read catalog, sample events uniformly, build records."""
        is_exp = particle_type == "exp"

        if is_exp:
            cols = [
                "event_id", "h5_part",
                "hit_start_idx", "hit_end_idx", "n_hits",
            ]
        else:
            cols = [
                "event_id", "h5_part_num",
                "hit_start_idx", "hit_end_idx",
                "n_hits", "n_signal_hits",
            ]

        lf = pl.scan_parquet(catalog_path).select(cols)

        # Get total row count (metadata-only operation)
        total_rows = lf.select(pl.len()).collect().item()
        n_to_load = min(n_requested, total_rows)

        if n_to_load < total_rows:
            # Uniform random sampling across all parts:
            # generate random row indices, filter via lazy scan
            logger.info(
                f"  Sampling {n_to_load:,} / {total_rows:,} rows "
                f"(scanning parquet, may take a while)..."
            )
            sample_idx = np.sort(
                self.rng.choice(
                    total_rows, size=n_to_load, replace=False
                )
            )
            df = (
                lf.with_row_index("_idx")
                .filter(pl.col("_idx").is_in(sample_idx))
                .drop("_idx")
                .collect()
            )
        else:
            logger.info(
                f"  Loading all {total_rows:,} rows from catalog"
            )
            df = lf.collect()

        # Extract columns
        event_ids = df["event_id"].to_list()
        hit_starts = df["hit_start_idx"].to_list()
        hit_ends = df["hit_end_idx"].to_list()
        n_hits_list = df["n_hits"].to_list()

        if is_exp:
            h5_parts = df["h5_part"].to_list()
        else:
            # h5_part_num (int) → HDF5 key string
            # Simple: 1000 → "part_1000"
            # Compound (nue2_2019): 1000003 → "part_1000_3"
            #   (encoded as num1*1000 + num2)
            h5_parts = [
                f"part_{num // 1000}_{num % 1000}"
                if num >= 1_000_000
                else f"part_{num}"
                for num in df["h5_part_num"].to_list()
            ]

        if is_exp:
            n_signal_list = [0] * len(df)
        else:
            n_signal_list = df["n_signal_hits"].to_list()

        records: List[EventRecord] = []
        for i in range(len(df)):
            n_sig = n_signal_list[i]
            label = self._compute_soft_label(n_sig, is_neutrino)
            eid = event_ids[i]
            if isinstance(eid, bytes):
                eid = eid.decode("utf-8", errors="replace")
            records.append(EventRecord(
                particle_type=particle_type,
                h5_part_key=h5_parts[i],
                hit_start_idx=hit_starts[i],
                hit_end_idx=hit_ends[i],
                n_hits=n_hits_list[i],
                n_signal_hits=n_sig,
                label=label,
                event_id=eid,
            ))
        return records

    # ------------------------------------------------------------------
    # Soft labels
    # ------------------------------------------------------------------

    def _compute_soft_label(
        self, n_signal_hits: int, is_neutrino: bool
    ) -> float:
        """Compute label from signal hit count and config."""
        if not is_neutrino:
            return 0.0
        mode = self.soft_label_config.get("mode", "hard")
        if mode == "hard":
            return 1.0 if n_signal_hits >= self.h_min else 0.0
        elif mode == "linear":
            h_sat = self.soft_label_config.get("h_saturate", 15)
            if h_sat <= 0:
                return 1.0 if n_signal_hits > 0 else 0.0
            return min(1.0, n_signal_hits / h_sat)
        elif mode == "sigmoid":
            # Sigmoid centered at h_saturate/2, steepness k.
            # label ≈ 0 for n < h_saturate/3,
            #        = 0.5 at n = h_saturate/2,
            #        ≈ 1 for n > 2*h_saturate/3.
            # Default k = 30./h_saturate gives ~0.05..0.95 transition
            # across the middle third of [0, h_saturate].
            h_sat = self.soft_label_config.get("h_saturate", 15)
            k = self.soft_label_config.get(
                "steepness", 30. / max(h_sat, 1)
            )
            x = k * (n_signal_hits - h_sat / 2.0)
            return 1.0 / (1.0 + math.exp(-x))
        else:
            raise ValueError(f"Unknown soft label mode: {mode}")

    # ------------------------------------------------------------------
    # Class balancing
    # ------------------------------------------------------------------

    def _balance_index(
        self, records: List[EventRecord]
    ) -> List[EventRecord]:
        """
        Balance classes using catalog metadata.

        Signal: neutrino events where soft_label >= 0.5
        Background: muon events + neutrino events where soft_label < 0.5

        Constraints:
        1. signal_nuatm == signal_nue2
        2. total_background == total_signal
        3. muons_high (signal_hits >= h_min) == rest_of_background
        """
        signal_nuatm: List[int] = []
        signal_nue2: List[int] = []
        bg_muon_high: List[int] = []
        bg_muon_low: List[int] = []
        bg_nuatm: List[int] = []
        bg_nue2: List[int] = []

        for i, rec in enumerate(records):
            is_signal = rec.label >= 0.5
            pt = rec.particle_type

            if is_signal:
                if "nuatm" in pt:
                    signal_nuatm.append(i)
                elif "nue2" in pt:
                    signal_nue2.append(i)
                else:
                    signal_nuatm.append(i)
            else:
                if "muatm" in pt or "mu" in pt:
                    if rec.n_signal_hits >= self.h_min:
                        bg_muon_high.append(i)
                    else:
                        bg_muon_low.append(i)
                elif "nuatm" in pt:
                    bg_nuatm.append(i)
                elif "nue2" in pt:
                    bg_nue2.append(i)
                else:
                    bg_muon_low.append(i)

        logger.info(
            f"Before balancing: "
            f"sig_nuatm={len(signal_nuatm)}, "
            f"sig_nue2={len(signal_nue2)}, "
            f"bg_muon_high={len(bg_muon_high)}, "
            f"bg_muon_low={len(bg_muon_low)}, "
            f"bg_nuatm={len(bg_nuatm)}, "
            f"bg_nue2={len(bg_nue2)}"
        )

        n_sig_per = min(len(signal_nuatm), len(signal_nue2))
        if n_sig_per == 0:
            logger.warning(
                "No signal events of one type — cannot balance"
            )
            return records

        # Background quotas
        n_muon_high = min(n_sig_per, len(bg_muon_high))
        n_muon_low = min(n_sig_per // 2, len(bg_muon_low))
        n_bg_nu = n_sig_per // 2
        n_bg_nuatm = min(n_bg_nu // 2, len(bg_nuatm))
        n_bg_nue2 = min(n_bg_nu - n_bg_nuatm, len(bg_nue2))

        # Check limiting factor
        targets = [
            (n_muon_high, n_sig_per),
            (n_muon_low, n_sig_per // 2),
            (n_bg_nuatm, n_bg_nu // 2),
            (n_bg_nue2, n_bg_nu - n_bg_nu // 2),
        ]
        limiting = min(
            (actual / target if target > 0 else 1.0)
            for actual, target in targets
        )
        if limiting < 1.0:
            logger.warning(
                f"Insufficient bg, scaling by {limiting:.3f}"
            )
            n_sig_per = int(n_sig_per * limiting)
            n_muon_high = n_sig_per
            n_muon_low = n_sig_per // 2
            n_bg_nu = n_sig_per // 2
            n_bg_nuatm = n_bg_nu // 2
            n_bg_nue2 = n_bg_nu - n_bg_nuatm

        # Random selection
        for lst in [
            signal_nuatm, signal_nue2, bg_muon_high,
            bg_muon_low, bg_nuatm, bg_nue2,
        ]:
            self.rng.shuffle(lst)

        selected = (
            signal_nuatm[:n_sig_per]
            + signal_nue2[:n_sig_per]
            + bg_muon_high[:n_muon_high]
            + bg_muon_low[:n_muon_low]
            + bg_nuatm[:n_bg_nuatm]
            + bg_nue2[:n_bg_nue2]
        )
        balanced = [records[i] for i in selected]

        total_sig = 2 * n_sig_per
        total_bg = (
            n_muon_high + n_muon_low + n_bg_nuatm + n_bg_nue2
        )
        logger.info(
            f"After balancing: signal={total_sig} "
            f"(nuatm={n_sig_per}, nue2={n_sig_per}), "
            f"background={total_bg} "
            f"(muon_high={n_muon_high}, muon_low={n_muon_low}, "
            f"nuatm={n_bg_nuatm}, nue2={n_bg_nue2}), "
            f"total={len(balanced)}"
        )
        return balanced

    # ------------------------------------------------------------------
    # HDF5 reading
    # ------------------------------------------------------------------

    def _open_h5(self) -> None:
        """Lazily open HDF5 file handle."""
        logger.debug(f"Opening HDF5 file: {self.h5_path} ...")
        self._h5_file = h5py.File(self.h5_path, "r")
        logger.debug("HDF5 file opened")

    def _read_event(
        self, record: EventRecord
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read hits, magic_numbers, channels for one event via direct HDF5 slice."""
        if self._h5_file is None:
            self._open_h5()

        grp = self._h5_file[record.particle_type]["raw"]
        pk = record.h5_part_key
        s, e = record.hit_start_idx, record.hit_end_idx
        hits = grp["data"][pk]["data"][s:e]
        magic = grp["labels"][pk]["data"][s:e]
        chans = grp["channels"][pk]["data"][s:e]
        return hits, magic, chans

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._event_index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self._event_index[idx]
        hits, magic, chans = self._read_event(rec)

        # Apply max_hits truncation
        if self.max_hits and len(hits) > self.max_hits:
            hits = hits[: self.max_hits]
            magic = magic[: self.max_hits]
            chans = chans[: self.max_hits]

        features = torch.tensor(hits, dtype=torch.float32)
        return {
            "features": features,
            "labels": torch.tensor(
                rec.label, dtype=torch.float32
            ),
            "magic_numbers": magic,
            "lengths": torch.tensor(
                len(features), dtype=torch.long
            ),
            "event_id": rec.event_id,
            "channels_ids": chans,
            "signal_hit_count": rec.n_signal_hits,
        }

    # ------------------------------------------------------------------
    # Stats / logging
    # ------------------------------------------------------------------

    def _calculate_stats(self) -> None:
        """Calculate dataset statistics from index."""
        labels = np.array([r.label for r in self._event_index])
        n_hits = np.array([r.n_hits for r in self._event_index])
        sig_hits = np.array(
            [r.n_signal_hits for r in self._event_index]
        )

        self.n_signal = int(np.sum(labels >= 0.5))
        self.n_background = len(labels) - self.n_signal
        self.class_balance = (
            self.n_signal / len(labels) if len(labels) > 0 else 0.0
        )

        self.min_hits = int(n_hits.min())
        self.max_hits_stat = int(n_hits.max())
        self.mean_hits = float(n_hits.mean())
        self.std_hits = float(n_hits.std())

        self.min_signal_hits = int(sig_hits.min())
        self.max_signal_hits = int(sig_hits.max())
        self.mean_signal_hits = float(sig_hits.mean())

    def _log_summary(self) -> None:
        ptypes = ", ".join(self.particle_types)
        logger.info(
            f"PrefilterDataset [{ptypes}]: {len(self)} events "
            f"(signal={self.n_signal}, bg={self.n_background}, "
            f"balance={self.class_balance:.3f})"
        )
        logger.info(
            f"  Hits: min={self.min_hits}, "
            f"max={self.max_hits_stat}, "
            f"mean={self.mean_hits:.1f}"
        )
        logger.info(
            f"  Signal hits: min={self.min_signal_hits}, "
            f"max={self.max_signal_hits}, "
            f"mean={self.mean_signal_hits:.1f}"
        )
        mode = self.soft_label_config.get("mode", "hard")
        logger.info(f"  Soft label mode: {mode}")

    def _get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return {
            "total_events": len(self),
            "signal_events": self.n_signal,
            "background_events": self.n_background,
            "class_balance": self.class_balance,
            "min_hits": self.min_hits,
            "max_hits": self.max_hits_stat,
            "mean_hits": self.mean_hits,
            "std_hits": self.std_hits,
            "min_signal_hits": self.min_signal_hits,
            "max_signal_hits": self.max_signal_hits,
            "mean_signal_hits": self.mean_signal_hits,
            "soft_label_config": self.soft_label_config,
        }

    def __del__(self) -> None:
        if self._h5_file is not None:
            try:
                self._h5_file.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def prefilter_collate_fn(
    batch: List[Dict[str, Any]],
    max_hits: int = 500,
    normalization_config: Optional[Dict[str, List[float]]] = None,
    augmentation_config: Optional[Dict[str, Any]] = None,
    shuffle_batch: bool = True,
    use_polar_coords: bool = False,
    device: str = "cpu",
    rng: Optional[np.random.RandomState] = None,
    torch_rng: Optional[torch.Generator] = None,
) -> Dict[str, Any]:
    """
    Standalone collate function for PrefilterDataset.

    Returns dict with keys:
        features, labels, lengths, original_lengths, mask,
        hits_lost, magic_numbers, channels_ids, signal_hit_counts
    """
    if rng is None:
        rng = np.random.RandomState()

    if shuffle_batch:
        batch = list(batch)
        rng.shuffle(batch)

    features = [item["features"] for item in batch]
    labels = [item["labels"] for item in batch]
    magic_numbers = [item["magic_numbers"] for item in batch]
    channels = [item["channels_ids"] for item in batch]
    signal_hit_counts = [
        item["signal_hit_count"] for item in batch
    ]

    original_lengths = torch.tensor(
        [len(f) for f in features], dtype=torch.long, device=device
    )

    # Truncation
    truncated_features: List[torch.Tensor] = []
    hits_lost = torch.zeros(
        len(features), dtype=torch.long, device=device
    )
    for i, feat in enumerate(features):
        if max_hits and len(feat) > max_hits:
            truncated_features.append(feat[:max_hits])
            hits_lost[i] = len(feat) - max_hits
        else:
            truncated_features.append(feat)

    lengths = torch.tensor(
        [len(f) for f in truncated_features],
        dtype=torch.long, device=device,
    )
    max_len = int(lengths.max().item())

    feature_dim = 5
    batch_size = len(truncated_features)
    padded = torch.zeros(
        batch_size, max_len, feature_dim,
        dtype=torch.float32, device=device,
    )
    for i, feat in enumerate(truncated_features):
        seq_len = len(feat)
        padded[i, :seq_len] = feat.to(device)

    mask = (
        torch.arange(max_len, device=device)[None, :]
        < lengths[:, None]
    )

    # --- Augmentation ---

    if (
        augmentation_config is not None
        and augmentation_config.get("rotation_enabled", False)
    ):
        angles = (
            torch.rand(batch_size, generator=torch_rng).to(device)
            * 2 * torch.pi
        )
        for b in range(batch_size):
            cos_a = torch.cos(angles[b])
            sin_a = torch.sin(angles[b])
            em = mask[b]
            x = padded[b, :, 2]
            y = padded[b, :, 3]
            xr = cos_a * x - sin_a * y
            yr = sin_a * x + cos_a * y
            padded[b, :, 2] = torch.where(em, xr, x)
            padded[b, :, 3] = torch.where(em, yr, y)

    if (
        augmentation_config is not None
        and "noise_std" in augmentation_config
    ):
        noise_std = torch.tensor(
            augmentation_config["noise_std"],
            dtype=torch.float32, device=device,
        )
        noise = torch.randn(
            padded[:, :, :5].shape,
            dtype=padded.dtype, generator=torch_rng,
        ).to(device) * noise_std
        padded = torch.where(
            mask.unsqueeze(-1), padded + noise, padded,
        )
        # Re-sort by time (index 1)
        si = padded[:, :, 1].argsort(dim=1)
        se = si.unsqueeze(-1).expand(-1, -1, feature_dim)
        padded = padded.gather(dim=1, index=se)
        mask = mask.gather(dim=1, index=si)

    # Polar coordinates
    if use_polar_coords:
        x = padded[:, :, 2]
        y = padded[:, :, 3]
        r = torch.sqrt(x ** 2 + y ** 2)
        cos_a = x / (r + 1e-6)
        sin_a = y / (r + 1e-6)
        polar = torch.zeros(
            batch_size, max_len, 3, device=device,
        )
        polar[:, :, 0] = r * mask
        polar[:, :, 1] = cos_a * mask
        polar[:, :, 2] = sin_a * mask
        padded = torch.cat([padded, polar], dim=-1)
        feature_dim = 8

    # --- Normalization ---
    if normalization_config is not None:
        means = torch.tensor(
            normalization_config["means"],
            dtype=torch.float32, device=device,
        )
        stds = torch.tensor(
            normalization_config["stds"],
            dtype=torch.float32, device=device,
        )
        if use_polar_coords:
            means = torch.cat(
                [means, torch.tensor([30, 0.0, 0.0], device=device)]
            )
            stds = torch.cat(
                [stds, torch.tensor([30, 1.0, 1.0], device=device)]
            )
        padded = torch.where(
            mask.unsqueeze(-1),
            (padded - means) / (stds + 1e-8),
            padded,
        )

    # --- Labels ---
    label_tensors = []
    for lab in labels:
        if hasattr(lab, "to"):
            label_tensors.append(lab.to(device))
        else:
            label_tensors.append(
                torch.tensor(
                    lab, dtype=torch.float32, device=device
                )
            )
    labels_tensor = torch.stack(label_tensors)

    return {
        "features": padded,
        "labels": labels_tensor,
        "lengths": lengths,
        "original_lengths": original_lengths,
        "mask": mask,
        "hits_lost": hits_lost,
        "magic_numbers": magic_numbers,
        "channels_ids": channels,
        "signal_hit_counts": torch.tensor(
            signal_hit_counts, dtype=torch.long, device=device,
        ),
    }


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def _get_base_dataset(ds) -> 'PrefilterDataset':
    """Unwrap Subset to get the underlying PrefilterDataset."""
    while hasattr(ds, "dataset"):
        ds = ds.dataset
    return ds


def _worker_init_fn(worker_id: int) -> None:
    """Re-open HDF5 handle and seed RNGs per worker for reproducibility."""
    info = torch.utils.data.get_worker_info()
    if info is not None:
        base = _get_base_dataset(info.dataset)
        base._h5_file = None  # force re-open in worker process
        # Deterministic per-worker seeding
        worker_seed = (
            base.seed + worker_id if base.seed is not None
            else worker_id
        )
        base.rng = np.random.RandomState(worker_seed)
        base.torch_rng = torch.Generator()
        base.torch_rng.manual_seed(worker_seed)


def create_prefilter_dataloader(
    dataset: PrefilterDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    normalization_config: Optional[Dict[str, List[float]]] = None,
    augmentation_config: Optional[Dict[str, Any]] = None,
    shuffle_batch: bool = True,
    use_polar_coords: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Create DataLoader from a PrefilterDataset (or Subset wrapping one)."""
    base = _get_base_dataset(dataset)

    # With num_workers > 0, collate runs in forked workers that
    # cannot use CUDA.  Build batches on CPU; the trainer moves to GPU.
    collate_device = "cpu" if num_workers > 0 else base.device

    def collate_wrapper(batch: List[Dict]) -> Dict[str, Any]:
        return prefilter_collate_fn(
            batch,
            max_hits=base.max_hits,
            normalization_config=normalization_config,
            augmentation_config=augmentation_config,
            shuffle_batch=shuffle_batch,
            use_polar_coords=use_polar_coords,
            device=collate_device,
            rng=base.rng,
            torch_rng=base.torch_rng,
        )

    # Seeded generator for DataLoader shuffle order
    shuffle_generator = None
    if shuffle and base.seed is not None:
        shuffle_generator = torch.Generator()
        shuffle_generator.manual_seed(base.seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_wrapper,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=shuffle_generator,
        worker_init_fn=(
            _worker_init_fn if num_workers > 0 else None
        ),
    )
