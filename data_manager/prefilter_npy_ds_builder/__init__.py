"""Prefilter NPY dataset builder — converts HDF5 parts into flat .npy training arrays."""

from .soft_labels import compute_soft_label
from .balance import balance_classes
from .io import read_part_metadata, read_all_metadata, write_selected_features
