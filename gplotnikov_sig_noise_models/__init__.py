"""Backward-compatibility shim for the external sig-noise model.

The model tree moved from the repository root to ``trained_models/sig_noise/``.
Old dotted imports keep resolving there, for example::

    from gplotnikov_sig_noise_models.k_nsol_labelneq0_da_hs128_k0p0001.sig_noise_model_v3 import load_model

New code should import from
``trained_models.sig_noise.k_nsol_labelneq0_da_hs128_k0p0001.sig_noise_model_v3``.
"""
from pathlib import Path

__path__ = [str(Path(__file__).resolve().parents[1] / "trained_models" / "sig_noise")]
