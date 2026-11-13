"""Fixed provenance for the released angle/track/uncertainty bundle.

The loaders in this directory verify these values. If a file is replaced by a
different checkpoint, fail loudly instead of silently scoring with the wrong
weights.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

BUNDLE_SHA256 = (
    "7cb8f77cfdb2a7cf021d090be5246b83f8a94c7513cc24c793ab019c67686f41"
)
SN_SHA256 = (
    "be9cc39b93dd6a57a69233faf6755278e3f202c56de3de7d1cc8d2d4a8b1e93d"
)
ANGLE_NORM_SHA256 = (
    "8053d7579d1f0e065a5c4bf1092ce8e513e3985b37c086799b169f5c08207bb7"
)
CLASSIFIER_NORM_SHA256 = (
    "28e1dae133db7bacce5a908807b2892c8af675fc20a6ba27fc7fd95b3b38f9ea"
)

BUNDLE_FORMAT_VERSION = 2
BUNDLE_MODEL_TYPE = "photon_corrected_uncertainty"
BUNDLE_STATE_ENTRIES = 97
BUNDLE_PARAMETERS = 3_079_266
SN_PARAMETERS = 992_513

EXPECTED_MAX_SEQ_LEN = 128
EXPECTED_CENTER_TIME = False
EXPECTED_FEATURE_ORDER = ["charge", "time_ns", "x_m", "y_m", "z_m"]
EXPECTED_NORM_MEAN = [
    10.060193061828613,
    -43.222816467285156,
    1.3752371072769165,
    -0.37781569361686707,
    15.266818046569824,
]
EXPECTED_NORM_STD = [
    21.777206420898438,
    371.0409240722656,
    39.418434143066406,
    38.6718864440918,
    147.00804138183594,
]
EXPECTED_CALIBRATION = {
    "angle": {"68": 2.6017538254422115, "95": 6.456679605404715},
    "line": {"68": 1.8528922487425732, "95": 4.22721294656512},
}

# The released SN checkpoint has a deeper t_res head (tres_head.0/2) than the
# plain Linear in EncoderTwoHead. It is unused for hit probabilities.
ALLOWED_SN_MISSING = {"tres_head.weight", "tres_head.bias"}
ALLOWED_SN_UNEXPECTED = {
    "tres_head.0.weight",
    "tres_head.0.bias",
    "tres_head.2.weight",
    "tres_head.2.bias",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: str | Path, expected: str, label: str) -> None:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"{label} SHA256 mismatch: expected {expected}, got {actual} ({path})"
        )
