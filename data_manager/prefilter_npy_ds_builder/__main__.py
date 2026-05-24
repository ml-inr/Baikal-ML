"""Allow running as: python -m data_manager.prefilter_npy_ds_builder --config ..."""

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

from .build import main

main()
