import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
import yaml
from data_manager.nu_classifier_ds_builder.build import build_nu_classifier_npy
with open('data_manager/nu_classifier_ds_builder/test_config.yaml') as f:
    cfg = yaml.safe_load(f)
build_nu_classifier_npy(cfg)