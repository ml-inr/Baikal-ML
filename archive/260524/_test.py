import json, yaml, sys, logging
sys.path.insert(0, '.')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')

from data_manager.nu_classifier_ds_builder.build import build_nu_classifier_npy

with open('data_manager/nu_classifier_ds_builder/compliment_config.yaml') as f:
    cfg = yaml.safe_load(f)

# Load and slice parts
with open(cfg['parts_json']) as f:
    parts = json.load(f)
cfg['_parts_override'] = {k: v[:5] for k, v in parts.items()}
cfg['output_dir'] = './nu_classifier_smoke_test'

build_nu_classifier_npy(cfg)