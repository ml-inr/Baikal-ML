from datetime import datetime
from pathlib import Path
import logging

from clearml import Task
import torch

import data.settings_manager as cfgm
from data.batch_generators import MCMuNuSepBatchGenerator
from nnetworks.models.config_manager import model_from_yaml, save_model_cfg
from learning.config_manager import yaml2trainercfg, save_trainer_cfg
from data.settings_manager import save_paths, save_dict2yaml
from learning.trainers import MuNuSepTrainer
from nnetworks.models.munusep_resnet import MuNuSepResNet
from nnetworks.models.munusep_lstm import MuNuSepLstm
from nnetworks.models.munusep_transformer import TransformerClassifier

# Experiment name
project_name = "MuNuSepAll"
dttm = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
task_name=f"{dttm}_SmallTransformer_4Enc_4H_256DM_256DFF_MaxPool_Small2020MC"
device = torch.device("cuda:1")
# task_name=f"{dttm}_MediumResNet_k3-3-3_AvPool_SmallDS_binary_lr1e-4_LayerNorm"
# task_name=f"{dttm}_TinyLSTM1_RSTrue_AvPool_SmallDS_binary_lr1e-4_BatchNorm"

# Local logging settings
experiment_path = f"/home/albert/Baikal-ML/experiments/{project_name}/{task_name}"
experiment_folder = Path(experiment_path)
experiment_folder.mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename= experiment_folder / "logs.log", 
                    filemode='a', 
                    level=logging.DEBUG, 
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    force=True)

# ClearML settings
tags = ['ApproxGeom', 'Small', 'Transformer', 'SmallDS']
# tags = ['Medium', 'ResNet']
clearml_task = Task.init(project_name, task_name, tags=tags, auto_connect_arg_parser=False, auto_connect_frameworks=False, auto_resource_monitoring=False, auto_connect_streams=False)

# data
name_of_dataset = "munusep_all_small_approxgeom"
train_paths = cfgm.read_paths(f"/home/albert/Baikal-ML/data/datasets/{name_of_dataset}/train_paths.csv")
train_mu_paths = [p for p in train_paths if 'muatm' in p]
train_nu_paths = [p for p in train_paths if 'muatm' not in p]
test_paths = cfgm.read_paths(f"/home/albert/Baikal-ML/data/datasets/{name_of_dataset}/test_paths.csv")
test_mu_paths = [p for p in test_paths if 'muatm' in p]
test_nu_paths = [p for p in test_paths if 'muatm' not in p]
data_kwargs = cfgm.load_yaml2dict(f"/home/albert/Baikal-ML/data/datasets/{name_of_dataset}/cfg.yaml")
# # change batchsize manually if needed
# data_kwargs['batch_size'] = 128
train_gen = MCMuNuSepBatchGenerator(
    train_mu_paths
    , train_nu_paths
    , device=device
    , **data_kwargs)
test_gen = MCMuNuSepBatchGenerator(test_mu_paths
                                   , test_nu_paths
                                   , device=device
                                   , **data_kwargs)

# model
model = model_from_yaml(TransformerClassifier, "/home/albert/Baikal-ML/nnetworks/models/configurations/munusep_all_transformer.yaml").to(device)
# model = model_from_yaml(MuNuSepResNet, "/home/albert/Baikal-ML/nnetworks/models/configurations/munusep_all_resnet.yaml")
# model = model_from_yaml(MuNuSepLstm, "/home/albert/Baikal-ML/nnetworks/models/configurations/munusep_all_rnn.yaml")

# trainer
trainer_config = yaml2trainercfg("/home/albert/Baikal-ML/learning/configurations/munusepall_long.yaml")

# Locally log all configs
save_model_cfg(model.config, experiment_folder / "model_cfg.yaml")
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
logging.info(f"Model's parameters number: {count_parameters(model)}")
save_trainer_cfg(trainer_config, experiment_folder / "learning_cfg.yaml")
save_dict2yaml(data_kwargs, experiment_folder / "train_dataset_cfg.yaml")
save_paths(train_paths, experiment_folder/"train_root_files.csv")
logging.info(f"Train dataset path: {Path(train_paths[0]).parent}")
if test_gen is not None:
    save_dict2yaml(data_kwargs, experiment_folder / "test_dataset_cfg.yaml")
    save_paths(test_paths, experiment_folder / "test_root_files.csv")
    logging.info(f"Test dataset path: {Path(test_paths[0]).parent}")

# Log in ClearML
clearml_task.connect({"experiment_path": experiment_path}, name="Local path to experiment")
clearml_task.connect({**model.config.to_dict(), "NumParams": f"{count_parameters(model)}"}, name="Model's architecture")
clearml_task.connect(trainer_config.to_dict(), name="Learning config")  # Log training hyperparameters
clearml_task.connect({f"Train dataset path": f"{Path(train_paths[0]).parent}", **data_kwargs}, name="Train data generator's config")
if test_gen is not None: clearml_task.connect({f"Test dataset path": f"{Path(test_paths[0]).parent}", **data_kwargs}, name="Test data generator's config")

# Launch experiment
fitter = MuNuSepTrainer(
                        model, 
                        train_batches=train_gen, 
                        test_batches=test_gen, 
                        experiment_folder=experiment_folder,
                        train_config=trainer_config,
                        clearml_task=clearml_task
                        )

fitter.train()
