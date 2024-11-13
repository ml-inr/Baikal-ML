from datetime import datetime

from clearml import Task
import torch

import data.config_manager as cfgm
from data.batch_generator import BatchGenerator
from nnetworks.models.config_manager import model_from_yaml
from learning.config_manager import yaml2trainercfg
from learning.trainers import MuNuSepTrainer
from nnetworks.models.munusep_resnet import MuNuSepResNet
from nnetworks.models.munusep_lstm import MuNuSepLstm


# data
name_of_dataset = "munusep_all_small"
train_paths = cfgm.read_paths(f"/home/albert/Baikal-ML/data/configurations/{name_of_dataset}/train_paths.csv")
train_paths = train_paths[:5] + train_paths[-5:]
test_paths = cfgm.read_paths(f"/home/albert/Baikal-ML/data/configurations/{name_of_dataset}/test_paths.csv")
cfg = cfgm.load_cfg(f"/home/albert/Baikal-ML/data/configurations/{name_of_dataset}/cfg.yaml")
train_gen, test_gen = BatchGenerator(train_paths, cfg), BatchGenerator(test_paths, cfg)

# model
model = model_from_yaml(MuNuSepResNet, "/home/albert/Baikal-ML/nnetworks/models/configurations/munusep_all_resnet.yaml")
# model = model_from_yaml(MuNuSepLstm, "/home/albert/Baikal-ML/nnetworks/models/configurations/munusep_all_rnn.yaml")

# ClearML
project_name = "MuNuSepAll"
dttm = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# task_name=f"{dttm}_TinyResNet_k3-3-5_SmallDS_binary_lr1e-4_LayerNorm2"
task_name=f"CheckTestWorking2"
# task_name=f"{dttm}_TinyLSTM1_RSTrue_AvPool_SmallDS_binary_lr1e-4_LayerNorm12"
task = Task.init(project_name, task_name) 

# trainer
trainer_config = yaml2trainercfg("/home/albert/Baikal-ML/learning/configurations/munusepall_long.yaml")
trainer_config.experiment_path = f"/home/albert/Baikal-ML/experiments/{project_name}/{task_name}"
device = torch.device("cuda:0")
fitter = MuNuSepTrainer(
                        model, 
                        train_gen=train_gen, 
                        test_gen=test_gen, 
                        train_config=trainer_config,
                        clearml_task=task,
                        device=device
                        )

fitter.train()
