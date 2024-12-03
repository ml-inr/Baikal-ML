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
from nnetworks.models.munusep_transformer import TransformerClassifier

# data
name_of_dataset = "munusep_all_small"
train_paths = cfgm.read_paths(f"/home/albert/Baikal-ML/data/configurations/{name_of_dataset}/train_paths.csv")
test_paths = cfgm.read_paths(f"/home/albert/Baikal-ML/data/configurations/{name_of_dataset}/test_paths.csv")
cfg = cfgm.load_cfg(f"/home/albert/Baikal-ML/data/configurations/{name_of_dataset}/cfg.yaml")
train_gen, test_gen = BatchGenerator(train_paths, cfg), BatchGenerator(test_paths, cfg)

# model
# model = model_from_yaml(TransformerClassifier, "/home/albert/Baikal-ML/nnetworks/models/configurations/munusep_all_transformer.yaml")
model = model_from_yaml(MuNuSepResNet, "/home/albert/Baikal-ML/nnetworks/models/configurations/munusep_all_resnet.yaml")
# model = model_from_yaml(MuNuSepLstm, "/home/albert/Baikal-ML/nnetworks/models/configurations/munusep_all_rnn.yaml")

# ClearML
project_name = "MuNuSepAll"
dttm = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# task_name=f"{dttm}_SmallTransformer_2H_128DM_128DFF_MaxPool_SmallDS_binary_lr1e-4"
task_name=f"{dttm}_MediumResNet_k3-3-3_AvPool_SmallDS_binary_lr1e-4_LayerNorm"
# task_name=f"{dttm}_TinyLSTM1_RSTrue_AvPool_SmallDS_binary_lr1e-4_BatchNorm"

# tags = ['Small', 'Transformer']
tags = ['Medium', 'ResNet']
task = Task.init(project_name, task_name, tags=tags, auto_connect_arg_parser=False, auto_connect_frameworks=False, auto_resource_monitoring=False, auto_connect_streams=False)

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
