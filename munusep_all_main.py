from clearml import Task

import data.config_manager as cfgm
from data.batch_generator import BatchGenerator
from nnetworks.models.config_manager import model_from_yaml
from nnetworks.learning.config_manager import yaml2trainercfg
from nnetworks.models.munusep_resnet import MuNuSepResNet
from nnetworks.learning.trainers import MuNuSepTrainer

# data
name_of_dataset = "munusep_all_small"
train_paths = cfgm.read_paths(f"/home/albert/Baikal-ML/data/configurations/{name_of_dataset}/train_paths.csv")
test_paths = cfgm.read_paths(f"/home/albert/Baikal-ML/data/configurations/{name_of_dataset}/test_paths.csv")
cfg = cfgm.load_cfg(f"/home/albert/Baikal-ML/data/configurations/{name_of_dataset}/cfg.yaml")
train_gen, test_gen = BatchGenerator(train_paths, cfg), BatchGenerator(test_paths, cfg)

# model
model = model_from_yaml(MuNuSepResNet, "/home/albert/Baikal-ML/nnetworks/models/configurations/munusep_all_resnet.yaml")


project_name = "MuNuSepAll"
task_name="ResNet_SmallDS_FirstTrial"
# ClearML
task = Task.init(project_name, task_name) 
# trainer
learning_cfg = yaml2trainercfg("/home/albert/Baikal-ML/nnetworks/learning/configurations/munusepall_short.yaml")
learning_cfg.experiment_path = f"/home/albert/Baikal-ML/experiments/{project_name}/{task_name}"
fitter = MuNuSepTrainer(model, 
                        train_gen, 
                        test_gen=test_gen, 
                        train_config=learning_cfg,
                        clearml_task=task)
try:
    fitter.train()
except Exception as e:
    print(e)
    task.close()