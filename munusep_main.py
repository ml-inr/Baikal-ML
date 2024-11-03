import data.config_manager as cfgm
from data.batch_generator import BatchGenerator
from nnetworks.models.config_manager import model_from_yaml
from nnetworks.models.munusep_resnet import MuNuSepResNet
from nnetworks.learning.trainers import MuNuSepTrainer
from nnetworks.learning.config import TrainerConfig

name_of_dataset = "numusep_signal_small"
train_paths = cfgm.read_paths(f"./data/configurations/{name_of_dataset}/train_paths.csv")
test_paths = cfgm.read_paths(f"./data/configurations/{name_of_dataset}/test_paths.csv")
cfg = cfgm.load_cfg(f"./data/configurations/{name_of_dataset}/cfg.yaml")
train_gen, test_gen = BatchGenerator(train_paths, cfg), BatchGenerator(test_paths, cfg)


model = model_from_yaml(MuNuSepResNet, "/home/albert/Baikal-ML/nnetworks/configurations/munusep_resnet.yaml")

fitter = MuNuSepTrainer(model, train_gen, test_gen, TrainerConfig())
fitter.train()