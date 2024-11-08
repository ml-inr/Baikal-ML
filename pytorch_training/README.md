## Project for training NN's on baikal MC data based on torch and torch_geometric

### Currently supported training types are
- noise-signal hits classification
- track-cascade hits classification
- angle-reconstruction
- t_res
- track-cascade & angle-reconstruction || track-cascade & t_res 

### How to run
```
python train.py -c=train_configs/<your_config> -dw
```
remove `-dw` if you want to track your run in wandb (login from terminal required before that)

### Project structure

- trainining
    - train_utils.py - all high-level logic for model training & evaluation

- data_utils
    - dataloaders.py - create torch datasets and dataloaders, collator logic
    - preprocessors.py - preparing data for different tasks, add noise, filter etc
    - readers.py - read data from H5 file and run preprocessor

- metrics - all logic for metrics calculation 

- models - each file contains some architecture 

- train_configs - yaml configs describing task type, model etc

