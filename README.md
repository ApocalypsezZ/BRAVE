# VQSAN

Getting started
This section shows an example of a process of using the codes.

Prepare the dataset (use miniImageNet as an example)
download and extract miniimagent–ravi.

check the structure of the dataset：

The dataset must be in the following structure:

dataset_folder/
├── images/
│   ├── images_1.jpg
│   ├── ...
│   └── images_n.jpg
├── train.csv *
├── test.csv *
└── val.csv *
Modify the config file
Use ProtoNet as an example：

create a new yaml file getting_started.yaml in config/

write the following commands into the created file:

includes:
  - headers/data.yaml
  - headers/device.yaml
  - headers/losses.yaml
  - headers/misc.yaml
  - headers/model.yaml
  - headers/optimizer.yaml
  - classifiers/Proto.yaml
  - backbones/Conv64F.yaml
More details can be referred to write a config yaml.

Run
set the config as follows in run_trainer.py:

config = Config("./config/getting_started.yaml").get_config_dict()
train with the console command:

python run_trainer.py
wait for the end of training.

View the log files
After running the program, you can find a symlink of results/ProtoNet-miniImageNet-Conv64F-5-1 and a directory of results/ProtoNet-miniImageNet-Conv64F-5-1-$TS, where TS means the timestamp. The directory contains two folders: checkpoint/ and log_files/, and a configuration file: config.yaml. Note that the symlink will always link to the directory created at the last time, when you train the model with the same few-shot learning configuration for multiple times.

config.yaml contains all the settings used in the training phase.

log_files/ contains tensorboard files, training log files and test log files.

checkpoints/ contains model checkpoints saved at $save_insterval intervals, the last model checkpoint (used to resume) and the best model checkpoint (used to test). The checkpoint files are generally divided into emb_func.pth, classifier.pth, and model.pth (a combination of the first two), respectively.
