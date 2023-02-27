import json
import argparse
from argparse import ArgumentParser
from datasets import load_dataset
from transformers import AutoTokenizer
import torch
import numpy as np
import os

# Functions 
def shuffled_indices(dataset, random_seed):
    num_samples = len(dataset)
    generator = torch.Generator()
    generator.manual_seed(random_seed)
    return torch.randperm(num_samples, generator=generator).tolist()

def subsample(dataset, n_obs=None, random_seed=42):
    num_samples = len(dataset)
    indices = shuffled_indices(dataset, random_seed)
    indices = indices[:n_obs]
    return dataset.select(indices)

# 0. Getting configurations
parser = ArgumentParser()
parser.add_argument('--config', default=None, type=str)
arg_ = parser.parse_args()
if arg_.config is None:
    raise NameError("Include a config file in the argument please.")
config_path = arg_.config
with open(config_path) as config_file:
    config = json.load(config_file)
config = argparse.Namespace(**config)
if not os.path.exists(config.save_dataset_path): # Make directory path if does not exist
    os.makedirs(config.save_dataset_path)

# 0-1. Check whether the number of train & eval datasets given are aligned
if len(config.train_datasets) == len(config.train_dataset_configs) == len(config.train_instances) == len(config.train_input_column) == len(config.train_output_column):
    train_dataset_num = len(config.train_datasets)
    print(f'The same number of training datasets given: {train_dataset_num}')
else:
    raise NameError("Please provide the same number of entries for the train datasets")
if len(config.eval_datasets) == len(config.eval_dataset_configs) == len(config.eval_instances) == len(config.eval_input_column) == len(config.eval_output_column):
    eval_dataset_num = len(config.eval_datasets)
    print(f'The same number of evaluation datasets given: {eval_dataset_num}')
else:
    raise NameError("Please provide the same number of entries for the eval datasets")

# 1. Loading & Saving Training Datasets
train_entries = []
for i in range(train_dataset_num):  
    # Load dataset from the hub
    train_dataset = load_dataset(config.train_datasets[i], name=config.train_dataset_configs[i], split="train")
    if config.train_instances[i] > len(train_dataset):
        raise Exception('The number of designated instances exceeds the original dataset size')
    print(f"{config.train_datasets[i]} | Original train dataset size: {len(train_dataset)}, Limit size to: {config.train_instances[i]}")
    train_dataset = subsample(train_dataset, config.train_instances[i])
    for row in train_dataset:
        input_ = row[config.train_input_column[i]]
        output_ = row[config.train_output_column[i]]
        entry = {
            "input": input_,
            "output": output_
        }
        train_entries.append(entry)
with open(f"{config.save_dataset_path}/train.json", "w") as file:
    json.dump(train_entries, file)
    
# 2. Loading & Saving Evaluation Dataset
eval_entries = []
for i in range(eval_dataset_num):  
    # Load dataset from the hub
    eval_dataset = load_dataset(config.eval_datasets[i], name=config.eval_dataset_configs[i], split="validation")
    if config.eval_instances[i] > len(eval_dataset):
        raise Exception('The number of designated instances exceeds the original dataset size')
    print(f"{config.eval_datasets[i]} | Original evaluation dataset size: {len(eval_dataset)}, Limit size to: {config.eval_instances[i]}")
    eval_dataset = subsample(eval_dataset, config.eval_instances[i])
    if config.eval_type[i] == 'generation':
        for row in eval_dataset:
            input_ = row[config.eval_input_column[i]]
            output_ = row[config.eval_output_column[i]]
            entry = {
                "input": input_,
                "output": output_
            }
            eval_entries.append(entry)
    else:
        raise Exception(f'Currently {config.eval_type} format is not yet implemented..')
with open(f"{config.save_dataset_path}/eval.json", "w") as file:
    json.dump(eval_entries, file)