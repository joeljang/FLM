import json
import argparse
from argparse import ArgumentParser
from datasets import load_dataset, get_dataset_config_names, Dataset
from transformers import AutoTokenizer
import torch
import numpy as np
import pandas as pd
import os
import random
from promptsource.templates import DatasetTemplates

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

def select_template(dataset_name, config, k):
    template_path = f'{dataset_name}/{config}'
    prompt_template = DatasetTemplates(template_path)
    all_prompts = prompt_template.all_template_names
    if not all_prompts:
        first_prompt = None 
    else:
        first_prompt = all_prompts[0]
    return prompt_template, first_prompt

def load_dataset_exception(eval_metadata, eval_name, eval_config, eval_split):
    if eval_name=='story_cloze':
        df = pd.read_csv(eval_metadata['path'])
        hf_dataset = Dataset.from_pandas(df)
        return hf_dataset
    else:
        raise Exception(f'Please find load_dataset support for {eval_name}/{eval_config}/{eval_split}')

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

# 1. Loading & Saving Training Datasets
if config.make_train:
    # Checking if the same number of entries were given for configs
    if len(config.train_datasets) == len(config.train_dataset_configs) == len(config.train_instances):
        train_dataset_num = len(config.train_datasets)
        print(f'The same number of training datasets given: {train_dataset_num}')
    else:
        raise NameError("Please provide the same number of entries for the train datasets")

    # Concatenating the training data into one single blob for easy training
    train_entries = []
    for i in range(train_dataset_num):  
        # Load dataset from the hub
        train_dataset = load_dataset(config.train_datasets[i], name=config.train_dataset_configs[i], split="train", ignore_verifications=True)
        if config.train_instances[i] > len(train_dataset):
            raise Exception('The number of designated instances exceeds the original dataset size')
        print(f"{config.train_datasets[i]} | Original train dataset size: {len(train_dataset)}, Limit size to: {config.train_instances[i]}")
        train_dataset = subsample(train_dataset, config.train_instances[i])
        if config.use_promptsource:
            print(f'Using Promptsource to preprocess the training dataset!')
        else:
            for row in train_dataset:
                input_ = row[config.train_input_column[i]]
                output_ = row[config.train_output_column[i]]
                entry = {
                    "source": input_,
                    "target": output_
                }
                train_entries.append(entry)
    with open(f"{config.save_dataset_path}/train.json", "w") as file:
        json.dump(train_entries, file)
    
# 2. Loading & Saving Evaluation Dataset
if config.make_eval:
    # Concatenting the evaluation data, still dividied into each dataset for easy logging
    all_evals = {}
    no_template_list = []
    for i in range(len(config.eval_datasets)):  
        eval_metadata = config.eval_datasets[i]
        eval_dataset_name = eval_metadata['name']
        eval_num_instances = eval_metadata['num_instances']
        # Expand "all" configs
        if eval_metadata['configs'] == ['all']:
            all_configs = get_dataset_config_names(eval_dataset_name)
            eval_metadata['configs'] = all_configs       
        for eval_config in eval_metadata['configs']: # 2.1 Loop through all of the configs
            for eval_split in eval_metadata['splits']: # 2.2 Loop through all of the splits (e.g. anli)
                eval_entries = []
                try:
                    eval_dataset = load_dataset(eval_dataset_name, name=eval_config, split=eval_split, ignore_verifications=True)
                except:
                    eval_dataset = load_dataset_exception(eval_metadata, eval_dataset_name, eval_config, eval_split)
                if eval_num_instances > len(eval_dataset):
                    print(f'The number of designated instances exceeds the original dataset size. Requested: {eval_num_instances}, Original: {len(eval_dataset)}')
                    eval_num_instances = len(eval_dataset)
                print(f"{eval_dataset_name}/{eval_config}/{eval_split} | Original evaluation dataset size: {len(eval_dataset)}, Limit size to: {eval_num_instances}")
                eval_dataset = subsample(eval_dataset, eval_num_instances)
                if config.use_promptsource:
                    prompt_templates, selected_prompt = select_template(eval_dataset_name, eval_config, 1) # Select random template
                    if selected_prompt==None: 
                        print(f'Could not find a matching template for {eval_dataset_name, eval_config}')
                        no_template_list.append(f'{eval_dataset_name}/{eval_config}')
                        continue # If matching prompt template could not be found, skip it
                    prompt = prompt_templates[selected_prompt]
                    for row in eval_dataset:
                        result = prompt.apply(row)
                        input_ = result[0]
                        output_ = result[1][0] # change this to result[1] if you are using the original promptsource library
                        options = prompt.get_answer_choices_list(row)
                        label = options.index(output_)
                        entry = {
                            "source": input_,
                            "target": output_,
                            "labels_list": options
                        }
                        eval_entries.append(entry)
                else:
                    raise Exception('Currently not supporting preprocessing method other than using promptsource.')
                all_evals[f'{eval_dataset_name}/{eval_config}/{eval_split}'] = eval_entries        
    
    print(f'Combined a total of {len(all_evals)} number of evaluation datasets')
    print(f'Could not find matching templates for {len(no_template_list)} number of evaluation datasets. Here is the list: {no_template_list}')
    with open(f"{config.save_dataset_path}/eval.json", "w") as file:
        json.dump(all_evals, file)