import os
import argparse
from argparse import ArgumentParser
import json
import numpy as np
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    AutoTokenizer,
    set_seed,
)
from datasets import Dataset, load_from_disk, load_metric
import torch
import evaluate
import nltk
import numpy as np
import multiprocessing
import wandb

from huggingface_hub import HfFolder
from transformers import Seq2SeqTrainingArguments
from third_party.trainers import Seq2SeqTrainer
from third_party.trainers import TaskDataCollatorForSeq2Seq
from third_party.trainers import PostProcessor

def check_args(config):
    """Check the configurations"""
    # REQUIRED configs
    if 'mode' not in config:
        raise Exception('Please provide the mode of the run. Choose between `train` & `eval`.')
    if 'model_id' not in config:
        raise Exception('Please provide the model_id provide in huggingface models')
    if 'dataset_path' not in config:
        raise Exception('Please provide the dataset path that contains train.json & eval.json')
    if 'epochs' not in config:
        raise Exception('Please provide the epoch of the training data')
    if 'output_dir' not in config:
        raise Exception('Please provide the output directory to save the log files & model checkpoint')
    
    # DEFAULT values for other configs
    if 'per_device_train_batch_size' not in config:
        config.per_device_train_batch_size = 8 # Batch size to use for training.
    if 'per_device_eval_batch_size' not in config:
        config.per_device_eval_batch_size = 8 # Batch size to use for testing.
    if 'max_input_length' not in config:
        config.max_input_length = 512 # Maximum length to use for generation
    if 'max_output_length' not in config:
        config.max_output_length = 128 # Maximum length to use for generation
    if 'generation_num_beams' not in config:
        config.generation_num_beams = 1 # Number of beams to use for generation
    if 'lr' not in config:
        config.lr = 1e-4 # Learning rate to use for training.
    if 'seed' not in config:
        config.seed = 42 # Random seed for all things random
    if 'deepspeed' not in config:
        config.deepspeed = "gpu_configs/z3_bf16.json" # Directory to the deepspeed configuration. Details in https://www.deepspeed.ai/tutorials/zero/
    if 'gradient_checkpointing' not in config:
        config.gradient_checkpointing = True # Whether to use gradient checkpointing. 
    if 'bf16' not in config:
        config.bf16 = True if torch.cuda.get_device_capability()[0] == 8 else False # Whether to use bf16.
    if 'num_workers' not in config:
        config.num_workers = multiprocessing.cpu_count()
    
    # etc.
    if 'repository_id' not in config:
        config.repository_id = None # Hugging Face Repository id for uploading models
    if 'hf_token' not in config:
        config.hf_token = HfFolder.get_token() # Token to use for uploading models to Hugging Face Hub.
    if 'wandb' not in config:
        config.wandb =  False
    if 'wandb_entity' not in config:
        config.wandb_entity = 'wkddydpf' # Default wandb entity to log experiments to. Change with your wandb entity
    if 'wandb_project' not in config:
        config.wandb_project = 'flm' # Change depending on your project name
    if 'wandb_run_name' not in config:
        config.wandb_run_name = 'random' # Provide name to the run
    return config

nltk.download("punkt", quiet=True)

# Metric
#metric = evaluate.load("rouge")
metric = evaluate.load("accuracy")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

def training_run(args):
    # Set Random Seed :)
    set_seed(args.seed)
    
    # Load model & tokenizer from huggingface
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_id,
        use_cache=False if args.gradient_checkpointing else True,  # this is needed for gradient checkpointing
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # Load train & eval datasets
    train_dataset = load_from_disk(os.path.join(args.dataset_path, "train"))
    eval_datasets = {}
    with open(f"{args.dataset_path}/eval.json", 'r') as f:
        all_evals = json.load(f)
        for key in all_evals:
            eval_dataset = load_from_disk(os.path.join(args.dataset_path, key))
            key = key.replace('/', '_')
            print(key, eval_dataset)
            eval_datasets[key] = eval_dataset

    # we want to ignore tokenizer pad token in the loss
    label_pad_token_id = -100
    # Data collator
    data_collator = TaskDataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=label_pad_token_id, pad_to_multiple_of=8
    )

    def get_accuracy(preds, labels):
        total_cnt = 0
        correct = 0
        for i in range(len(preds)):
            total_cnt+=1
            if preds[i] == labels[i]:
                correct+=1
        return {'accuracy': correct / total_cnt}

    # Define compute metrics function
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        post_processor = PostProcessor(tokenizer, ignore_pad_token_for_loss=True)
        decoded_preds, decoded_labels = post_processor.process(preds, labels)
        result = get_accuracy(preds=decoded_preds, labels=decoded_labels)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        print(result)
        return result

    # Define training args
    # output_dir = args.repository_id if args.repository_id else args.model_id.split("/")[-1]
    output_dir = args.output_dir
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        predict_with_generate=True,
        generation_max_length=args.max_output_length,
        generation_num_beams=args.generation_num_beams,
        fp16=False,  # T5 overflows with fp16
        bf16=args.bf16,  # Use BF16 if available
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        deepspeed=args.deepspeed,
        gradient_checkpointing=args.gradient_checkpointing,
        # logging & evaluation strategies
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=500,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        # push to hub parameters
        report_to="wandb",
        push_to_hub=True if args.repository_id else False,
        hub_strategy="every_save",
        hub_model_id=args.repository_id if args.repository_id else None,
        hub_token=args.hf_token,
    )

    # Create Trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_datasets,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    if args.mode=='train':
        print('Starting Training!')
        trainer.train()

        # Save our tokenizer and create model card
        tokenizer.save_pretrained(output_dir)
        trainer.create_model_card()

        # Push the results to the hub
        if args.repository_id:
            trainer.push_to_hub()
    elif args.mode=='eval':
        print('Starting Evaluation!')
        for task, eval_dataset in eval_datasets.items():
            print(task)
            trainer.evaluate(eval_dataset = eval_dataset, metric_key_prefix=task, config=args)
    else:
        raise Exception('Currently only supporting train & eval.')

def main():
    parser = ArgumentParser()
    parser.add_argument('--config', default=None, type=str)
    arg_, _ = parser.parse_known_args()
    if arg_.config is None:
        raise NameError("Include a config file in the argument please.")
    config_path = arg_.config
    with open(config_path) as config_file:
        config = json.load(config_file)
    config = check_args(argparse.Namespace(**config))
    if config.wandb:
        wandb.init(entity=config.wandb_entity, project=config.wandb_project, name=config.wandb_run_name)
    training_run(config)

if __name__ == "__main__":
    main()