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
from datasets import Dataset
import torch
import evaluate
import nltk
import numpy as np
import multiprocessing
import functools

from huggingface_hub import HfFolder
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

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
    
    # DEFAULT values for other configs
    if 'repository_id' not in config:
        config.repository_id = None # Hugging Face Repository id for uploading models
    if 'hf_token' not in config:
        config.hf_token = HfFolder.get_token() # Token to use for uploading models to Hugging Face Hub.
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
    return config

def preprocess_function(examples, config, tokenizer): 
    model_inputs = tokenizer(examples['source'], max_length=config.max_input_length, padding=padding, truncation=True)
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['target'], max_length=max_output_length, padding=padding, truncation=True)
    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def load_train_dataset(dataset, config, tokenizer):
    tmp_dict = {"source":[],"target":[]}
    for entry in dataset:
        tmp_dict['source'].append(entry['input'])
        tmp_dict['target'].append(entry['output'])
    dataset = Dataset.from_dict(tmp_dict)
    dataset = dataset.map(
        functools.partial(preprocess_function, config=config, tokenizer=tokenizer),
        batched=True,
        num_proc=config.num_workers,
    )
    return dataset

def load_eval_dataset(dataset, config, tokenizer):
    tmp_dict = {"source":[],"target":[], "label_list":[]}
    for entry in dataset:
        source = entry['input']
        target = entry['output']
        label_list = None # Not yet implemented
        tmp_dict['source'].append(entry['input'])
        tmp_dict['target'].append(entry['output'])
        tmp_dict['label_list'].append(None)
    dataset = Dataset.from_dict(tmp_dict)
    dataset = dataset.map(
        functools.partial(preprocess_function, config=config, tokenizer=tokenizer),
        batched=True,
        num_proc=config.num_workers
    )
    return dataset

nltk.download("punkt", quiet=True)

# Metric
metric = evaluate.load("rouge")
# evaluation generation args
gen_kwargs = {
    "early_stopping": True,
    "length_penalty": 2.0,
    "max_new_tokens": 50,
    "min_length": 30,
    "no_repeat_ngram_size": 3,
    "num_beams": 4,
}

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
    with open(f"{args.dataset_path}/train.json", 'r') as f:
        train_dataset = load_train_dataset(json.load(f), config=args, tokenizer = tokenizer)
    with open(f"{args.dataset_path}/eval.json", 'r') as f:
        eval_dataset = load_eval_dataset(json.load(f), config=args, tokenizer = tokenizer)


    # we want to ignore tokenizer pad token in the loss
    label_pad_token_id = -100
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=label_pad_token_id, pad_to_multiple_of=8
    )

    # Define compute metrics function
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    # Define training args
    # output_dir = args.repository_id if args.repository_id else args.model_id.split("/")[-1]
    output_dir = args.model_id.split("/")[-1]
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        predict_with_generate=True,
        generation_max_length=args.generation_max_length,
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
        report_to="tensorboard",
        push_to_hub=True if args.repository_id else False,
        hub_strategy="every_save",
        hub_model_id=args.repository_id if args.repository_id else None,
        hub_token=args.hf_token,
    )

    # Create Trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Start training
    trainer.train()

    # Save our tokenizer and create model card
    tokenizer.save_pretrained(output_dir)
    trainer.create_model_card()
    # Push the results to the hub
    if args.repository_id:
        trainer.push_to_hub()

def main():
    parser = ArgumentParser()
    parser.add_argument('--config', default=None, type=str)
    arg_ = parser.parse_args()
    if arg_.config is None:
        raise NameError("Include a config file in the argument please.")
    config_path = arg_.config
    with open(config_path) as config_file:
        config = json.load(config_file)
    config = check_args(argparse.Namespace(**config))
    if config.mode == 'train':
        training_run(config)
    else:
        raise Exception(f'{config.mode} not yet implemented..')

if __name__ == "__main__":
    main()