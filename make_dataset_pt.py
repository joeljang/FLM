import argparse
from argparse import ArgumentParser
import json
import os
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
import torch

# Functions
def preprocess_text(text):
    text = text.replace('\n', '')
    return text

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
    
def mlm(source, tokenizer, config):
    num_noise_tokens = int(np.round(len(source) * config.noise_density))

    # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
    num_noise_tokens = min(max(num_noise_tokens, 1), len(source) - 1)
    num_noise_spans = int(np.round(num_noise_tokens / config.mean_noise_span_length))

    # avoid degeneracy by ensuring positive number of noise spans
    num_noise_spans = max(num_noise_spans, 1)
    num_nonnoise_tokens = len(source) - num_noise_tokens
    
    # pick the lengths of the noise spans and the non-noise spans
    def _random_segmentation(num_items, num_segments):
        """Partition a sequence of items randomly into non-empty segments.
        Args:
            num_items: an integer scalar > 0
            num_segments: an integer scalar in [1, num_items]
        Returns:
            a Tensor with shape [num_segments] containing positive integers that add
            up to num_items
        """
        mask_indices = np.arange(num_items - 1) < (num_segments - 1)
        np.random.shuffle(mask_indices)
        first_in_segment = np.pad(mask_indices, [[1, 0]])
        segment_id = np.cumsum(first_in_segment)
        # count length of sub segments assuming that list is sorted
        _, segment_length = np.unique(segment_id, return_counts=True)
        return segment_length

    noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
    nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)

    interleaved_span_lengths = np.reshape(
        np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2]
    )
    span_starts = np.cumsum(interleaved_span_lengths)[:-1]
    span_start_indicator = np.zeros((len(source),), dtype=np.int8)
    span_start_indicator[span_starts] = True
    span_num = np.cumsum(span_start_indicator)
    is_noise = np.equal(span_num % 2, 1)

    mask_indices =  np.asarray([is_noise[:len(source)]])
    labels_mask = ~mask_indices

    def create_sentinel_ids(mask_indices):
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices)
        sentinel_ids = np.where(sentinel_ids != 0, (len(tokenizer) - sentinel_ids), 0)
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    input_ids_sentinel = create_sentinel_ids(mask_indices.astype(np.int8))
    labels_sentinel = create_sentinel_ids(labels_mask.astype(np.int8))

    def filter_input_ids(input_ids, sentinel_ids):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        input_ids = input_ids_full[input_ids_full >= 0].reshape((batch_size, -1))
        input_ids = np.concatenate(
            [input_ids, np.full((batch_size, 1), tokenizer.eos_token_id, dtype=np.int32)], axis=-1
        )
        return input_ids

    inputs_ = filter_input_ids(np.asarray([source]), input_ids_sentinel)
    outputs_ = filter_input_ids(np.asarray([source]), labels_sentinel)

    source = tokenizer.decode(inputs_[0], clean_up_tokenization_spaces=True)
    target = tokenizer.decode(outputs_[0], clean_up_tokenization_spaces=True)

    return source, target

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

# 0-1. Initialize configs & tokenizer
if config.mode not in ["mlm"]: # Need to support ul2 & ssm in the future
    raise Exception(f'{config.mode} not yet implemented.')
tokenizer = AutoTokenizer.from_pretrained(config.model_id)

# 1. Preprocessing the (Continued) Pretraining Corpora. 
pretrain_dataset = load_dataset(config.train_dataset, config.train_dataset_config, split='train')
print(f'Total number of rows for the pretraining corpora {config.train_dataset}/{config.train_dataset_config}: {len(pretrain_dataset)}')
if config.num_instances > len(pretrain_dataset):
    raise Exception('The number of designated instances exceeds the original dataset size')
print(f"Original train dataset size: {len(pretrain_dataset)}, Limit size to: {config.num_instances}")
pretrain_dataset = subsample(pretrain_dataset, config.num_instances)

train_entries = []
continued_text = ""
for row in pretrain_dataset:
    text = preprocess_text(row['text'])
    text = continued_text + ' ' + text
    input_t = tokenizer.encode(text)
    
    while len(input_t) > config.input_length:
        input_ = input_t[:config.input_length]
        input_t = input_t[config.input_length:]
        if config.mode == 'mlm': # masked language modeling
            input_, output_ = mlm(input_, tokenizer, config)
        else: # autoregressive language modeling
            input_, output_ = tokenizer.decode(input_, skip_special_tokens=True)
        entry = {
            "source": input_,
            "target": output_
        }
        train_entries.append(entry)
    continued_text = tokenizer.decode(input_t, skip_special_tokens=True)

with open(f"{config.save_dataset_path}/train.json", "w") as file:
    json.dump(train_entries, file, ensure_ascii=False)

print('Done! :)')