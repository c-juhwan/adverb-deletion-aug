# Standard Library Modules
import os
import sys
import pickle
import argparse
# 3rd-party Modules
import pandas as pd
from tqdm.auto import tqdm
# Pytorch Modules
import torch
# Huggingface Modules
from transformers import AutoTokenizer
# Custom Modules
from .augmentation_utils import run_eda, run_aeda, run_adverb_aug
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.utils import check_path, get_huggingface_model_name

def load_preprocessed_data(args: argparse.Namespace) -> dict:
    """
    Open preprocessed train pickle file from local directory.

    Args:
        args (argparse.Namespace): Arguments.

    Returns:
        train_data (dict): Preprocessed training data.
    """

    preprocessed_path = os.path.join(args.preprocess_path, args.task, args.task_dataset, args.model_type, 'train_ORI.pkl')

    with open(preprocessed_path, 'rb') as f:
        train_data = pickle.load(f)

    return train_data

def augmentation(args: argparse.Namespace) -> None:
    """
    1. Load preprocessed train data
    2. Apply augmentation by pre-defined augmentation strategy & Give soft labels for soft_eda method
    3. Concatenate original data & augmented data
    4. Save total data

    Args:
        args (argparse.Namespace): Arguments.
    """

    # Load preprocessed train data
    train_data = load_preprocessed_data(args)
    augmented_data = {
        'input_text': [],
        'labels': [],
        'soft_labels': [],
        'vocab_size': train_data['vocab_size'],
        'num_classes': train_data['num_classes'],
        'tokenizer': train_data['tokenizer']
    }

    for idx in tqdm(range(len(train_data['input_text'])), desc=f'Augmenting with {args.augmentation_type}'):
        original_text = train_data['input_text'][idx]

        # Apply augmentation by pre-defined augmentation strategy
        if args.augmentation_type == 'hard_eda':
            augmented_sent = run_eda(original_text, args)
            augmented_data['soft_labels'].append(train_data['soft_labels'][idx]) # Hard EDA: Keep original soft labels (Actually one-hot labels)
        elif args.augmentation_type == 'soft_eda':
            augmented_sent = run_eda(original_text, args)
            soft_labels = train_data['soft_labels'][idx] * (1 - args.augmentation_label_smoothing) + args.augmentation_label_smoothing / train_data['num_classes']
            augmented_data['soft_labels'].append(soft_labels) # SoftEDA: Apply soft labels for soft_eda method using label smoothing
        elif args.augmentation_type == 'aeda':
            augmented_sent = run_aeda(original_text, args)
            augmented_data['soft_labels'].append(train_data['soft_labels'][idx]) # AEDA: Keep original soft labels (Actually one-hot labels)
        elif args.augmentation_type in ['adverb_aug', 'adverb_aug_curriculum']:
            augmented_sent, aug_count = run_adverb_aug(original_text, args)
            if aug_count == 0:
                continue # Skip if no adverb is augmented
            augmented_data['soft_labels'].append(train_data['soft_labels'][idx]) # AEDA: Keep original soft labels (Actually one-hot labels)

        augmented_data['input_text'].append(augmented_sent)
        augmented_data['labels'].append(train_data['labels'][idx]) # Keep the hard labels as they are

        # Merge original data & augmented data
        total_dict = {
            'input_text': train_data['input_text'] + augmented_data['input_text'],
            'labels': train_data['labels'] + augmented_data['labels'],
            'soft_labels': train_data['soft_labels'] + augmented_data['soft_labels'],
            'vocab_size': augmented_data['vocab_size'],
            'num_classes': augmented_data['num_classes'],
            'tokenizer': augmented_data['tokenizer']
        }

        # Save total data as pickle file
        save_path = os.path.join(args.preprocess_path, args.task, args.task_dataset, args.model_type)
        with open(os.path.join(save_path, f'train_{args.augmentation_type}.pkl'), 'wb') as f:
            pickle.dump(total_dict, f)
