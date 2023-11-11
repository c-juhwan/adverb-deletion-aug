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
from transformers import AutoTokenizer, AutoConfig
from datasets import load_dataset
# Custom Modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.utils import check_path, get_huggingface_model_name

def load_data(args: argparse.Namespace) -> tuple: # (dict, dict, dict, int)
    """
    Load data from huggingface datasets.
    If dataset is not in huggingface datasets, takes data from local directory.

    Args:
        dataset_name (str): Dataset name.
        args (argparse.Namespace): Arguments.
        train_valid_split (float): Train-valid split ratio.

    Returns:
        train_data (dict): Training data. (text, label)
        valid_data (dict): Validation data. (text, label)
        test_data (dict): Test data. (text, label)
        num_classes (int): Number of classes.
    """

    name = args.task_dataset.lower()
    train_valid_split = args.train_valid_split

    train_data = {
        'text': [],
        'label': [],
        'soft_label': []
    }
    valid_data = {
        'text': [],
        'label': [],
        'soft_label': []
    }
    test_data = {
        'text': [],
        'label': [],
        'soft_label': []
    }

    if name == 'sst2':
        dataset = load_dataset('SetFit/sst2')

        train_df = pd.DataFrame(dataset['train'])
        valid_df = pd.DataFrame(dataset['validation'])
        test_df = pd.DataFrame(dataset['test'])
        num_classes = 2

        train_data['text'] = train_df['text'].tolist()
        train_data['label'] = train_df['label'].tolist()
        valid_data['text'] = valid_df['text'].tolist()
        valid_data['label'] = valid_df['label'].tolist()
        test_data['text'] = test_df['text'].tolist()
        test_data['label'] = test_df['label'].tolist()
    if name == 'sst5':
        dataset = load_dataset('SetFit/sst5')

        train_df = pd.DataFrame(dataset['train'])
        valid_df = pd.DataFrame(dataset['validation'])
        test_df = pd.DataFrame(dataset['test'])
        num_classes = 5

        train_data['text'] = train_df['text'].tolist()
        train_data['label'] = train_df['label'].tolist()
        valid_data['text'] = valid_df['text'].tolist()
        valid_data['label'] = valid_df['label'].tolist()
        test_data['text'] = test_df['text'].tolist()
        test_data['label'] = test_df['label'].tolist()
    elif name == 'cola':
        dataset = load_dataset('linxinyuan/cola')

        train_df = pd.DataFrame(dataset['train'])
        #valid_df = pd.DataFrame(dataset['validation']) # No pre-defined validation set
        test_df = pd.DataFrame(dataset['test'])
        num_classes = 2

        train_df = train_df.sample(frac=1).reset_index(drop=True) # Shuffle train data before split
        valid_df = train_df[:int(len(train_df) * train_valid_split)]
        train_df = train_df[int(len(train_df) * train_valid_split):]

        train_data['text'] = train_df['text'].tolist()
        train_data['label'] = train_df['label'].tolist()
        valid_data['text'] = valid_df['text'].tolist()
        valid_data['label'] = valid_df['label'].tolist()
        test_data['text'] = test_df['text'].tolist()
        test_data['label'] = test_df['label'].tolist()
    elif name == 'trec':
        dataset = load_dataset('trec')

        train_df = pd.DataFrame(dataset['train'])
        #valid_df = pd.DataFrame(dataset['validation']) # No pre-defined validation set
        test_df = pd.DataFrame(dataset['test'])
        num_classes = 6

        # train-valid split
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        valid_df = train_df[:int(len(train_df) * train_valid_split)]
        train_df = train_df[int(len(train_df) * train_valid_split):]

        train_data['text'] = train_df['text'].tolist()
        train_data['label'] = train_df['coarse_label'].tolist()
        valid_data['text'] = valid_df['text'].tolist()
        valid_data['label'] = valid_df['coarse_label'].tolist()
        test_data['text'] = test_df['text'].tolist()
        test_data['label'] = test_df['coarse_label'].tolist()
    elif name == 'subj':
        dataset = load_dataset('SetFit/subj')

        train_df = pd.DataFrame(dataset['train'])
        #valid_df = pd.DataFrame(dataset['validation']) # No pre-defined validation set
        test_df = pd.DataFrame(dataset['test'])
        num_classes = 2

        # train-valid split
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        valid_df = train_df[:int(len(train_df) * train_valid_split)]
        train_df = train_df[int(len(train_df) * train_valid_split):]

        train_data['text'] = train_df['text'].tolist()
        train_data['label'] = train_df['label'].tolist()
        valid_data['text'] = valid_df['text'].tolist()
        valid_data['label'] = valid_df['label'].tolist()
        test_data['text'] = test_df['text'].tolist()
        test_data['label'] = test_df['label'].tolist()
    elif name == 'mr':
        # There is no MR dataset in HuggingFace datasets, so we use custom file - check dataset/MR
        train_path = os.path.join(args.data_path, 'MR', 'train.csv')
        test_path = os.path.join(args.data_path, 'MR', 'test.csv')
        train_df = pd.read_csv(train_path, header=None)
        test_df = pd.read_csv(test_path, header=None)
        num_classes = 2

        # train-valid split
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        valid_df = train_df[:int(len(train_df) * train_valid_split)]
        train_df = train_df[int(len(train_df) * train_valid_split):]

        train_data['text'] = train_df[1].tolist()
        train_data['label'] = train_df[0].tolist()
        valid_data['text'] = valid_df[1].tolist()
        valid_data['label'] = valid_df[0].tolist()
        test_data['text'] = test_df[1].tolist()
        test_data['label'] = test_df[0].tolist()
    elif name == 'cr':
        dataset = load_dataset('SetFit/SentEval-CR')

        train_df = pd.DataFrame(dataset['train'])
        #valid_df = pd.DataFrame(dataset['validation']) # No pre-defined validation set
        test_df = pd.DataFrame(dataset['test'])
        num_classes = 2

        # train-valid split
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        valid_df = train_df[:int(len(train_df) * train_valid_split)]
        train_df = train_df[int(len(train_df) * train_valid_split):]

        train_data['text'] = train_df['text'].tolist()
        train_data['label'] = train_df['label'].tolist()
        valid_data['text'] = valid_df['text'].tolist()
        valid_data['label'] = valid_df['label'].tolist()
        test_data['text'] = test_df['text'].tolist()
        test_data['label'] = test_df['label'].tolist()
    elif name == 'proscons':
        # There is no MR dataset in HuggingFace datasets, so we use custom file - check dataset/ProsCons
        train_path = os.path.join(args.data_path, 'ProsCons', 'train.csv')
        test_path = os.path.join(args.data_path, 'ProsCons', 'test.csv')
        train_df = pd.read_csv(train_path, header=None)
        test_df = pd.read_csv(test_path, header=None)
        num_classes = 2

        # train-valid split
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        valid_df = train_df[:int(len(train_df) * train_valid_split)]
        train_df = train_df[int(len(train_df) * train_valid_split):]

        train_data['text'] = train_df[1].tolist()
        train_data['label'] = train_df[0].tolist()
        valid_data['text'] = valid_df[1].tolist()
        valid_data['label'] = valid_df[0].tolist()
        test_data['text'] = test_df[1].tolist()
        test_data['label'] = test_df[0].tolist()

    # Convert integer label to soft label
    for data in [train_data, valid_data]:
        for i, label in enumerate(data['label']):
            soft_label = [0.0] * num_classes
            soft_label[label] = 1.0
            data['soft_label'].append(soft_label)

    # For test data
    for data in [test_data]:
        for i, label in enumerate(data['label']):
            soft_label = [0.0] * num_classes
            if label == -1:
                pass # Ignore unlabeled data
            else:
                soft_label[label] = 1.0
            data['soft_label'].append(soft_label)

    return train_data, valid_data, test_data, num_classes

def preprocessing(args: argparse.Namespace) -> None:
    """
    Main function for preprocessing.

    Args:
        args (argparse.Namespace): Arguments.
    """

    # Load data
    train_data, valid_data, test_data, num_classes = load_data(args)

    # Define tokenizer & config
    model_name = get_huggingface_model_name(args.model_type)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)

    # Preprocessing - Define data_dict
    data_dict = {
        'train': {
            'input_text': [],
            'labels': [],
            'soft_labels': [],
            'vocab_size': config.vocab_size,
            'num_classes': num_classes,
            'tokenizer': tokenizer,
        },
        'valid': {
            'input_text': [],
            'labels': [],
            'soft_labels': [],
            'vocab_size': config.vocab_size,
            'num_classes': num_classes,
            'tokenizer': tokenizer,
        },
        'test': {
            'input_text': [],
            'labels': [],
            'soft_labels': [],
            'vocab_size': config.vocab_size,
            'num_classes': num_classes,
            'tokenizer': tokenizer,
        },
    }

    # Save data as pickle file
    preprocessed_path = os.path.join(args.preprocess_path, args.task, args.task_dataset, args.model_type)
    check_path(preprocessed_path)

    for split_data, split in zip([train_data, valid_data, test_data], ['train', 'valid', 'test']):
        for idx in range(len(split_data['text'])):
            # Get text and label
            text = split_data['text'][idx]
            label = split_data['label'][idx]
            soft_label = split_data['soft_label'][idx]

            # Append data to data_dict
            data_dict[split]['input_text'].append(text)
            data_dict[split]['labels'].append(label)
            data_dict[split]['soft_labels'].append(soft_label)

        # Save data_dict as pickle file
        with open(os.path.join(preprocessed_path, f'{split}_ORI.pkl'), 'wb') as f:
            pickle.dump(data_dict[split], f)
