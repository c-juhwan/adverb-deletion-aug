# Standard Library Modules
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" # This prevents tokenizers from taking all cpus
import sys
import shutil
import logging
import argparse
# 3rd-party Modules
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
import pandas as pd
# Pytorch Modules
import torch
torch.set_num_threads(2) # This prevents Pytorch from taking all cpus
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# Custom Modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.classification.model import ClassificationModel
from model.classification.dataset import CustomDataset
from utils.utils import TqdmLoggingHandler, write_log, get_tb_exp_name, get_wandb_exp_name, get_torch_device

def testing(args: argparse.Namespace) -> tuple:
    device = get_torch_device(args.device)

    # Define logger and tensorboard writer
    logger = logging.getLogger(__name__)
    if len(logger.handlers) > 0:
        logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False

    if args.use_tensorboard:
        writer = SummaryWriter(os.path.join(args.log_path, get_tb_exp_name(args)))
        writer.add_text('args', str(args))

    # Load dataset and define dataloader
    write_log(logger, "Loading dataset")
    dataset_dict, dataloader_dict = {}, {}
    dataset_dict['test'] = CustomDataset(args, os.path.join(args.preprocess_path, args.task, args.task_dataset, args.model_type, f'test_ORI.pkl'))

    dataloader_dict['test'] = DataLoader(dataset_dict['test'], batch_size=args.batch_size, num_workers=args.num_workers,
                                         shuffle=False, pin_memory=True, drop_last=False)
    args.vocab_size = dataset_dict['test'].vocab_size
    args.num_classes = dataset_dict['test'].num_classes

    write_log(logger, "Loaded data successfully")
    write_log(logger, f"Test dataset size / iterations: {len(dataset_dict['test'])} / {len(dataloader_dict['test'])}")

    # Get model instance
    write_log(logger, "Building model")
    model = ClassificationModel(args)

    # Load model weights
    write_log(logger, "Loading model weights")
    final_model_save_name = f'final_model_{args.augmentation_type}.pt'
    load_model_name = os.path.join(args.model_path, args.task, args.task_dataset, args.model_type, final_model_save_name)
    model = model.to('cpu')
    checkpoint = torch.load(load_model_name, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    write_log(logger, f"Loaded model weights from {load_model_name}")
    del checkpoint

    # Load Wandb
    if args.use_wandb:
        import wandb
        from wandb import AlertLevel
        wandb.init(project=args.proj_name,
                   name=get_wandb_exp_name(args) + f' - Test',
                   config=args,
                   notes=args.description,
                   tags=["TEST",
                         f"Dataset: {args.task_dataset}",
                         f"Model: {args.model_type}",
                         f"Aug: {args.augmentation_type}"])

    # Test - Start testing on test set
    model = model.eval()
    test_acc_cls = 0
    test_f1_cls = 0
    for test_iter_idx, data_dicts in enumerate(tqdm(dataloader_dict['test'], total=len(dataloader_dict['test']), desc="Testing on TEST Set", position=0, leave=True)):
        # Test - Get input data
        input_data = {k: v.to(device) for k, v in data_dicts['input_data'].items()}
        hard_labels = data_dicts['label'].to(device)

        # Test - Forward pass
        with torch.no_grad():
            classification_logits = model(input_ids=input_data['input_ids'],
                                          attention_mask=input_data['attention_mask'],
                                          token_type_ids=input_data['token_type_ids'] if args.model_type not in ['roberta'] else None)

        # Test - Calculate loss & accuracy/f1 score
        batch_acc_cls = (classification_logits.argmax(dim=-1) == hard_labels).float().mean()
        batch_f1_cls = f1_score(hard_labels.cpu().numpy(), classification_logits.argmax(dim=-1).cpu().numpy(), average='macro')

        # Test - Logging
        test_acc_cls += batch_acc_cls.item()
        test_f1_cls += batch_f1_cls

        if test_iter_idx % args.log_freq == 0 or test_iter_idx == len(dataloader_dict['test']) - 1:
            write_log(logger, f"TEST(T) - Iter [{test_iter_idx}/{len(dataloader_dict['test'])}] - Acc: {batch_acc_cls.item():.4f}")
            write_log(logger, f"TEST(T) - Iter [{test_iter_idx}/{len(dataloader_dict['test'])}] - F1: {batch_f1_cls:.4f}")

    # Test - Check metric
    test_acc_cls /= len(dataloader_dict['test'])
    test_f1_cls /= len(dataloader_dict['test'])

    # Final - End of testing
    write_log(logger, f"Done! - TEST SET - - Acc: {test_acc_cls:.4f} - F1: {test_f1_cls:.4f}")
    if args.use_tensorboard:
        writer.add_scalar('TEST/Acc', test_acc_cls, 0)
        writer.add_scalar('TEST/F1', test_f1_cls, 0)
        writer.close()
    if args.use_wandb:
        wandb_df = pd.DataFrame({
            'Dataset': [args.task_dataset],
            'Model': [args.model_type],
            'Augmentation': [args.augmentation_type],
            'Test_Acc': [test_acc_cls],
            'Test_F1': [test_f1_cls],
        })
        wandb_table = wandb.Table(dataframe=wandb_df)
        wandb.log({'TEST_Result': wandb_table})

        wandb.finish()

    return test_acc_cls, test_f1_cls
