# Standard Library Modules
import pickle
# 3rd-party Modules
from tqdm.auto import tqdm
# Pytorch Modules
import torch
from torch.utils.data.dataset import Dataset

class CustomDataset(Dataset):
    def __init__(self, args, data_path:str) -> None:
        super(CustomDataset, self).__init__()
        self.args = args
        with open(data_path, 'rb') as f:
            data_ = pickle.load(f)

        self.data_list = []
        self.tokenizer = data_['tokenizer']
        self.num_classes = data_['num_classes']
        self.vocab_size = data_['vocab_size']

        for idx in tqdm(range(len(data_['input_text'])), desc=f'Loading data from {data_path}'):
            self.data_list.append({
                'input_text': data_['input_text'][idx],
                'label': data_['labels'][idx],
                'soft_label': data_['soft_labels'][idx],
                'index': idx,
            })

        del data_

    def __getitem__(self, idx:int) -> dict:
        # Tokenize input text
        input_tokenized = self.tokenizer(self.data_list[idx]['input_text'],
                                         padding='max_length', truncation=True,
                                         max_length=self.args.max_seq_len, return_tensors='pt')

        input_tokenized = {k: v.squeeze(0) for k, v in input_tokenized.items()}
        label_tensor = torch.tensor(self.data_list[idx]['label'], dtype=torch.long)
        soft_label_tensor = torch.tensor(self.data_list[idx]['soft_label'], dtype=torch.float)

        return {
            'input_data': input_tokenized,
            'label': label_tensor,
            'soft_label': soft_label_tensor,
            'index': idx,
        }

    def __len__(self) -> int:
        return len(self.data_list)
