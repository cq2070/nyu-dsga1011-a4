import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):
    def __init__(self, data_folder, split):
        self.tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
        self.bos_id = self.tokenizer.convert_tokens_to_ids("<extra_id_0>")
        self.encoder_ids, self.sql_ids = self.process_data(
            data_folder, split, self.tokenizer
        )

    def process_data(self, data_folder, split, tokenizer):
        texts = load_lines(os.path.join(data_folder, f"{split}.nl"))
        encoder_ids = tokenizer(texts, add_special_tokens=False)["input_ids"]

        if split == "test":
            return encoder_ids, None

        queries = load_lines(os.path.join(data_folder, f"{split}.sql"))
        if len(queries) != len(texts):
            raise ValueError(f"{split}: NL and SQL counts differ")

        sql_ids = tokenizer(queries, add_special_tokens=False)["input_ids"]
        return encoder_ids, sql_ids

    def __len__(self):
        return len(self.encoder_ids)

    def __getitem__(self, idx):
        eos_id = self.tokenizer.eos_token_id
        encoder = torch.tensor(self.encoder_ids[idx] + [eos_id], dtype=torch.long)
        start = torch.tensor([self.bos_id], dtype=torch.long)

        if self.sql_ids is None:
            return encoder, start

        sql = self.sql_ids[idx]
        decoder_input = torch.tensor([self.bos_id] + sql, dtype=torch.long)
        decoder_target = torch.tensor(sql + [eos_id], dtype=torch.long)
        return encoder, decoder_input, decoder_target, start


def normal_collate_fn(batch):
    encoders, decoder_inputs, decoder_targets, starts = zip(*batch)
    encoder_ids = pad_sequence(encoders, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = encoder_ids.ne(PAD_IDX).long()
    decoder_ids = pad_sequence(decoder_inputs, batch_first=True, padding_value=PAD_IDX)
    targets = pad_sequence(decoder_targets, batch_first=True, padding_value=PAD_IDX)
    return encoder_ids, encoder_mask, decoder_ids, targets, torch.stack(starts)


def test_collate_fn(batch):
    encoders, starts = zip(*batch)
    encoder_ids = pad_sequence(encoders, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = encoder_ids.ne(PAD_IDX).long()
    return encoder_ids, encoder_mask, torch.stack(starts)


def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # TODO
    return train_x, train_y, dev_x, dev_y, test_x