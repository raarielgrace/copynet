import os
import random
from torch.utils.data import Dataset
from language import Language
from utils import tokens_to_seq, contains_digit
from operator import itemgetter

### CURRENTLY UNUSED. TO USE THIS DATASET FOR TRAINING, THE TRAIN FILES WILL NEED TO BE EDITED ###

### Dataset for data where the input and output are on the same line and tab-separated

class SequencePairDataset(Dataset):
    data_path='./data/copynet_data_v2.txt'

    with open(data_path, "r") as f:
        lines = f.readlines()

    # Split our data files with a 20:80 split
    split_idx = len(lines) // 5
    test_src = lines[:split_idx]
    train_src = lines[split_idx:]

    def __init__(self,
                 maxlen=200,
                 lang=None,
                 vocab_limit=None,
                 val_size=0.1,
                 seed=42,
                 is_val=False,
                 is_test=False,
                 use_cuda=False,
                 save_lang=False,
                 use_extended_vocab=True):

        self.maxlen = maxlen
        self.use_cuda = use_cuda
        self.parser = None
        self.val_size = val_size
        self.seed = seed
        self.is_val = is_val
        self.is_test = is_test
        self.use_extended_vocab = use_extended_vocab

        self.data = [] # Will hold all data from a single file

        if self.is_test:
            lines = SequencePairDataset.test_src
        else:
            lines = SequencePairDataset.train_src

        for line in lines:
            inputs, outputs, _ = line.split('\t')
            inputsL = inputs.split(',')
            outputsL = outputs.split(',')
            self.data.append([inputsL, outputsL])

        if lang is None:
            lang = Language(vocab_limit, self.data)

        self.lang = lang

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        :arg
        idx: int

        :returns
        input_token_list: list[int]
        output_token_list: list[int]
        token_mapping: binary array"""

        data_pair = self.data[idx]

        input_token_list = (['<SOS>'] + data_pair[0] + ['<EOS>'])[:self.maxlen]
        output_token_list = (['<SOS>'] + data_pair[1] + ['<EOS>'])[:self.maxlen]

        input_seq = tokens_to_seq(input_token_list, self.lang.tok_to_idx, self.maxlen, self.use_extended_vocab)
        output_seq = tokens_to_seq(output_token_list, self.lang.tok_to_idx, self.maxlen, self.use_extended_vocab, input_tokens=input_token_list)

        return input_seq, output_seq, ' '.join(input_token_list), ' '.join(output_token_list)
