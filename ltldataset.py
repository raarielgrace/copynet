import os
import sys
import random
from torch.utils.data import Dataset
from language import Language
from utils import tokens_to_seq, contains_digit, shuffle_correlated_lists
from operator import itemgetter

### Dataset class for data where the input (source) and output (target)
### are stored as two separate files.

class SequencePairDataset(Dataset):

    def __init__(self,
                 src_lines,
                 tgt_lines,
                 maxlen=200,
                 lang=None,
                 vocab_limit=None,
                 use_extended_vocab=True):

        self.maxlen = maxlen
        self.parser = None
        self.use_extended_vocab = use_extended_vocab

        self.data = [] # Will hold all data

        for i in range(len(src_lines)):
            inputs = src_lines[i]
            outputs = tgt_lines[i]
            inputsL = inputs.replace('\n', '').split(' ')
            outputsL = outputs.replace('\n', '').split(' ')
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
