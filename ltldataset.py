import os
import sys
import random
from torch.utils.data import Dataset
from utils import tokens_to_seq, contains_digit
from operator import itemgetter


class Language(object):
    def __init__(self, vocab_limit, data):
        self.data = data

        self.vocab = self.create_vocab()

        truncated_vocab = sorted(self.vocab.items(), key=itemgetter(1), reverse=True)[:vocab_limit]

        self.tok_to_idx = dict()
        self.tok_to_idx['<MSK>'] = 0
        self.tok_to_idx['<SOS>'] = 1
        self.tok_to_idx['<EOS>'] = 2
        self.tok_to_idx['<UNK>'] = 3
        for idx, (tok, _) in enumerate(truncated_vocab):
            self.tok_to_idx[tok] = idx + 4
        self.idx_to_tok = {idx: tok for tok, idx in self.tok_to_idx.items()}

    def create_vocab(self):
        vocab = dict()
        for data_pair in self.data:
            tokens = data_pair[0] + data_pair[1]
            for token in tokens:
                # Track frequency of each word in the vocab
                vocab[token] = vocab.get(token, 0) + 1
        return vocab


class SequencePairDataset(Dataset):
    def __init__(self,
                 src_data_path='./data/hard_pc_src_syn2.txt',
                 tgt_data_path='./data/hard_pc_tar_syn2.txt',
                 maxlen=200,
                 lang=None,
                 vocab_limit=None,
                 val_size=0.1,
                 seed=42,
                 is_val=False,
                 use_cuda=False,
                 use_extended_vocab=True):

        self.src_data_path = src_data_path
        self.tgt_data_path = tgt_data_path
        self.maxlen = maxlen
        self.use_cuda = use_cuda
        self.parser = None
        self.val_size = val_size
        self.seed = seed
        self.is_val = is_val
        self.use_extended_vocab = use_extended_vocab

        with open(self.src_data_path, "r") as sf:
            src_lines = sf.readlines()

        with open(self.tgt_data_path, "r") as tf:
            tgt_lines = tf.readlines()

        if not len(src_lines) == len(tgt_lines):
            sys.exit("ERROR: Data files have inconsistent lengths. Make sure your labels are aligned correctly.")

        self.data = [] # Will hold all data

        for i in range(len(src_lines)):
            inputs = src_lines[i]
            outputs = tgt_lines[i]
            inputsL = inputs.split(' ')
            outputsL = outputs.split(' ')
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
        #print(input_token_list)
        #print(output_token_list)

        input_seq = tokens_to_seq(input_token_list, self.lang.tok_to_idx, self.maxlen, self.use_extended_vocab)
        output_seq = tokens_to_seq(output_token_list, self.lang.tok_to_idx, self.maxlen, self.use_extended_vocab, input_tokens=input_token_list)

        if self.use_cuda:
            input_seq = input_seq.cuda()
            output_seq = output_seq.cuda()

        return input_seq, output_seq, ' '.join(input_token_list), ' '.join(output_token_list)