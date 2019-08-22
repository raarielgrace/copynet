import os
import sys
import random
from torch.utils.data import Dataset
from utils import tokens_to_seq, contains_digit, shuffle_correlated_lists
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
        # Note that this vocab is case sensitive
        vocab = dict()
        for data_pair in self.data:
            tokens = data_pair[0] + data_pair[1]
            for token in tokens:
                # Track frequency of each word in the vocab
                vocab[token] = vocab.get(token, 0) + 1
        return vocab
