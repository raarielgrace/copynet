import os
import sys
import random
from torch.utils.data import Dataset
from ltldataset import Language
from utils import tokens_to_seq, contains_digit, shuffle_correlated_lists, chunks
from operator import itemgetter

class OneFoldSequencePairDataset(Dataset):

    def __init__(self,
                 unprocessed_data,
                 maxlen,
                 vocab_limit,
                 use_extended_vocab):

        self.maxlen = maxlen
        self.parser = None
        self.use_extended_vocab = use_extended_vocab
        self.k = k

        self.data = [] # Will hold all data

        for i in range(len(unprocessed_data)):
            inputs = unprocessed_data[i][0]
            outputs = unprocessed_data[i][1]
            inputsL = inputs.replace('\n', '').split(' ')
            outputsL = outputs.replace('\n', '').split(' ')
            self.data.append([inputsL, outputsL])

        self.lang = Language(vocab_limit, self.data)

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


def generateKFoldDatasets(maxlen=200,
             lang=None,
             vocab_limit=None,
             use_extended_vocab=True,
             k=5,
             src_data_path='./data/twophrase_south_shuffled_src.txt',
             tgt_data_path='./data/twophrase_south_shuffled_tar.txt'):

    with open(src_data_path, "r") as sf:
        src_lines = sf.readlines()

    with open(tgt_data_path, "r") as tf:
        tgt_lines = tf.readlines()

    if not len(src_lines) == len(tgt_lines):
        sys.exit("ERROR: Data files have inconsistent lengths. Make sure your labels are aligned correctly.")

    shuffle_correlated_lists(src_lines, tgt_lines)

    data = [(src_lines[i], tgt_lines[i]) for i in len(src_lines)]

    # Divide the data into k chunks
    chunked = chunks(data, k)
    folds = []
    for _ in range(k):
        folds.append(chunked.next())

    # Build the k training and testing datasets
    datasets = []
    for i in range(k):
        # Build out data for the datasets
        train_data = []
        test_data = []
        # For each fold
        for j in range(len(folds)):
            if i == j: # Pick one fold for testing data
                test_data += folds[j]
            else: # Add other folds to training data
                train_data += folds[i]

        # Make the testing and training dataset objects
        training_dataset = OneFoldSequencePairDataset(train_data, maxlen, vocab_limit, use_extended_vocab)
        test_dataset = OneFoldSequencePairDataset(test_data, maxlen, vocab_limit, use_extended_vocab)

        datasets.append((training_dataset, test_dataset))

    return datasets
