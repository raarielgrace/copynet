import numpy as np
import torch
from torch.autograd import Variable
from abc import ABC
from torch import nn
import bcolz
import pickle
import random


class DecoderBase(ABC, nn.Module):
    def forward(self, encoder_outputs, inputs, final_encoder_hidden, targets=None, teacher_forcing=1.0):
        raise NotImplementedError

def get_glove():
    ''' Loads the glove database and returns it as word vectors.
    '''
    vectors = bcolz.open('data/glove.6B.50.dat')[:]
    words = pickle.load(open('data/glove.6B.50_words.pkl', 'rb'))
    word2idx = pickle.load(open('data/glove.6B.50_idx.pkl', 'rb'))

    # Vectors are 50 long
    glove = {w: vectors[word2idx[w]] for w in words}
    return glove

def load_complete_data(data_name):
    ''' Loads source and target lines of data files.
    '''
    with open('./data/' + data_name + '_src.txt', "r") as sf:
        src_lines = sf.readlines()

    with open('./data/' + data_name + '_tar.txt', "r") as tf:
        tgt_lines = tf.readlines()

    # No processing is done on the data, just loads it

    if not len(src_lines) == len(tgt_lines):
        sys.exit("ERROR: Data files have inconsistent lengths. Make sure your labels are aligned correctly.")

    return ((src_lines, tgt_lines))


def load_split_eighty_twenty(data_name, seed):
    ''' Loads data and makes a training and testing dataset.
    '''
    src_lines, tgt_lines = load_complete_data(data_name)

    # Shuffles lines before splitting to get randomized split
    src_lines, tgt_lines, _ = shuffle_correlated_lists(src_lines, tgt_lines, seed)

    # Split our source and target files with a 20:80 split
    split_idx = len(src_lines) // 5
    test_src = src_lines[:split_idx]
    train_src = src_lines[split_idx:]
    test_tgt = tgt_lines[:split_idx]
    train_tgt = tgt_lines[split_idx:]

    return (train_src, train_tgt, test_src, test_tgt)


def shuffle_correlated_lists(l1, l2, seed):
    ''' Shuffles data split into two lists with corresponding entries.
    '''

    #Lists must have the same length, as the entries correspond
    if not len(l1) == len(l2):
        print("ERROR: Lists of unequal length! Returning without shuffling.")
        return (l1, l2)

    # Seeds the randomizer
    random.seed(seed)
    # Make a list of all the indexes
    idxes = list(range(len(l1)))
    # Shuffle this index list
    random.shuffle(idxes)

    new_l1 = []
    new_l2 = []
    # Make new shuffled lists by reading in the data
    # In the randomized index order
    for idx in idxes:
        new_l1.append(l1[idx])
        new_l2.append(l2[idx])

    # Returns the list of indexes so the order can be tracked
    return (new_l1, new_l2, idxes)

# Source: https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
def chunks(l, n):
    """ Yield n successive chunks from l.
    """
    newn = int(len(l) / n)
    for i in range(0, n-1):
        yield l[i*newn:i*newn+newn]
    yield l[n*newn-newn:]

### BELOW ARE THE ORIGINAL UTIL FILES ###

def to_np(x):
    return x.data.cpu().numpy()


def seq_to_string(seq, idx_to_tok, input_tokens=None):
    vocab_size = len(idx_to_tok)
    seq_length = (seq != 0).sum()
    words = []
    for idx in seq[:seq_length]:
        if idx < vocab_size:
            words.append(idx_to_tok[idx])
        elif input_tokens is not None:
            words.append(input_tokens[idx - vocab_size])
        else:
            words.append('<???>')
    string = ' '.join(words)
    return string


def largest_indices(ary, n):
    """Returns the indicies of the n largest values in a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)


def to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).contiguous().view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot


def tokens_to_seq(tokens, tok_to_idx, max_length, use_extended_vocab, input_tokens=None):
    seq = torch.zeros(max_length).long()
    tok_to_idx_extension = dict()

    for pos, token in enumerate(tokens):
        if token in tok_to_idx:
            idx = tok_to_idx[token]

        elif token in tok_to_idx_extension:
            idx = tok_to_idx_extension[token]

        elif use_extended_vocab and input_tokens is not None:
            # If the token is not in the vocab and an input token sequence was provided
            # find the position of the first occurance of the token in the input sequence
            # the token index in the output sequence is size of the vocab plus the position in the input sequence.
            # If the token cannot be found in the input sequence use the unknown token.
            tok_to_idx_extension[token] = tok_to_idx_extension.get(token,
                                 next((pos + len(tok_to_idx)
                                       for pos, input_token in enumerate(input_tokens)
                                       if input_token == token), max_length + pos + len(tok_to_idx)))
            idx = tok_to_idx_extension[token]

        elif use_extended_vocab:
            # unknown tokens in the input sequence use the position of the first occurence + vocab_size as their index
            idx = pos + len(tok_to_idx)
        else:
            idx = tok_to_idx['<UNK>']

        seq[pos] = idx

    return seq


def trim_seqs(seqs):
    trimmed_seqs = []
    for output_seq in seqs:
        trimmed_seq = []
        for idx in to_np(output_seq):
            trimmed_seq.append(idx[0])
            if idx == 2:
                break
        trimmed_seqs.append(trimmed_seq)
    return trimmed_seqs


def get_seq_lengths(seqs):
    lengths = []
    for seq in seqs:
        seq = list(to_np(seq))
        lengths.append(seq.index(2) + 1 if 2 in seq else len(seq))
    return lengths


def contains_digit(string):
    return any(char.isdigit() for char in string)
