import torch
from torch import nn
#from torch.autograd import Variable
import argparse
import numpy as np

### Adds a initial word vectors option and updates to Pytorch 0.4 ###

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size, device, init_weight_dict=None, vocab_to_idx=None):
        super(EncoderRNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

        self.embedding = nn.Embedding(input_size, self.embedding_size).to(device)

        # If there's a dictionary with initial weights for words
        if not init_weight_dict == None:

            weights_matrix = np.zeros((input_size, embedding_size))
            for word in vocab_to_idx.keys():
                try:
                    # Put those word vectors into our weight matrix
                    weights_matrix[vocab_to_idx[word]] = init_weight_dict[word]
                except KeyError:
                    # Use a normalized vector if the word isn't in the dictionary
                    weights_matrix[vocab_to_idx[word]] = np.random.normal(scale=0.6, size=(embedding_size, ))

            # Load this into our embedding layer
            self.embedding.load_state_dict({'weight': torch.from_numpy(weights_matrix)})
        else:
            # Otherwise, start with just normalized vectors for the embedding
            self.embedding.weight.data.normal_(0, 1 / self.embedding_size**0.5)

        self.gru = nn.GRU(embedding_size, hidden_size, bidirectional=True, batch_first=True).to(device)

    def forward(self, iput, hidden, lengths):
        # iput batch must be sorted by sequence length
        iput = iput.masked_fill(iput > self.embedding.num_embeddings, 3)  # replace OOV words with <UNK> before embedding
        # Send the data through an embedding
        embedded = self.embedding(iput)
        packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True)
        self.gru.flatten_parameters()
        # Then an RNN
        output, hidden = self.gru(packed_embedded, hidden)
        output = output.to(self.device)
        hidden = hidden.to(self.device)
        output, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hidden

    def init_hidden(self, batch_size):
        ''' Creates a tensor for the proper hidden size.
        '''
        hidden = torch.zeros(2, batch_size, self.hidden_size).to(self.device)  # bidirectional rnn
        return hidden
