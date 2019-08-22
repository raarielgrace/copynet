import argparse
import os
import time
import numpy as np

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch import nn

from ltldataset import SequencePairDataset
from model.encoder_decoder import EncoderDecoder
from evaluate import evaluate
from utils import to_np, trim_seqs, load_complete_data, load_split_eighty_twenty

from tensorboardX import SummaryWriter
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

def train(encoder_decoder: EncoderDecoder,
          train_data_loader: DataLoader,
          model_name,
          val_data_loader: DataLoader,
          keep_prob,
          teacher_forcing_schedule,
          lr,
          max_length,
          device):

    global_step = 0
    loss_function = torch.nn.NLLLoss(ignore_index=0)
    optimizer = optim.Adam(encoder_decoder.parameters(), lr=lr)
    model_path = './model/' + model_name + '/'

    for epoch, teacher_forcing in enumerate(teacher_forcing_schedule):
        print('epoch %i' % epoch, flush=True)

        correct_predictions = 0.0
        all_predictions = 0.0
        for batch_idx, (input_idxs, target_idxs, input_tokens, target_tokens) in enumerate(tqdm(train_data_loader)):
            # input_idxs and target_idxs have dim (batch_size x max_len)
            # they are NOT sorted by length

            lengths = (input_idxs != 0).long().sum(dim=1)
            sorted_lengths, order = torch.sort(lengths, descending=True)

            input_variable = input_idxs[order, :][:, :max(lengths)]
            input_variable = input_variable.to(device)
            target_variable = target_idxs[order, :]
            target_variable = target_variable.to(device)

            optimizer.zero_grad()
            output_log_probs, output_seqs = encoder_decoder(input_variable,
                                                            list(sorted_lengths),
                                                            targets=target_variable,
                                                            keep_prob=keep_prob,
                                                            teacher_forcing=teacher_forcing)

            batch_size = input_variable.shape[0]

            output_sentences = output_seqs.squeeze(2)

            flattened_outputs = output_log_probs.view(batch_size * max_length, -1)

            batch_loss = loss_function(flattened_outputs, target_variable.contiguous().view(-1))
            batch_loss.backward()
            optimizer.step()

            batch_outputs = trim_seqs(output_seqs)

            batch_targets = [[list(seq[seq > 0])] for seq in list(to_np(target_variable))]

            for i in range(len(batch_outputs)):
                y_i = batch_outputs[i]
                tgt_i = batch_targets[i][0]

                if y_i == tgt_i:
                    correct_predictions += 1.0

                all_predictions += 1.0

            batch_bleu_score = corpus_bleu(batch_targets, batch_outputs, smoothing_function=SmoothingFunction().method1)

            if global_step % 100 == 0:

                writer.add_scalar('train_batch_loss', batch_loss, global_step)
                writer.add_scalar('train_batch_bleu_score', batch_bleu_score, global_step)

                for tag, value in encoder_decoder.named_parameters():
                    tag = tag.replace('.', '/')
                    writer.add_histogram('weights/' + tag, value, global_step, bins='doane')
                    writer.add_histogram('grads/' + tag, to_np(value.grad), global_step, bins='doane')

            global_step += 1

        with torch.no_grad():
            val_loss, val_bleu_score = evaluate(encoder_decoder, val_data_loader, device)

        writer.add_scalar('val_loss', val_loss, global_step=global_step)
        writer.add_scalar('val_bleu_score', val_bleu_score, global_step=global_step)

        encoder_embeddings = encoder_decoder.encoder.embedding.weight.data
        encoder_vocab = encoder_decoder.lang.tok_to_idx.keys()
        writer.add_embedding(encoder_embeddings, metadata=encoder_vocab, global_step=0, tag='encoder_embeddings')

        decoder_embeddings = encoder_decoder.decoder.embedding.weight.data
        decoder_vocab = encoder_decoder.lang.tok_to_idx.keys()
        writer.add_embedding(decoder_embeddings, metadata=decoder_vocab, global_step=0, tag='decoder_embeddings')

        print('training accuracy %.5f' % (100.0 * (correct_predictions / all_predictions)))
        print('val loss: %.5f, val BLEU score: %.5f' % (val_loss, val_bleu_score), flush=True)
        torch.save(encoder_decoder, "%s%s_%i.pt" % (model_path, model_name, epoch))

        print('-' * 100, flush=True)

    torch.save(encoder_decoder, "%s%s_final.pt" % (model_path, model_name))
    return encoder_decoder

def test(encoder_decoder: EncoderDecoder, test_data_loader: DataLoader, max_length, device, log_file=None):

    correct_predictions = 0.0
    all_predictions = 0.0
    for batch_idx, (input_idxs, target_idxs, input_tokens, target_tokens) in enumerate(tqdm(test_data_loader)):
        # input_idxs and target_idxs have dim (batch_size x max_len)
        # they are NOT sorted by length

        lengths = (input_idxs != 0).long().sum(dim=1)
        sorted_lengths, order = torch.sort(lengths, descending=True)

        input_variable = input_idxs[order, :][:, :max(lengths)]
        input_variable = input_variable.to(device)
        input_tokens = [input_tokens[o] for o in order]
        target_variable = target_idxs[order, :]
        target_variable = target_variable.to(device)
        target_tokens = [target_tokens[o] for o in order]

        output_log_probs, output_seqs = encoder_decoder(input_variable,
                                                        list(sorted_lengths),
                                                        targets=target_variable)

        batch_size = input_variable.shape[0]

        output_sentences = output_seqs.squeeze(2)

        flattened_outputs = output_log_probs.view(batch_size * max_length, -1)

        batch_outputs = trim_seqs(output_seqs)

        batch_inputs = [[list(seq[seq > 0])] for seq in list(to_np(input_variable))]

        batch_targets = [[list(seq[seq > 0])] for seq in list(to_np(target_variable))]

        for i in range(len(batch_outputs)):
            # Get the input and output tokens
            y_i = batch_outputs[i]
            tgt_i = batch_targets[i][0]
            src_i = batch_inputs[i][0]

            # Make dictionaries of the unknown tokens to their words
            src_token_list = input_tokens[i].split()
            tar_token_list = target_tokens[i].split()
            src_unk_to_tok = {src_i[j]:src_token_list[j] for j in range(len(src_i)) if not src_i[j] in encoder_decoder.lang.idx_to_tok}
            tar_unk_to_tok = {tgt_i[j]:tar_token_list[j] for j in range(len(tgt_i)) if not tgt_i[j] in encoder_decoder.lang.idx_to_tok}

            # Translate the tokens back to language
            correct_seq = [encoder_decoder.lang.idx_to_tok[n] if n in encoder_decoder.lang.idx_to_tok else tar_unk_to_tok[n] for n in tgt_i]
            incorrect_seq = [encoder_decoder.lang.idx_to_tok[n] if n in encoder_decoder.lang.idx_to_tok else src_unk_to_tok[n] for n in y_i]
            input_seq = [encoder_decoder.lang.idx_to_tok[n] if n in encoder_decoder.lang.idx_to_tok else src_unk_to_tok[n] for n in src_i]

            if correct_seq == incorrect_seq:
                correct_predictions += 1.0

            else:
                # Write wrong outputs to a log file
                if not log_file == None:
                    log_file.write("INPUT PHRASE:      {}\n".format(' '.join(input_seq)))
                    log_file.write("CORRECT LABEL:     {}\n".format(' '.join(correct_seq)))
                    log_file.write("PREDICTED LABEL:   {}\n".format(' '.join(incorrect_seq)))
                    log_file.write("\n\n")

            all_predictions += 1.0

    print('TESTING ACCURACY %.5f' % (100.0 * (correct_predictions / all_predictions)))


def main(model_name, use_cuda, batch_size, teacher_forcing_schedule, keep_prob, val_size, lr, decoder_type, vocab_limit, hidden_size, embedding_size, max_length, main_data, test_data, device, seed=42):
    print("Max Length is: ", max_length)
    model_path = './model/' + model_name + '/'

    print("training %s with use_cuda=%s, batch_size=%i"% (model_name, use_cuda, batch_size), flush=True)
    print("teacher_forcing_schedule=", teacher_forcing_schedule, flush=True)
    print("keep_prob=%f, val_size=%f, lr=%f, decoder_type=%s, vocab_limit=%i, hidden_size=%i, embedding_size=%i, max_length=%i, seed=%i" % (keep_prob, val_size, lr, decoder_type, vocab_limit, hidden_size, embedding_size, max_length, seed), flush=True)

    train_src, train_tgt, val_src, val_tgt = load_split_eighty_twenty(main_data, seed)

    if test_data:
        test_src, test_tgt = load_complete_data(test_data)
    
    if os.path.isdir(model_path):

        print("loading encoder and decoder from model_path", flush=True)
        encoder_decoder = torch.load(model_path + model_name + '_final.pt')

        print("creating training, validation, and testing datasets with saved languages", flush=True)
        train_dataset = SequencePairDataset(train_src, train_tgt,
                                            lang=encoder_decoder.lang,
                                            use_extended_vocab=(encoder_decoder.decoder_type=='copy'))

        val_dataset = SequencePairDataset(val_src, val_tgt,
                                          lang=encoder_decoder.lang,
                                          use_extended_vocab=(encoder_decoder.decoder_type=='copy'))

        test_dataset = SequencePairDataset(test_src, test_tgt,
                                            lang=train_dataset.lang,
                                            use_extended_vocab=(encoder_decoder.decoder_type=='copy'))

    else:
        os.mkdir(model_path)

        print("creating training, validation, and testing datasets", flush=True)
        train_dataset = SequencePairDataset(train_src, train_tgt,
                                            vocab_limit=vocab_limit,
                                            use_extended_vocab=(decoder_type=='copy'))

        val_dataset = SequencePairDataset(val_src, val_tgt,
                                          lang=train_dataset.lang,
                                          use_extended_vocab=(decoder_type=='copy'))

        test_dataset = SequencePairDataset(test_src, test_tgt,
                                            lang=train_dataset.lang,
                                            use_extended_vocab=(decoder_type=='copy'))

        print("creating encoder-decoder model", flush=True)
        encoder_decoder = EncoderDecoder(train_dataset.lang,
                                         max_length,
                                         embedding_size,
                                         hidden_size,
                                         decoder_type,
                                         device)

        torch.save(encoder_decoder, model_path + '/%s.pt' % model_name)

    encoder_decoder = encoder_decoder.to(device)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size)

    mixed_data_loader = DataLoader(mixed_dataset, batch_size=batch_size)

    trained_model = train(encoder_decoder,
          train_data_loader,
          model_name,
          val_data_loader,
          keep_prob,
          teacher_forcing_schedule,
          lr,
          encoder_decoder.decoder.max_length,
          device)

    trained_model = torch.load(model_path + model_name + '_final.pt')

    # Write final model errors to an output log
    if test_data:
        f = open("./logs/log_" + model_name + ".txt", "w")
        f.write("MODEL {}\n\n".format(model_name))
        f.write("UNSEEN ACCURACY\n")
        with torch.no_grad():
            accuracy = test(trained_model, test_data_loader, encoder_decoder.decoder.max_length, device, log_file=f)
        f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse training parameters')
    parser.add_argument('model_name', type=str,
                        help='the name of a subdirectory of ./model/ that '
                             'contains encoder and decoder model files')
    
    parser.add_argument('train_data', type=str,
                        help='The training data. Should be saved in the form/data/[name]_src/tar.txt')

    parser.add_argument('--epochs', type=int, default=50,
                        help='the number of epochs to train')

    parser.add_argument('--use_cuda', action='store_true',
                        help='flag indicating that cuda will be used')

    parser.add_argument('--batch_size', type=int, default=100,
                        help='number of examples in a batch')

    parser.add_argument('--teacher_forcing_fraction', type=float, default=0.5,
                        help='fraction of batches that will use teacher forcing during training')

    parser.add_argument('--scheduled_teacher_forcing', action='store_true',
                        help='Linearly decrease the teacher forcing fraction '
                             'from 1.0 to 0.0 over the specified number of epocs')

    parser.add_argument('--keep_prob', type=float, default=1.0,
                        help='Probablity of keeping an element in the dropout step.')

    parser.add_argument('--val_size', type=float, default=0.1,
                        help='fraction of data to use for validation')

    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')

    parser.add_argument('--decoder_type', type=str, default='copy',
                        help="Allowed values 'copy' or 'attn'")

    parser.add_argument('--vocab_limit', type=int, default=5000,
                        help='When creating a new Language object the vocab'
                             'will be truncated to the most frequently'
                             'occurring words in the training dataset.')

    parser.add_argument('--hidden_size', type=int, default=256,
                        help='The number of RNN units in the encoder. 2x this '
                             'number of RNN units will be used in the decoder')

    parser.add_argument('--embedding_size', type=int, default=128,
                        help='Embedding size used in both encoder and decoder')

    parser.add_argument('--max_length', type=int, default=200,
                        help='Sequences will be padded or truncated to this size.')    

    parser.add_argument('--test_data', type=str, default=None,
                        help='The data to test with. Should be saved in the form /data/[name]_src/tar.txt')

    args = parser.parse_args()

    writer = SummaryWriter('./logs/%s_%s' % (args.model_name, str(int(time.time()))))
    if args.scheduled_teacher_forcing:
        schedule = np.arange(1.0, 0.0, -1.0/args.epochs)
    else:
        schedule = np.ones(args.epochs) * args.teacher_forcing_fraction
        
    device = None
    if args.use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print("device is: ", device)

    main(args.model_name, args.use_cuda, args.batch_size, schedule, args.keep_prob, args.val_size, args.lr, args.decoder_type, args.vocab_limit, args.hidden_size, args.embedding_size, args.max_length, args.train_data, args.test_data, device)
