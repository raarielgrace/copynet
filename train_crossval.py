import argparse
import os
import time
import numpy as np
import re
import matplotlib.pyplot as plt
import datetime
import math

import torch
from torch import optim
#from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch import nn

from ltldataset import SequencePairDataset
from kfoldltldataset import OneFoldSequencePairDataset, generateKFoldDatasets
#from mjcdataset import SequencePairDataset
from model.encoder_decoder import EncoderDecoder
from evaluate import evaluate
from utils import to_np, trim_seqs, get_glove, load_complete_data

from tensorboardX import SummaryWriter
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

#torch.backends.cudnn.enabled = False
#torch.set_printoptions(profile="full")

def train(encoder_decoder: EncoderDecoder,
          train_data_loader: DataLoader,
          model_name,
          val_data_loader: DataLoader,
          keep_prob,
          teacher_forcing_schedule,
          lr,
          max_length,
          device,
          test_data_loader: DataLoader):

    global_step = 0
    loss_function = torch.nn.NLLLoss(ignore_index=0)
    optimizer = optim.Adam(encoder_decoder.parameters(), lr=lr)
    model_path = './model/' + model_name + '/'
    trained_model = encoder_decoder

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
            batch_outputs = trim_seqs(output_seqs)

            batch_inputs = [[list(seq[seq > 0])] for seq in list(to_np(input_variable))]
            batch_targets = [[list(seq[seq > 0])] for seq in list(to_np(target_variable))]

            for i in range(len(batch_outputs)):
                y_i = batch_outputs[i]
                tgt_i = batch_targets[i][0]

                if y_i == tgt_i:
                    correct_predictions += 1.0

                all_predictions += 1.0

            batch_loss.backward()
            optimizer.step()

            batch_bleu_score = corpus_bleu(batch_targets, batch_outputs, smoothing_function=SmoothingFunction().method1)

            if global_step % 100 == 0:

                writer.add_scalar('train_batch_loss', batch_loss, global_step)
                writer.add_scalar('train_batch_bleu_score', batch_bleu_score, global_step)

                for tag, value in encoder_decoder.named_parameters():
                    tag = tag.replace('.', '/')
                    writer.add_histogram('weights/' + tag, value, global_step, bins='doane')
                    writer.add_histogram('grads/' + tag, to_np(value.grad), global_step, bins='doane')

            global_step += 1

        #val_loss, val_bleu_score = evaluate(encoder_decoder, val_data_loader)

        #writer.add_scalar('val_loss', val_loss, global_step=global_step)
        #writer.add_scalar('val_bleu_score', val_bleu_score, global_step=global_step)

        encoder_embeddings = encoder_decoder.encoder.embedding.weight.data
        encoder_vocab = encoder_decoder.lang.tok_to_idx.keys()
        writer.add_embedding(encoder_embeddings, metadata=encoder_vocab, global_step=0, tag='encoder_embeddings')

        decoder_embeddings = encoder_decoder.decoder.embedding.weight.data
        decoder_vocab = encoder_decoder.lang.tok_to_idx.keys()
        writer.add_embedding(decoder_embeddings, metadata=decoder_vocab, global_step=0, tag='decoder_embeddings')

        print('training accuracy %.5f' % (100.0 * (correct_predictions / all_predictions)))
        #print('val loss: %.5f, val BLEU score: %.5f' % (val_loss, val_bleu_score), flush=True)

        '''
        val_acc = test(encoder_decoder, val_data_loader, max_length, device)
        print('validation accuracy {}'.format(val_acc))
        val_accus.append(val_acc)

        test_acc = test(encoder_decoder, test_data_loader, max_length, device)
        print('test accuracy {}'.format(test_acc))
        test_accus.append(test_acc)
        #test_struct_accus.append(test_struct_acc)
        '''
        torch.save(encoder_decoder, "%s%s_%i.pt" % (model_path, model_name, epoch))
        trained_model = encoder_decoder

        print('-' * 100, flush=True)

    torch.save(encoder_decoder, "%s%s_final.pt" % (model_path, model_name))
    return trained_model

def test(encoder_decoder: EncoderDecoder, test_data_loader: DataLoader, max_length, device, log_files=None):

    correct_predictions = 0.0
    struct_correct_only = 0.0
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
            y_i = batch_outputs[i]
            tgt_i = batch_targets[i][0]
            src_i = batch_inputs[i][0]
            src_token_list = input_tokens[i].split()
            tar_token_list = target_tokens[i].split()
            src_unk_to_tok = {src_i[j]:src_token_list[j] for j in range(len(src_i)) if not src_i[j] in encoder_decoder.lang.idx_to_tok}
            tar_unk_to_tok = {tgt_i[j]:tar_token_list[j] for j in range(len(tgt_i)) if not tgt_i[j] in encoder_decoder.lang.idx_to_tok}
            correct_seq = [encoder_decoder.lang.idx_to_tok[n] if n in encoder_decoder.lang.idx_to_tok else tar_unk_to_tok[n] for n in tgt_i]
            incorrect_seq = [encoder_decoder.lang.idx_to_tok[n] if n in encoder_decoder.lang.idx_to_tok else src_unk_to_tok[n] for n in y_i]
            input_seq = [encoder_decoder.lang.idx_to_tok[n] if n in encoder_decoder.lang.idx_to_tok else src_unk_to_tok[n] for n in src_i]

            if y_i == tgt_i:
                correct_predictions += 1.0

                if not log_files == None:
                    log_files[1].write("INPUT PHRASE:      {}\n".format(' '.join(input_seq)))
                    log_files[1].write("CORRECT LABEL:     {}\n".format(' '.join(correct_seq)))
                    log_files[1].write("PREDICTED LABEL:   {}\n".format(' '.join(incorrect_seq)))
                    log_files[1].write("\n\n")
            else:
                if not log_files == None:
                    log_files[0].write("INPUT PHRASE:      {}\n".format(' '.join(input_seq)))
                    log_files[0].write("CORRECT LABEL:     {}\n".format(' '.join(correct_seq)))
                    log_files[0].write("PREDICTED LABEL:   {}\n".format(' '.join(incorrect_seq)))
                    log_files[0].write("\n\n")


            all_predictions += 1.0
    return 100.0 * (correct_predictions / all_predictions)


def main(model_name, use_cuda, batch_size, teacher_forcing_schedule, keep_prob, val_size, lr, decoder_type, vocab_limit, hidden_size, embedding_size, max_length, train_data, test_data, device, seed=42):

    # TODO: Change logging to reflect loaded parameters
    model_path = './model/' + model_name
    print("training %s with use_cuda=%s, batch_size=%i"% (model_name, use_cuda, batch_size), flush=True)
    print("teacher_forcing_schedule=", teacher_forcing_schedule, flush=True)
    print("keep_prob=%f, val_size=%f, lr=%f, decoder_type=%s, vocab_limit=%i, hidden_size=%i, embedding_size=%i, max_length=%i, seed=%i" % (keep_prob, val_size, lr, decoder_type, vocab_limit, hidden_size, embedding_size, max_length, seed), flush=True)

    glove = get_glove()

    currentDT = datetime.datetime.now()
    seeds = [8, 23, 10, 41, 32]

    all_means_seen = []
    all_means_unseen = []
    all_means_mixed = []

    print('Testing using ' + test_data)

    mixed_src, mixed_tgt = load_complete_data('twophrase_1seen1unseen_clean')
    test_src, test_tgt = load_complete_data(test_data)

    # Repeat for the listed iterations #
    for it in range(5):
        print("creating training, and validation datasets", flush=True)
        all_datasets = generateKFoldDatasets(train_data, vocab_limit=vocab_limit, use_extended_vocab=(decoder_type=='copy'), seed=seeds[it])
        all_accuracies_seen = []
        all_accuracies_unseen = []
        all_accuracies_mixed = []

        # Repeat for all k dataset folds #
        for k in range(len(all_datasets)):
            train_dataset = all_datasets[k][0]
            val_dataset = all_datasets[k][1]

            print("creating {}th encoder-decoder model".format(k), flush=True)
            encoder_decoder = EncoderDecoder(train_dataset.lang,
                                            max_length,
                                            hidden_size,
                                            embedding_size,
                                            decoder_type,
                                            device,
                                            glove)

            test_dataset = SequencePairDataset(test_src, test_tgt,
                                            lang=train_dataset.lang,
                                            use_extended_vocab=(encoder_decoder.decoder_type=='copy'))

            mixed_dataset = SequencePairDataset(mixed_src, mixed_tgt,
                                            lang=train_dataset.lang,
                                            use_extended_vocab=(encoder_decoder.decoder_type=='copy'))

            encoder_decoder = encoder_decoder.to(device)

            train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
            val_data_loader = DataLoader(val_dataset, batch_size=batch_size)
            mixed_data_loader = DataLoader(mixed_dataset, batch_size=batch_size)
            test_data_loader = DataLoader(test_dataset, batch_size=batch_size)

            # Make a dir to hold this fold's model
            os.mkdir(model_path + str(k) + str(it) + '/')
            # Train the model
            train(encoder_decoder,
                train_data_loader,
                model_name + str(k) + str(it),
                val_data_loader,
                keep_prob,
                teacher_forcing_schedule,
                lr,
                encoder_decoder.decoder.max_length,
                device,
                test_data_loader)

            # Change the model path so the proper model can be loaded
            model_path = './model/' + model_name + str(k) + str(it) +  '/'
            # Load the trained model before testing, just in case
            trained_model = torch.load(model_path + model_name + '{}{}_final.pt'.format(k, it))

            ### WRITE TO ALL THE LOG FILES ##
            s_f = open("./logs/log_" + model_name + "seen" + currentDT.strftime("%Y%m%d%H%M%S") + ".txt", "w")
            s_f.write("TRAINING MODEL {}\nUSING SEED VALUE {}\n\n".format(model_name, seeds[it]))
            sc_f = open("./logs/log_" + model_name + "seencorrect" + currentDT.strftime("%Y%m%d%H%M%S") + ".txt", "w")
            seen_accuracy = test(trained_model, val_data_loader, encoder_decoder.decoder.max_length, device, log_files=(s_f, sc_f))
            s_f.close()
            sc_f.close()
        
            m_f = open("./logs/log_" + model_name + "1seen1unseen" + currentDT.strftime("%Y%m%d%H%M%S") + ".txt", "w")
            m_f.write("TRAINING MODEL {}\nUSING SEED VALUE {}\n\n".format(model_name, seeds[it]))
            mc_f = open("./logs/log_" + model_name + "1seen1unseencorrect" + currentDT.strftime("%Y%m%d%H%M%S") + ".txt", "w")
            mixed_accuracy = test(trained_model, mixed_data_loader, encoder_decoder.decoder.max_length, device, log_files=(m_f, mc_f))
            m_f.close()
            mc_f.close()

            u_f = open("./logs/log_" + model_name + "unseen" + currentDT.strftime("%Y%m%d%H%M%S") + ".txt", "w")
            u_f.write("TRAINING MODEL {}\nUSING SEED VALUE {}\n\n".format(model_name, seeds[it]))
            uc_f = open("./logs/log_" + model_name + "unseencorrect" + currentDT.strftime("%Y%m%d%H%M%S") + ".txt", "w")
            unseen_accuracy = test(trained_model, test_data_loader, encoder_decoder.decoder.max_length, device, log_files=(u_f, uc_f))
            u_f.close()
            uc_f.close()
        
            # Append the accuracies for this model so they can be averaged
            all_accuracies_seen.append(seen_accuracy)
            all_accuracies_unseen.append(unseen_accuracy)
            all_accuracies_mixed.append(mixed_accuracy)

            # Reset model path
            model_path = './model/' + model_name
        
        # Average all the accuracies for this round of cross val and record the mean
        s_mean = sum(all_accuracies_seen) / len(all_accuracies_seen)
        m_mean = sum(all_accuracies_mixed) / len(all_accuracies_mixed)
        u_mean = sum(all_accuracies_unseen) / len(all_accuracies_unseen)
        all_means_seen.append(s_mean)
        all_means_mixed.append(m_mean)
        all_means_unseen.append(u_mean)

    # PRINT FINAL CROSSVAL RESULTS #
    currentDT = datetime.datetime.now()
    acc_f = open("./results/results_" + model_name + currentDT.strftime("%Y%m%d%H%M%S") + ".txt", "w")

    # Seen accuracies
    acc_f.write("SEEN ACCURACIES:\n")
    for acc in all_means_seen:
        acc_f.write("{}\n".format(acc))

    s_mean = sum(all_means_seen) / len(all_means_seen)
    s_std_dev = math.sqrt(sum([math.pow(x - s_mean, 2) for x in all_means_seen]) / len(all_means_seen))
    acc_f.write("\nMean: {}\nStandard Deviation: {}\n".format(s_mean, s_std_dev))

    # One seen one unseen accuracies
    acc_f.write("ONE SEEN ONE UNSEEN ACCURACIES:\n")
    for acc in all_means_mixed:
        acc_f.write("{}\n".format(acc))

    m_mean = sum(all_means_mixed) / len(all_means_mixed)
    m_std_dev = math.sqrt(sum([math.pow(x - m_mean, 2) for x in all_means_mixed]) / len(all_means_mixed))
    acc_f.write("\nMean: {}\nStandard Deviation: {}\n".format(m_mean, m_std_dev))

    # Unseen accuracies
    acc_f.write("\nUNSEEN ACCURACIES:\n")
    for acc in all_means_unseen:
        acc_f.write("{}\n".format(acc))

    u_mean = sum(all_means_unseen) / len(all_means_unseen)
    u_std_dev = math.sqrt(sum([math.pow(x - u_mean, 2) for x in all_means_unseen]) / len(all_means_unseen))
    acc_f.write("\nMean: {}\nStandard Deviation: {}\n".format(u_mean, u_std_dev))
    acc_f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse training parameters')
    parser.add_argument('model_name', type=str,
                        help='the name of a subdirectory of ./model/ that '
                             'contains encoder and decoder model files')

    parser.add_argument('train_data', type=str, default=None,
                        help='The training data. Should be of the form /data/[name]_src/tar.txt')

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

    parser.add_argument('--seed', type=int, default=42,
                        help='The seed for the randomizer.')

    parser.add_argument('--test_data', type=str, default=None,
                        help='The data to use as testing data. Should be of the form /data/[name]_src/tar.txt')

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

    main(args.model_name, args.use_cuda, args.batch_size, schedule, args.keep_prob, args.val_size, args.lr, args.decoder_type, args.vocab_limit, args.hidden_size, args.embedding_size, args.max_length, args.train_data, args.test_data, device, args.seed)
