import argparse
import os
import time
import numpy as np
import re
import matplotlib.pyplot as plt

import torch
from torch import optim
#from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch import nn

from ltldataset import SequencePairDataset
#from mjcdataset import SequencePairDataset
from model.encoder_decoder import EncoderDecoder
from evaluate import evaluate
from utils import to_np, trim_seqs, get_glove

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
    #f = open("losses.txt", "w")
    trained_model = encoder_decoder
    epochs = []
    train_accus = []
    val_accus = []
    test_accus = []
    test_struct_accus = []
    for epoch, teacher_forcing in enumerate(teacher_forcing_schedule):
        print('epoch %i' % epoch, flush=True)
        #f.write('epoch {}\n'.format(epoch))
        correct_predictions = 0.0
        all_predictions = 0.0
        epochs.append(epoch)
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
            #f.write('{}\n'.format(batch_loss))
            batch_outputs = trim_seqs(output_seqs)

            batch_inputs = [[list(seq[seq > 0])] for seq in list(to_np(input_variable))]
            batch_targets = [[list(seq[seq > 0])] for seq in list(to_np(target_variable))]

            wrong_landmarks = 0.0
            for i in range(len(batch_outputs)):
                y_i = batch_outputs[i]
                tgt_i = batch_targets[i][0]

                if y_i == tgt_i:
                    correct_predictions += 1.0
                #else:
                #    src_i = batch_inputs[i][0]
                #    src_token_list = input_tokens[i].split()
                #    tar_token_list = target_tokens[i].split()
                #    src_unk_to_tok = {src_i[j]:src_token_list[j] for j in range(len(src_i)) if not src_i[j] in encoder_decoder.lang.idx_to_tok}
                #    tar_unk_to_tok = {tgt_i[j]:tar_token_list[j] for j in range(len(tgt_i)) if not tgt_i[j] in encoder_decoder.lang.idx_to_tok}
                #    correct_seq = [encoder_decoder.lang.idx_to_tok[n] if n in encoder_decoder.lang.idx_to_tok else tar_unk_to_tok[n] for n in tgt_i]
                #    incorrect_seq = [encoder_decoder.lang.idx_to_tok[n] if n in encoder_decoder.lang.idx_to_tok else src_unk_to_tok[n] for n in y_i]
                #    if not correct_seq == incorrect_seq:
                #        wrong_landmarks += 1.0

                all_predictions += 1.0

            #wrong_percent = wrong_landmarks / len(batch_outputs)
            #batch_loss = batch_loss + wrong_percent

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

        train_acc = 100.0 * (correct_predictions / all_predictions)
        print('training accuracy %.5f' % (100.0 * (correct_predictions / all_predictions)))
        train_accus.append(train_acc)
        #print('val loss: %.5f, val BLEU score: %.5f' % (val_loss, val_bleu_score), flush=True)

        val_acc = test(encoder_decoder, val_data_loader, max_length, device)
        print('validation accuracy {}'.format(val_acc))
        val_accus.append(val_acc)

        test_acc = test(encoder_decoder, test_data_loader, max_length, device)
        print('test accuracy {}'.format(test_acc))
        test_accus.append(test_acc)
        #test_struct_accus.append(test_struct_acc)

        torch.save(encoder_decoder, "%s%s_%i.pt" % (model_path, model_name, epoch))
        trained_model = encoder_decoder

        print('-' * 100, flush=True)

    #f.close()
    plt.plot(epochs, train_accus, marker='o', label='Training')
    plt.plot(epochs, val_accus, marker='^', label='Validation')
    plt.plot(epochs, test_accus, marker='>', label='Testing')
    #plt.plot(epochs, test_struct_accus, marker='<', label='Test LTL Only')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig('plot.png')
    return trained_model

def test(encoder_decoder: EncoderDecoder, test_data_loader: DataLoader, max_length, device):
    
    correct_predictions = 0.0
    struct_correct_only = 0.0
    all_predictions = 0.0
    #f = open("issues.txt", "w")
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

            if y_i == tgt_i:
                correct_predictions += 1.0
            else:
                src_i = batch_inputs[i][0]
                src_token_list = input_tokens[i].split()
                tar_token_list = target_tokens[i].split()
                src_unk_to_tok = {src_i[j]:src_token_list[j] for j in range(len(src_i)) if not src_i[j] in encoder_decoder.lang.idx_to_tok}
                tar_unk_to_tok = {tgt_i[j]:tar_token_list[j] for j in range(len(tgt_i)) if not tgt_i[j] in encoder_decoder.lang.idx_to_tok}
                correct_seq = [encoder_decoder.lang.idx_to_tok[n] if n in encoder_decoder.lang.idx_to_tok else tar_unk_to_tok[n] for n in tgt_i]
                incorrect_seq = [encoder_decoder.lang.idx_to_tok[n] if n in encoder_decoder.lang.idx_to_tok else src_unk_to_tok[n] for n in y_i]
                c_minus_ldmks = re.sub('lm(.+?)lm', '',  ' '.join(correct_seq))
                i_minus_ldmks = re.sub('lm(.+?)lm', '',  ' '.join(incorrect_seq))

                #if c_minus_ldmks == i_minus_ldmks:
                    #struct_correct_only += 1.0

                #f.write("CORRECT PLACES: {}\n".format(c_result))
                #f.write("INCORRECT PLACES: {}\n".format(i_result))
                #f.write("-----------------------------------------------------------------------------------------------\n")


            all_predictions += 1.0
    #f.close()
    return 100.0 * (correct_predictions / all_predictions)


def main(model_name, use_cuda, batch_size, teacher_forcing_schedule, keep_prob, val_size, lr, decoder_type, vocab_limit, hidden_size, embedding_size, max_length, save_lang, test_data_substitute,device, seed=42):

    model_path = './model/' + model_name + '/'

    # TODO: Change logging to reflect loaded parameters

    print("training %s with use_cuda=%s, batch_size=%i"% (model_name, use_cuda, batch_size), flush=True)
    print("teacher_forcing_schedule=", teacher_forcing_schedule, flush=True)
    print("keep_prob=%f, val_size=%f, lr=%f, decoder_type=%s, vocab_limit=%i, hidden_size=%i, embedding_size=%i, max_length=%i, seed=%i" % (keep_prob, val_size, lr, decoder_type, vocab_limit, hidden_size, embedding_size, max_length, seed), flush=True)

    glove = get_glove()
    if os.path.isdir(model_path):

        print("loading encoder and decoder from model_path", flush=True)
        encoder_decoder = torch.load(model_path + model_name + '.pt')

        print("creating training, validation, and testing datasets with saved languages", flush=True)
        train_dataset = SequencePairDataset(lang=encoder_decoder.lang,
                                            use_cuda=use_cuda,
                                            is_val=False,
                                            is_test=False,
                                            val_size=val_size,
                                            use_extended_vocab=(encoder_decoder.decoder_type=='copy'))

        val_dataset = SequencePairDataset(lang=encoder_decoder.lang,
                                          use_cuda=use_cuda,
                                          is_val=True,
                                          is_test=False,
                                          val_size=val_size,
                                          use_extended_vocab=(encoder_decoder.decoder_type=='copy'))

        test_dataset = SequencePairDataset(lang=train_dataset.lang,
                                            use_cuda=use_cuda,
                                            is_val=False,
                                            is_test=True,
                                            val_size=val_size,
                                            use_extended_vocab=(encoder_decoder.decoder_type=='copy'),
                                            data_substitute=test_data_substitute)

    else:
        os.mkdir(model_path)

        print("creating training, validation, and testing datasets", flush=True)
        train_dataset = SequencePairDataset(vocab_limit=vocab_limit,
                                            use_cuda=use_cuda,
                                            is_val=False,
                                            is_test=False,
                                            val_size=val_size,
                                            seed=seed,
                                            use_extended_vocab=(decoder_type=='copy'))

        val_dataset = SequencePairDataset(lang=train_dataset.lang,
                                          use_cuda=use_cuda,
                                          is_val=True,
                                          is_test=False,
                                          val_size=val_size,
                                          seed=seed,
                                          use_extended_vocab=(decoder_type=='copy'))

        test_dataset = SequencePairDataset(vocab_limit=vocab_limit,
                                            use_cuda=use_cuda,
                                            is_val=False,
                                            is_test=True,
                                            val_size=val_size,
                                            seed=seed,
                                            use_extended_vocab=(decoder_type=='copy'))

        print("creating encoder-decoder model", flush=True)
        encoder_decoder = EncoderDecoder(train_dataset.lang,
                                         max_length,
                                         hidden_size,
                                         embedding_size,
                                         decoder_type,
                                         device,
                                         glove)

        torch.save(encoder_decoder, model_path + '/%s.pt' % model_name)

    # CHANGE 2
    # #########################
    # if use_cuda and torch.cuda.device_count() > 1:
    #     print("Using ", torch.cuda.device_count(), " GPUs.")
    #     encoder_decoder = nn.DataParallel(encoder_decoder)
    encoder_decoder = encoder_decoder.to(device)
    #########################

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size)

    trained_model = train(encoder_decoder,
          train_data_loader,
          model_name,
          val_data_loader,
          keep_prob,
          teacher_forcing_schedule,
          lr,
          encoder_decoder.decoder.max_length,
          device,
          test_data_loader)

    print('TESTING ACCURACY %.5f' % test(trained_model, test_data_loader, encoder_decoder.decoder.max_length, device)[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse training parameters')
    parser.add_argument('model_name', type=str,
                        help='the name of a subdirectory of ./model/ that '
                             'contains encoder and decoder model files')

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

    parser.add_argument('--save_lang', action='store_true',
                        help='Flag to save the training vocabulary to a pkl file.')

    parser.add_argument('--test_data_substitute', type=str, default=None,
                        help='The data to use instead of the training data. Should be of the form /data/[name]_src/tar.txt')

    args = parser.parse_args()

    writer = SummaryWriter('./logs/%s_%s' % (args.model_name, str(int(time.time()))))
    if args.scheduled_teacher_forcing:
        schedule = np.arange(1.0, 0.0, -1.0/args.epochs)
    else:
        schedule = np.ones(args.epochs) * args.teacher_forcing_fraction

    # CHANGE 1
    #########################
    device = None
    if args.use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print("device is: ", device)

    main(args.model_name, args.use_cuda, args.batch_size, schedule, args.keep_prob, args.val_size, args.lr, args.decoder_type, args.vocab_limit, args.hidden_size, args.embedding_size, args.max_length, args.save_lang, args.test_data_substitute, device)
