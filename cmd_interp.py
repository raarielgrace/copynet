import io
import os
import argparse
import gensim
import nltk
from scipy.spatial.distance import cosine
from gensim.models import FastText
import numpy as np
import pandas as pd
import json
import torch
from utils import trim_seqs

def load_json(path):
	"""
	A function to load a JSON file specifically extracted from OSM.

	input: path, str, a path where the JSON file is located
	output: sem_dict, dict, a dictionary where the keys are landmark names and
							the values are dictionaries made of semantic tags
							with values
	"""
	sem_dict = {}
	with open(path, 'r') as f:
		sem_dict = json.load(f)
	# f.close() ?
	return sem_dict

def get_phrase_vec(wrd_lst, w2v):
	"""
	Gets the vector representation of a phrase, meaning a list of strings.
	Does it by averaging the vectors of the words in the list.

	input:
		- wrd_lst, str list, a list of words
		- w2v, dict, a dictionary from words to their vectors
	output: vector, numpy array, the vector representation of the phrase
	"""
	phrase_size = len(wrd_lst)
	vector = np.copy(w2v[wrd_lst[0]])
	for i in range(1, phrase_size):
		new_w = np.copy(w2v[wrd_lst[i]])
		vector += new_w
	vector = vector / phrase_size
	return vector

def sem_dict2lists(sem_dict, word2vec, use_sem):
	"""
	A function to extract an ordered list of landmark names and landmark
	semantic vectors to be used when we would like to get the most similar
	landmark.

	input:
		- sem_dict, dict, a dictionary where the keys are landmark names and the
						  values are dictionaries made of semantic tags with
						  values
		- word2vec, the loaded word2vec/fastText model
	output:
		- lm_list, list, an ordered list of the landmarks' name (which is a
						 string list)
		- vec_lm_list, list, an ordered list of the landmarks' semantic vectors
	"""
	lm_list = []
	vec_lm_list = []
	for lm in sem_dict:
		curr_dict = sem_dict[lm]
		if use_sem:
			list_values = [v.lower().strip().replace("_", " ").split() for v in curr_dict.values()]
			list_values = [item for sublist in list_values for item in sublist]
			lm_vec = get_phrase_vec(list_values, word2vec)
			vec_lm_list.append(lm_vec)
		else:
			lm_vec = get_phrase_vec(lm.lower().strip().split(), word2vec)
			vec_lm_list.append(lm_vec)
		lm_list.append(lm.lower().strip().split())
	return (lm_list, vec_lm_list)

def tokens_to_seq(tokens, tok_to_idx, max_length, use_extended_vocab, input_tokens=None):
	"""
	TODO: document
	"""
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
									  if input_token == token), 3))
			idx = tok_to_idx_extension[token]
		else:
			# unknown tokens in the input sequence use the position of the first occurence + vocab_size as their index
			idx = pos + len(tok_to_idx)
		seq[pos] = idx
	return seq

def most_similar_landmark(target, w2v, vec_lm_list):
	"""
	Finds the most similar landmark index to the target inside the vec_lm_list.

	input:
		- target, str list, a landmark phrase
		- w2v, dict, a dictionary from words to their vectors
		- vec_lm_list, list, an ordered list of the landmarks' semantic vectors
	"""
	target_vec = get_phrase_vec(target, w2v)
	dist_list = []
	for i, lm in enumerate(vec_lm_list):
		dist = cosine(target_vec, lm)
		dist_list.append((i, dist))
	dist_list.sort(key=lambda x: x[1])
	return dist_list[0][0]

def get_true_formula(ltl_list, word2vec, lm_list, vec_lm_list):
	"""
	Turns an ltl_list into a parsable ltl string.
	Honestly, here just for the sake of future error handling, get_true_formula()
	does most of the work.

	input:
		- ltl_list, str list, a list of LTL tokens
		- word2vec, dict, a dictionary from words to their vectors
		- lm_list, list, an ordered list of the landmarks' name (which is a
						 string list)
		- vec_lm_list, list, an ordered list of the landmarks' semantic vectors
	output: either "error" or the most probable LTL formula with the list of
			landmarks in the formula
	"""
	start_idx_lst = []
	end_idx_lst = []
	cnt = 0
	for i in ltl_list:
		if i == "lm(":
			start_idx_lst.append(cnt)
		elif i == ")lm":
			end_idx_lst.append(cnt)
		cnt += 1
	if len(start_idx_lst) == 1 and len(end_idx_lst) == 1:
		idx1 = start_idx_lst[0] + 1
		idx2 = end_idx_lst[0]
		lm1 = ltl_list[idx1:idx2]
		lm1 = [lm.lower() for lm in lm1]

		sim_lm1 = most_similar_landmark(lm1, word2vec, vec_lm_list)

		formatted_sim_lm1 = "_".join(lm_list[sim_lm1])
		ltl_list[idx1 - 1 :idx2 + 1] = []
		ltl_list.insert(idx1 - 1, formatted_sim_lm1)
		new_formula = " ".join(ltl_list)
		return (new_formula, [lm1])
	elif len(start_idx_lst) == 2 and len(end_idx_lst) == 2:
		idx1 = start_idx_lst[0] + 1
		idx2 = end_idx_lst[0]
		lm1 = ltl_list[idx1:idx2]
		lm1 = [lm.lower() for lm in lm1]
		idx3 = start_idx_lst[1] + 1
		idx4 = end_idx_lst[1]
		lm2 = ltl_list[idx3:idx4]
		lm2 = [lm.lower() for lm in lm2]

		sim_lm1 = most_similar_landmark(lm1, word2vec, vec_lm_list)
		sim_lm2 = most_similar_landmark(lm2, word2vec, vec_lm_list)

		formatted_sim_lm1 = "_".join(lm_list[sim_lm1])
		formatted_sim_lm2 = "_".join(lm_list[sim_lm2])
		ltl_list[idx3 - 1 :idx4 + 1] = []
		ltl_list.insert(idx3 - 1, formatted_sim_lm2)
		ltl_list[idx1 - 1 :idx2 + 1] = []
		ltl_list.insert(idx1 - 1, formatted_sim_lm1)
		new_formula = " ".join(ltl_list)
		return (new_formula, [lm1, lm2])
	else:
		return ("error", [])

def main(ltl_list, word2vec, lm_list, vec_lm_list):
	"""
	Turns an ltl_list into a parsable ltl string.
	Honestly, here just for the sake of future error handling, get_true_formula()
	does most of the work.

	input:
		- ltl_list, str list, a list of LTL tokens
		- word2vec, dict, a dictionary from words to their vectors
		- lm_list, list, an ordered list of the landmarks' name (which is a
						 string list)
		- vec_lm_list, list, an ordered list of the landmarks' semantic vectors
	output: either print's that there is an error or outputs the most probable
			LTL formula and returns the list of landmarks in the formula along
			with it
	"""
	output = get_true_formula(ltl_list, word2vec, lm_list, vec_lm_list)
	if output[0] == "error":
		print("ERROR: Command not supported. It can only contain 1 <= f <= 2 landmarks.")
		return output
	else:
		print("The closest LTL formula is: ", output[0])
		print("The list of landmarks in the LTL formula is: ", output[1])
		return output

if __name__ == "__main__":
	"""
	A python program to translate a NL command into a parsable LTL formula.
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument("-cmd", type=str, action='store', help="A natural language command.", default="Go to Kabob and Curry.")
	parser.add_argument("-lms_path", type=str, action='store', help="Path to a JSON file containing information about landmarks extracted from OSM.", default="fasttext/all_landmarks_sem.json")
	parser.add_argument("-copynet_path", type=str, action='store', help="Path to a saved pt file of the CopyNet model.", default="./model/onephrase_propercrossval12/onephrase_propercrossval12_final.pt")
	parser.add_argument("-word2vec_path", type=str, action='store', help="Path to a saved bin file of the fastText model.", default="fasttext/cc_en_300.bin")
	parser.add_argument('-use_sem', action='store_true', help='Flag indicating that semantics will be used for landmark representation.', default=False)
	parser.add_argument('-use_cuda', action='store_true', help='Flag indicating that cuda will be used.', default=False)
	args = parser.parse_args()
	#################################################

	print("The command is: ", args.cmd)
	print("Status of using semantics is: ", args.use_sem)

	# 1) Loading language models:
	word2vec_model = FastText.load_fasttext_format(args.word2vec_path)
	# TODO: make it work across devices
	device = None
	if args.use_cuda:
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')
	copynet_model = torch.load(args.copynet_path, map_location=device)
	copynet_model.eval()

	# output_string = copynet_model.get_response(args.cmd)
	# print("LTL formula is: ", output_string)

	# 3) Translating the cmd into a list of LTL formulae:
	maxlen = copynet_model.decoder.max_length
	input = args.cmd.replace('\n', '').split(' ')
	input_token_list = (['<SOS>'] + input + ['<EOS>'])[:maxlen]
	input_seq = tokens_to_seq(input_token_list, copynet_model.lang.tok_to_idx, maxlen, True)
	# print(input_seq)
	# print(torch.LongTensor([len(input_token_list)]).shape)

	input_idxs = input_seq.unsqueeze(0)

	lengths = (input_idxs != 0).long().sum(dim=1)
	sorted_lengths, order = torch.sort(lengths, descending=True)

	input_variable = input_idxs[order, :][:, :max(lengths)]
	input_variable = input_variable.to(device)

	_, output_seq = copynet_model(input_variable, list(sorted_lengths))
	output_sentence = trim_seqs(output_seq)[0]

	src_unk_to_tok = {input_seq[i].item():input_token_list[i] for i in range(len(input_token_list)) if not input_seq[i] in copynet_model.lang.idx_to_tok}

	ltl_list = [copynet_model.lang.idx_to_tok[n] if n in copynet_model.lang.idx_to_tok else src_unk_to_tok[n] for n in output_sentence]
	# print("ltl_list before filtering is: ", ltl_list)
	ltl_list = list(filter(lambda x: not(x == '<SOS>') and not(x == '<EOS>'), ltl_list))
	print("CopyNet's output ltl_list is: ", ltl_list)

	# 2) Loading landmarks and their semantic information into ordered lists:
	# TODO: better pre-processing for the sem_dict
	lm_list, vec_lm_list = sem_dict2lists(load_json(args.lms_path), word2vec_model, args.use_sem)
	# print("lm_list is: ", lm_list)
	# print("vec_lm_list is: ", vec_lm_list)

	main(ltl_list, word2vec_model, lm_list, vec_lm_list)
