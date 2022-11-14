import numpy as np
import pandas as pd
import random
import re
import os
import sys
import argparse
import json
import subprocess

from wg_utils.lex_remove import LexRemover


def read_from_csv(datafile):
	df = pd.read_csv(datafile)
	rules = df['Rules'].dropna(how='all')
	vowels = df.iloc[:1,1:].dropna(axis=1)
	vowel_list = vowels.values.flatten().tolist()
	vowel_weights = df.iloc[1:2,1:len(vowel_list)+1].values.flatten().tolist()
	# Replace null consonant nan with underscore:
	if df.iloc[2:3,1:2].isnull().values.any():
		df.iloc[2:3,1:2] = '_'
	consonants = df.iloc[2:3,1:].dropna(axis=1)
	cons_list = consonants.values.flatten().tolist()
	cons_weights_map = {}
	idx = 0
	for rule in rules.values.flatten().tolist():
		cons_weights = []
		if len(rule.split(',')) > 1:
			cons_weights = df.iloc[idx:idx+1,1:len(cons_list)+1].fillna(value=0.0).values.flatten().tolist()
			cons_weights = [float(i) for i in cons_weights]
			cons_weights_map[rule] = cons_weights
		idx += 1
	vowel_df = pd.DataFrame(vowel_weights, index=vowel_list)
	cons_df = pd.DataFrame(cons_weights_map, index=cons_list)
	return vowel_df, cons_df


def random(chars, weights):
	normw = [w/sum(weights) for w in weights]
	choice = np.random.choice(chars, 1, p=normw)
	return choice


def get_weights(df, rule):
	return [float(i) for i in df[rule].tolist()]


def generate_nucleus(vowel_df):
	return random(list(vowel_df.index), [float(i) for i in vowel_df.iloc[:,0].tolist()])


def generate_onset(oldsyl, cons_df):
	onsc1 = ['' if c == '_' else c for c in list(cons_df.index)]
	# First syllable: #
	if oldsyl == '':
		onsc1weights = get_weights(cons_df, "onsc1,first syl")
		return random(onsc1, onsc1weights)
	# After first syllable: #
	else:
		if oldsyl[0] == 'p' or oldsyl[0] == 'b' or oldsyl[0] == 'f':
			onsc1weights = get_weights(cons_df, "onsc1,prec labial ons")
			return random(onsc1, onsc1weights)
		elif oldsyl[0] == 't' or oldsyl[0] == 'd' or oldsyl[0] == 's':
			onsc1weights = get_weights(cons_df, "onsc1,prec alveolar ons")
			return random(onsc1, onsc1weights)
		elif oldsyl[0] == 'k' or oldsyl[0] == 'g':
			onsc1weights = get_weights(cons_df, "onsc1,prec velar ons")
			return random(onsc1, onsc1weights)
		elif oldsyl[0] == 'm' or oldsyl[0] == 'n' or oldsyl[0] == 'ɲ' or oldsyl[0] == 'ŋ':
			onsc1weights = get_weights(cons_df, "onsc1,prec nasal ons")
			return random(onsc1, onsc1weights)
		elif oldsyl[0] == 'l' or oldsyl[0] == 'r':
			onsc1weights = get_weights(cons_df, "onsc1,prec liquid ons")
			return random(onsc1, onsc1weights)
		elif oldsyl[0] == 'z' or oldsyl[0] == 'ʃ' or oldsyl[0] == 'ʧ':
			onsc1weights = get_weights(cons_df, "onsc1,prec sibilant ons")
			return random(onsc1, onsc1weights)
		elif oldsyl[0] == 'h' or oldsyl[0] == 'ʔ':
			onsc1weights = get_weights(cons_df, "onsc1,prec glottal ons")
			return random(onsc1, onsc1weights)
		elif oldsyl[0] == 'w' or oldsyl[0] == 'j':
			onsc1weights = get_weights(cons_df, "onsc1,prec semivowel ons")
			return random(onsc1, onsc1weights)
		else:
			onsc1weights = get_weights(cons_df, "onsc1,prec no ons")
			return random(onsc1, onsc1weights)


def generate_coda(oldsyl, cons_df, syl_idx, sylnum):
	coda = ['' if c == '_' else c for c in list(cons_df.index)]
	# Monosyllabic word:
	if sylnum == 1:
		codaweights = get_weights(cons_df, "coda,monosyllable")
		return random(coda, codaweights)
	# Before penultimate syllable: #
	if syl_idx < sylnum-2:
		if len(oldsyl) > 2:
			if oldsyl[2] == 'm' or oldsyl[2] == 'n' or oldsyl[2] == 'ŋ':
				codaweights = get_weights(cons_df, "coda,before penult prec nasal coda")
				return random(coda, codaweights)
			elif oldsyl[2] == 'p':
				codaweights = get_weights(cons_df, "coda,before penult prec p coda")
				return random(coda, codaweights)
			elif oldsyl[2] == 't':
				codaweights = get_weights(cons_df, "coda,before penult prec t coda")
				return random(coda, codaweights)
			elif oldsyl[2] == 'k':
				codaweights = get_weights(cons_df, "coda,before penult prec k coda")
				return random(coda, codaweights)
			elif oldsyl[2] == 's':
				codaweights = get_weights(cons_df, "coda,before penult prec s coda")
				return random(coda, codaweights)
			elif oldsyl[2] == 'l' or oldsyl[2] == 'r':
				codaweights = get_weights(cons_df, "coda,before penult prec liquid coda")
				return random(coda, codaweights)
			else:
				codaweights = get_weights(cons_df, "coda,before penult prec semivowel coda")
				return random(coda, codaweights)
		else:
			codaweights = get_weights(cons_df, "coda,before penult prec no coda")
			return random(coda, codaweights)
	# Penultimate syllable: #
	if syl_idx == sylnum-2:
		if len(oldsyl) > 2:
			if oldsyl[2] == 'm' or oldsyl[2] == 'n' or oldsyl[2] == 'ŋ':
				codaweights = get_weights(cons_df, "coda,penult prec nasal coda")
				return random(coda, codaweights)
			elif oldsyl[2] == 'p':
				codaweights = get_weights(cons_df, "coda,penult prec p coda")
				return random(coda, codaweights)
			elif oldsyl[2] == 't':
				codaweights = get_weights(cons_df, "coda,penult prec t coda")
				return random(coda, codaweights)
			elif oldsyl[2] == 'k':
				codaweights = get_weights(cons_df, "coda,penult prec k coda")
				return random(coda, codaweights)
			elif oldsyl[2] == 's':
				codaweights = get_weights(cons_df, "coda,penult prec s coda")
				return random(coda, codaweights)
			elif oldsyl[2] == 'l' or oldsyl[2] == 'r':
				codaweights = get_weights(cons_df, "coda,penult prec liquid coda")
				return random(coda, codaweights)
			else:
				codaweights = get_weights(cons_df, "coda,penult prec semivowel coda")
				return random(coda, codaweights)
		else:
			codaweights = get_weights(cons_df, "coda,penult prec no coda")
			return random(coda, codaweights)
	# Final syllable: #
	if syl_idx == sylnum-1:
		if len(oldsyl) > 2:
			if oldsyl[2] == 'm' or oldsyl[2] == 'n' or oldsyl[2] == 'ŋ':
				codaweights = get_weights(cons_df, "coda,last syl prec nasal coda")
				return random(coda, codaweights)
			elif oldsyl[2] == 'p':
				codaweights = get_weights(cons_df, "coda,last syl prec p coda")
				return random(coda, codaweights)
			elif oldsyl[2] == 't':
				codaweights = get_weights(cons_df, "coda,last syl prec t coda")
				return random(coda, codaweights)
			elif oldsyl[2] == 'k':
				codaweights = get_weights(cons_df, "coda,last syl prec k coda")
				return random(coda, codaweights)
			elif oldsyl[2] == 's':
				codaweights = get_weights(cons_df, "coda,last syl prec s coda")
				return random(coda, codaweights)
			elif oldsyl[2] == 'l' or oldsyl[2] == 'r':
				codaweights = get_weights(cons_df, "coda,last syl prec liquid coda")
				return random(coda, codaweights)
			else:
				codaweights = get_weights(cons_df, "coda,last syl prec semivowel coda")
				return random(coda, codaweights)
		else:
			codaweights = get_weights(cons_df, "coda,last syl prec no coda")
			return random(coda, codaweights)


def generate_words(vowel_df, cons_df, sylnum, outputlines):
	wordlist = []
	for i in range (0, outputlines):
		word = ''
		# Do this for each syllable: #
		if sylnum == 0:
			sylnum_weights = [5,10,7.5,4,1.5,0.1,0.01]
			temp_sylnum = np.random.choice([1,2,3,4,5,6,7], p=[w/sum(sylnum_weights) for w in sylnum_weights])
		else:
			temp_sylnum = sylnum
		for j in range (0, temp_sylnum):
			if j == 0:
				oldsyl = ''
			syl = ''
			nuc = generate_nucleus(vowel_df)
			onsc1 = generate_onset(oldsyl, cons_df)
			coda = generate_coda(oldsyl, cons_df, j, temp_sylnum)
			# Write to output file: ##
			if temp_sylnum == 1:
				syl += onsc1[0] + nuc[0] + coda[0]
			else:
				syl += onsc1[0] + nuc[0] + coda[0] + '.'
			oldsyl = syl
			word += syl
		wordlist.append(word)
	return wordlist


def write_file(wordlist, outfile):
	# Writes a wordlist output file: #
	with open(outfile, "w") as f:
		for i in range (0, len(wordlist)):
			f.write(wordlist[i] + '\n')


def post_process(patterns, infile, outfile):
	# Load patterns from file:
	with open(patterns, "r") as patf:
		pats = json.load(patf)
	# Post-process output: #
	with open(infile, "r") as f1:
		seen = set()
		with open(outfile, "w") as f2:
			for line in f1:
				if line not in seen:
					seen.add(line)
					for k, v in pats.items():
						sub = re.sub(re.compile(k), v, line)
						line = sub
					f2.write(line)


def clean_up():
	# Trash intermediate output:
	if os.path.exists("output.txt"):
		os.remove("output.txt")


def clean_up_ascii(wordlist_fname):
	# Replace ASCII output with wordlist filename:
	os.remove(wordlist_fname)
	os.rename("ascii_output.txt", wordlist_fname)


def sample(wordlist, outputlines):
	return np.random.choice(wordlist, outputlines, replace=False)


def sample_run(vowel_df, cons_df, sylnum, outputlines, sample_n):
	full_wordlist = []
	for i in range(0, sample_n):
		print('Sample %d' % (i+1))
		part_wordlist = generate_words(vowel_df, cons_df, sylnum, outputlines)
		full_wordlist.append(part_wordlist)
	full_wordlist = [item for part in full_wordlist for item in part]
	return sample(full_wordlist, outputlines)


def remove_from_lex(sylnum, lex_filepath):
	remover = LexRemover(sylnum, lex_filepath)
	remover.populate_wordset()
	remover.populate_found()
	remover.write_new_list()
	remover.post_process()


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-c", "--csvfile", help="Input file with phoneme weights and phonotactic rules.", default="data/example.csv")
	parser.add_argument("-m", "--mode", help="Choose between deterministic rules or character-level LM (LSTM) modes.", default="rules")
	parser.add_argument("-n", "--sylnum", help="Number of syllables in generated words.", default=0)
	parser.add_argument("-o", "--outputlines", help="Number of output words generated.", default=3000)
	parser.add_argument("-p", "--patterns", help="Optional json file for post-processing rules.", action='store_true', default=False)
	parser.add_argument("-s", "--sampling", help="Option to sample from n runs of WordGen (default n=10).", action='store_true', default=False)
	parser.add_argument("-r", "--remove", help="Option to remove words from the output according to a provided wordlist.", action='store_true', default=False)
	parser.add_argument("-a", "--ascii_only", help="Option to convert IPA to ASCII-only representation.", action='store_true', default=False)
	args = parser.parse_args()
	
	datafile = args.csvfile
	mode = args.mode
	sylnum = int(args.sylnum)
	outputlines = int(args.outputlines)
	patterns = args.patterns
	sampling = args.sampling
	remove = args.remove
	ascii_only = args.ascii_only

	if mode == "rules":
		vowel_df, cons_df = read_from_csv(datafile)
		if sampling:
			sample_n = int(input("Enter number of samples: (default=10) ") or 10)
			sampled_wordlist = sample_run(vowel_df, cons_df, sylnum, outputlines, sample_n)
			write_file(sampled_wordlist, "output.txt")
		else:
			wordlist = generate_words(vowel_df, cons_df, sylnum, outputlines)
			write_file(wordlist, "output.txt")
		wl_fname = "wordlist-%dsyl.txt" % sylnum
		if sylnum == 0:
			wl_fname = "wordlist-randnsyl.txt"
		if patterns:
			pats_file = input("Enter filepath for patterns: (default=data/patterns.json) ") or "data/patterns.json"
			post_process(pats_file, "output.txt", wl_fname)
		else:
			os.rename("output.txt", wl_fname)
		clean_up()
		if ascii_only:
			ascii_map = input("Enter filepath for ASCII map: (default=data/ascii_map.json) ") or "data/ascii_map.json"
			post_process(ascii_map, wl_fname, "ascii_output.txt")
			clean_up_ascii(wl_fname)
	elif mode == "lstm":
		subprocess.call(['./lstm_run.sh'])
	if remove:
		lex_filepath = input("Enter filepath for lexicon: (default=data/lexicon.txt) ") or "data/lexicon.txt"
		remove_from_lex(sylnum, lex_filepath)
		

if __name__ == '__main__':
	main()
