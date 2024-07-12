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
    """
    Reads data from a CSV file and processes it into vowel and consonant DataFrames.

    Args:
        datafile (str): Path to the CSV file.

    Returns:
        tuple: A tuple containing:
            - vowel_df (pd.DataFrame): DataFrame with vowels and their weights.
            - cons_df (pd.DataFrame): DataFrame with consonants and their weights based on rules.
    """
    try:
        # Attempt to read the CSV file into a DataFrame
        df = pd.read_csv(datafile)
    except FileNotFoundError:
        # Handle the case where the file is not found
        print(f"Error: The file {datafile} was not found.")
        return None, None

    # Extract rules from the 'Rules' column and drop NaN values
    rules = df['Rules'].dropna(how='all')

    # Extract vowels and their weights
    vowels = df.iloc[:1, 1:].dropna(axis=1)  # Select first row (vowels) excluding NaN columns
    vowel_list = vowels.values.flatten().tolist()  # Flatten to a list of vowels
    vowel_weights = df.iloc[1:2, 1:len(vowel_list)+1].values.flatten().tolist()  # Extract weights for vowels

    # Replace null consonants with underscores
    if df.iloc[2:3, 1:2].isnull().values.any():
        df.iloc[2:3, 1:2] = '_'

    # Extract consonants and their weights
    consonants = df.iloc[2:3, 1:].dropna(axis=1)  # Select third row (consonants) excluding NaN columns
    cons_list = consonants.values.flatten().tolist()  # Flatten to a list of consonants

    # Map consonant weights to rules
    cons_weights_map = {}
    idx = 0
    for rule in rules.values.flatten().tolist():
        cons_weights = []
        if len(rule.split(',')) > 1:
            with pd.option_context("future.no_silent_downcasting", True):
                cons_weights = df.iloc[idx:idx+1, 1:len(cons_list)+1].fillna(0.0).infer_objects(copy=False).values.flatten().tolist()
            cons_weights = [float(i) for i in cons_weights]
            cons_weights_map[rule] = cons_weights
        idx += 1

    # Create DataFrames for vowels and consonants
    vowel_df = pd.DataFrame(vowel_weights, index=vowel_list)
    cons_df = pd.DataFrame(cons_weights_map, index=cons_list)

    return vowel_df, cons_df


def weighted_random_choice(chars, weights):
    """
    Selects a random character from a list of characters based on given weights.

    Args:
        chars (list): List of characters to choose from.
        weights (list): List of weights corresponding to the characters.

    Returns:
        str: A randomly selected character based on the provided weights.
    """
    total_weight = sum(weights)
    if total_weight == 0:
        raise ValueError("The sum of weights must not be zero.")
    
    normalized_weights = [w / total_weight for w in weights]
    choice = np.random.choice(chars, 1, p=normalized_weights)
    
    return choice[0]


def get_weights(df, rule):
    """
    Retrieves the weights for a specific rule from a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the weights.
        rule (str): The rule for which to retrieve the weights.

    Returns:
        list: A list of weights corresponding to the given rule.
    """
    if rule not in df:
        raise KeyError(f"Rule '{rule}' not found in DataFrame.")
    
    return df[rule].astype(float).tolist()


def get_onset_weights(oldsyl, cons_df):
    """
    Retrieves onset weights based on the previous syllable.

    Args:
        oldsyl (str): The previous syllable.
        cons_df (pd.DataFrame): DataFrame containing consonant weights.

    Returns:
        list: A list of weights for the onset.
    """
    onset_rules = {
        'p': "onsc1,prec labial ons", 'b': "onsc1,prec labial ons", 'f': "onsc1,prec labial ons",
        't': "onsc1,prec alveolar ons", 'd': "onsc1,prec alveolar ons", 's': "onsc1,prec alveolar ons",
        'k': "onsc1,prec velar ons", 'g': "onsc1,prec velar ons",
        'm': "onsc1,prec nasal ons", 'n': "onsc1,prec nasal ons", 'ɲ': "onsc1,prec nasal ons", 'ŋ': "onsc1,prec nasal ons",
        'l': "onsc1,prec liquid ons", 'r': "onsc1,prec liquid ons",
        'z': "onsc1,prec sibilant ons", 'ʃ': "onsc1,prec sibilant ons", 'ʧ': "onsc1,prec sibilant ons",
        'h': "onsc1,prec glottal ons", 'ʔ': "onsc1,prec glottal ons",
        'w': "onsc1,prec semivowel ons", 'j': "onsc1,prec semivowel ons",
    }
    
    if oldsyl == '':
        return get_weights(cons_df, "onsc1,first syl")
    else:
        return get_weights(cons_df, onset_rules.get(oldsyl[0], "onsc1,prec no ons"))


def get_coda_weights(oldsyl, cons_df, syl_idx, sylnum):
    """
    Retrieves coda weights based on the syllable index and total syllable count.

    Args:
        oldsyl (str): The previous syllable.
        cons_df (pd.DataFrame): DataFrame containing consonant weights.
        syl_idx (int): The index of the current syllable.
        sylnum (int): The total number of syllables.

    Returns:
        list: A list of weights for the coda.
    """
    coda_rules = {
        'm': "coda,before penult prec nasal coda", 'n': "coda,before penult prec nasal coda", 'ŋ': "coda,before penult prec nasal coda",
        'p': "coda,before penult prec p coda",
        't': "coda,before penult prec t coda",
        'k': "coda,before penult prec k coda",
        's': "coda,before penult prec s coda",
        'l': "coda,before penult prec liquid coda", 'r': "coda,before penult prec liquid coda",
        'w': "coda,before penult prec semivowel coda", 'j': "coda,before penult prec semivowel coda"
    }

    coda_weights_key = None

    if sylnum == 1:
        coda_weights_key = "coda,monosyllable"
    elif syl_idx < sylnum - 2:
        if len(oldsyl) > 2:
            if oldsyl[2] in coda_rules:
                coda_weights_key = coda_rules[oldsyl[2]]
            else:
                coda_weights_key = "coda,before penult prec no coda"
        else:
            coda_weights_key = "coda,before penult prec no coda"
    elif syl_idx == sylnum - 2:
        if len(oldsyl) > 2:
            if oldsyl[2] in coda_rules:
                coda_weights_key = coda_rules[oldsyl[2]]
            else:
                coda_weights_key = "coda,penult prec no coda"
        else:
            coda_weights_key = "coda,penult prec no coda"
    elif syl_idx == sylnum - 1:
        if len(oldsyl) > 2:
            if oldsyl[2] in coda_rules:
                coda_weights_key = coda_rules[oldsyl[2]]
            else:
                coda_weights_key = "coda,last syl prec no coda"
        else:
            coda_weights_key = "coda,last syl prec no coda"

    if coda_weights_key is not None:
        return get_weights(cons_df, coda_weights_key)
    else:
        raise ValueError("Invalid syllable index or total syllable count.")


def generate_nucleus(vowel_df):
    """
    Generates a nucleus (vowel) for a syllable based on vowel weights.

    Args:
        vowel_df (pd.DataFrame): DataFrame containing vowel weights.

    Returns:
        str: A randomly selected vowel.
    """
    return weighted_random_choice(list(vowel_df.index), vowel_df.iloc[:, 0].astype(float).tolist())


def generate_onset(oldsyl, cons_df):
    """
    Generates an onset (consonant or cluster) for a syllable based on previous syllable and consonant weights.

    Args:
        oldsyl (str): The previous syllable.
        cons_df (pd.DataFrame): DataFrame containing consonant weights.

    Returns:
        str: A randomly selected onset.
    """
    onsets = ['' if c == '_' else c for c in list(cons_df.index)]
    weights = get_onset_weights(oldsyl, cons_df)
    return weighted_random_choice(onsets, weights)


def generate_coda(oldsyl, cons_df, syl_idx, sylnum):
    """
    Generates a coda (consonant or cluster) for a syllable based on previous syllable, syllable index, and total syllable count.

    Args:
        oldsyl (str): The previous syllable.
        cons_df (pd.DataFrame): DataFrame containing consonant weights.
        syl_idx (int): The index of the current syllable.
        sylnum (int): The total number of syllables.

    Returns:
        str: A randomly selected coda.
    """
    codas = ['' if c == '_' else c for c in list(cons_df.index)]
    weights = get_coda_weights(oldsyl, cons_df, syl_idx, sylnum)
    return weighted_random_choice(codas, weights)


def generate_words(vowel_df, cons_df, sylnum, outputlines):
    """
    Generates a list of words based on given vowel and consonant dataframes, number of syllables, and output lines.

    Args:
        vowel_df (pd.DataFrame): DataFrame containing vowel weights.
        cons_df (pd.DataFrame): DataFrame containing consonant weights.
        sylnum (int): The number of syllables in each word. If 0, a random number of syllables is chosen.
        outputlines (int): The number of words to generate.

    Returns:
        list: A list of generated words.
    """
    def get_random_syllable_count():
        """
        Gets a random syllable count based on predefined weights.

        Returns:
            int: A random syllable count.
        """
        sylnum_weights = [5, 10, 7.5, 4, 1.5, 0.1, 0.01]
        return np.random.choice([1, 2, 3, 4, 5, 6, 7], p=[w / sum(sylnum_weights) for w in sylnum_weights])

    wordlist = []
    
    for _ in range(outputlines):
        word = ''
        num_syllables = sylnum if sylnum > 0 else get_random_syllable_count()
        oldsyl = ''
        
        for j in range(num_syllables):
            nucleus = generate_nucleus(vowel_df)
            onset = generate_onset(oldsyl, cons_df)
            coda = generate_coda(oldsyl, cons_df, j, num_syllables)
            
            syllable = onset + nucleus + coda
            word += syllable + '.' if num_syllables > 1 else syllable
            oldsyl = syllable
        
        wordlist.append(word.rstrip('.'))
    
    return wordlist


def write_file(wordlist, outfile):
    """
    Writes a list of words to an output file.

    Args:
        wordlist (list): List of words to write to the file.
        outfile (str): Path to the output file.
    """
    with open(outfile, "w") as f:
        for word in wordlist:
            f.write(word + '\n')


def post_process(patterns, infile, outfile):
    """
    Post-processes a text file by applying regex patterns and removing duplicate lines.

    Args:
        patterns (str): Path to the JSON file containing regex patterns.
        infile (str): Path to the input file to be post-processed.
        outfile (str): Path to the output file to write the processed text.
    """
    # Load patterns from file
    with open(patterns, "r") as patf:
        pats = json.load(patf)
    
    # Process input file and write to output file
    with open(infile, "r") as f1:
        seen_lines = set()
        with open(outfile, "w") as f2:
            for line in f1:
                if line not in seen_lines:
                    seen_lines.add(line)
                    for pattern, replacement in pats.items():
                        line = re.sub(re.compile(pattern), replacement, line)
                    f2.write(line)


def clean_up():
    """
    Removes intermediate output files if they exist.
    """
    intermediate_files = ["output.txt"]
    for file in intermediate_files:
        if os.path.exists(file):
            os.remove(file)


def clean_up_ascii(wordlist_fname):
    """
    Replaces ASCII output file with the wordlist filename.

    Args:
        wordlist_fname (str): The desired name for the wordlist file.
    """
    ascii_output = "ascii_output.txt"
    if os.path.exists(ascii_output):
        os.remove(wordlist_fname)
        os.rename(ascii_output, wordlist_fname)


def sample(wordlist, outputlines):
    """
    Samples a specified number of unique words from the wordlist.

    Args:
        wordlist (list): List of words to sample from.
        outputlines (int): Number of words to sample.

    Returns:
        list: A list of sampled words.
    """
    return np.random.choice(wordlist, outputlines, replace=False).tolist()


def sample_run(vowel_df, cons_df, sylnum, outputlines, sample_n):
    """
    Runs multiple samples of word generation and returns a combined sample.

    Args:
        vowel_df (pd.DataFrame): DataFrame containing vowel weights.
        cons_df (pd.DataFrame): DataFrame containing consonant weights.
        sylnum (int): The number of syllables in each word.
        outputlines (int): The number of words to generate per sample.
        sample_n (int): The number of samples to run.

    Returns:
        list: A list of sampled words.
    """
    full_wordlist = []
    
    for i in range(sample_n):
        print(f'Sample {i + 1}')
        part_wordlist = generate_words(vowel_df, cons_df, sylnum, outputlines)
        full_wordlist.extend(part_wordlist)
    
    return sample(full_wordlist, outputlines)


def remove_from_lex(sylnum, lex_filepath):
    """
    Removes words from a lexicon file based on syllable number and other criteria.

    Args:
        sylnum (int): The number of syllables to filter by.
        lex_filepath (str): Path to the lexicon file.
    """
    remover = LexRemover(sylnum, lex_filepath)
    remover.populate_wordset()
    remover.populate_found()
    remover.write_new_list()
    remover.post_process()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--csvfile", help="Input file with phoneme weights and phonotactic rules.", default="data/example.csv")
    parser.add_argument("-m", "--mode", help="Choose between deterministic rules or character-level LM (LSTM) modes.", default="rules")
    parser.add_argument("-n", "--sylnum", help="Number of syllables in generated words.", type=int, default=0)
    parser.add_argument("-o", "--outputlines", help="Number of output words generated.", type=int, default=3000)
    parser.add_argument("-p", "--patterns", help="Optional json file for post-processing rules.", action='store_true', default=False)
    parser.add_argument("-s", "--sampling", help="Option to sample from n runs of WordGen (default n=10).", action='store_true', default=False)
    parser.add_argument("-r", "--remove", help="Option to remove words from the output according to a provided wordlist.", action='store_true', default=False)
    parser.add_argument("-a", "--ascii_only", help="Option to convert IPA to ASCII-only representation.", action='store_true', default=False)
    return parser.parse_args()


def handle_patterns(wordlist_file, patterns_file, output_file, args):
    post_process(patterns_file, wordlist_file, output_file)
    clean_up()
    if args.ascii_only:
        ascii_map = input("Enter filepath for ASCII map: (default=data/ascii_map.json) ") or "data/ascii_map.json"
        post_process(ascii_map, output_file, "ascii_output.txt")
        clean_up_ascii(output_file)


def generate_wordlist(vowel_df, cons_df, sylnum, outputlines, sampling):
    if sampling:
        sample_n = int(input("Enter number of samples: (default=10) ") or 10)
        return sample_run(vowel_df, cons_df, sylnum, outputlines, sample_n)
    else:
        return generate_words(vowel_df, cons_df, sylnum, outputlines)


def main():
    args = get_args()

    if args.mode == "rules":
        vowel_df, cons_df = read_from_csv(args.csvfile)
        wordlist = generate_wordlist(vowel_df, cons_df, args.sylnum, args.outputlines, args.sampling)
        write_file(wordlist, "output.txt")

        wl_fname = f"wordlist-{args.sylnum if args.sylnum != 0 else 'randnsyl'}.txt"
        if args.patterns:
            patterns_file = input("Enter filepath for patterns: (default=data/patterns.json) ") or "data/patterns.json"
            handle_patterns("output.txt", patterns_file, wl_fname, args)
        else:
            os.rename("output.txt", wl_fname)
            clean_up()
        
        if args.ascii_only:
            ascii_map = input("Enter filepath for ASCII map: (default=data/ascii_map.json) ") or "data/ascii_map.json"
            post_process(ascii_map, wl_fname, "ascii_output.txt")
            clean_up_ascii(wl_fname)

    elif args.mode == "lstm":
        subprocess.call(['./lstm_run.sh'])

    if args.remove:
        lex_filepath = input("Enter filepath for lexicon: (default=data/lexicon.txt) ") or "data/lexicon.txt"
        remove_from_lex(args.sylnum, lex_filepath)


if __name__ == '__main__':
    main()
