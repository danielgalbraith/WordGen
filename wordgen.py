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
    rules = df['Rules'].dropna(how='all').tolist()

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

    cons_rules = rules[rules.index('C')+1:]
    for rule in cons_rules:
        row_index = df.index[df['Rules'] == rule].tolist()[0]
        cons_weights = df.iloc[row_index, 1:].fillna(0.0).tolist()
        cons_weights = [float(i) for i in cons_weights]
        cons_weights_map[rule] = cons_weights

    # Create DataFrames for vowels and consonants
    vowel_df = pd.DataFrame(vowel_weights, index=vowel_list)
    cons_df = pd.DataFrame(cons_weights_map, index=cons_list)

    return vowel_df, cons_df


def read_place_file(place_file):
    """
    Reads data from a CSV file and processes it into place features DataFrames.

    Args:
        place_file (str): Path to the CSV file.

    Returns:
        place_df (pd.DataFrame): DataFrame with consonants and their place features.
    """
    try:
        # Attempt to read the CSV file into a DataFrame
        place_df = pd.read_csv(place_file)
    except FileNotFoundError:
        # Handle the case where the file is not found
        print(f"Error: The file {place_file} was not found.")
        return None
    return place_df


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


def adjust_weights_by_place(weights, reduce_indices, reduction_factor):
    """
    Adjust weights by reducing selected weights and proportionally scaling the rest.

    Args:
        weights (list of float): The original list of weights.
        reduce_indices (list of int): Indices of weights to be reduced.
        reduction_factor (float): The factor by which to reduce the selected weights.

    Returns:
        list of float: The adjusted list of weights.
    """
    # Reduce the selected weights by given factor
    reduced_weights = [weights[i] * reduction_factor if i in reduce_indices else weights[i] for i in range(len(weights))]
    # Calculate the total of the reduced weights
    total_reduced = sum(reduced_weights[i] for i in reduce_indices)
    # Determine the total reduction
    original_reduced_sum = sum(weights[i] for i in reduce_indices)
    total_reduction = original_reduced_sum - total_reduced
    # Calculate the total of the unchanged values
    unchanged_indices = [i for i in range(len(weights)) if i not in reduce_indices]
    total_unchanged = sum(weights[i] for i in unchanged_indices)
    # Adjust the unchanged values proportionally
    scaling_factor = (total_unchanged + total_reduction) / total_unchanged
    adjusted_weights = [weights[i] * scaling_factor if i in unchanged_indices else reduced_weights[i] for i in range(len(weights))]
    return adjusted_weights


def get_cons_features(cons_df, place_df):
    """
    Retrieves place features of consonants from DataFrame.

    Args:
        cons_df (pd.DataFrame): DataFrame containing the consonants.
        place_df (pd.DataFrame): DataFrame containing consonants with corresponding place features.

    Returns:
        cons_place_map: A dict mapping consonants to their place features.
    """

    def get_place(cons, place_df):
        if cons in place_df['Consonant'].values:
            return place_df.loc[place_df['Consonant'] == cons, 'Place'].iloc[0]
        return 'other'

    cons_list = ['' if c == '_' else c for c in list(cons_df.index)]
    cons_place_map = { cons: get_place(cons, place_df) for cons in cons_list }
    
    return cons_place_map


def get_reduce_indices(oldsyl, cons_place_map, cons_df, is_onset):
    """
    Get indices of consonants with the same place value as the target consonant in a syllable.

    Args:
        oldsyl (str): The previous syllable.
        cons_place_map (dict): Mapping from consonants to place features.
        cons_df (pandas.DataFrame): DataFrame containing the consonants.
        is_onset (bool): A flag indicating if the target consonant is an onset (True) or a coda (False).

    Returns:
        list: A list of indices of consonants with the same place value in the DataFrame.
    """
    idx = 0 if is_onset else -1
    place_value = cons_place_map.get(oldsyl[idx])
    return [cons_df.index.get_loc(key) for key, value in cons_place_map.items() if value == place_value and key in cons_df.index]


def get_onset_weights(oldsyl, cons_df, syl_idx, sylnum, cons_place_map):
    """
    Retrieves onset weights based on the previous syllable.

    Args:
        oldsyl (str): The previous syllable.
        cons_df (pd.DataFrame): DataFrame containing consonant weights.
        syl_idx (int): The index of the current syllable.
        sylnum (int): The total number of syllables.
        cons_place_map (dict): Mapping from consonants to place features.

    Returns:
        list: A list of weights for the onset.
    """
    ons_weights = []
    if oldsyl == '':
        ons_weights = get_weights(cons_df, "initial onset")
    else:
        ons_weights = get_weights(cons_df, "medial onset")
        if oldsyl[0] in cons_place_map.keys() and cons_place_map[oldsyl[0]] != 'other':
            reduce_indices = get_reduce_indices(oldsyl, cons_place_map, cons_df, True)
            ons_weights = adjust_weights_by_place(ons_weights, reduce_indices, 0.1)
    
    if ons_weights:
        return ons_weights
    else:
        raise ValueError("Invalid syllable index or total syllable count.")


def get_coda_weights(oldsyl, cons_df, syl_idx, sylnum, cons_place_map):
    """
    Retrieves coda weights based on the syllable index and total syllable count.

    Args:
        oldsyl (str): The previous syllable.
        cons_df (pd.DataFrame): DataFrame containing consonant weights.
        syl_idx (int): The index of the current syllable.
        sylnum (int): The total number of syllables.
        cons_place_map (dict): Mapping from consonants to place features.

    Returns:
        list: A list of weights for the coda.
    """
    def adjust_coda_weights(weights, no_coda_value, remaining_sum):
        weights[0] = no_coda_value
        scaling_factor = remaining_sum / sum(weights[1:])
        weights[1:] = [x * scaling_factor for x in weights[1:]]
        return weights

    coda_weights = []
    if sylnum == 1:
        # For monosyllabic words, reduce coda likelihood from 30% to 15%:
        coda_weights = get_weights(cons_df, "final coda")
        coda_weights = adjust_coda_weights(coda_weights, 0.85, 0.15)
    else:
        # Polysyllabic word
        if syl_idx == sylnum - 1:
            # Final syllable
            coda_weights = get_weights(cons_df, "final coda")
        elif syl_idx == sylnum - 2:
            # For penultimate syllable, increase coda likelihood from 20% to 30%:
            coda_weights = get_weights(cons_df, "nonfinal coda")
            coda_weights = adjust_coda_weights(coda_weights, 0.7, 0.3)
        else:
            coda_weights = get_weights(cons_df, "nonfinal coda")

        # Reduce weights with preceding coda of same place feature:
        if len(oldsyl) > 1:
            if oldsyl[-1] in cons_place_map.keys() and cons_place_map[oldsyl[-1]] != 'other':
                reduce_indices = get_reduce_indices(oldsyl, cons_place_map, cons_df, False)
                coda_weights = adjust_weights_by_place(coda_weights, reduce_indices, 0.1)

    if coda_weights:
        return coda_weights
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


def generate_onset(oldsyl, cons_df, syl_idx, sylnum, cons_place_map):
    """
    Generates an onset (consonant or cluster) for a syllable based on previous syllable and consonant weights.

    Args:
        oldsyl (str): The previous syllable.
        cons_df (pd.DataFrame): DataFrame containing consonant weights.
        syl_idx (int): The index of the current syllable.
        sylnum (int): The total number of syllables.
        cons_place_map (dict): Mapping from consonants to place features.

    Returns:
        str: A randomly selected onset.
    """
    onsets = ['' if c == '_' else c for c in list(cons_df.index)]
    weights = get_onset_weights(oldsyl, cons_df, syl_idx, sylnum, cons_place_map)
    return weighted_random_choice(onsets, weights)


def generate_coda(oldsyl, cons_df, syl_idx, sylnum, cons_place_map):
    """
    Generates a coda (consonant or cluster) for a syllable based on previous syllable, syllable index, and total syllable count.

    Args:
        oldsyl (str): The previous syllable.
        cons_df (pd.DataFrame): DataFrame containing consonant weights.
        syl_idx (int): The index of the current syllable.
        sylnum (int): The total number of syllables.
        cons_place_map (dict): Mapping from consonants to place features.

    Returns:
        str: A randomly selected coda.
    """
    codas = ['' if c == '_' else c for c in list(cons_df.index)]
    weights = get_coda_weights(oldsyl, cons_df, syl_idx, sylnum, cons_place_map)
    return weighted_random_choice(codas, weights)


def generate_words(vowel_df, cons_df, sylnum, outputlines, cons_place_map):
    """
    Generates a list of words based on given vowel and consonant dataframes, number of syllables, and output lines.

    Args:
        vowel_df (pd.DataFrame): DataFrame containing vowel weights.
        cons_df (pd.DataFrame): DataFrame containing consonant weights.
        sylnum (int): The number of syllables in each word. If 0, a random number of syllables is chosen.
        outputlines (int): The number of words to generate.
        cons_place_map (dict): Mapping from consonants to place features.

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
            onset = generate_onset(oldsyl, cons_df, j, num_syllables, cons_place_map)
            coda = generate_coda(oldsyl, cons_df, j, num_syllables, cons_place_map)
            
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


def sample_run(vowel_df, cons_df, sylnum, outputlines, cons_place_map, sample_n):
    """
    Runs multiple samples of word generation and returns a combined sample.

    Args:
        vowel_df (pd.DataFrame): DataFrame containing vowel weights.
        cons_df (pd.DataFrame): DataFrame containing consonant weights.
        sylnum (int): The number of syllables in each word.
        outputlines (int): The number of words to generate per sample.
        cons_place_map (dict): Mapping from consonants to place features.
        sample_n (int): The number of samples to run.

    Returns:
        list: A list of sampled words.
    """
    full_wordlist = []
    
    for i in range(sample_n):
        print(f'Sample {i + 1}')
        part_wordlist = generate_words(vowel_df, cons_df, sylnum, outputlines, cons_place_map)
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


def generate_wordlist(vowel_df, cons_df, sylnum, outputlines, cons_place_map, sampling):
    if sampling:
        sample_n = int(input("Enter number of samples: (default=10) ") or 10)
        return sample_run(vowel_df, cons_df, sylnum, outputlines, cons_place_map, sample_n)
    else:
        return generate_words(vowel_df, cons_df, sylnum, outputlines, cons_place_map)


def main():
    args = get_args()

    if args.mode == "rules":
        vowel_df, cons_df = read_from_csv(args.csvfile)
        place_df = read_place_file("data/place.csv")
        cons_place_map = get_cons_features(cons_df, place_df)
        wordlist = generate_wordlist(vowel_df, cons_df, args.sylnum, args.outputlines, cons_place_map, args.sampling)
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
