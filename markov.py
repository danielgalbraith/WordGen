import csv
import re
import numpy as np
import json
import os
import random
import sys
import argparse

from collections import defaultdict, Counter


PHONEME_SET = ['a', 'e', 'i', 'o', 'u', 'm', 'n', 'ng', 'ny', 'p', 't', 'k', 'q', 'b', 'd', 'g', 'f', 's', 'z', 'x', 'c', 'h', 'l', 'r', 'y', 'w']
VOWELS = ['a', 'e', 'i', 'o', 'u']


class HigherOrderMarkovModel:
    def __init__(self, order):
        self.order = order
        self.transition_probs = defaultdict(Counter)

    def train(self, words):
        for word in words:
            padded_word = f"{' ' * self.order}{word}{' ' * self.order}"
            for i in range(len(padded_word) - self.order):
                context = padded_word[i:i + self.order]
                next_char = padded_word[i + self.order]
                self.transition_probs[context][next_char] += 1

        # Convert counts to probabilities
        for context, counter in self.transition_probs.items():
            total = float(sum(counter.values()))
            for next_char in counter:
                self.transition_probs[context][next_char] /= total

    def generate(self, num_words, max_length):
        words = []
        for _ in range(num_words):
            context = ' ' * self.order
            word = []
            for _ in range(max_length):
                if context not in self.transition_probs:
                    break
                next_char = random.choices(
                    list(self.transition_probs[context].keys()),
                    list(self.transition_probs[context].values())
                )[0]
                if next_char == ' ':
                    break
                word.append(next_char)
                context = (context + next_char)[1:]
            words.append(''.join(word))
        return words


def read_input_file(inputfile):
    with open(inputfile, "r") as f:
        return [line.strip() for line in f if line.strip()]


def create_and_train_model(words, model_order):
    markov_model = HigherOrderMarkovModel(model_order)
    markov_model.train(words)
    return markov_model


def process_markov_output(words, phonotactics_file):
    phonotactics = load_phonotactics(phonotactics_file)
    permitted_onsets = set(phonotactics['onset'])
    permitted_codas = set(phonotactics['coda'])
    permitted_nuclei = set(phonotactics['nucleus'])
    output_words = []
    for word in words:
        processed_word = remove_invalid_phonemes(word)
        applied_phonotactics = apply_phonotactics(processed_word, permitted_onsets, permitted_nuclei, permitted_codas)
        output_words.append(applied_phonotactics)
    return output_words


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


def load_phonotactics(phonotactics_file):
    phonotactics = {}
    onset = []
    nucleus = []
    coda = []
    with open(phonotactics_file, 'r', newline='') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        next(reader) # Skip header
        for row in reader:
            if row[0]:
                onset.append(row[0])
            if row[1]:
                nucleus.append(row[1])
            if row[2]:
                coda.append(row[2])
    phonotactics['onset'] = onset
    phonotactics['nucleus'] = nucleus
    phonotactics['coda'] = coda
    return phonotactics


def read_words(file_path):
    with open(file_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def split_to_phonemes(word):
    phonemes = []
    i = 0
    while i < len(word):
        found_phoneme = False
        for phoneme in sorted(PHONEME_SET, key=len, reverse=True):
            if word[i:i + len(phoneme)] == phoneme:
                phonemes.append(phoneme)
                i += len(phoneme)
                found_phoneme = True
                break
        if not found_phoneme:
            # If no matching phoneme is found, treat the character as a single phoneme.
            phonemes.append(word[i])
            i += 1
    return phonemes


def remove_invalid_phonemes(word):
    phonemes = split_to_phonemes(word)
    output_phonemes = []
    for ph in phonemes:
        if ph in PHONEME_SET:
            output_phonemes.append(ph)
    return "".join(output_phonemes)


def next_vowel(word, index):
    for i in range(index, len(word)):
        if word[i] in VOWELS:
            return word[i]
    return 'a'


def previous_vowel(word, index):
    for i in range(index, -1, -1):
        if word[i] in VOWELS:
            return word[i]
    return 'a'


def has_no_vowels(word):
    for ph in word:
        if ph in VOWELS:
            return False
    return True


def generate_word_structure(phonemes):
    word_structure = {
        "onset": None,
        "nuclei": [],
        "medials": [],
        "coda": None
    }
    # Get the positions of all the vowels in the phonemes
    vowel_indices = [i for i, ph in enumerate(phonemes) if ph in VOWELS]
    if not vowel_indices:
        # No vowels in the word; return just the phonemes as onset
        word_structure["onset"] = {"type": "onset", "segment": ''.join(phonemes), "start": 0, "end": len(phonemes)-1}
        return [word_structure["onset"]]
    # Identify onset
    if phonemes[0] not in VOWELS:
        word_structure["onset"] = {"type": "onset", "segment": ''.join(phonemes[:vowel_indices[0]]), "start": 0, "end": vowel_indices[0]-1}
    # Identify nuclei and medials
    curr_nucleus = {"start": vowel_indices[0], "end": vowel_indices[0]}
    for i in range(1, len(vowel_indices)):
        curr_vowel = vowel_indices[i]
        prev_vowel = vowel_indices[i-1]           
        if curr_vowel - prev_vowel == 1:  # extend current nucleus
            curr_nucleus["end"] = curr_vowel
        else:
            # Append nucleus
            nucleus_segment = ''.join(phonemes[curr_nucleus["start"]:curr_nucleus["end"]+1])
            word_structure["nuclei"].append({"type": "nucleus", "segment": nucleus_segment, "start": curr_nucleus["start"], "end": curr_nucleus["end"]})              
            # Append medial
            medial_segment = ''.join(phonemes[curr_nucleus["end"]+1:curr_vowel])
            word_structure["medials"].append({"type": "medial", "segment": medial_segment, "start": curr_nucleus["end"]+1, "end": curr_vowel-1})                
            # Start new nucleus
            curr_nucleus = {"start": curr_vowel, "end": curr_vowel}
    # Finalize the last nucleus after loop ends
    nucleus_segment = ''.join(phonemes[curr_nucleus["start"]:curr_nucleus["end"]+1])
    word_structure["nuclei"].append({"type": "nucleus", "segment": nucleus_segment, "start": curr_nucleus["start"], "end": curr_nucleus["end"]})
    # Identify coda
    if vowel_indices[-1] < len(phonemes)-1:
        word_structure["coda"] = {"type": "coda", "segment": ''.join(phonemes[vowel_indices[-1]+1:]), "start": vowel_indices[-1], "end": len(phonemes)-1}
    # Produce ordered list of word structure elements
    result = []
    if word_structure["onset"]: 
        result += [word_structure["onset"]]
    nuclei_and_medials = [item for pair in zip(word_structure["nuclei"], word_structure["medials"]) for item in pair] + word_structure["nuclei"][len(word_structure["medials"]):] + word_structure["medials"][len(word_structure["nuclei"]):]
    result += nuclei_and_medials
    if word_structure["coda"]:
        result += [word_structure["coda"]]
    return result


def isViolation(struct, permitted_onsets, permitted_nuclei, permitted_codas):
    if struct["type"] == "onset":
        if struct["segment"] in permitted_onsets:
            return False
    elif struct["type"] == "nucleus":
        # Deal with vowel hiatus violations using patterns json
        return False
    elif struct["type"] == "medial":
        if struct["segment"] in permitted_onsets or len(struct["segment"]) == 1:
            return False
        elif len(struct["segment"]) == 2 and struct["segment"] in permitted_codas:
            return False
        elif len(struct["segment"]) == 2 and struct["segment"][0] in permitted_codas and struct["segment"][1] in permitted_onsets:
            return False
        elif len(struct["segment"]) == 3 and struct["segment"][0] in permitted_codas and struct["segment"][1:] in permitted_onsets:
            return False
        elif len(struct["segment"]) == 3 and struct["segment"][:2] in permitted_codas and struct["segment"][2] in permitted_onsets:
            return False
        elif len(struct["segment"]) == 4 and struct["segment"][:2] in permitted_codas and struct["segment"][2:] in permitted_onsets:
            return False
    elif struct["type"] == "coda":
        if struct["segment"] in permitted_codas:
            return False
    return True


def fix_violation(struct, phonemes, permitted_onsets, permitted_nuclei, permitted_codas):
    if struct["type"] == "onset":
        if struct["segment"].startswith('s'):
            phonemes.insert(0, 'i')
        elif has_no_vowels(''.join(phonemes)):
            insertion_position = len(phonemes) // 2
            phonemes.insert(insertion_position, 'a')
        else:
            insertion_position = struct["start"] + 1
            phonemes.insert(insertion_position, next_vowel(phonemes, insertion_position))
    elif struct["type"] == "medial":
        if len(struct["segment"]) >= 3:
            # Handle potential CC coda
            if struct["segment"][:2] in permitted_codas and struct["segment"][2] in permitted_onsets:
                insertion_position = struct["start"] + 2
                phonemes.insert(insertion_position, previous_vowel(phonemes, insertion_position))
            else:
                # Split into potential coda C and onset CC
                coda_candidate = struct["segment"][0]
                onset_candidate = struct["segment"][1:3]
                # Check if split results in valid coda and onset
                if coda_candidate in permitted_codas and onset_candidate in permitted_onsets:
                    insertion_position = struct["start"] + 2
                    phonemes.insert(insertion_position, previous_vowel(phonemes, insertion_position))
                else:
                    if struct["segment"][:2] in permitted_onsets or (struct["segment"][0] in permitted_codas and struct["segment"][1] in permitted_onsets):
                        insertion_position = struct["start"] + 1
                        phonemes.insert(insertion_position, previous_vowel(phonemes, insertion_position))
                    else:
                        insertion_position = struct["start"] + 1
                        phonemes.insert(insertion_position, previous_vowel(phonemes, insertion_position))
        # Handle two consonants
        else:
            insertion_position = struct["start"] + 1
            phonemes.insert(insertion_position, previous_vowel(phonemes, insertion_position))
    else:  #coda
        if len(struct["segment"]) > 3:
            insertion_position = struct["start"] + 2
            phonemes.insert(insertion_position, previous_vowel(phonemes, insertion_position))
        elif len(struct["segment"]) == 3:
            # Handle potential CC coda
            if struct["segment"][:2] in permitted_codas and struct["segment"][2] in permitted_onsets:
                insertion_position = struct["start"] + 2
                phonemes.insert(insertion_position, previous_vowel(phonemes, insertion_position))
            else:
                if struct["segment"][0] in permitted_codas:
                    if struct["segment"][1:] in permitted_onsets:
                        phonemes.append(previous_vowel(phonemes, len(phonemes) - 1))
                    else:  # second & third consonants not in permitted onsets
                        insertion_position = struct["start"] + 2
                        phonemes.insert(insertion_position, previous_vowel(phonemes, insertion_position))
                else:  # first consonant not in permitted codas
                    insertion_position = struct["start"] + 2
                    phonemes.insert(insertion_position, previous_vowel(phonemes, insertion_position))
        elif len(struct["segment"]) == 2:
            if struct["segment"][0] in permitted_codas or struct["segment"] in permitted_onsets:
                phonemes.append(previous_vowel(phonemes, len(phonemes) - 1))
            else:  # first consonant not in permitted codas
                insertion_position = struct["start"] + 2
                phonemes.insert(insertion_position, previous_vowel(phonemes, insertion_position))
        else:  # one consonant
            phonemes.append(previous_vowel(phonemes, len(phonemes) - 1))
    return phonemes


def apply_phonotactics(word, permitted_onsets, permitted_nuclei, permitted_codas):
    phonemes = split_to_phonemes(word)
    word_structure = generate_word_structure(phonemes)
    i = 0
    while i < len(word_structure):
        if isViolation(word_structure[i], permitted_onsets, permitted_nuclei, permitted_codas):
            phonemes = fix_violation(word_structure[i], phonemes, permitted_onsets, permitted_nuclei, permitted_codas)
            word_structure = generate_word_structure(phonemes)
            i = 0
        else:
            i += 1
    return "".join(phonemes)


def post_process(patterns, infile, outfile):
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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inputfile", help="Input file with wordlist for training Markov model.", default="data/swadesh.txt")
    parser.add_argument("-m", "--model_order", help="Order of Markov model (preceding character window).", type=int, default=3)
    parser.add_argument("-l", "--max_length", help="Maximum length of input sequence.", type=int, default=10)
    parser.add_argument("-o", "--outputlines", help="Number of output words generated.", type=int, default=3000)
    parser.add_argument("-t", "--phonotactics_file", help="TSV file containing phonotactic rules.", default="data/phonotactics.tsv")
    parser.add_argument("-p", "--patterns", help="Optional json file for post-processing rules.", action='store_true', default=False)
    return parser.parse_args()


def main():
    args = get_args()

    words = read_input_file(args.inputfile)
    markov_model = create_and_train_model(words, args.model_order)
    generated_words = markov_model.generate(args.outputlines, args.max_length)
    processed_words = process_markov_output(generated_words, args.phonotactics_file)
    for word in processed_words:
        print(word)
    write_file(processed_words, "output.txt")

    markov_fname = f"markov_{args.model_order}_output.txt"
    if args.patterns:
        patterns_file = input("Enter filepath for patterns: (default=data/markov_patterns.json) ") or "data/markov_patterns.json"
        post_process(patterns_file, "output.txt", markov_fname)
        clean_up()
    else:
        os.rename("output.txt", markov_fname)
        clean_up()

if __name__ == '__main__':
    main()