# WordGen
A program written in Python 3.7 for generating random words based on a predefined phonology. Both deterministic rule-based and character model (Markov chain) modes available.

## Features

Wordgen has the following features by default:

### Rule-based mode
1. Example phoneme inventory: /a e i o u m n ŋ ɲ p t k ʔ b d g f s z ʃ h tʃ l r j w/
2. Example phoneme weights based on an average across Swadesh lists from an 100-language sample; statistical differentiation between word-initial onsets, medial onsets, nonfinal and final coda consonants
3. Example syllable structure: (C)V(C) with possible onsets /m n ŋ ɲ p t k b d g f s z ʃ h tʃ l r j w/ and possible codas /m n ŋ p t k s l r j w/
4. Can specify syllable count and number of output words
5. Example weights have built-in dispreference for repeated syllables with same point of articulation (reduced weight if previous syllable has same POA)
6. Output file format: .txt with newline for each word; syllables separated by dot
7. Post-processing according to assimilation rules included in `patterns.json`

These default settings can be changed by altering the phoneme inventory and weights in the input CSV, or by changing the number of lines/syllables in the output.

## Additional tools
1. `lex_remove.py` removes a specific set of words from the generated wordlist; this functionality is included in `wordgen.py` with the `-r` flag.
2. The `-a` flag can be used to replace IPA characters with ASCII-only according to patterns in `data/ascii_map.json`.
3. `find_overlap.py` can be run to print overlapping lexemes between two wordlists.

# Installation
For easiest setup, create a Python environment using [Miniconda](https://docs.conda.io/en/latest/miniconda.html). After installing conda, run the following to set up your environment:

```
conda create -n wordgen python=3.7
conda activate wordgen
pip install -r requirements.txt
```

This will install all necessary Python dependencies and ensure you are using the correct versions.

# Usage

## Rule-based

The script should be called from root directory as follows:

```
python wordgen.py [options]
```

Rules mode is run by default. The logic for the rule-based mode is hard-coded in `wordgen.py`. The phoneme weights are read from a CSV-formatted file, for example the provided `data/example.csv`. The `read_from_csv` function in `wordgen.py` assumes the following structure, where the phonotactic contexts are represented by rules in the leftmost column and weights for each rule in the respective phoneme columns.

| Rules | | | | | | 
| --- | --- | --- | --- | --- | --- | 
| V | a | e | i | o | ... |
| "all" | 1.0 | 1.0 | 1.0 | 1.0 | ... |
| C | _ | m | n | p | ... |
| "initial onset" | 1.0 | 1.0 | 1.0 | 1.0 | ... |
| "medial onset" | 1.0 | 1.0 | 1.0 | 1.0 | ... |
| "nonfinal coda" | 1.0 | 1.0 | 1.0 | 1.0 | ... |
| "final coda" | 1.0 | 1.0 | 1.0 | 1.0 | ... |

The hard-coded logic in `wordgen.py` must be changed if different syllable structures are desired, such as additional onset or coda consonants, vowel length rules, diphthongs, tones etc.

### Options

#### Custom phoneme weights

A custom input CSV file with different rules and weights can be provided by the `-c` flag; the default path is `data/example.csv`.

```
python wordgen.py -c path/to/myweights.csv
```

If you want to use a different input format, edit the `read_from_csv` function in `wordgen.py`.

#### Number of syllables and output words

By default, a random number of syllables is chosen for each output word (option `-n 0`), weighted by the typical distribution of number of syllables across English words, but a specific number can be specified using the `-n` flag. The default number of output words is 3000, but a different number can be specified using the `-o` flag.

```
python wordgen.py -n 2 -o 1000
```

#### Custom post-processing rules

The `-p` flag specifies a json file of replacement rules, applied before the final output wordlist file is generated. The default path to the patterns file is `data/patterns.json`; the user will be prompted to input a different path if desired. The format of the patterns file is assumed to be json, with the structure { "[REGEX]": "[STRING REPLACEMENT]" }. The regular expression on the left for each rule should be formatted as a raw string input for the Python 3+ `re.compile()` function.

By default, without the `-p` flag rules mode will permit duplicates, but with the `-p` flag the logic will remove duplicates due to adding to the set of 'seen' lexemes. This behaviour can be changed by editing the `post_process` function in `wordgen.py`.

#### Sampling

The output wordlist can be sampled from several runs of WordGen with the `-s` flag. The user will be prompted for the number of sample runs, where the default is 10. (Sampling calls the entire rule-based flow each time, so it is not advised to use a high n).

#### Removal of specific lexemes

The `-r` flag prompts the user to specify a lexicon text file, by default the provided `data/lexicon.txt`, to remove those lexemes from the output wordlist.

#### Convert IPA to ASCII-only representation

The provided example CSV represents phonemes using IPA. The `-a` flag can be used to convert IPA to ASCII characters, following the rules in `data/ascii_map.json`. The logic is the same as that of the `-p` flag post-processing rules, where the user is prompted for the json rules file.

## Markov chain mode

A Markov chain character model can be trained using `python markov.py`. The default is a third-order model, i.e. a window of three preceding characters. By default, phonotactic rules are applied using `data/phonotactics.tsv` to enforce the same syllable structure as provided in the rule-based mode; this file may be modified to permit different phonemes in the syllabic positions.

### Options

#### Model order

The order (i.e. number of characters considered in the preceding window) can be changed using the `-m` flag.

#### Max length

The maximum length of the output sequence can be specified using the `-l` flag (default 10 characters). This does not consider post-processing of invalid sequences according to the phonotactic rules, which may instert epenthetic segments and result in a longer sequence.

#### Number of output words

As in rule-based mode, the number of output words can be specified using the `-o` flag.

#### Custom phonotactics

A different phonotactic rules file may be specified using the `-t` flag. The expected file format is as follows:

| Onset | Nucleus | Coda |
| --- | --- | --- |
| m | a | m |
| n | e | n |
| p | i | ng |
| t | o | l |
| ... | ... | ... |

where the columns indicate the permitted onsets, nuclei and codas respectively.

#### Custom post-processing rules

As in rule-based mode, the `-p` flag specifies a JSON file containing post-processing rules; in Markov mode, the default file is `data/markov_patterns.json`.

