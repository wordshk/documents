# Extracting 4-gram words

## Overview

We use a two-pass algorithm to extract 4-grams that occur above a threshold.

Then we extract the frequencies of the 4-grams and the 1,2,3-gram substrings of
these 4grams, also whether subgrams are recognized as words, and feed it use a
very simple NN network, to generate a model that guesses whether the given
4-gram is a word.

The lihkg-corpus we used contains data from Nov 2016 to Feb 2022.

Since the aim was not to show the model has high accuracy, but as part of a
task to find 4-gram words from lihkg corpus, we just used all the ground truth
data available to us, and did not set aside a subset for testing.

The model can be trained to ~90% accuracy when evaluated with the training data.

The final guessed_4grams.txt is included in this repo as a reference. In
addition to getting a feel of how the model performs, it may be useful as a
list of "word-like" Cantonese 4-grams. Note that 4-grams that are known to be
actual words (at the time of execution) is not in the list, but our intended
next step is to go through the list and add the 4-grams we manually identify as
words into our dictionary.

Note that some 4-grams may be offensive to some. Note that the list is not
curated and is the direct result of running the described algorithms/processses
to a snapshot of publicly available data. The authors do not intend to condone,
support, or otherwise support any perceived ideas in this list of 4-grams. The
words in guessed_4grams.txt are ordered by frequency that they appear in the
underlying corpus.

## Pipeline

1. pypy3 two_pass_4grams.py 'lihkg-corpus/*gz'
  - creates 4gh.txt
2. pypy3 grams_from_4gh.py 'lihkg-corpus/*gz'
  - reads 4gh.txt and extracts the 1,2,3-gram substrings of the selected 4grams in 4gh.txt
3. pypy3 generate_training_data.py
  - this will generate not_known_as_words.jsonl known_words.jsonl
4. pick some lines from not_known_as_words.jsonl into not_known_as_words_picked.jsonl (should be roughly equal to lines in known_words.jsonl, perhaps a bit more to bias the model to outputting NO)
5. run my_model.py -- using tinygrad as ML library
  - outputs to a results.txt
6. grep '1$' results.txt  | perl -pe 'chomp; s/\s+.*/\t/; print("\n") if $a++ % 8 == 0' >! guessed_4grams.txt
