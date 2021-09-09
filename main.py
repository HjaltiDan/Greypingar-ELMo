import os
from bilm.training import train, test, load_options_latest_checkpoint, load_vocab
from bilm.data import LMDataset, BidirectionalLMDataset
# The following imports may or may not be required, depending on which functions are executed
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras


# Constants:
FULL_CORPUS = "all_sentences_lower.txt"
CORPUS_MINUS_TEST_SET = "corpus_minus_test_set.txt"
TEST_SET = "full_test_set.txt"
VOCAB_FILE = "full_vocab.txt"
TEST_SET_PERCENTAGE_SIZE = 0.05
MAX_VOCAB_TOKENS = 800000

BATCH_SIZE = 128
# IMPORTANT: SET N_GPUS and N_CHARACTERS ACCORDINGLY
# N_GPUS depends on whatever hardware is being used to run this program.
# N_CHARACTERS should be 261 for training, or 262 for prediction.
N_GPUS = 0
N_CHARACTERS = 261
N_TRAIN_TOKENS = 1475857948
N_EPOCHS = 1

# Debug one-off. Create a small corpus useable for quicker testing, plus a small test set
def createSmallerCorpus(totalLines=1000000):
    lineCount = 0
    corpusStartingLineNumber = totalLines // 10
    with open("all_sentences_lower_lemmatized.txt", "r", encoding="utf-8") as f_in, \
            open("1m_lemmatized_corpus.txt", "w", encoding="utf-8") as f_out_corpus, \
            open("1m_test_set.txt", "w", encoding="utf-8") as f_out_test:
        for line in f_in:
            if (lineCount <= corpusStartingLineNumber):
                f_out_test.write(line)
            elif (lineCount > corpusStartingLineNumber) and (lineCount <= totalLines):
                f_out_corpus.write(line)
            else:
                return
            lineCount += 1


# Support one-off function. Creates a test set as a percentage of
def createTestSet(input_corpus=FULL_CORPUS, output_corpus=CORPUS_MINUS_TEST_SET, output_test=TEST_SET, percentage=TEST_SET_PERCENTAGE_SIZE):
    moduloPercent = int(percentage*100)
    lineCount = 0
    with open(input_corpus, "r", encoding="utf-8") as f_in, \
            open(output_corpus, "w", encoding="utf-8") as f_out_corpus, \
            open(output_test, "w", encoding="utf-8") as f_out_test:
        for line in f_in:
            if lineCount % 100 <= moduloPercent:
                f_out_test.write(line)
            else:
                f_out_corpus.write(line)
            lineCount += 1


# Support one-off. Create vocabulary based on corpus.
# Based on https://github.com/PhilipMay/de-wiki-text-corpus-tools and https://github.com/t-systems-on-site-services-gmbh/german-elmo-model
# Note that the latter page recommends NOT having a vocabulary larger than 800k tokens, lest we run into out-of-memory errors.
#
# For a LEMMATIZED version of the full corpus, we get the following values:
# Found so many token: 4724931
# Selecting so many of the top token: 800000
# New number of token: 800000
# Removed so many token: 3924931
# Token count of input file: 1475855582
#
# For a NON-LEMMATIZED version of the full corpus (which is the DEFAULT right now), we get the following:
# Found so many token: 5993158
# Selecting so many of the top token: 800000
# New number of token: 800000
# Removed so many token: 5193158
# Token count of input file: 1475857948
def vocab_file_writer():
    vocab_dict = {}
    token_count = 0

    with open(FULL_CORPUS, "r", encoding="utf-8") as input_file:
        with open(VOCAB_FILE, "w", encoding="utf-8") as output_file:
            for line in input_file:
                tokens = line.split()
                token_count += len(tokens)
                for token in tokens:
                    value = vocab_dict.get(token, 0)
                    vocab_dict[token] = value + 1

            data = [(value, key) for key, value in vocab_dict.items()]
            data.sort(reverse=True)

            ori_data_len = len(data)
            print("Found so many token:", ori_data_len)

            print("Selecting so many of the top token:", MAX_VOCAB_TOKENS)

            data = data[:MAX_VOCAB_TOKENS]

            print("New number of token:", len(data))
            print("Removed so many token:", ori_data_len - len(data))

            output_file.write("<S>\n</S>\n<UNK>")
            for d in data:
                value, key = d
                output_file.write("\n")
                output_file.write(key)
    print("Token count of input file:", token_count)

# Based on https://github.com/allenai/bilm-tf/blob/master/bin/train_elmo.py
# Additional info from https://github.com/allenai/bilm-tf/blob/master/README.md is as follows:
# export CUDA_VISIBLE_DEVICES=0,1,2
# python bin/train_elmo.py \
# --train_prefix='/path/to/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/*' \
# --vocab_file /path/to/vocab-2016-09-10.txt \
# --save_dir /output_path/to/checkpoint
#def train_elmo(vocab_file=VOCAB_FILE, train_prefix=CORPUS_MINUS_TEST_SET, save_dir):
def train_elmo(save_dir):
    vocab = load_vocab(VOCAB_FILE, 50)

    batch_size = BATCH_SIZE
    n_gpus = N_GPUS

    # number of tokens in training data (this for 1B Word Benchmark)
    n_train_tokens = N_TRAIN_TOKENS

    options = {
        'bidirectional': True,

        'char_cnn': {'activation': 'relu',
                     'embedding': {'dim': 16},
                     'filters': [[1, 32],
                                 [2, 32],
                                 [3, 64],
                                 [4, 128],
                                 [5, 256],
                                 [6, 512],
                                 [7, 1024]],
                     'max_characters_per_token': 50,
                     'n_characters': N_CHARACTERS,
                     'n_highway': 2},

        'dropout': 0.1,

        'lstm': {
            'cell_clip': 3,
            'dim': 4096,
            'n_layers': 2,
            'proj_clip': 3,
            'projection_dim': 512,
            'use_skip_connections': True},

        'all_clip_norm_val': 10.0,

        'n_epochs': N_EPOCHS,
        'n_train_tokens': n_train_tokens,
        'batch_size': batch_size,
        'n_tokens_vocab': vocab.size,
        'unroll_steps': 20,
        'n_negative_samples_batch': 8192,
    }

    prefix = CORPUS_MINUS_TEST_SET
    data = BidirectionalLMDataset(prefix, vocab, test=False,
                                  shuffle_on_load=True)

    tf_save_dir = save_dir
    tf_log_dir = save_dir
    train(options, data, n_gpus, tf_save_dir, tf_log_dir)


def run_test(save_dir):
    options, ckpt_file = load_options_latest_checkpoint(save_dir)

    # load the vocab
    if 'char_cnn' in options:
        max_word_length = options['char_cnn']['max_characters_per_token']
    else:
        max_word_length = None
    vocab = load_vocab(VOCAB_FILE, max_word_length)

    test_prefix = CORPUS_MINUS_TEST_SET

    kwargs = {
        'test': True,
        'shuffle_on_load': False,
    }

    if options.get('bidirectional'):
        data = BidirectionalLMDataset(test_prefix, vocab, **kwargs)
    else:
        data = LMDataset(test_prefix, vocab, **kwargs)

    return test(options, ckpt_file, data, batch_size=BATCH_SIZE)


def restart(save_dir):
    options, ckpt_file = load_options_latest_checkpoint(save_dir)

    if 'char_cnn' in options:
        max_word_length = options['char_cnn']['max_characters_per_token']
    else:
        max_word_length = None
    vocab = load_vocab(VOCAB_FILE, max_word_length)

    prefix = CORPUS_MINUS_TEST_SET

    kwargs = {
        'test': False,
        'shuffle_on_load': True,
    }

    if options.get('bidirectional'):
        data = BidirectionalLMDataset(prefix, vocab, **kwargs)
    else:
        data = LMDataset(prefix, vocab, **kwargs)

    tf_save_dir = save_dir
    tf_log_dir = save_dir

    # set optional inputs
    if N_TRAIN_TOKENS > 0:
        options['n_train_tokens'] = N_TRAIN_TOKENS
    if N_EPOCHS > 0:
        options['n_epochs'] = N_EPOCHS
    if BATCH_SIZE > 0:
        options['batch_size'] = BATCH_SIZE

    train(options, data, N_GPUS, tf_save_dir, tf_log_dir,
          restart_ckpt_file=ckpt_file)


### MAIN STARTS

# Construction code
# Should only be run when explicitly required
# createTestSet()
# vocab_file_writer()

# Training code
train_elmo(os.getcwd())
lastComplexity = run_test(os.getcwd())
newComplexity = lastComplexity
while ( newComplexity <= lastComplexity ):
    restart(os.getcwd())
    lastComplexity = newComplexity
    newComplexity = run_test(os.getcwd())
    complString = "Completed while loop. newComplexity is " + str(newComplexity) + "; lastComplexity is " + str(lastComplexity)
    print(complString)


# Evaluation code
# ...