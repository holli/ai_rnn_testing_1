
import importlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import random
import time
import os
import pandas as pd
import csv
import math
import bcolz
import pickle
import re
import pathlib
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

print("Pytorch utils oh:", os.path.basename(__file__))
print("Pytorch: {}".format(torch.__version__))


SAMPLE_WORD_TOKEN = '<SAMPLE>'
EOS_TOKEN = '<EOS>' # End Of Word
SOS_TOKEN = '<SOS>' # Start Of Word
VERBATIM_CHAR = '☒'
NUMBER_WORD_TOKEN = '<0000>'
UNKNOWN_WORD_TOKEN = '<UNK>'

###############################################################

re_tok_apos = re.compile(r"(\w)'s\b")         # make 's a separate word
re_tok_mw_punc = re.compile(r"(\w[’'])(\w)")  # other ' in a word creates 2 words
re_tok_punc = re.compile("([\"().,;:/_?!—])") # add spaces around punctuation
re_tok_mult_space = re.compile(r"  *")        # replace multiple spaces with just one

def simple_tokeniser(sent):
    """Split string to array"""
    sent = re_tok_apos.sub(r"\1 's", sent)
    sent = re_tok_mw_punc.sub(r"\1 \2", sent)
    sent = re_tok_punc.sub(r" \1 ", sent).replace('-', ' - ')
    sent = re_tok_punc.sub(r" \1 ", sent)
    sent = re_tok_mult_space.sub(' ', sent)
    return sent.lower().split()

###############################################################
### TENSOR CREATIONS


def char_to_tensor(char, chars_index):
    """Onehot encoded tensor"""
    tensor = torch.zeros(1, len(chars_index))
    tensor[0, chars_index[char]] = 1
    return tensor


def string_to_tensor(str, chars_index, include_eos=True):
    """Onehot encoded tensor of string"""
    tensor_length = len(str) + 1 if include_eos else len(str)
    tensor = torch.zeros(1, tensor_length, len(chars_index))
    for li, letter in enumerate(str):
        tensor[0, li, chars_index[letter]] = 1
    if include_eos:
        tensor[0, -1, chars_index['<EOS>']] = 1
    return tensor

re_includes_numbers = re.compile('[0-9]')


def words_to_tensor(words, words_lookup_index, include_eos=True, numbers_replace=True, unknown_replace=True):
    """Onehot encoded tensor of list of words"""
    # if type(words) != list: raise Exception("Expected a list (not a string etc)")
    tensor_length = len(words) + 1 if include_eos else len(words)
    # tensor = np.zeros((1, tensor_length, len(words_lookup_index)), dtype=np.float32)
    tensor = torch.zeros(1, tensor_length, len(words_lookup_index))

    for i, w in enumerate(words):
        if w in words_lookup_index:
            tensor[0, i, words_lookup_index[w]] = 1
        elif numbers_replace and re_includes_numbers.search(w):
            tensor[0, i, words_lookup_index[NUMBER_WORD_TOKEN]] = 1
        elif unknown_replace:
            tensor[0, i, words_lookup_index[UNKNOWN_WORD_TOKEN]] = 1
        else:
            raise Exception("Did not find item in index")

    if include_eos:
        tensor[0, -1, words_lookup_index[EOS_TOKEN]] = 1
    return tensor


def load_common_words_100():
    """common_words, common_words_index = load_common_words_100()"""
    common_words = pickle.load(open("data/en_train_words_before_over_100.pk", "rb" ))
    common_words = [EOS_TOKEN, SOS_TOKEN, UNKNOWN_WORD_TOKEN, NUMBER_WORD_TOKEN, SAMPLE_WORD_TOKEN] + common_words
    common_words_index = dict((c, i) for i, c in enumerate(common_words))
    return common_words, common_words_index


def load_characters_all(path):
    """characters_all, characters_all_index = load_characters_all("data/en_train_characters_all_1.pkl")"""
    print("USE BETTER CHARACTERS SETTINGS")
    characters_all = pickle.load(open(path, "rb" ))
    characters_all_index = dict((c, i) for i, c in enumerate(characters_all))
    return characters_all, characters_all_index


def load_glove(name):
    """wv_vecs, wv_words, wv_idx = load_glove('/home/ohu/koodi/data/glove_wordvec/glove.6B.50d.txt')"""
    with open(name, 'r') as f: lines = [line.split() for line in f]
    words = [d[0] for d in lines]
    vecs = np.stack(np.array(d[1:], dtype=np.float32) for d in lines)
    wordidx = {o:i for i,o in enumerate(words)}
    return vecs, words, wordidx


def words_to_word_vectors_tensor(words, wv_vecs, wv_idx):
    """Tensor including word vectors, load wv_vecs and wv_idx with load_glove"""
    word_vect = np.zeros((1, len(words), wv_vecs.shape[1]), dtype=np.float32)
    for i, w in enumerate(words):
        if w==SAMPLE_WORD_TOKEN:
            word_vect[0][i] = np.zeros((1, wv_vecs.shape[1]))
        else:
            try:
                word_vect[0][i] = wv_vecs[wv_idx[w]]
            except KeyError:
                word_vect[0][i] = np.random.rand(1, wv_vecs.shape[1])

    return torch.from_numpy(word_vect)


if False: # or True:
    print("Loadig pytorch_utils_oh defaults")

    wv_vecs, wv_words, wv_idx = load_glove('/home/ohu/koodi/data/glove_wordvec/glove.6B.50d.txt')
    def sentence_word_vectorize(str_list):
        return sentence_word_vectorize_no_defaults(str_list, wv_vecs, wv_idx)


###############################################################
### AI STUFF

class ModelTraining:
    iterations = 0
    losses = []
    accuracy = []
    learning_rates = []

    def __init__(self, save_path, models):
        self.save_path_addition = os.path.join('data/models', save_path)
        if os.path.exists(self.save_path_addition):
            for i in range(1, 99999):
                new_path = self.save_path_addition + '_' + str(i)
                if not os.path.exists(new_path):
                    self.save_path_addition = new_path
                    break

        self.save_path = os.path.abspath(self.save_path_addition)
        print("Save path: {}".format(self.save_path_addition))

        self.models = models
        self.model_names = []

        for m in models:
            m_name = str(m).split(" ")[0]
            if m_name in self.model_names:
                for i in range(1, 99999):
                    new_name = m_name + '_' + str(i)
                    if new_name not in self.model_names:
                        m_name = new_name
                        break
            self.model_names.append(m_name)

        self.dir_description = ""
        for i in range(len(self.models)):
            self.dir_description += self.model_names[i] + "\n"
            self.dir_description += str(self.models[i]) + "\n"
            self.dir_description += "\n"

    def save_models(self):
        pathlib.Path(self.save_path).mkdir(parents=True, exist_ok=True)

        text_file = open(os.path.join(self.save_path, "description.txt"), "w")
        text_file.write(self.dir_description)
        text_file.close()

        for i in range(len(self.models)):
            path = os.path.join(self.save_path, (str(self.iterations) + "_" + self.model_names[i]))
            torch.save(self.models[i].state_dict(), path)

        print("Saved model to {}_({})".format(
            os.path.join(self.save_path_addition, str(self.iterations)),
            '/'.join(self.model_names)))

    def load_models(self, models, iteration = None):
        """Load parameters, newest if no iteration given"""
        print("NOT IMPLEMENTED YET")

def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

###############################################################
### General training

def category_from_output(output, categories):
    top_n, top_i = output.data.topk(1)
    category_i = top_i[0][0]
    return categories[category_i], category_i

def test_model_accuracy(model, test_model_single_sample_fn, n_sample=10000):
    model_train_org = model.training
    model.eval()

    n_correct = 0
    for iteration in range(n_sample):
        output, guess, correct, sample = test_model_single_sample_fn(model)

        if guess == correct:
            n_correct += 1

    print("Accuracy: {:>4.2%} ({:>8d}/{:>8d})".format(n_correct/n_sample, n_correct, n_sample))

    if model_train_org:
        model.train()

    return(n_correct/n_sample)

def plot_category_confusion_matrix(model, categories, test_model_single_sample_fn, n_confusion=50000, remove_diagonal=False):
    # Keep track of correct guesses in a confusion matrix
    confusion = torch.zeros(len(categories), len(categories))

    n_correct = 0

    # Go through a bunch of examples and record which are correctly guessed
    for i in range(n_confusion):
        output, (guess, guess_i), (correct, correct_i), sample = test_model_single_sample_fn(model)

        if guess == correct:
            n_correct += 1

        confusion[correct_i][guess_i] += 1

    if remove_diagonal:
        for i in range(len(categories)):
            confusion[i, i] = 0

    # Normalize by dividing every row by its sum
    for i in range(len(categories)):
        confusion[i] = confusion[i] / confusion[i].sum()

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + list(categories), rotation=90)
    ax.set_yticklabels([''] + list(categories))

    # Force label at every tick
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.show()

    print("Accuracy: {:>4.2%} ({:>8d}/{:>8d})".format(n_correct/n_confusion, n_correct, n_confusion))


def get_some_wrong_predictions(model, test_model_single_sample_fn, max_iterations=50000, max_results=10):
    wrong_arr = []
    for _ in range(max_iterations):
        output, guess, correct, sample = test_model_single_sample_fn(model)
        if guess != correct:
            wrong_arr.append([sample, guess, output])
            if len(wrong_arr) > max_results:
                break

    return wrong_arr
