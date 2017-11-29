
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
VERBATIM_CHAR = UNKNOWN_CHAR = '☒'
NUMBER_WORD_TOKEN = '<0000>'
UNKNOWN_WORD_TOKEN = '<UNK>'

###############################################################

# https://github.com/alexanderkuk/log-progress
def log_progress(sequence, every=None, size=None, name='Items'):
    from ipywidgets import IntProgress, HTML, VBox
    from IPython.display import display

    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = int(size / 200)     # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = '{name}: {index} / ?'.format(
                        name=name,
                        index=index
                    )
                else:
                    progress.value = index
                    label.value = u'{name}: {index} / {size}'.format(
                        name=name,
                        index=index,
                        size=size
                    )
            yield record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = "{name}: {index}".format(
            name=name,
            index=str(index or '?')
        )

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


def string_to_tensor(str, chars_index, include_eos=True, unknown_replace=True):
    """Onehot encoded tensor of string"""
    tensor_length = len(str) + 1 if include_eos else len(str)
    tensor = torch.zeros(1, tensor_length, len(chars_index))
    for li, letter in enumerate(str):
        if letter in chars_index:
            tensor[0, li, chars_index[letter]] = 1
        elif unknown_replace:
            tensor[0, li, chars_index[UNKNOWN_CHAR]] = 1
    if include_eos:
        tensor[0, -1, chars_index[EOS_TOKEN]] = 1
    return tensor


re_includes_numbers = re.compile('[0-9]')
def words_to_tensor(words, words_lookup_index, include_eos=True, numbers_replace=True, unknown_replace=True):
    """Onehot encoded tensor of list of words"""
    if type(words) != list: raise Exception("Expected a list (not a string etc)")
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

###############################################################
### MODEL INFOS AND STUFF

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

