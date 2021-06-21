# based on https://www.youtube.com/watch?v=vGG4N6YR7Ic&list=WL&index=5

import numpy as np
import matplotlib.pyplot as plt

from collections import Counter

from sklearn.utils import shuffle
from datetime import datetime

import os
import sys

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def init_weights(shape):
    return np.random.randn(*shape).astype(np.float32) / np.sqrt(sum(shape))


class Model:
    def __init__(self, D: int, V: int, context_sz: int):
        self.D = D
        self.V = V
        self.context_sz = context_sz

    def _get_pnw(self, X):
        flat_word_list = [word for sentence in X for word in sentence]
        tokens = [flat_word_list[0], flat_word_list[-1]]
        flat_word_list = flat_word_list[1:-1]
        word_count = len(flat_word_list)

        word_freq = dict(Counter(flat_word_list))
        word_freq = {k: (v / word_count)**0.75 for k, v in word_freq.items()}
        self.Pnw = list(word_freq.values)
        self.Pnw.insert(0,0)
        self.Pnw.append(0)
        return self.Pnw

    def _get_negative_samples(self, context, num_neg_samples):
        
        

