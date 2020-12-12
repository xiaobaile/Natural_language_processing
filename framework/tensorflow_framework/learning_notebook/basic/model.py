import os
import sys
import time

import tensorflow as tf 
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib.rnn import LSTMCell


class BiLSTM_CRF(object):
    """docstring for BiLSTM_CRF"""
    def __init__(self, args, embedding):
        self.embedding = embedding


    def build_graph(self):
        pass


    def add_placeholder(self):
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
        


