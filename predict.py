# coding: utf-8

from __future__ import print_function

import os
import re
from collections import Counter

import tensorflow as tf
import tensorflow.contrib.keras as kr
from nltk import word_tokenize

from cnn_model import TCNNConfig, TextCNN
from data.review_data_loader import read_category, read_vocab, read_file, build_vocab, process_file \
    # , build_sentences, build_word_vec

try:
    bool(type(unicode))
except NameError:
    unicode = str

save_dir = 'checkpoints/reviews'
save_path = os.path.join(save_dir, 'best_validation')


class DataModel (object):
    base_dir = 'data/function'
    vocab_dir = os.path.join(base_dir, 'vocab.txt')
    train_file = os.path.join(base_dir, 'train.txt')
    categories = ['SIDE_FEATURE', 'SECURITY', 'OTHERS', 'SOCIAL']
    number_classes = 4

    def __init__(self, which):
        if which == 'reviews':
            self.base_dir = 'data/reviews'
            self.vocab_dir = os.path.join(self.base_dir, 'vocab.txt')
            self.train_file = os.path.join(self.base_dir, 'train.txt')
            self.categories = ['FUNCTIONAL_DOMAIN', 'OUT_OF_DOMAIN', 'GENERAL_REVIEW']
            self.number_classes = 3


class CnnModel:
    def __init__(self, which):
        self.dataModel = which
        self.config = TCNNConfig()
        self.config.num_classes = self.dataModel.number_classes
        self.categories, self.cat_to_id = read_category(self.dataModel.categories)
        self.words, self.word_to_id = read_vocab(self.dataModel.vocab_dir)
        self.config.vocab_size = len(self.words)
        self.model = TextCNN(self.config)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=save_path)  # load model

    def predict(self, message):
        content = unicode(message)
        data = [self.word_to_id[x] for x in content if x in self.word_to_id]

        feed_dict = {
            self.model.input_x: kr.preprocessing.sequence.pad_sequences([data], self.config.seq_length),
            self.model.keep_prob: 1.0
        }

        y_pred_cls = self.session.run(self.model.y_pred_cls, feed_dict=feed_dict)
        return self.categories[y_pred_cls[0]]


if __name__ == '__main__':
    reviews_data_model = DataModel("reviews")
    function_data_model = DataModel("function")

    # To pick the data model to construct the cnn model
    cnn_model = CnnModel(function_data_model)

    in_file = "data/output/review_sentences.txt"
    output_file = "data/output/generated_output.csv"

    out = open(output_file, "w+")
    with open(in_file, "r") as input_file:
        for line in input_file:
            out.write(cnn_model.predict(line) + ", " + line)
        out.close()




