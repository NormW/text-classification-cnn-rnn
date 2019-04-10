# coding: utf-8

from __future__ import print_function

import os
from collections import Counter

import tensorflow as tf
import tensorflow.contrib.keras as kr
from nltk import word_tokenize

from cnn_model import TCNNConfig, TextCNN
from data.cnews_loader import read_category, read_vocab, read_file, build_vocab, process_file
from text_pre_processor import split_text, tokenize_word

try:
    bool(type(unicode))
except NameError:
    unicode = str

base_dir = 'data/reviews'
vocab_dir = os.path.join(base_dir, 'reviews.vocab.txt')
train_file = os.path.join(base_dir, 'reviews.train.txt')

save_dir = 'checkpoints/reviews'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径


class CnnModel:
    def __init__(self):
        self.config = TCNNConfig()
        self.categories, self.cat_to_id = read_category()
        self.words, self.word_to_id = read_vocab(vocab_dir)
        self.config.vocab_size = len(self.words)
        self.model = TextCNN(self.config)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=save_path)  # 读取保存的模型

    def predict(self, message):
        # 支持不论在python2还是python3下训练的模型都可以在2或者3的环境下运行
        content = unicode(message)
        data = [self.word_to_id[x] for x in content if x in self.word_to_id]

        feed_dict = {
            self.model.input_x: kr.preprocessing.sequence.pad_sequences([data], self.config.seq_length),
            self.model.keep_prob: 1.0
        }

        y_pred_cls = self.session.run(self.model.y_pred_cls, feed_dict=feed_dict)
        return self.categories[y_pred_cls[0]]


if __name__ == '__main__':
    # cnn_model = CnnModel()
    # test_demo = ['works great, much better than whatsapp',
    #              'and cant join the square']
    # for i in test_demo:
    #     print(cnn_model.predict(i))

    # contents, labels = read_file(train_file)
    # print("")
    # print(contents)
    # build_vocab(train_file, vocab_dir)
    # all_data = []
    # for content in contents:
    #     all_data.extend(content)
    #
    # counter = Counter(contents)
    # print(all_data)
    # print(counter)

    # test = ["this is a test", "this is another test"]
    # for content in test:
    #     all_data.extend(word_tokenize(content))
    #
    # print(all_data)
    categories, cat_to_id = read_category()
    words, word_to_id = read_vocab(vocab_dir)
    process_file(train_file, word_to_id, cat_to_id, 600)

