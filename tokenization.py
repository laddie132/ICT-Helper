#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

from pyhanlp import *
import os
import pandas as pd
from tqdm import tqdm
import numpy as np


HanLP.segment('我爱祖国天门')


class Tokenizer:
    """
    Tokenization for documents data
    """
    @staticmethod
    def hanlp_tokenize(s):
        s_split = HanLP.segment(s)
        s_split = list(map(lambda x: x.word, s_split))
        return s_split


class LabelVocab:
    OOV = 'unk'

    def __init__(self, key, data_path, path_prefix='data/labels'):
        self.data_path = data_path
        self.key = key
        self.path = path_prefix + key + '.txt'

        if os.path.exists(self.path):
            print('reading labels %s...' % key)
            self.labels = self.read_labels()
        else:
            print('building labels %s...' % key)
            self.labels = self.build_labels()
            print('saving labels %s...' % key)
            self.save_labels()

        self.label2id = dict(zip(self.labels,
                                 range(len(self.labels))))

    def __len__(self):
        return len(self.labels)

    def read_labels(self):
        labels = []
        with open(self.path, 'r') as f:
            for line in f.readlines():
                labels.append(line.strip())
        return labels

    def build_labels(self):
        labels = []
        raw_data = pd.read_csv(self.data_path)

        cur_col = raw_data[self.key]
        cur_col = cur_col[cur_col.notnull()]
        cur_col = cur_col[cur_col != '0']
        labels += list(set(cur_col.tolist()))

        return labels

    def save_labels(self):
        with open(self.path, 'w') as wf:
            tmp_write = list(map(lambda x: x + '\n', self.labels))
            wf.writelines(tmp_write)

    def label_to_id(self, l):
        if l not in self.labels:
            # l = self.OOV
            raise ValueError('Not found label %s' % l)
        return self.label2id[l]

    def id_to_label(self, id):
        if id >= len(self.labels):
            raise ValueError('id should <= %d' % len(self.labels))
        return self.labels[id]


class Vocabulary:
    """
    Construct vocabulary for word and id transforming
    """
    PAD_IDX = 0
    BOS = '_BOS_'
    EOS = '_EOS_'
    OOV = '_OOV_'
    PAD = '_PAD_'

    def __init__(self, data_path, vocab_path):
        self.data_path = data_path

        if not os.path.exists(vocab_path):
            print('building vocab...')

            # PAD should be the first index
            self.vocab = [self.PAD, self.BOS, self.EOS, self.OOV, '_UNUSED1_', '_UNUSED2_']
            self.vocab += self.build_vocab()

            print('saving vocab...')
            self.save_vocab(vocab_path)
        else:
            print('reading vocab...')
            self.vocab = self.read_vocab(vocab_path)

        print('Vocabulary size: ', len(self.vocab))

        self.word2id = dict(zip(self.vocab,
                                range(len(self.vocab))))

    def __len__(self):
        return len(self.vocab)

    def build_vocab(self):
        raw_data = pd.read_csv(self.data_path)
        data = raw_data['问题现象'].tolist()

        words_set = set()

        for line in tqdm(data):
            line_split = Tokenizer.hanlp_tokenize(line)

            for word in line_split:
                word = word.strip()
                if word != '':
                    words_set.add(word)

        return words_set

    def read_vocab(self, vocab_path):
        """
        read the vocab from vocabulary path
        :return:
        """
        vocab = []
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                vocab.append(line.strip())
        return vocab

    def save_vocab(self, vocab_path):
        self._save_words(vocab_path, self.vocab)

    def _save_words(self, path, words):
        with open(path, 'w', encoding='utf-8') as wf:
            wvocab = list(map(lambda x: x + '\n', words))
            wf.writelines(wvocab)

    def word_to_id(self, word):
        if word not in self.vocab:
            return self.word2id[self.OOV]
        return self.word2id[word]

    def text_to_id(self, text):
        text_split = Tokenizer.hanlp_tokenize(text)
        return list(map(lambda w: self.word_to_id(w), text_split))

    def handle_emb(self, emb_dim, emb_path, save_emb_path, oov_path):
        """
        handle glove embeddings: reading and saving embeddings
        :return:
        """
        # reading the embeddings
        print("reading glove from text file %s" % emb_path)
        embeddings = np.random.rand(len(self), emb_dim)
        non_oov_words = []

        with open(emb_path, mode='rb') as f:
            for line_b in tqdm(f):
                try:
                    line = line_b.decode('utf-8')
                except UnicodeDecodeError:
                    # line = line_b.decode('latin-1')
                    continue

                line_split = line.strip().split(' ')
                word = line_split[0]
                vec = [float(x) for x in line_split[1:]]

                if word in self.vocab:
                    embeddings[self.word2id[word]] = vec
                    non_oov_words.append(word)

        oov_words = list(set(self.vocab).difference(set(non_oov_words)))

        # save the embeddings
        print("saving emb to file %s" % save_emb_path)
        np.save(save_emb_path, embeddings)

        print('OOV words num: %d/%d' % (len(oov_words), len(self)))
        self._save_words(oov_path, oov_words)


def preprocess():
    DATA_PATH = 'data/ict_fault.csv'
    VOCAB_PATH = 'data/vocab_words.txt'
    EMB_PATH = '/home/sth/code/Text-Classification-Pytorch/data/embedding/merge_sgns_bigram_char300.txt'
    SAVE_EMB_PATH = 'data/vocab_emb.npy'
    OOV_PATH = 'data/oov_words.txt'
    EMB_DIM = 300

    vocab = Vocabulary(data_path=DATA_PATH, vocab_path=VOCAB_PATH)
    vocab.handle_emb(EMB_DIM, EMB_PATH, SAVE_EMB_PATH, OOV_PATH)


if __name__ == '__main__':
    preprocess()
