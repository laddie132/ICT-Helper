#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"


from pyhanlp import *
from data_reader import TextClsReader


def test_reader():
    reader = TextClsReader(raw_data_path='data/ict_fault.csv',
                           vocab_path='data/vocab_words.txt',
                           train_path='data/train.csv',
                           dev_path='data/dev.csv',
                           test_path='data/test.csv',
                           max_text_length=128)

    train_data_loader = reader.get_dataloader_train(batch_size=32,
                                                    num_workers=1)

    for batch in train_data_loader:
        text = batch[0]
        labels = batch[1]

        print(text.shape, labels.shape)


def test_split():
    s_split = HanLP.segment('我爱祖国天安门')
    print([x.word for x in s_split])


if __name__ == '__main__':
    test_reader()
    # test_split()
