#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import time
from pyhanlp import *
from utils.data_reader import TextClsReader
from utils.data_util import set_seed
from utils.preprocess import Preprocess
from CNNSentenceClassifier import *

def test_reader():
    reader = TextClsReader(raw_data_path='data/dataset/full_data.csv',
                           vocab_path='data/dataset/vocab_words.txt',
                           train_path='data/dataset/train.csv',
                           dev_path='data/dataset/dev.csv',
                           test_path='data/dataset/test.csv',
                           max_text_length=256)

    train_data_loader = reader.get_dataloader_train(batch_size=32,
                                                    num_workers=1)

    for batch in train_data_loader:
        text = batch[0]
        labels = batch[1]

        print(text, labels)


def test_split():
    s_split = HanLP.segment('我爱祖国天安门')
    print([x.word for x in s_split])


def test_classifier():
    start_time = time.clock()

    set_seed(42)
    max_text_lenth = 256
    batch_size = 32
    num_workers = 1
    num_of_epoch = 30
    k = 1
    connection = [-1, -1, -1, -1, -1]

    pre = Preprocess(raw_data_path='data/dataset/full_data.csv',
                     vocab_path='data/dataset/vocab_words.txt',
                     embedding_path='data/embedding/vocab_emb.npy',
                     train_path='data/dataset/train.csv',
                     dev_path='data/dataset/dev.csv',
                     test_path='data/dataset/test.csv',
                     max_text_length=max_text_lenth,
                     batch_size=batch_size,
                     num_workers=num_workers)

    vocabulary, vocab_size, label_length, train_data_loader, dev_data_loader, test_data_loader, reader = pre.data_reader()
    embedding_weight = pre.load_embedding()

    preloading_time = time.clock()
    print("Preloading time: {}".format(preloading_time - start_time))

    model = SentenceClassifier(train_data_loader, dev_data_loader, test_data_loader, batch_size, embedding_weight, reader, use_cuda=True, k=k, connection=connection)
    model.train(num_of_epoch)
    model.save_model('models/saved_model')
    train_time = time.clock()
    print("Training time: {}".format(train_time - start_time))

    model.load_model('models/saved_model')
    test_loss, test_acc = model.evaluate(test_data_loader)
    print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%')
    example = "8月14日14:15-16:35，国网蒙东电力I6000所有系统监控状态异常。期间，系统健康运行时长及在线用户数指标缺失，URL探测异常。"
    final_output = model.decode(example)
    print(final_output)

    total_time = time.clock()
    print("Finish testing. Total time: {}".format(total_time - start_time))


if __name__ == '__main__':
    # test_reader()
    # test_split()
    test_classifier()
