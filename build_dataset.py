#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import json
import random
import pandas as pd
import numpy as np
from utils import set_seed

VALID_KEYS = ['异常类型', '异常类型细项', '问题现象', '故障组件', '形成原因', '形成原因细类', '原因分析', '故障处理过程']
OUT_KEYS = ['异常类型', '异常类型细项', '故障组件', '形成原因', '形成原因细类']
NEED_KEYS = ['问题现象'] + OUT_KEYS


def build(in_path, train_path, dev_path, test_path, dev_per=0.1, test_per=0.1):
    raw_data = pd.read_csv(in_path)
    data = raw_data[VALID_KEYS]
    data = data.sample(frac=1).reset_index(drop=True)
    print(data.shape[0])

    # filter
    for key in NEED_KEYS:
        data = data[data[key].notnull()]
        data = data[data[key] != '0']

    num_rows = data.shape[0]
    dev_data_len = int(num_rows * dev_per)
    test_data_len = int(num_rows * test_per)

    print(num_rows, num_rows - dev_data_len - test_data_len, dev_data_len, test_data_len)

    dev_data = data.iloc[:dev_data_len, :].reset_index(drop=True)
    test_data = data.iloc[dev_data_len:(dev_data_len + test_data_len), :].reset_index(drop=True)
    train_data = data.iloc[(dev_data_len + test_data_len):, :].reset_index(drop=True)

    train_data.to_csv(train_path)
    dev_data.to_csv(dev_path)
    test_data.to_csv(test_path)


def build_labels(in_path):
    raw_data = pd.read_csv(in_path)
    for k in OUT_KEYS:
        cur_col = raw_data[k]
        cur_col = cur_col[cur_col.notnull()]
        cur_col = cur_col[cur_col != '0']
        cur_labels = set(cur_col.tolist())

        with open('../data/labels/' + k + '.txt', 'w') as wf:
            tmp_write = list(map(lambda x: x + '\n', cur_labels))
            wf.writelines(tmp_write)


def csv_to_tsv(in_path, out_path):
    df = pd.read_csv(in_path, index_col=0)
    df.to_csv(out_path, sep='\t')


if __name__ == '__main__':
    set_seed(123)

    build('data/ict_fault.csv',
          'data/train.csv',
          'data/dev.csv',
          'data/test.csv')
    build_labels('data/ict_fault.csv')

    # csv_to_tsv('data/train.csv', 'data/train.tsv')
    # csv_to_tsv('data/dev.csv', 'data/dev.tsv')
    # csv_to_tsv('data/test.csv', 'data/test.tsv')
