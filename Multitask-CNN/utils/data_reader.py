#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import pandas as pd
import torch
import torch.utils.data
from utils.tokenization import Vocabulary, LabelVocab
from utils.data_util import pad_sequences


class TextClsReader:
    """
    Text Classification dataset reader
    """

    KEYS = ['异常类型', '异常类型细项', '故障组件', '形成原因', '形成原因细类']

    def __init__(self, raw_data_path, vocab_path, train_path, dev_path, test_path, max_text_length):
        self._data_train_path = train_path
        self._data_dev_path = dev_path
        self._data_test_path = test_path

        self.vocab = Vocabulary(raw_data_path, vocab_path)

        self.max_text_length = max_text_length

        self.label_vocab_list = []
        for k in self.KEYS:
            self.label_vocab_list.append(LabelVocab(k, raw_data_path))

    def text_to_tensor(self, text):
        """
        Convert raw text to torch tensor with word id
        :param text:
        :return:
        """
        text_id = self.vocab.text_to_id(text)
        text_id_array = pad_sequences([text_id],
                                      maxlen=self.max_text_length,
                                      padding='post',
                                      value=Vocabulary.PAD_IDX)
        text_tensor = torch.tensor(text_id_array, dtype=torch.long).squeeze(0)
        return text_tensor

    def _get_dataloader(self, data_path, batch_size, num_workers, shuffle):
        """
        Get dataloader
        :param data_path:
        :param batch_size:
        :param num_workers:
        :param shuffle:
        :return:
        """
        dataset = TextClsDataset(data_path,
                                 self.KEYS,
                                 self.text_to_tensor,
                                 [x.label_to_id for x in self.label_vocab_list])

        return torch.utils.data.DataLoader(dataset,
                                           batch_size=batch_size,
                                           collate_fn=TextClsDataset.collect_fun,
                                           num_workers=num_workers,
                                           shuffle=shuffle)

    def get_dataloader_train(self, batch_size, num_workers):
        """
        Train dialog data loader for supervised training
        :param batch_size:
        :param num_workers:
        :return:
        """
        return self._get_dataloader(self._data_train_path, batch_size, num_workers, shuffle=True)

    def get_dataloader_dev(self, batch_size, num_workers):
        """
        Dev dialog data loader for supervised training
        :param batch_size:
        :param num_workers:
        :return:
        """
        return self._get_dataloader(self._data_dev_path, batch_size, num_workers, shuffle=False)

    def get_dataloader_test(self, batch_size, num_workers):
        """
        Test dialog data loader for supervised training
        :param batch_size:
        :param num_workers:
        :return:
        """
        return self._get_dataloader(self._data_test_path, batch_size, num_workers, shuffle=False)


class TextClsDataset(torch.utils.data.Dataset):
    """
    Dataset for text classification
    """

    def __init__(self, data_path, keys, text2tensor, labels2id):
        super(TextClsDataset, self).__init__()

        self.keys = keys
        self.text2tensor = text2tensor
        self.labels2id = labels2id

        self.df_data = pd.read_csv(data_path, index_col=0)

    def __len__(self):
        return self.df_data.shape[0]

    def __getitem__(self, index):
        cur_col = self.df_data.iloc[index, :]

        text = cur_col['问题现象']
        text_tensor = self.text2tensor(text)

        labels = [cur_col[k] for k in self.keys]
        labels_id = [self.labels2id[i](l) for i, l in enumerate(labels)]

        labels_tensor = torch.tensor(labels_id, dtype=torch.long)

        return text_tensor, labels_tensor

    @staticmethod
    def collect_fun(batch):
        batch_text = []
        batch_labels = []

        for ele in batch:
            batch_text.append(ele[0])
            batch_labels.append(ele[1])

        batch_text = torch.stack(batch_text, dim=0)
        batch_labels = torch.stack(batch_labels, dim=0)

        return batch_text, batch_labels
