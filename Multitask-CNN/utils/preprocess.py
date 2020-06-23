import numpy as np
import torch
from utils.data_reader import TextClsReader


class Preprocess:
    def __init__(self, raw_data_path, vocab_path, embedding_path,
                 train_path, dev_path, test_path, max_text_length, batch_size,
                 num_workers):
        self.raw_data_path = raw_data_path
        self.vocab_path = vocab_path
        self.embedding_path = embedding_path
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.max_text_length = max_text_length
        self.batch_size = batch_size
        self.num_workers = num_workers

    def data_reader(self):
        reader = TextClsReader(raw_data_path=self.raw_data_path,
                               vocab_path=self.vocab_path,
                               train_path=self.train_path,
                               dev_path=self.dev_path,
                               test_path=self.test_path,
                               max_text_length=self.max_text_length)

        train_data_loader = reader.get_dataloader_train(batch_size=self.batch_size,
                                                        num_workers=self.num_workers)
        dev_data_loader = reader.get_dataloader_dev(batch_size=self.batch_size,
                                                    num_workers=self.num_workers)
        test_data_loader = reader.get_dataloader_test(batch_size=self.batch_size,
                                                      num_workers=self.num_workers)

        vocabulary = reader.vocab
        vocab_size = len(vocabulary.vocab)
        label_length = [len(label.labels) for label in reader.label_vocab_list]

        return vocabulary, vocab_size, label_length, train_data_loader, dev_data_loader, test_data_loader, reader

    def load_embedding(self):
        embedding_weight = torch.tensor(np.load(self.embedding_path), dtype=torch.float32)
        print("Embedding size: {}".format(embedding_weight.size()))
        return embedding_weight

    def label_len(self):
        with open(self.label_path) as f:
            label_itos = f.readlines()
        return len(label_itos)
