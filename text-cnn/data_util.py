import pandas as pd
import torch
import torch.nn as nn
from torchtext import data, vocab
from pyhanlp import *


def label_len(label_file):
    label_file_prefix = 'data/dataset/label/'
    with open(label_file_prefix + label_file + '.txt', 'r') as f:
        label_itos = f.readlines()
    return len(label_itos)


def label_numericalize(data_file, label_file, column):
    # Replace label string in data file with indices.
    with open(label_file, 'r') as f:
        label_itos = f.readlines()
        label_itos = [label.strip('\n') for label in label_itos]
    df = pd.read_csv(data_file)
    df[column] = df[column].replace(label_itos, list(range(0, len(label_itos))))
    df.to_csv(data_file, index=False)
    return df[column]


def replace_files():
    # Replace all files and columns
    data_file_list = ['train.csv', 'dev.csv', 'test.csv']
    label_file_list = ['异常类型', '形成原因', '故障组件']
    data_file_prefix = 'data/dataset/'
    label_file_prefix = 'data/dataset/label/'
    for data_file in data_file_list:
        for label_file in label_file_list:
            label_numericalize(data_file_prefix + data_file, label_file_prefix + label_file + '.txt', label_file)


def label_denumericalize(index, label):
    # Replace a index with its original label string.
    label_file_prefix = 'data/dataset/label/'
    with open(label_file_prefix + label + '.txt', 'r') as f:
        label_itos = f.readlines()
        label_itos = [label.strip('\n') for label in label_itos]
    return label_itos[index]


def load_dataset(args):
    # 自然语言处理预处理的工作流程：
    # 1. Train/Validation/Test数据集分割（已做好）
    # 2. 分词（Tokenization） 文本字符串切分为词语列表
    def hanlp_tokenizer(sentence):
        return [term.word for term in HanLP.segment(sentence) if len(term.word) >= 1]

    # test case
    # print(hanlp_tokenizer("8月24日16:25，国网信通调度监控发现总部人资辅助决策、物资辅助决策、营销辅助决策与综合查询IAS系统IMS健康运行时长及在线用户数指标缺失，同时数据中心ODS库报出实例无法连接告警。17:00，经国网信通公司运维人员重启数据库后，系统指标恢复正常。"))

    # 3. 文件数据导入 (File Loading)
    full_data = pd.read_csv("data/dataset/full_data.csv")
    # Get max length of sentences in dataset
    max_length = max([len(sentence) for sentence in full_data["问题现象"].to_list()])

    label = data.LabelField(dtype=torch.float)
    text = data.Field(sequential=True,
                      tokenize=hanlp_tokenizer,
                      lower=True,
                      include_lengths=True,
                      batch_first=True,
                      fix_length=max_length)
    datafields = [("id", None),
                  ("异常类型", label),
                  ("异常类型细项", None),
                  ("问题现象", text),
                  ("故障组件", None),
                  ("形成原因", None),
                  ("形成原因细类", None),
                  ("原因分析", None),
                  ("故障处理过程", None)]
    train_data, valid_data, test_data = data.TabularDataset.splits(path='data/dataset',
                                                                   train='train.csv',
                                                                   validation='dev.csv',
                                                                   test='test.csv',
                                                                   format='csv',
                                                                   skip_header=True,
                                                                   fields=datafields)
    # Print a sample
    # print(train_data[0].__dict__["问题现象"])

    # 4. 构建词典 (Vocab) 根据训练的语料数据集构建词典
    # 5. 数字映射(Numericalize/Indexify) 根据词典，将数据从词语映射成数字，方便机器学习
    # 6. 导入预训练好的词向量(word vector)
    # 7. 向量映射（Embedding Lookup） 根据预处理好的词向量数据集，将5的结果中每个词语对应的索引值变成词语向量
    if args.embedding == "merge":
        embedding_path = "data/embedding/merge_sgns_bigram_char300.txt"
    elif args.embedding == "baidu":
        embedding_path = "data/embedding/sgns.baidubaike.bigram-char"
    else:
        embedding_path = "data/embedding/sgns.wiki.bigram-char"

    vectors = vocab.Vectors(name=embedding_path, cache='data/embedding')
    text.build_vocab(train_data, vectors=vectors, unk_init=nn.init.normal_)
    label.build_vocab(train_data, vectors=vectors, unk_init=nn.init.normal_)

    # Print vocab sample
    # print(text.vocab.itos[:20])
    # print(text.vocab.stoi)
    # print(text.vocab.vectors[13])

    # Print vocab info
    print("Length of Text Vocabulary: " + str(len(text.vocab)))
    print("Vector size of Text Vocabulary: ", text.vocab.vectors.size())
    print("Label Length: " + str(len(label.vocab)))

    # 8. 分批(Batch) 数据集太大的话，不能一次性让机器读取，否则机器会内存崩溃。解决办法就是将大的数据集分成更小份的数据集，分批处理
    train_iter, valid_iter, test_iter = data.BucketIterator.splits((train_data, valid_data, test_data),
                                                                   batch_size=args.batch_size,
                                                                   sort_key=lambda x: len(x.问题现象),
                                                                   repeat=False,
                                                                   shuffle=True)

    word_embeddings = text.vocab.vectors
    vocab_size = len(text.vocab)

    return text, vocab_size, word_embeddings, train_iter, valid_iter, test_iter
