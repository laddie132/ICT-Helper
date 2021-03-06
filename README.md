# ICT故障辅助研判模型
基于Text-CNN模型以及BERT模型，实现从问题现象，预测五个输出：异常类型、异常类型细项、故障组件、形成原因和形成原因细类。

## 依赖
- Python 3.6
```shell script
pip install -r requirements.txt
```

## 制作数据集

```shell script
python build_dataset.py
```

数据集输出路径在`data`文件夹下，包含训练集、验证集、测试集，以及词向量和词典。

### 分词

- 中文HanLP分词 + ChineseWordVectors
- 词典大小1043，其中OOV单词为26个

### 数据集统计

|         | 原始数据大小 | 过滤空数据后 | train | dev  | test |
| - | - | - | - | - | - |
| 3个字段 | 1023         | 600          | 480   | 60   | 60   |
| 5个字段 | 1023         | 596          | 478   | 59   | 59   |

> - 3个字段指输出：异常类型、故障组件、形成原因
> - 5个字段指输出：异常类型、异常类型细项、故障组件、形成原因、形成原因细类

|字段| 异常类型 | 异常类型细项 | 故障组件 | 形成原因 | 形成原因细类 |
|-| - | - | - | - | - |
|标签数| 3 | 11 | 12 | 12 | 109 |

## BERT模型
```shell script
./run_bert.sh
```
使用上述脚本，训练BERT模型，相关代码在`bert`文件夹下，已增加ICT数据的预处理模块，可以直接使用。

> 修改脚本中`TASK_NAME`和`OUTPUT_DIR`可以选择训练任务和输出路径，`TASK_NAME`=[ict_et/ict_fc/ict_rs]，分别表示异常类型、故障组件、形成原因

- 模型输入为`问题现象`
- 模型输出为`异常类型`、`故障组件`和`形成原因`

其中，使用BERT的中文字符级别预训练模型`chinese_L-12_H-768_A-12`，作为参数初始化，并在本ICT数据集上进行fine-tune，参数设置在上述脚本中。

对于三个字段，分别进行单独的训练，由于数据量较少，性能受限，较CNN有一定差距，实验结果见下文。

<img src="https://github.com/laddie132/ICT-Helper/raw/master/imgs/bert.png" width="300" alt=""/>

## Text-CNN模型
```
调用方法及测试用例可参考 text-cnn/main.py
```

使用上述脚本，训练Text-CNN模型，相关代码在`Multitask-CNN`文件夹下。使用HanLP分词器，以及ChineseWordVectors(merge)中文词向量，实现多个单分类器，以及一个多任务分类模型。模型如下：

<img src="https://github.com/laddie132/ICT-Helper/raw/master/imgs/text-cnn.jpg" width="600" alt=""/>

## 多任务模型 (Multitask-CNN)
```
调用方法及测试用例可参考 Multitask-CNN/test_example.py
```

<img src="https://github.com/laddie132/ICT-Helper/raw/master/imgs/multitask-cnn.jpg" width="600" alt=""/>

多任务模型基于上述CNN模型进行改进，采用了多任务学习的方式，使所有分类器都共享同一个Text-CNN编码器，因此使用一个分类模型就可以同时对多个字段进行分类，并且可以通过在不同字段分类器之间共享参数提高模型的整体性能和泛化能力。

## 超参数设置

### BERT
Fine-tune阶段设置如下超参数，其余保持默认。
- 最大句子长度: 128
- Learning rate: 2e-5
- Batch size: 32
- Epochs: 3

### CNN & Multitask-CNN
两个模型使用同一套超参数：
- 激活函数：ReLU
- 预训练词向量维度：300
- 卷积核大小：共有3种大小分别为3, 4, 5的卷积核
- 特征图数量：每种卷积核各有128个特征图，共计384个
- Dropout比率：0.2
- Batch size：32

## 实验结果

1. Baseline（多个单分类器）

| Model    | 异常类型-Acc | 故障组件-Acc | 形成原因-Acc |
| - | - | - | - |
| BERT     | 76.27        | 38.98        | 62.71        |
| Text-CNN | 81.67        | 51.50        | 71.67        |

2. 多任务模型

| Model | 异常类型-Acc | 异常类型细项-Acc | 故障组件-Acc | 形成原因-Acc | 形成原因细项-Acc |
| - | - | - | - | - | - |
| Multitask-CNN |    79.38    |      68.75     |    46.88    |    89.38   |     21.25      |

3. 多任务模型（Top-3）

| Model | 异常类型-Acc | 异常类型细项-Acc | 故障组件-Acc | 形成原因-Acc | 形成原因细项-Acc |
| - | - | - | - | - | - |
| Multitask-CNN |  100.00     |      89.38     |    87.50    |   100.00    |     38.13      |

## 参考

- ChineseWordVectors: https://github.com/Embedding/Chinese-Word-Vectors
- BERT: https://github.com/google-research/bert
- Text-CNN: https://github.com/prakashpandey9/Text-Classification-Pytorch
