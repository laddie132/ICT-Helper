# ICT故障辅助研判模型
基于Text-CNN模型以及BERT模型，实现从问题现象，预测五个输出：异常类型、异常类型细项、故障组件、形成原因和形成原因细类。

## 依赖
- Python 3.6
```shell script
pip install -r requirements.txt
```

## 流程图

TODO

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

## CNN模型
```shell script
TODO
```
使用上述脚本，训练Text-CNN模型，相关代码在`text-cnn`文件夹下。使用HanLP分词器，以及ChineseWordVectors(merge)中文词向量，实现多个单分类器，以及一个多任务分类模型。模型如下：

## 实验结果

1. Baseline（多个单分类器）

| Model    | 异常类型-Acc | 故障组件-Acc | 形成原因-Acc |
| - | - | - | - |
| BERT     | 76.27        | 38.98        | 62.71        |
| Text-CNN | 81.67        | 51.50        | 71.67        |

2. 多任务模型

| Model | 异常类型-Acc | 异常类型细项-Acc | 故障组件-Acc | 形成原因-Acc | 形成原因细项-Acc |
| - | - | - | - | - | - |
| CNN   |              |                  |              |              |

3. 多任务模型（Top-3）

| Model | 异常类型-Acc | 异常类型细项-Acc | 故障组件-Acc | 形成原因-Acc | 形成原因细项-Acc |
| - | - | - | - | - | - |
| CNN   |              |                  |              |              |

## 参考

- ChineseWordVectors: https://github.com/Embedding/Chinese-Word-Vectors
- BERT: https://github.com/google-research/bert
- Text-CNN: https://github.com/prakashpandey9/Text-Classification-Pytorch
