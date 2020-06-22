# ICT故障辅助研判模型
基于Text-CNN模型以及BERT模型，实现从"问题现象"智能预测"异常类型"、"故障组件"和"形成原因"。

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

## BERT模型
```shell script
./run_bert.sh
```
使用上述脚本，训练BERT模型，相关代码在`bert`文件夹下，已增加ICT数据的预处理模块，可以直接使用。

> 修改脚本中`TASK_NAME`和`OUTPUT_DIR`可以选择训练任务和输出路径

- 模型输入为`问题现象`
- 模型输出为`异常类型`、`故障组件`和`形成原因`

## CNN模型
```shell script
TODO
```
使用上述脚本，训练Text-CNN模型，相关代码在`text-cnn`文件夹下

## 实验结果
TODO

## 参考
- BERT: https://github.com/google-research/bert
- Text-CNN: https://github.com/prakashpandey9/Text-Classification-Pytorch
