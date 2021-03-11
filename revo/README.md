# Rasa EVOlution
流程其实主要还是两个train和process
通过指定1. 数据集名字， 2. NLU components
## train
不支持一步到位的train。只能通过指定数据集名字以及训练的组件名进行训练。  
**需要保存模型以及得到测试集上的表现。**
1. preprocessor
通过数据集的名字，整理数据集，输出在temp/data/"数据集名字"/.../“component任务”/下。
2. components
各个components通过各自的任务名加载各自的数据集。 

## process
**应该能支持批处理和流处理**