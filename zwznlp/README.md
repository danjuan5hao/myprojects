# 为基本的NLP任务制作框架
用于应付快速制作baseline， 代码绝大部分抄袭[fancy_nlp](https://github.com/boat-group/fancy-nlp)

## 文本分类任务

## 语义相似度任务

## 实体标注任务
### 输入类型1： 字符级别的标注


### 输入类型2： 词级别的标注


## 信息抽取任务
### 输入类型1： 短文本，其中实体只出现一次 
**sample**:   
{'text': '《邪少兵王》是冰火未央写的网络小说连载于旗峰天下', 'spo_list': [{'predicate': '作者', 'object_type': {'@value': '人物'}, 'subject_type': '图书作品', 'object': {'@value': '冰火未央'}, 'subject': '邪少兵王'}]}

### 输入类型2： 长文本，其中实体可能出现多次
**sample**:  
file1:   

{
    "train_1": "Safety and efficacy of intravenous bimagrumab in inclusion body myositis (RESILIENT): a randomised, double-blind, placebo-controlled phase 2b trial\tBimagrumab showed a good safety profile, relative to placebo, in individuals with inclusion body myositis but did not improve 6MWD. The strengths of our study are that, to the best of our knowledge, it is the largest randomised controlled trial done in people with inclusion body myositis, and it provides important natural history data over 12 months."
}

file2:  
{
    "train_1": [
            {
                "head": {
                    "word": "Bimagrumab",
                    "start": 148,
                    "end": 158
                },
                "rel": "PositivelyRegulates",
                "tail": {
                    "word": "inclusion body myositis",
                    "start": 230,
                    "end": 253
                }
            }
        ]
}

## isA抽取任务

## 阅读理解任务

## 文本摘要任务