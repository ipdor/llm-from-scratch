---
title: Chapter 06
date: 2026-05-02
tags:
  - LLM
  - AI
  - DeepLearning
---

## 总体流程

![road map](https://raw.githubusercontent.com/ipdor/Pictures/master/20260502092049296.png)


## fine-tuning 分类

![LLM fine tuning](https://raw.githubusercontent.com/ipdor/Pictures/master/20260502090254021.png)


**Instruction fine-tuning**: training a language model on a set of tasks using specific instructions to improve its ability to understand and execute tasks described in natural language prompts.    
指令微调涉及使用特定的指令在一系列任务上训练语言模型，以提高其理解和执行自然语言提示中所描述任务的能力


在分类微调中（如果你有机器学习的背景，可能已经熟悉这个概念），模型被训练来识别一组特定的类别标签，例如“垃圾信息”和“非垃圾信息”。

关键点在于，分类微调模型仅限于预测其在训练期间遇到过的类别，而指令微调模型通常可以承担更广泛的任务。

## 准备数据

存在几个问题需要处理：

1. 两种类别的样本数量不同。"ham"(not spam)的样本数量远远大于"spam"        
  解决方案： 对"ham"抽样，采集和"spam"相同数量的样本

2. 分割不同数据集，训练集，验证集和测试集    
  解决方案：`random_split`函数shuffle后，按比例分别划分为 `train`, `valid`, `test` set  


```python
def random_split(df, train_frac, validation_frac):
    # 分为 train, valid, test集，
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    train_df = df[: train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df

train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
```

3. 保存文件     

```python
train_df. to_csv("train.csv", index=None)
```

## 数据加载器

![text to padded token ids](https://raw.githubusercontent.com/ipdor/Pictures/master/20260502113416459.png)

由于每条数据长度不一致，需要处理，有两种办法：    
1. 统一截断为最短序列长度   
2. 统一填充为最大序列长度

为了不损失信息，这里采用方法2填充。

### 数据集

创建数据集重点是：   
1. 使用tokenizer编码    
2. 计算最大序列的长度，注意模型最大支持1024的序列      
3. 把超过最大token id序列长度的序列截断（主要是1024）
4. `__getitem__` 返回 `(token ids, label)` tuple

这里验证集和测试集最长长度使用训练集的最大长度，但这是可选的。可以不截断，只要保证不超过模型的 `context_length = 1024`



![training batch process](https://raw.githubusercontent.com/ipdor/Pictures/master/20260502170814780.png)

注意这里的数据 `(x, y)`，`x` 是最大序列长度的token id序列 token ids； `y` 是标签0或者1，表示是否为诈骗邮件。

数据集创建好之后直接可以创建数据加载器

```python
from torch.utils.data import DataLoader

num_workers = 0     # 确保兼容性
batch_size = 8
torch.manual_seed(123)

train_loader = DataLoader(
    dataset = train_dataset,
    batch_size = batch_size,
    shuffle = True,   # 验证集和测试集不需要
    num_workers = num_workers,
    drop_last = True  # 验证集和测试集设为False
)
```

