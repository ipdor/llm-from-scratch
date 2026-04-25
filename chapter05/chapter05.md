---
title: Chapter 05
date: 2026-04-25
tags:
  - LLM
  - AI
  - DeepLearning
---

## 总体流程

![road map](https://raw.githubusercontent.com/ipdor/Pictures/master/20260425112048986.png)

![chapter 5 overview](https://raw.githubusercontent.com/ipdor/Pictures/master/20260425113116619.png)



## 模型评估与交叉熵损失


模型评估的作用：   

1. 通过比较生成的文本和目标文本，用定量的方式确定生成token的质量，表示它们之间的"距离"；   
2. 后期使用的训练函数将会使用这种"距离"来更新模型权重，实现优化的目的。

定量评估模型质量（生成的文本和目标文本的差异）需要使用负的平均对数概率，也即[交叉熵损失cross entropy loss](#交叉熵损失)



## 计算步骤

![Calculating the loss ](https://raw.githubusercontent.com/ipdor/Pictures/master/20260425171205961.png)



**1. 获取模型的未归一化输出 (Logits)**
*   将输入数据送入模型，得到输出，即每个位置上词汇表中所有词的分数（Logits）。
*   这是神经网络最后一层的原始数值输出，但由于数值大小不受限制，无法直接表示为“概率”。

**2. 转换为概率分布 (Softmax)**
*   对 Logits 应用 Softmax 函数。
*   **为什么：** 将无界的 Logits 转换成 0 到 1 之间的概率值，且保证所有可能出现的下一个词的概率总和刚好为 1，方便进行数值评估。

**3. 提取正确目标词的概率 (Target Probabilities)**
*   从所有的概率分数中，单独把“正确的下一个词”（Target）对应的概率提取出来。
*   **为什么：** 训练的终极目标就是让模型准确预测下一个词，所以我们只关心模型给正确答案分配了多高的概率。

**4. 取对数 (Log Probabilities)**
*   对上一步提取出的目标概率应用对数函数（`torch.log`）。
*   **为什么：** 在数学和深度学习优化中，处理对数比直接处理极小的概率连乘要容易得多，能有效避免计算机处理极小数时发生“数值下溢”（Underflow）现象。

**5. 计算平均值 (Average Log Probability)**
*   把当前批次（Batch）里所有词的对数概率相加，然后求平均值（`torch.mean`）。
*   **为什么：** 这样可以将多个词、多个句子的评估结果合并成一个单一的综合评分，代表模型在当前批次上的整体表现。

**6. 乘以 -1 (Negative Average Log Probability)**
*   将算出的平均对数概率乘以 -1，这就得到了交叉熵损失（Cross Entropy Loss）。
*   **为什么：** 因为概率是 0 到 1 之间的小数，取对数后必然是负数。深度学习的习惯是“最小化损失函数”（将损失降到 0）。乘以负号将负数变正后，模型就可以通过不断减小这个正向的损失值，来反向逼近目标词 100% 的预测概率。

在实际编写代码时，PyTorch 的内置函数 `torch.nn.functional.cross_entropy` 会自动在底层一步到位地完成这完整的六个步骤。

<br>

### 交叉熵损失


交叉熵损失*cross entropy loss*: 把平均对数概率average log probability变为负的平均对数概率negative average log probability的操作。

核心上，交叉熵损失是机器学习和深度学习中一种流行的度量方法，用于衡量两个概率分布之间的差异——通常是标签的真实分布（在这里，是数据集中的标记）和来自模型的预测分布（例如，LLM 生成的标记概率）。

实践中，“交叉熵”和“负平均对数概率”这两个术语相关并且经常互换使用。



### 困惑度 (Perplexity) 

困惑度是一种经常与交叉熵损失一起使用的指标，用于评估模型在语言建模等任务中的性能。

它同样衡量用来衡量模型预测的概率分布与数据集中词汇的实际分布之间的匹配程度，但它提供了一种更具解释性的方式，来理解模型在预测序列中下一个标记时的不确定性。与损失相似，较低的困惑度表明模型的预测更接近实际分布。 

困惑度可以通过 `perplexity = torch.exp(loss)` 来计算。 困惑度通常被认为比原始损失值更具解释性，因为它代表了模型在每一步中对其不确定的有效词汇表大小。


## 模型训练

目的：   
模型训练旨在提高与正确目标标记 ID 相对应的索引位置的 `softmax` 概率。（提高生成正确token的概率）


```python
# 打印训练前和目标标记相对应的初始 softmax 概率分数（从模型输出的概率分布中，取出“正确 token 的概率”)
# [0,1,2]               三个 token 的位置索引
# targets[text_idx]     三个位置的正确 token ID !
text_idx = 0
target_probas_1 = probas[text_idx, [0,1,2], targets[text_idx]]
print("Text 1:", target_probas_1)

text_idx = 1
target_probas_2 = probas[text_idx, [0,1,2], targets[text_idx]]
print("Text 2:", target_probas_2)
```

```
Text 1: tensor([7.4540e-05, 3.1061e-05, 1.1563e-05])
Text 2: tensor([1.0337e-05, 5.6776e-05, 4.7559e-06])
```







