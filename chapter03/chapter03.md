---
title: Chapter 03
date: 2026-04-09
tags:
  - LLM
  - AI
  - DeepLearning
---

## 总体流程

1. 为输入的每个token $x_i$计算关于$x_j$的注意力分数$W_{ij}$   
2. 把注意力分数归一化，计算注意力权重    
3. 计算context vector上下文向量。  

对于下标是i的Target词元$Z_i = z^{(1)} + z^{(2)} + .... z^{(n)}$。其中$z^{(1)} = x^{(i)} * w_{1i}$,  $z^{(2)} = x^{(i)} * w_{2i}$, .......


## The “self” in self-attention
“self” = 在同一个序列内部做关联

self-attention = 每个元素根据“同一输入内部的其他元素”来更新自己

比如：

The animal didn’t cross the street because it was too tired.

一句话中it要指谁，👉 要看句子里的其他词（animal）


Self-Attention 做的事情是：   
- 对每个词（token）
- 去“看”同一句话里的其他词
- 判断谁和当前词语更相关

👉 比如：  
- “it” 会关注 “animal”
- “tired” 也会关注 “animal”

---

所以：

> **“self” = 从“自己这句话内部”找关系**

## RNN和self-attention异同

传统模型（RNN）：  
- 只能一步一步传信息（慢 + 容易丢）

self-attention：   
- **任意两个位置直接交互**

本质：  
> 从“链式传递” → “全连接关系图”


从计算机学上，self-attention本质是：
> **构建一个“序列内部的全连接加权图”**


**旧 attention = 跨序列（input → output）**  
**self-attention = 序列内部**


## 为什么只用self-attention

可以用多层self-attention实现更复杂结构

- 第1层：词和词关系
- 第2层：短语关系
- 第3层：句子结构

## 点积=向量相似程度

计算score就是计算Query 和 K有多匹配
score=Q⋅K

ab两个向量分别是从原点出发的箭头/射线，他们的点积a⋅b=∣a∣∣b∣cosθ


关键在 cosθ：   
cosθ = 1 → 完全同方向（最相似）   
cosθ = 0 → 垂直（无关）  
cosθ = -1 → 完全相反（最不相似）


两个向量越相似 -> 空间中两条直线方向越一致，他们的点积就越大，本质上是“一个向量在另一个方向上的投影有多大”


## Transformer模型的注意力机制

![self-attention computation](https://raw.githubusercontent.com/ipdor/Pictures/master/20260427133537619.png)



### 计算过程


1. 注意力分数 `attn_scores = queries @ keys`

2. 注意力权重 `attn_weights = softmax(attn_scores)`

3. 上下文向量 `context_vec = attn_weights @ value`


**第一步：计算相关性分数**

用当前 token 的 Q 与序列中所有 token 的 K 做点积：

$$\text{score}_i = Q \cdot K_i$$

点积越大，表示这两个 token 在语义上越"匹配"。

实际使用时会除以 $\sqrt{d_k}$（K 的维度的平方根）做缩放，防止维度过大时点积值过大导致梯度消失：

$$\text{score}_i = \frac{Q \cdot K_i}{\sqrt{d_k}}$$

**第二步：归一化为权重**

对所有分数做 softmax，使权重之和为 1：

$$\alpha_i = \text{softmax}(\text{score}_i)$$

这一步把"相关程度"变成了一个概率分布，即注意力权重。

**第三步：加权汇总**

用权重对所有 token 的 V 做加权求和：

$$\text{output} = \sum_i \alpha_i V_i$$

结果是一个上下文向量，融合了当前 token"最关心的"那些位置的信息。


### 上下文向量context vector是什么，代表初步预测的、和每个token关联最近的词表中的token吗？**

它不是初步预测的词，跟词表没有关系。

每个token进入attention之前，是一个孤立的向量，只包含"这个词本身"的信息。经过attention之后，每个token得到一个新的向量——这就是context vector——它现在包含的是"**这个词在当前上下文里应该是什么意思**"。

举个例子：

> "The bank by the river" vs "The bank approved the loan"

"bank"这个词的原始embedding是固定的，但经过attention之后，两句话里"bank"的context vector是不同的——因为它分别从"river"和"loan"这些词那里聚合了不同的信息。

所以context vector的本质是：**用周围token的信息，对当前token的表示做了一次加权重写，得到当前token在上下文中的含义**。   
它仍然在embedding空间里，跟词表完全没关系。真正跟词表发生关系，是最后一步——把最后一层的输出乘以一个`(emb_dim, vocab_size)`的矩阵，才变成logits，才有"预测哪个词"这件事。


### 注意力机制与 QKV

#### 它在解决什么问题

语言里的词不是孤立的。"它"这个词出现时，模型需要知道它指的是什么——这取决于上下文中其他词的含义。注意力机制要解决的就是：**处理当前这个 token 时，应该从序列里的哪些位置汲取信息，汲取多少。**

以这句话为例：

> The animal didn't cross the street because it was tired.

处理 "it" 时，"animal" 是关键，"the"、"street" 几乎无关。注意力机制的目标就是自动发现这种依赖关系，而不是手动规则。


#### Q、K、V 分别是什么

每个 token 的 embedding 经过三个不同的线性变换，得到三个向量：

**Query（Q）**：当前 token 的"提问"。表达的是"我在寻找什么样的信息"。处理 "it" 时，它的 Q 可以理解为"我需要找到我指代的对象"。

**Key（K）**：每个 token 对外暴露的"索引"。表达的是"我这个位置能匹配什么样的查询"。"animal" 的 K 大致表示"我是一个可被指代的名词主语"。

**Value（V）**：每个 token 真正携带的内容。如果查询命中了某个 Key，就从对应的 V 里取信息。

三者的关系可以用一个检索类比来理解：你拿着 Q 去图书馆查找，K 是每本书的标签，V 是书的正文内容。查找时你用 Q 匹配 K 来决定"看哪几本"，然后从 V 里综合内容。


#### 为什么要拆成三个向量而不是一个

这是设计里最关键的地方。如果直接用原始 embedding 互相比较，那么"用于匹配"和"用于传递内容"是同一个东西，表达能力受限。

分开之后，模型可以独立学习三件事：怎么提问（Q）、怎么被检索（K）、提供什么内容（V）。同一个 token 完全可以在不同维度上有不同的表达——例如某个词作为被指代的对象（K 方向）和作为信息来源（V 方向）可以是不同的语义特征。这种解耦让模型的表达能力大幅提升。

#### softmax + 加权求和为什么可训练

注意力的计算全程是连续可导的。softmax 输出连续权重，加权求和是线性操作，整个过程可以通过反向传播端到端训练。相比之下，如果用硬选择（比如只取得分最高的那个 token），梯度就断了，无法训练。

这种"软选择"机制是注意力机制能被集成进深度学习框架的根本原因。


## 掩码注意力 Masked attention

实践中使用掩码注意力(Masked attention)和 `Dropout` 改进单头注意力机制。

### Mask

![Mask](https://raw.githubusercontent.com/ipdor/Pictures/master/20260427133831229.png)

对于许多大语言模型（LLM）任务，当预测序列中的下一个词元时，你希望自注意力机制只考虑出现在当前位置和之前的词元。

1. 盖住矩阵对角线以上的token;   
2. 重新归一化

注意力分数（未归一化） -> 注意力权重（已归一化） -> 被掩盖的注意力分数（未归一化） -> 被掩盖的注意力权重（已归一化）

一种改良方法是： 

Attention scores(unnormalized)     
-> 把对角线以上元素变成−∞ -> Masked attention scores (unnormalized)   
 -> 应用Softmax归一化 -> Masked attention weights(normalized) 



### Dropout

深度学习中的 _Dropout_ 是一种在训练期间随机忽略（有效地“丢弃”）选定隐藏层单元的技术。这种方法通过确保模型不会过度依赖于任何特定的隐藏层单元集来帮助防止过拟合。只在训练时用

对掩码注意力权重应用dropout

![Dropout](https://raw.githubusercontent.com/ipdor/Pictures/master/20260427133738962.png)


### 多头注意力

最终维度 `(batch, seq_len, embedding_dim)` 每个头分别负责一部分维度的计算。

例如，`embedding_dim=4`， 有2个注意力头，则每个头的输出维度是`4/2=2`，各自计算数据的尺寸为 `(batch, seq_len, head_dim=2)` 。最终合并`(batch, seq_len, embedding_dim=4)`。


#### Multi-Head 的意义

单头注意力一次只能关注一种"相关性模式"。多头（Multi-Head Attention）就是把上面这整套 QKV 流程并行跑 $h$ 次，每个头有自己独立的 $W_Q, W_K, W_V$，最后把所有头的输出拼起来再做一次线性变换。

这样不同的头可以各司其职：有的头学语法依存关系，有的头学指代关系，有的头学位置邻近性。模型不需要你指定这些，它自己在训练中分工。


#### 怎么理解 embedding dimension?

embedding dimension = 每个 token 的特征向量长度


注意力机制

输入：inputs `[batch, num_tokens, token_dim]` 

每行用 `token_dim` 个维度描述一个token（embedding token）, 一共 `num_tokens` 行/个token。这样的数据有 `batch` 批

输出: context_vecs `[batch, sequence, embedding]]`

一次处理 `batch` 句话，每句话长度 `sequence`， 每句话中每个单词用 `embedding` 个特征描述


比如用4个维度描述一个人：  

```
[身高, 年龄, 收入, 幽默感]
```

语言模型在做类似事情：

```
[语法性,  
 情感,  
 抽象程度,  
 动作相关性,  
 ...]
```

（真实维度不可解释，但概念类似）

---

👉 embedding dimension = **语义空间维度**


为什么需要很多维？

如果只有：`embedding_dim = 1`

所有词只能排成一条线：

```
bad ---- neutral ---- good
```

表达能力极弱。

如果：`embedding_dim = 768`

词可以分布在：768维空间

模型能表达：

- 语义
- 语法
- 上下文
- 风格
- 情绪
- 领域知识
* .......


#### 数据维度处理


