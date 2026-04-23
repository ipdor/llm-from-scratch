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


## 计算流程

1. 注意力分数



2. 注意力权重


3. 上下文向量


## Transformer模型的注意力机制


### 一、先说本质（别急着看Q/K/V）

> **注意力机制在做一件事：**
> 
> 👉 对于当前这个 token：  
> **“我应该从其他 token 那里拿多少信息？”**

---

#### 举个例子

The animal didn’t cross the street because it was tired.

处理 **“it”** 时：

- 应该多看 “animal”
- 少看 “the”

👉 这就是 attention 的目标

### 二、Q / K / V 到底是什么（核心理解）

在 Transformer 里，每个 token 会变成三种向量：

#### 1）Query（Q）：我在找什么

👉 当前 token 的“需求描述”

- “我想找一个主语”
- “我想找和我有关的词”

#### 2）Key（K）：我能提供什么

👉 每个 token 的“标签 / 特征”

- “我是一个名词”
- “我是主语”
- “我是情绪词”

#### 3）Value（V）：我真正的内容

👉 如果你关注我，我给你什么信息

|向量|作用|
|---|---|
|Q|提问|
|K|索引|
|V|内容|


### 三、计算流程（一步一步来）

我们用一个 token（比如 “it”）来看。

#### Step 1：计算相似度（谁跟我相关）

$$ score=Q⋅K $$

👉 当前 token 的 Q  
👉 和所有 token 的 K 做点积

解释：

> “我（Q）和你（K）匹不匹配？”

结果：

- it · animal → 高
- it · tired → 中
- it · the → 低

#### Step 2：归一化（变成权重）

用 softmax：

$$ α_i​=softmax(Q⋅K_i​) $$

👉 变成：

- 所有权重加起来 = 1
- 可以理解为“注意力分布”

#### Step 3：加权求和（拿信息）

$$ context=∑α_i ​V_i $$​

👉 用刚才的权重：

- 多拿 animal 的信息
- 少拿 the 的信息

👉 得到：

> **context vector（上下文向量）**

### 四、为什么这样就能得到“上下文”？

#### 本质在这一步：

$$ context=加权平均（所有token的信息） $$

👉 权重不是乱来的，而是：

> **由“语义匹配（Q·K）”决定的**

所以：

- 和我相关的 → 权重大 → 影响大
- 不相关 → 权重小 → 几乎忽略

👉 结果：

> context = “我需要的那部分信息的汇总”

### 五、一个更直觉的类比（非常关键）

#### 想象你在查资料

你是 Query：   
- “我要找原因”

每本书是：  
- Key：这本书讲什么
- Value：书里的内容

#### 流程：

1. 用 Q 去匹配 K（找相关书）
2. 选出最相关的（softmax）
3. 把这些书的内容（V）综合起来

👉 最终得到：

> 你需要的信息（context）

### 六、为什么要分成 Q / K / V（三个而不是一个）

这是设计精髓。

#### 如果不用 Q/K/V，会怎样？

如果直接用 embedding：

👉 “匹配”和“内容”是同一个东西

问题：

- 不够灵活
- 无法表达复杂关系


#### 分开之后的好处

👉 可以学三种不同的映射：

- Q：我关心什么
- K：我是什么属性
- V：我提供什么信息

👉 本质：

> **解耦（decoupling）**


### 七、为什么点积 + softmax 这么有效？

因为它把问题变成：

> **“相似度 → 权重 → 加权求和”**


这在数学上是：

👉 一个“软选择（soft selection）机制”


- 不像 if-else（硬选择）
- 是连续可导的（可以训练）

### 八、再往深一点（你可以理解这个了）

整个 attention 可以看成：

> **一个可学习的动态路由系统**

每个 token：

- 动态决定：
    - “我该听谁的”
    - “听多少”

👉 这比：

- CNN（固定局部）
- RNN（固定顺序）

强太多

### 九、一句话打穿整个流程

> **Query 决定你想要什么，Key 决定别人能提供什么，Value 是实际信息，attention 用相似度决定权重，再加权汇总得到 context。**

## 标准注意力机制提升


### 因果注意力

#### 掩码
对于许多大语言模型（LLM）任务，当预测序列中的下一个词元时，你希望自注意力机制只考虑出现在当前位置和之前的词元。

1. 盖住矩阵对角线以上的token;   
2. 重新归一化

注意力分数（未归一化） -> 注意力权重（已归一化） -> 被掩盖的注意力分数（未归一化） -> 被掩盖的注意力权重（已归一化）

一种改良方法是： 

Attention scores(unnormalized)     
-> 把对角线以上元素变成−∞ -> Masked attention scores (unnormalized)   
 -> 应用Softmax归一化 -> Masked attention weights(normalized) 



#### Dropout

深度学习中的 _Dropout_ 是一种在训练期间随机忽略（有效地“丢弃”）选定隐藏层单元的技术。这种方法通过确保模型不会过度依赖于任何特定的隐藏层单元集来帮助防止过拟合。只在训练时用

对掩码注意力权重应用dropout




### 多头注意力


最终维度 `(batch, seq_len, embedding_dim)`

2个头，每个输出维度是2，因此最终embedding_dim是2\*2=4

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


