---
title: Chapter 04
date: 2026-04-14
tags:
  - LLM
  - AI
  - DeepLearning
---

# 总体架构

GPT模型包括：    
1. Embedding layer   
2. Transformer block   
	* Masked multi-head attention等   
3. Output layers   

![image.png](https://raw.githubusercontent.com/ipdor/Pictures/master/20260415161903894.png)


其中Transformer block 包括：   

![image.png](https://raw.githubusercontent.com/ipdor/Pictures/master/20260415161959247.png)

# Layer normalization 层归一化

梯度消失或梯度爆炸 vanishing or exploding gradients 等问题会导致不稳定的训练动态，并使网络难以有效地调整其权重，这意味着学习过程难以找到一组能够最小化损失函数的神经网络参数（权重）。换句话说，网络很难在足够高的程度上学习数据中的潜在模式，从而使其能够做出准确的预测或决策。

因此，可以使用 Layer normalization 层归一化来提高稳定性

层归一化背后的主要思想是调整神经网络层的激活（输出），使其均值为 0，方差为 1，这也称为单位方差。一般应用在 multi-head
attention module 多头注意力模块之前和之后，并且在 final output layer 之前。



![image.png](https://raw.githubusercontent.com/ipdor/Pictures/master/20260416215214727.png)

```python
torch.manual_seed(123)

batch_example = torch.randn(2,5) # 创建两个训练样本，每个样本具有五个维度（特征）

layer = nn.Sequential(nn.Linear(5,6), nn.ReLU())

out = layer(batch_example) # [2,5] * [5,6] + bias -> ReLU([2,6])

print(out)
```

### 层归一化函数

```python
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
```

层归一化对最后一个维度操作，其代表嵌入维度 `em_dim`  

变量 `eps` 是一个在归一化过程中加到方差上的小常数（epsilon），以防止除以零。

`scale` 和 `shift` 是两个可训练参数（与输入具有相同的维度），如果确定这样做可以提高模型在其训练任务上的性能，LLM 会在训练期间自动调整它们。


`unbiased` 默认使用贝塞尔校正（Bessel's correction），因为在分母中使用 `n – 1` 而不是 `n`,所以叫做有偏估计：

```
unbiased=True
```

即：  
$$\frac{1}{N-1}$$
而 LayerNorm 使用的是：  

$$\frac{1}{N}$$


也就是

```
unbiased=False
```

实际在大模型中由于N非常大，这两种计算方式的差异可以忽略。这里是为了和GPT-2模型保持一致而选择 `unbiased=False`


## Different Normalization

**Normalization = 调整数值尺度或形式。**

分为两类： **BN/LN 调分布，Softmax 造概率。**

| 方法             | 归一化对象   | 目的    | 归一化维度 | 解决的问题          |
| -------------- | ------- | ----- | ----- | -------------- |
| BatchNorm (BN) | batch   | 训练稳定  | 统计归一化 | 优化问题（让梯度好学）    |
| LayerNorm (LN) | feature | LLM稳定 | 统计归一化 | 优化问题（让梯度好学）    |
| Softmax        | 输出值     | 概率分布  | 概率归一化 | 建模问题（让输出符合概率论） |

Batch Normalization/Layer Normalization 归一化的目标：   
$$x_{norm} ​= \frac{x - mean}{std}$$

作用：    
- 均值≈0  
- 方差≈1  
- 提高训练稳定性
​
为了解决训练深层网络时**神经网络内部数值分布会不断漂移**：   
- 激活值越来越大  
- 不同层尺度差异巨大  
- 梯度不稳定  
- 学习率难调 

### Softmax Normalization

Softmax不同，是**概率归一化   
- 输出 ≥0 且总和=1   
- **把分数变成概率**

$$
\text{softmax}(z_i)=
\frac{e^{z_i}}{\sum_j e^{z_j}}
$$


用途：   
- Attention 权重   
- token 预测
- **解决建模问题，让输出符合概率论** (❗不用于稳定训练)


**为什么也叫 normalization？**

因为它满足：

$$
\sum p_i = 1
$$

叫：**probability normalization**

但它**不是**：   
- 标准化均值
- 标准化方差

### Layer Normalization（LN）

**跨 feature 归一化**  
- 每个样本 **独立** 计算均值和方差  
- 不依赖 batch

输入
```
(batch, feature)
```
LN需要：对**单个样本**内部的**所有 feature** 求均值

作用：稳定表示

特点：   
- ✔ batch=1 也正常  
- ✔ 训练/推理一致  
- ✔ 适合 Transformer / LLM


### Batch Normalization（BN）

**跨 batch 归一化**   

- 对**同一 feature** 在 **整个 batch** 中求均值和方差   
- 依赖 batch size  

```
(batch, feature) = (32 samples, 512 features)
```

比如有32 张猫图片， BN 问：“这一批猫里，第 j 个特征平均是多少？”，然后统一尺度。    
对整个batch中**所有样本**的**每个feature**分别求均值

作用：稳定梯度、加速训练

特点：   
- ✔ 大 batch 表现好   
- ❌ 小 batch / 推理不稳定
    
常见：CNN



## 激活函数（Activation Function）


![Activation Function](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*WohYvQmfbeH-yFbprE6jCQ.png)


**激活函数 = 给神经网络加入非线性能力的函数。**

如果没有激活函数：

```
Linear → Linear → Linear
```

多层线性层仍然等价于 **一个线性变换**，模型表达能力极弱。

激活函数的作用：

- 引入非线性
- 让网络能学习复杂模式
- 控制梯度传播

通常放在：

```
Linear → Activation → Linear
```

| 激活函数   | 类型    | 特点     | LLM 使用 |
| ------ | ----- | ------ | ------ |
| ReLU   | 硬非线性  | 简单、快   | 少      |
| GeLU   | 平滑非线性 | 连续、稳定  | 常见     |
| SwiGLU | 门控激活  | 信息筛选更强 | 主流     |
- **ReLU**：要么通过，要么丢掉
- **GeLU**：软判断，通过多少
- **SwiGLU**：加一个“开关”，决定让谁通过

### ReLU（Rectified Linear Unit）

$$ReLU(x)=max(0,x)$$

**特点**  
- 负数 → 0  
- 正数 → 保持不变

直觉：**只让正信号通过**，负数输出0

ReLU 在零处有一个尖角，这有时会使优化变得更加困难，尤其是在非常深或具有复杂架构的网络中。

### GeLU（Gaussian Error Linear Unit）

$$GeLU(x)=x⋅P(x)$$

其中 $P(x)$ 是输入为正的概率，$P$是标准高斯分布的累积分布函数cumulative distribution function。

特点：   
不再“硬截断”，而是**平滑地决定保留多少信息**：

```
小值 → 部分通过  
大值 → 基本通过  
负值 → 逐渐抑制
```

类似 **平滑门控（soft gating）**，这种平滑让它允许对模型参数进行更细微的调整。它的形状近似于 ReLU，但对于几乎所有负值（除了大约在 x = –0.75 处）都具有非零梯度。

常见于：Transformer、LLM

### SwiGLU（Swish Gated Linear Unit）

$$SwiGLU(x)=Swish(xW_1​)⊙(xW_2​)$$

核心思想是**门控（Gate）机制**

```
输入  
├─ Linear → Swish  
└─ Linear  
↓  
按元素相乘
```

作用：

- 一条分支决定“信息强度”
- 一条分支作为“门”



## 捷径连接

捷径连接(跳跃连接或残差连接)缓解梯度消失的挑战。具体做法是添加捷径，把一层的输出直接添加到后面层的输出，路径可选、更短，可以绕过某些层。

梯度消失 vanishing gradient 问题指的是梯度（在训练期间指导权重更新）在向后穿过层传播时变得越来越小，使得难以有效训练较早层的问题。

![shortcut connection](https://raw.githubusercontent.com/ipdor/Pictures/master/20260417152146891.png)



## nn.Module 和 nn.ModuleList


### nn.Module

`nn.Module` 是用作模型的模板。为了构建自己的网络，必须继承 `nn.Module` 实现子类。

它会自动跟踪类中定义的所有参数，允许你使用 `.to(device)` 将整个模型移动到 GPU，或者通过 `.parameters()` 提取所有权重以供优化器使用。

关键组件:   
* __init__: Used to define the layers and submodules (e.g., `nn.Linear`, `nn.Conv2d`).    
* `forward()`: A mandatory method that defines how input data passes through the layers.

### nn.ModuleList

[class torch.nn.ModuleList(modules=None)](https://docs.pytorch.org/docs/stable/generated/torch.nn.ModuleList.html)

`nn.ModuleList` 是一个用于以列表格式存储子模块的专业容器。它可以像 Python 列表一样索引，但其中包含的模块会被正确注册，并且可以通过所有 `Module` 方法访问。

解决的问题：标准的Python列表不会“注册”它们包含的模块。如果您将层存储在常规Python列表中，PyTorch将无法“看到”它们的权重，它们在训练期间也不会更新。

**关键组件**:   
* Registration：模块作为模型的一部分被正确注册，确保它们的参数包含在 `.parameters()` 调用中。  

* 无前向传递：与 `nn.Sequential` 不同，`nn.ModuleList` 没有这样的方法。它没有定义层之间的任何内部连接。


## GPT架构

![alt text](https://raw.githubusercontent.com/ipdor/Pictures/master/20260422221222577.png)


* 文本分词 Tokenizing text  
  * 嵌入层 Embedding layer  
  * 位置层 Positional layer  
  * Dropout   
* Transformer 块  
  * 多头注意力机制 Multihead Attention  
  * 神经网络 Feed forward（输入输出维度一致，但内部扩大维度）   
  * （两个模块都分别需要在前后进行进行层归一化和Dropout）
* 输出层  
  * 层归一化  
  * 无bias输出头 out_head ，映射向量

最后需要对输出按概率取对应的 token ID，映射回 Vocabulary，得到预测的单个字符


### Transformer block 结构


![An illustration of a transformer block](https://raw.githubusercontent.com/ipdor/Pictures/master/20260422214846246.png)

Transformer block 的输入输出维度相同。

虽然序列的物理维度（长度和特征大小）在穿过transformer块时保持不变，但每个输出向量的内容被重新编码，以整合来自整个输入序列的上下文信息。

整个block中结合了 `层归一化Layer normalization`, `多头注意力机制MultiHead Attention`，`Dropout`，`Feed Forward` 和`捷径连接shortcut connection`。

* Pre-LayerNorm: 层归一化（LayerNorm）应用于这两个组件的每一个之前，而丢弃法（dropout）应用于它们之后，以对模型进行正则化并防止过拟合。      
* Post-LayerNorm: 原始的transformer模型，在自注意力和前馈网络之后应用层归一化。这种效果更差。


### 独立权重的性能优于权重绑定

原始 GPT-2 架构中使用的一种称为权重绑定 `weight tying` 的概念。这意味着原始 GPT-2 架构在输出层  output layer 中重用了 token 嵌入层的权重。本书的GPT模型没有重用权重，所以本章中间计算得到参数数量为 163 million。

但是，使用独立的层来处理标记嵌入和最终输出通常会导致更好的训练和整体模型性能。   

虽然权重绑定是一种有效的技巧，可以减少模型的内存占用和计算复杂度——这也是原始GPT-2将参数数量精确控制在1.24亿的原因——但在现代大型语言模型中，保持这些层独立是标准做法，以实现更高质量的结果。


