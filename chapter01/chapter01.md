---
title: Chapter 01
date: 2026-04-09
tags:
  - LLM
  - AI
---

# Roadmap

![Roadmap](https://raw.githubusercontent.com/ipdor/Pictures/master/20260423194822006.png)


# 要点

1. 大语言模型 (LLM) 的核心概念

定义：LLM 是一种设计用来理解、生成和响应人类语言的深度神经网络。

基本原理：其核心训练任务通常非常简单——即利用语言的顺序特性，基于前文“预测序列中的下一个词”。

2. 构建与使用 LLM 的三个主要阶段

第一阶段：数据准备与架构实现。处理输入数据并编写模型的基础代码。 

第二阶段：预训练 (Pretraining)。在海量无标签的原始文本上进行自监督学习，得到一个具备基础语言生成能力的“基础模型 (Foundation model)”。

第三阶段：微调 (Fine-tuning)。在较小的特定带标签数据集上进一步训练模型。主要分为分类微调（如垃圾邮件识别）和指令微调（让模型学会遵循用户指令并回答问题）。

3. Transformer 架构与模型分类

![Transformer](https://raw.githubusercontent.com/ipdor/Pictures/master/20260423194721455.png)

基础架构：现代 LLM 主要基于 2017 年提出的 Transformer 架构，包含用于处理输入的编码器（Encoder）和用于生成输出的解码器（Decoder），核心是能够捕捉长距离依赖关系的自注意力机制（Self-attention）。

BERT 类模型：仅使用 Transformer 的编码器部分，通过预测句子中被掩码（隐藏）的单词进行训练，擅长文本分类等任务。

GPT 类模型：仅使用 Transformer 的解码器部分，是一种“自回归”模型，根据前面的输入从左到右逐个词生成文本。它们擅长文本生成，并在零样本（Zero-shot）和少样本（Few-shot）任务中表现出色。

4. 海量数据与涌现能力 (Emergent Behavior)

GPT 等模型在包含数千亿词汇的海量且多样化的数据集（如网页爬取数据、书籍、维基百科）上进行训练。   

虽然模型仅仅被训练用来“预测下一个词”，但它们自发学会了翻译、分类或代码编写等未被显式教导过的任务，这种现象被称为涌现能力。

# AI, ML, DL

Artificial intelligence(AI)泛指所有带有类人智能的系统human-like intelligence;

Machine learning(ML) 是指能自己从数据中学习规则的算法， 擅长处理结构化数据，包括 `[features, label]`；

Deep learning(DL) 使用神经网络来解析和生成文件, 擅长处理非结构化数据，包括人类语言、图片、音频、视频等。


![alt text](https://raw.githubusercontent.com/ipdor/Pictures/master/20260423193725840.png)
