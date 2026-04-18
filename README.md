# LLM From Scratch — Study & Implementation

This repository contains my study notes and code implementations while reading the book  
**"Build a Large Language Model (From Scratch)"**.

The goal of this project is to deeply understand how large language models work by re-implementing key components from scratch, rather than using high-level deep learning frameworks.

---

## 📚 Reference

- Book: Build a Large Language Model (From Scratch)
- Author: Sebastian Raschka
- Github Repositary: [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)

---

## 🎯 Purpose

- Understand the internal mechanisms of Transformer-based language models
- Reimplement core components step by step
- Strengthen low-level understanding of deep learning and LLMs
- Practice building models without relying on high-level abstractions

---

## 📂 Structure

The repository is organized by chapters of the book:

|Chapter | Maincontent | All Code + Supplementary|
|---|---|---|
|Chapter 1| Transformer architecture, GPT Vs. BERT, Roadmap of building a large language model(LLM)||
|Chapter 2| Tokenizing text, Data loader, Token embeddings, Byte pair encoding (BPE), Data sampling with a sliding window|[./chapter02](https://github.com/ipdor/llm-from-scratch/chapter02)|
|Chapter 3| Why we need self-attention mechanism, Attention weights, Causal attention, Dropout, Multi-head attention |[./chapter03](https://github.com/ipdor/llm-from-scratch/chapter03)|
|Chapter 4| GPT model,  Layer normalization, GELU activation, Feed forward network, Shortcut connections, Transformer block|[./chapter04](https://github.com/ipdor/llm-from-scratch/chapter04)|



![book structure](https://camo.githubusercontent.com/f3c959d1ac09015899f56611653558b85801475664b555413030cdddaa0ecf34/68747470733a2f2f73656261737469616e72617363686b612e636f6d2f696d616765732f4c4c4d732d66726f6d2d736372617463682d696d616765732f6d656e74616c2d6d6f64656c2e6a7067)

## ⚙️ Tech Stack

- Python 3.x
- NumPy
- PyTorch  
- Jupyter Notebook

## ⚠️ Disclaimer

This repository is for educational purposes only.  
It is not an official implementation of any production-grade LLM.

## 🧠 Notes

Each notebook includes:
- Step-by-step implementation
- Explanations in comments
- Small experiments for intuition building