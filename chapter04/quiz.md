# Quiz for first four chapters

Topics covered:    
* Large Language Model Architectures  
* Text Preprocessing and Tokenization  
* Self-Attention and Multi-Head Attention  
* Layer Normalization and Stability  
* Transformer Block Implementation  
* Gradient Management and Shortcut Connections  
* Text Generation Mechanics  

Source: Gemini notebooklm



1. In the context of the original Transformer architecture, which component is typically omitted when building a GPT-like model designed for text generation?

The Encoder module
```
That's right!   
GPT-style architectures simplify the original Transformer by using only the decoder modules, as they do not require an separate encoder for source-to-target mapping.
```

2. What is the primary advantage of Byte Pair Encoding (BPE) over simple word-based tokenization?

It can handle out-of-vocabulary words by breaking them into subword units.

```
That's right!
BPE avoids the 'unknown' token problem by iteratively merging frequent character sequences into subwords, allowing any word to be represented.
```

3. When implementing scaled dot-product attention, why do we divide the dot product by $\sqrt{d_K}$ ?

To prevent the dot products firom growing too large, which would lead tovanishing gradients in the softmax.

```
That's right!
Scaling ensures that the softmax function operates in a region where gradients aresignificant, preventing training from stagnating.
```


4. In a causal attention mechanism, how is the model prevented from 'looking ahead' at future tokens?     

B. By physically removing future tokens from the input tensor during everyforward pass.   

```
Not quite   
The entire sequence is processed at once for efficiency, and masking is used to simulatethe sequential constraint.
```

D. By applying a mask that sets future token attention scores to $-\infty$ beforethe softmax.   

```
Right answer  
Using $-\infty$ ensures that after the softmax operation, the probability of attending tofuture tokens becomes exactly zero.
```


5. What is the specific purpose of 'shortcut connections' (skip connections) in deep LLM architectures?

A. To facilitate the flow of gradients during backpropagation and mitigate the vanishing gradient problem.   

```
That's right!   
By providing an alternate path for gradients, shortcuts ensure that weights in earlier layers can be effectively updated.
```

6. In the GPT-2 architecture, how does the FeedForward module typically transform the hidden dimension (embedding size) internally?

C.
It expands the dimension by a factor of 4 and then contracts it back to the original size.

```
That's right!
The first linear layer projects the embedding into a higher-dimensional space (e.g., 768 to 3072) before the second layer restores the original size.
```

7. Which activation function is preferred in modern GPT models over the traditional Rectified Linear Unit (ReLU)?

GELU (Gaussian Error Linear Unit)

```
That's right!
GELU provides a smoother gradient and is the standard activation function for GPT-2 and other modern architectures.
```

8. When computing layer normalization, what is the role of the epsilon ($ϵ$) term?

It is a small constant added to the variance to prevent division by zero.

```
That's right!
Normalization requires dividing by the standard deviation; if variance is zero, epsilon ensures numerical stability.
```

9. How do GPT models represent the order of tokens in a sequence since the Transformer architecture is inherently permutation-invariant?

By adding positional embeddings to the token embeddings.

```
That's right!
Positional embeddings provide a unique vector for every index in the context window, allowing the model to distinguish between positions.
```


**置换不变性**（Permutation-invariant）是指改变输入数据的顺序，不会影响最终的输出结果。    
在 Transformer 的自注意力机制中，模型是同时并行处理所有的单词的，就好像它们被装在一个“词袋”里一样。如果你打乱一个句子中单词的顺序，自注意力机制计算出的这些词之间的关联得分和特征是完全一样的，因为它天生无法感知单词在句子中的先后顺序。
这就是为什么我们必须引入额外的机制来打破这种置换不变性。正如我们在 GPT 模型架构中看到的，输入文本在经过标记嵌入（Token embedding）之后，还必须加上位置嵌入（Positional embedding）
。这样组合后，输入到 Transformer 块中的向量就同时包含了“这个词是什么”以及“这个词在句子的哪里”这两种信息
。

10. In the implementation of multi-head attention, what is the 'head_dim' if the total embedding dimension is 768 and there are 12 heads?


768

```
Not quite
768 is the total embedding dimension ($d_{out}$).
```

64
```
Right answer
The head dimension is calculated as the total dimension divided by the number of heads: 768/12=64.
```

11. What does the term 'logits' refer to in the output of a GPT model?

The unnormalized probability scores for every token in the vocabulary.
```
That's right!
Logits are the raw values from the final linear layer; they are converted to probabilities using the softmax function.
```

12. True or False: Modern LLMs like GPT-2 apply layer normalization after the self-attention and feed-forward modules (Post-LayerNorm).

A.
True
```
Not quite   
Post-LayerNorm was used in the original Transformer, but modern models favor Pre-LayerNorm for better stability.
```

B.
False    
```
Right answer   
GPT-2 and modern Transformers apply layer normalization before the modules (Pre-LayerNorm) to improve training dynamics.

这里说的“after the self-attention and feed-forward modules”是在Transformer块内部，先层归一化再输入self-attention/feed-forward。而不是Transformer块后面的Final LayerNorm
```


13. Which process describes 'greedy decoding' in text generation?

D.
Consistently selecting the token with the highest probability/logit at each step.
```
That's right!
Greedy decoding picks the most likely next token without considering overall sequence probability or variance.
```

14. What is the primary function of the Query (Q), Key (K), and Value (V) matrices in self-attention?

They transform input embeddings into specialized vectors for calculating relevance and retrieving information.

```
That's right!
Queries are matched against Keys to find weights, which are then applied to Values to produce a context vector.
```

15. In the context of data preparation, why is a sliding window approach used?

To create input-target pairs where the target is shifted by one token from the input.
```
That's right!
This method generates the sequences needed for next-token prediction training.
```

16. When initializing a GPT model with 124 million parameters, what consumes the largest portion of those parameters?


The embedding layers and the final output layer.
```
That's right!
Due to the large vocabulary size (50,257), the weights mapping to and from this space account for a significant portion of the total parameters.
```


17. How does the 'MultiHeadAttention' class improve upon a single-head attention mechanism?

B.
It reduces the computational cost of the attention scores.
```
Multi-head attention is typically more computationally intensive than single-head attention.
```

C.
It enables the model to attend to information from different representation subspaces simultaneously.
```
That's right!
Each head can focus on different types of relationships (e.g., grammar, meaning, reference) at different positions.
```

18. In Chapter 4, what is the purpose of using the `.view()` and `.transpose()` methods in the multi-head attention implementation?

To reshape and align tensors so that matrix multiplication can be performed across multiple heads in parallel.
```
That's right!
Reshaping into (batch, heads, tokens, dim) allows PyTorch to use efficient batched matrix multiplications.
```

19. What happens to the mean and variance of activations after a layer normalization operation is applied (ignoring scale and shift)?

Mean becomes 0 and variance becomes 1.
```
That's right!
Normalization centers the data and scales it to unit variance to stabilize neural network training.
```


20. True or False: A GPT model can generate an infinite number of tokens in a single iteration of the forward pass.

A.
False
```
That's right!
GPT models generate text iteratively, one token at a time, by appending the new token to the input for the next cycle.
```



