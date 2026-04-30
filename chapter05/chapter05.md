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

![Train loop](https://raw.githubusercontent.com/ipdor/Pictures/master/20260426172033736.png)

模型训练有八个步骤，从迭代每个 epoch 开始，接着是处理批次、重置梯度、计算损失和新梯度、更新权重，最后以打印损失和生成文本样本等监控步骤结束。


目的：   
模型训练旨在提高与正确目标标记 ID 相对应的索引位置的 `softmax` 概率。（提高生成正确token的概率）。实践中是通过交叉熵损失*cross entropy loss*衡量的


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

### 训练集，验证集

书中附录A A.7 A typical training loop解释了使用验证集（Validation set）的两个主要原因：

1. **调整超参数**：我们在训练过程中会多次使用验证集来寻找最佳的模型设置（如学习率、训练轮数等），以此来优化模型，而不会污染最终只保留作一次性评估的测试集。    
2. **监控过拟合**：如果在训练时，训练集损失持续下降，但验证集损失却停滞或变大，这就直接说明模型只是在“死记硬背”训练数据（也就是过拟合）。


### cross_entropy 参数对shape的要求


下面代码为什么报错Expected target size [2, 50257], got [2, 256]。既然报错shape不匹配，为什么flatten之后就可以了，明明logits 元素要比target_batch更多，flatten之后不是应该更混乱吗

```python
def calc_loss_batch(input_batch, target_batch, model, device):
    # [2, 256]
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    # [2, 256, 50257]
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        #logits.flatten(0,1), target_batch.flatten()
        logits, target_batch
    )
    return loss
```

---

#### 根本原因：`cross_entropy` 强制要求 **C（类别数）在第 1 维**

PyTorch 的 `cross_entropy` 接受的 shape 规则是：

| 参数 | 期望 shape |
|------|-----------|
| `input` (logits) | `[N, C]` 或 `[N, C, d1, d2, ...]` |
| `target` | `[N]` 或 `[N, d1, d2, ...]` |

其中 **C 必须永远在第 1 维**，这是硬性规定，不能协商。

规律就一条：**target shape = input shape 去掉 C 那一维**。所以两个变量的其他维度 N, d1, d2, .. 必须完全匹配。

#### 为什么出错

你传入的是：
```
logits:  [2, 256, 50257]   → PyTorch 理解为 N=2, C=256, d1=50257
target:  [2, 256]          → PyTorch 理解为 N=2,        d1=256
```

PyTorch 把 **256 当成了类别数 C**，然后期望 target 的 d1 维度 = logits 的 d1 维度 = `50257`，所以报错：

```
Expected target size [2, 50257], got [2, 256]
```

它根本没意识到 50257 才是词表大小，因为你把它放错位置了。

---

#### 为什么 flatten 之后就对了

```python
logits.flatten(0, 1)  # [2, 256, 50257] → [512, 50257]
target.flatten()      # [2, 256]        → [512]
```

flatten 后：
```
logits:  [512, 50257]  → N=512, C=50257  ✅ C 在第 1 维
target:  [512]         → N=512           ✅ 匹配
```

现在 **50257（词表大小）终于出现在第 1 维**，PyTorch 才正确识别为类别数。

---

#### 另一种不 flatten 的写法

用 `permute` 把维度换到正确位置：

```python
# [2, 256, 50257] → [2, 50257, 256]
loss = torch.nn.functional.cross_entropy(
    logits.permute(0, 2, 1),  # [N, C, d1] 格式
    target_batch              # [N, d1] = [2, 256]
)
```

两种写法数学上完全等价，flatten 更常见，因为更简洁。

## 文本输出随机性

### 温度缩放

![temperature scale](https://raw.githubusercontent.com/ipdor/Pictures/master/20260428172536056.png)


温度缩放本质上就是将 logits 除以一个大于 0 的数字，这个数字被称为**温度（temperature）**


```python
def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)
```

大于 1 的温度会导致更均匀分布的 token 概率，而小于 1 的温度会导致更自信（更尖锐或更具峰值）的分布。

应用非常小的温度（例如 0.1）将导致更尖锐的分布，使得 multinomial 函数的行为几乎 100% 地选择最可能的 token（这里是 "forward"），这接近于 argmax 函数的行为。    
同样，温度为 5 会导致更均匀的分布，其中其他 token 被更频繁地选择。这可以为生成的文本增加更多样性，但也更经常导致毫无意义的文本。


结合 PyTorch 中的 `multinomial` 采样函数，可以按照温度缩放后的概率分数的概率来选择 token，以增加输出的多样性。

### Top-k sampling

为了提升生成文本之类，减少温度缩放导致的无意义文本，可以通过 Top-k sampling 缩小采样范围。

![topK sampling](https://raw.githubusercontent.com/ipdor/Pictures/master/20260428174821391.png)

In top-k sampling, we can restrict the sampled tokens to the top-k most likely tokens and exclude all other tokens from the selection process by masking their probability scores.    
在 Top-k 采样中，我们可以将采样范围限制在概率最高的 k 个标记内，并通过屏蔽其他标记的概率分数，将它们从选择过程中排除。

Top-k 方法将所有未选中的 logits 替换为负无穷大值 (-inf)，这样在计算 softmax 值时，非 top-k token 的概率分数为 0，而剩余的概率总和为 1。（和掩码计算的原理相同相同）



## 训练和推理差异

![text generation](https://raw.githubusercontent.com/ipdor/Pictures/master/20260430171400402.png)

训练阶段每个位置都在预测：   
"Hello" 对应位置应该预测","      
"，" 对应位置应该预测"I"     
"I" 对应位置应该预测"am"   
...... 

推理阶段只需要预测出结尾一个token "a"

无论训练还是推理，y都是x向后滑动一个窗口的内容。大小都是`context_size`





```python
inputs = torch.tensor([[16833, 3626, 6100],   # ["every effort moves",
                       [40,    1107, 588]])   #  "I really like"]

targets = torch.tensor([[3626, 6100, 345  ],  # [" effort moves you",
                        [1107,  588, 11311]]) #  " really like chocolate"]
```



![from input text to output text](https://raw.githubusercontent.com/ipdor/Pictures/master/20260425144904485.png)


图中"every"预测了"effort"    
"effort"预测了"moves"   
"moves"预测了"you"

因此生成 `y_pred = ["effort", "moves", "you"]`

### 生成词的差异

```
目标数据是输入数据向后slide一个token的数据。关键在于最后一个token，它是模型应该输出的正确数据。（X）
```

对，这个理解基本是对的。自监督学习，标注就藏在数据本身里，非常elegant。

实际上**每个位置都在同时训练**，不只是最后一个。

比如输入是 `[A, B, C, D]`，target是 `[B, C, D, E]`，那么：
- 位置0的logit要预测B
- 位置1的logit要预测C
- 位置2的logit要预测D
- 位置3的logit要预测E

cross_entropy在每个位置都算一个loss，最后取平均。**所以一个sequence里有多少个token，就有多少个训练信号**，这也是为什么语言模型训练数据利用率很高。

"最后一个token最关键"这个印象，可能是从推理时来的——推理时确实只看最后一个位置的输出来决定生成什么词。但训练时所有位置都参与。

（理解的差异来自书和作者视频，书和视频中多次提到一次生成一个token。所以误以为训练时也是只看最后生成的token。但实际上并不是，训练和推理不同）



## 自己的理解

* 梯度到底是个什么东西，backward之后发生了什么

backward是利用pytorch内部的计算图和链式法则，从 dL/dy_pred 一直计算到梯度。
梯度是gradient，严格来说是dL/dw，dL/db这种loss相对于训练参数的变化率。梯度越大说明受该参数影响越大，梯度为正说明改变方向一致，梯度为负说明loss和参数的改变是反向的。参数每次的更新是 w = w - lr * dL/dw，其中lr是学习率，代表每次向这loss下降的方向走多少。

**为了让loss减小，梯度为正说明w增大会让loss增大，所以要把w往反方向走——w = w - lr * grad，减去一个正数，w变小。梯度的符号告诉你loss的坡往哪边高，参数更新永远往坡低的方向走。**

* loss下降意味着模型内部什么变了

loss改变意味着模型的所有训练参数改变了，包括但不限于把token ID投影到高维空间的嵌入层参数、高维向量映射回词表的输出层参数、多头注意力模型参数、FeedForward层参数等。loss下降说明通过这些参数计算得到的输出token和目标token差距变小，简单来说就是进步了，准确率上升了。

* 为什么训练能work，模型怎么就"学会"了

我理解是1.每次训练后的loss定量衡量了模型输出质量好坏，可以用数学确定训练质量。2.反向传播计算gradient能知道参数对最终loss的影响（正反、大小），结合learning rate可以让参数每次逐渐向loss减小的方向改进。


> 第三个问题"为什么训练能work"，你说的是机制层面的答案——loss能衡量质量，梯度能指明方向，所以参数能收敛。这是对的。
但还有一个更深的问题你没提到，也是目前研究界都没完全想清楚的：
> 
> **为什么预测下一个token，能让模型学会推理、翻译、写代码这些东西？**
> 
> 训练目标只是"猜下一个词"，但训练出来的模型能做的远超这个。这背后的直觉是：人类写的文本本身就是人类思维的压缩，要在大量文本上持续猜对下一个词，模型被迫内化文本背后的结构、逻辑、知识——不然loss降不下去。预测任务只是表面，真正被优化的是对人类语言和思维模式的建模能力。
但这只是直觉，没有人能精确解释为什么scale up之后会出现emergent capabilities。这是目前LLM领域最有趣也最开放的问题之一。


## 模型保存和加载

### 自己的模型


如果只需要加载后推理比较简单，只要保存和加载模型本身权重。


```python
# 保存权重
torch.save(model.state_dict(), "model.pth")

# 加载权重
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()
```

如果需要后续训练，还需要优化器的权重参数


```python
# 保存权重，包含模型和优化器状态
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    },
    "model_and_optimizer.pth"
)


# 加载权重，包含模型和优化器状态
checkpoint = torch.load("model_and_optimizer.pth", map_location=device)
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
model.train()
```



### OpenAI 参数

利用作者提供的工具下载参数到本地，读取架构设置settings和模型权重params，然后分别加载。

省略`load_weights_into_gpt`，太长。

参考 [ch05.ipynb](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/01_main-chapter-code/ch05.ipynb)

```python
import urllib.request
url = (
    "https://raw.githubusercontent.com/rasbt/"
    "LLMs-from-scratch/main/ch05/"
    "01_main-chapter-code/gpt_download.py"
)
filename = url.split('/')[-1]
urllib.request.urlretrieve(url, filename)


# 架构设置 settings, 模型权重 params
from gpt_download import download_and_load_gpt2
settings, params = download_and_load_gpt2(
    model_size="124M", models_dir="gpt2"
)

# 更改架构设置
model_configs = {
    "gpt2-small(124M)":{"emb_dim":768, "n_layers":12, "n_heads":12},
    "gpt2-medium(355M)":{"emb_dim":1024, "n_layers":24, "n_heads":16},
    "gpt2-large(774M)":{"emb_dim":1280, "n_layers":36, "n_heads":20},
    "gpt2-xl(1558M)":{"emb_dim":1600, "n_layers":48, "n_heads":25},
}
model_name = "gpt2-small(124M)"
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])
NEW_CONFIG.update({"context_length": 1024})
NEW_CONFIG.update({"qkv_bias": True})
gpt = GPTModel(NEW_CONFIG)
gpt.eval()

# load_weights_into_gpt函数省略，太长
load_weights_into_gpt(gpt, params)
gpt.to(device)
```


