---
title: Chapter 07
date: 2026-05-07
tags:
  - LLM
  - AI
  - DeepLearning
---

## Roadmap

LLM指令和预期输出示例：

![example](https://raw.githubusercontent.com/ipdor/Pictures/master/20260507153028711.png)

Roadmap:

![road map](https://raw.githubusercontent.com/ipdor/Pictures/master/20260507153329098.png)

1. 准备数据集   
2. 微调LLM    
3. 评估LLM

## 为监督指令微调准备数据集

指令微调涉及在输入-输出对的数据集上训练模型，有不同的方法对记录进行格式化，图7.4是其中两种。

![two different ways of formatting](https://raw.githubusercontent.com/ipdor/Pictures/master/20260507171147257.png)

大型语言模型指令微调中提示风格的比较。Alpaca风格（左侧）采用结构化格式，为指令、输入和响应设置了明确的区域；而Phi-3风格（右侧）则使用更简洁的格式，通过指定的`<|user|>`和`<|assistant|>`标记来实现。


## 批处理

指令微调的批处理过程要稍微复杂一些，需要我们创建自定义的 `collate` 函数，稍后我们将其插入到 `DataLoader` 中。我们实现这个自定义的 `collate` 函数，是为了处理我们的指令微调数据集的特定要求和格式。

![batching process](https://raw.githubusercontent.com/ipdor/Pictures/master/20260507180806809.png)

图 7.6 实现批处理过程涉及的五个子步骤：
* (2.1) 应用提示模板    
* (2.2) 使用前面章节中的分词     
* (2.3) 添加填充标记   
* (2.4) 创建目标 token IDs，以及     
* (2.5) 替换 -100 占位符标记，以在损失函数中屏蔽填充标记。

## 填充方案

![sophisticated padding approach](https://raw.githubusercontent.com/ipdor/Pictures/master/20260507183106749.png)

和前面不同的是，指令微调会开发一个自定义的 `collate` 函数，它将每个批次中的训练示例填充到相同的长度，同时允许不同的批次具有不同的长度。    
**这种方法通过仅扩展序列以匹配每个批次中最长的序列，而不是匹配整个数据集中的最长序列，从而最大限度地减少了不必要的填充**。

### 2.1 组成完整Instruction + Response文本    


### 2.2 对每条完整文本编码成token ids序列


### 2.3 使用填充标记调整至相同长度  

需要对每个batch进行填充，长度按照组内最长长度

```python
# 处理一个batch的token_id数据
def custom_collate_draft_1(batch, pad_token_id=50256, device="cpu"):
    # 找到批次中最长的序列 
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst = []

    # 填充并准备输入
    for item in batch:
        new_item = item.copy()      # 避免对原始数据修改
        new_item += [pad_token_id]  # 额外填充标记，为下个版本做铺垫

        padded = (
            new_item + [pad_token_id] * (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1]) # 移除之前添加的额外填充标记
        inputs_lst.append(inputs)
    
    # 使用torrch.stack转换为 (batch_size, seq_len) 的标准矩阵格式
    # 将输入列表转换为张量并将其转移到目标设备
    inputs_tensor = torch.stack(inputs_lst).to(device)
    return inputs_tensor
```

### 2.4 为训练创建目标标记ID

除了每个 input_batch 之外，还需要对应的 target_batch，由输入批次右移一个token得到。也就是需要LLM每次预测下个单词。最右侧的空位填充`<|endoffile|>`。


![input to target token ids](https://raw.githubusercontent.com/ipdor/Pictures/master/20260507222535766.png)

类似 `custom_collate_draft_1`, `custom_collate_draft_2` 主要对结尾的padded裁去第一个token, 使input batch右移1得到output batch

```python
def custom_collate_draft_2(batch, pad_token_id=50256, device="cpu"):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id] # 在 [data, pad_token_id] 的基础上填充，最终一定会多一个pad_token_id

        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs_lst.append(torch.tensor(padded[:-1])) # 为输入batch去掉结尾多余token
        targets_lst.append(torch.tensor(padded[1:])) # 右移1得到输出batch
    
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor
```    

### 2.5 将填充令牌替换为占位符


![replace padding tokens with placeholders](https://raw.githubusercontent.com/ipdor/Pictures/master/20260507223256135.png)

除第一个填充标记外，其他所有替换为占位符`-100`

```python
logits_1 = torch.tensor(
    [[-1.0, 1.0],
    [-0.5, 1.5]]
)
targets_1 = torch.tensor([0, 1]) # Correct token indices to generate
loss_1 = torch.nn.functional.cross_entropy(logits_1, targets_1)
print(loss_1)
```

使用两个token，输出为
```
tensor(1.1269)
```

如我们所料，添加一个额外的 token ID 会影响损失计算： 

```python
logits_2 = torch.tensor(
[[-1.0, 1.0],
[-0.5, 1.5],
[-0.5, 1.5]]
)
targets_2 = torch.tensor([0, 1, 1])

targets_3 = torch.tensor([0, 1, 0]) # 替换成非0/1/-100的数值会报错
loss_4 = torch.nn.functional.cross_entropy(logits_2, targets_3)
print(loss_4)
```

使用3个token，但是targets最后一位被替换为 `-100`，输出为： 

```
tensor(1.1269)
loss_1 == loss_3: tensor(True)
```

进行这一步替换的原因是： PyTorch 中交叉熵函数的默认设置是 `cross_entropy(..., ignore_index=-100)`。**这意味着它会忽略targets中标记为 -100 的目标**。我们利用这个 `ignore_index` 来忽略那些我们用来填充训练示例，使得每个批次拥有相同长度的额外 end-of-text（填充）标记。


但是，我们希望在目标中保留一个 `50256` （end-of-text）标记 ID，因为它有助于 LLM 学习生成文本结束标记，我们可以将其用作响应完成的指示器。

<br>



还有一种训练技巧：屏蔽目标文本的指令部分以减少过拟合

![mask out the instruction section](https://raw.githubusercontent.com/ipdor/Pictures/master/20260507234724808.png)

图 7.13 左侧：我们在训练期间分词然后提供给 LLM 的格式化输入文本。右侧：我们为 LLM 准备的目标文本，我们可以选择性地屏蔽指令部分，这意味着用 -100 也就是 ignore_index 值替换相应的 token IDs。

通过屏蔽对应指令的 LLM 目标 token IDs，交叉熵损失仅针对生成的响应目标 IDs 进行计算。因此，模型被训练为专注于生成准确的响应，而不是死记硬背指令，这有助于减少过拟合。

但是这种方法存在分歧，有些研究人员证明不屏蔽指令有利于 LLM 的性能。如 “Instruction Tuning With Loss Over Instructions” (https://arxiv.org/abs/
2405.14394)










