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

## 增加分类头


![image.png](https://raw.githubusercontent.com/ipdor/Pictures/master/20260503130019840.png)

Fine-tune 之前需要先增加分类头，把原来映射到50257大小的词表的输出层out layer改为映射到只有两个分类0和1的输出层。


We could technically use a single output node since we are dealing with a binary classification task. However, it would require modifying the loss function, as I discuss in “Losses Learned—Optimizing Negative Log-Likelihood and CrossEntropy in PyTorch” (https://mng.bz/NRZ2).      
这里是二进制，所以也可以改为只用1个输出头表示0或者1，但是这样需要修改loss函数。并且为了更具有通用性，这里使用单个头。   
通用性是指把输入划分为任意数量的类别，如：将新闻分为3种”科技“，”运动“，”政治“。


> **微调选定层与所有层的对比**        
> 由于我们从预训练模型开始，因此没有必要微调所有的模型层。在基于神经网络的语言模型中，较低层通常捕获适用于广泛任务和数据集的基本语言结构和语义。因此，仅微调最后的层（即靠近输出的层，这些层更针对细微的语言模式和特定任务的特征）通常足以使模型适应新任务。一个很好的附带好处是，仅微调少量层的计算效率更高。感兴趣的读者可以在附录 B 中找到更多关于应该微调哪些层的信息，包括相关实验。



把模型整个冻结之后替换成新的 `out_head`，默认 `requires_grad = True`，这样也就是只有输出层能被训练。

```python
for param in model.parameters():
param.requires_grad = False

num_classes = 2
model.out_head = torch.nn.Linear(
    in_features=BASE_CONFIG["emb_dim"], # 输入不变，还是768
    out_features=num_classes            # 输出改为2，代表0和1
)
```

从技术上讲，训练我们刚刚添加的输出层就足够了。然而，正如我在实验中发现的那样，微调额外的层可以显著提高模型的预测性能。我们还将最后一个 Transformer 块以及连接该块到输出层的最终 LayerNorm 模块配置为可训练状态。（有关更多详细信息，请参阅附录 B）



![image.png](https://raw.githubusercontent.com/ipdor/Pictures/master/20260503133303113.png)


```python
# "从技术上讲，训练我们刚刚添加的输出层就足够了。
# 然而，正如我在实验中发现的那样，微调额外的层可以显著提高模型的预测性能。"

for param in model.trf_blocks[-1].parameters():
  param.requires_grad = True
for param in model.final_norm.parameters():
  param.requires_grad = True
```

对于 `batch_size=1, context_length=4` 的一个输入:     
因为替换了 `out_head` 输出层，以前会输出 `[1, 4, 50257]` 形状的张量，现在变为 `[1, 4, 2]`

对于每个输入，只需要最后关注一个token，代表是否是诈骗，而不是每个输出行(这里一共有4行)

![image.png](https://raw.githubusercontent.com/ipdor/Pictures/master/20260503135137396.png)


GPT 模型使用的因果注意力掩码（causal attention mask）将标记的注意力焦点限制在其当前位置及其之前的位置，确保每个标记只能受其自身和之前标记的影响。因此，每个序列中的最后一个标记累积了最多的信息，因为它是唯一一个能够访问前面所有标记数据的标记。


## **计算分类损失和准确率 (Calculating the classification loss and accuracy)**


![image.png](https://raw.githubusercontent.com/ipdor/Pictures/master/20260503140054260.png)


从输出logits到label（是否是诈骗）的转换和之前的算法相同，`softmax` 转换为概率后 `argmax` 获取最大值的下标，代表标签0/1，是否是诈骗。唯一的区别是维度是2维而不是50257维。

在初始随机权重的情况下，正确率接近 50% , 相当于二分类下随机猜测。因此我们需要微调训练提升准确率。然而，在开始微调模型之前，我们必须定义在训练期间要优化的损失函数。


## 在监督数据上微调模型


训练循环和之前用于预训练的整体训练循环相同；唯一的区别是，我们现在计算的是分类准确率，而不是生成样本文本来评估模型。

![image.png](https://raw.githubusercontent.com/ipdor/Pictures/master/20260503153552508.png)


```python
def train_classifier_simple(model, train_loader, val_loader, optimizer, device, 
                       num_epochs, eval_freq, eval_iter):
    train_losses, val_losses, train_accs, val_accs = [], [], [], [] #1
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs): # 1. each training epoch
        model.train()
        for i,(input_batch, target_batch) in enumerate(train_loader): # 2. each bach
            optimizer.zero_grad()   # 3. reset loss
            loss = calc_loss_batch(input_batch, target_batch, model, device) # 4. calc loss on batch
            loss.backward()  # 5. backward
            optimizer.step()  # 6. update
            global_step += 1
            examples_seen += input_batch.shape[0]   # new: 跟踪样本数量

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss.item())  # 放入数据而不是tensor，避免后面numpy出现gpu兼容性问题
                val_losses.append(val_loss.item())
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}"
                )

        # 7. print losses 每个epoch之后跟踪准确率
        train_accuracy = calc_accuracy_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_accuracy = calc_accuracy_loader(
            val_loader, model, device, num_batches=eval_iter
        )
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen

# 计算 train_loss, val_loss 用于模型评估
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )
    model.train()
    return train_loss, val_loss
```


打印训练过程中Loss和准确率变化：

![image.png](https://raw.githubusercontent.com/ipdor/Pictures/master/20260503194656448.png)


打印最终准确率

```python
train_accuracy = calc_accuracy_loader(train_loader, model, device)
val_accuracy = calc_accuracy_loader(val_loader, model, device)
test_accuracy = calc_accuracy_loader(test_loader, model, device)
print(f"Training accuracy: {train_accuracy*100:.2f}%")
print(f"Validation accuracy: {val_accuracy*100:.2f}%")
print(f"Test accuracy: {test_accuracy*100:.2f}%")
```

```
Training accuracy: 97.21% 
Validation accuracy: 95.97% 
Test accuracy: 96.33%
```


### 选择 epoch 数量

之前，在我们启动训练时，我们将 epoch 数量设置为 5。epoch 的数量取决于数据集和任务的难度，并没有普遍适用的解决方案或建议，尽管通常将 5 个 epoch 作为一个好的起点。如果模型在最初的几个 epoch 后过拟合，你可能需要减少 epoch 的数量。相反，如果趋势线表明验证损失可能会随着进一步的训练而改善，你应该增加 epoch 的数量。在这个具体案例中，5 个 epoch 是一个合理的数字，因为没有出现早期过拟合的迹象，且验证损失接近于 0。


![image.png](https://raw.githubusercontent.com/ipdor/Pictures/master/20260503195001094.png)


### 过拟合问题

从训练集和验证集的准确率，明显判断出训练过程中出现过拟合问题，

```
Ep 1 (Step 000100): Train loss 0.020, Val loss 2.592
Training accuracy: 100.00% | Validation accuracy: 47.50%
Ep 2 (Step 000250): Train loss 0.004, Val loss 3.756
....

Ep 5 (Step 000600): Train loss 0.000, Val loss 5.464
Training accuracy: 100.00% | Validation accuracy: 50.00%
Training completed in 2.90 minutes.
```

原因排查：

1. 冻结模型    
出现过拟合的原因是没有冻结模型，在对一个大型预训练模型的全部参数做微调，但训练数据量很小。    

结果是模型直接记住了训练集，预训练学到的语言特征被破坏，验证集上退化为乱猜。


**排查发现一开始忘记了冻结模型，也就是在整个模型上训练。但是改正后还有同样问题，说明这个不是真正导致过拟合的原因**。

```python
for param in model.parameters():
param.requires_grad = False
```

2. 冻结模型出问题  

可能没有正确冻结。确认后没问题


3. 模型参数没有成功加载

打印参数确认，没问题

```python
# GPT-2 预训练权重的 token embedding 有非常明显的结构
# 随机初始化的 std 约为 0.02，预训练权重的 std 约为 0.04~0.05

print(model.tok_emb.weight.mean().item())
print(model.tok_emb.weight.std().item())
```

4. 数据问题

排查了函数等没问题之后，说明模型没问题，是数据问题。

训练集 loss 趋近于 0，验证集准确率却 50%，说明很可能是 train/val 数据切分有问题。

然后检查了两个 loader 是否有数据重叠、dataset和dataloader创建之后，终于发现是typo导致`SpamDataset`数据集类出现变量污染问题。

**应该使用正确切分后的数据`self.data`，结果忽略前缀导致使用了原始数据`data`，包含所有样本**。

```python
class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
self.encoded_texts = [tokenizer.encode(x) for x in self.data["Text"]]   # 这里data要写self.data!!!!
```


结果就是：

`self.encoded_texts` → 来自全局 `data`，包含所有样本的 token

`self.data["Label"]` → 来自对应的 train/val CSV，label 是正确切分后的

三个 Dataset 实例都在对同一批 token 编码，但 label 各自不同，导致 val/test 的输入和标签完全错位，准确率自然退化为随机的 50%。训练集恰好因为顺序一致所以能强行记住，val 就完全乱了。


## 推理

推理函数的代码和之前准备数据集的函数`SpamDataset`类似：   
1. 数据编码   
2. 截断（如果超长）   
3. 填充填充符id（这里是50256）    
4. 转换格式`[batch,seq_token]`并转移设备     
5. 输入模型     
6. 对`logits`使用`argmax`得到结果


```python
# 预期 spam
text_1 = (
"You are a winner you have been specially"
" selected to receive $1000 cash or a $2000 award."
)

print(classify_review(text_1, model, tokenizer, device, 
                      max_length=train_dataset.max_length
))

# 预期 not spam
text_2 = (
"Hey, just wanted to check if we're still on"
" for dinner tonight? Let me know!"
)
print(classify_review(
text_2, model, tokenizer, device, max_length=train_dataset.max_length
))
```


### 准确率没问题，但是测试用例全部spam?

注意填充的padding token id！！！！

训练时用 5026 填充，推理时用 50256 填充，模型在训练中学到了 5026 是 padding 的语义，推理时看到一堆 50256 就完全蒙了，输出退化成固定预测 spam。

解决方案：改完重新训练！

```python
# ❌ 训练时
class SpamDataset(Dataset):
    def __init__(self, ..., pad_token_id=5026):   # typo！少了一个 2

# ✅ 推理时
def classify_review1(..., pad_token_id=50256):    # 正确的 GPT-2 eos token
```