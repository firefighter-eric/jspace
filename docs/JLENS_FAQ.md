# J-lens 常见问题与界面读法

本文集中回答本项目实际使用 J-lens 时最容易混淆的问题。完整数学定义与
J-space 的实验边界参见[方法原理](METHOD.md)，运行方式参见
[快速开始](QUICKSTART.md)。

## 1. 当前项目使用的模型和 Lens 是什么

- 模型：本地 `Qwen/Qwen3.5-4B`，默认路径为
  `~/models/Qwen/Qwen3.5-4B`，也可用 `JSPACE_MODEL_PATH` 覆盖。
- J-lens 核心代码：Anthropic `anthropics/jacobian-lens` 参考实现。
- 预拟合 Lens：`Qwen3.5-4B_jacobian_lens_n1000.pt`。
- 权重来源：Anthropic 官方 walkthrough 指定的
  `neuronpedia/jacobian-lens` 仓库、`qwen-n1000` revision。
- Lens 元数据：`d_model=2560`、`n_prompts=1000`、源层 `0..30`。

因此，准确说法是“Anthropic 官方教程采用、由 Neuronpedia 托管的预拟合
Lens”，而不是“由 Anthropic Hugging Face 账号发布的权重”。最终 `.pt`
只记录矩阵、层号、hidden size 和 prompt 数，不足以单独证明全部拟合参数。

## 2. Qwen tokenizer 怎样划分中文输入

本地 Qwen3.5-4B tokenizer 对下面文本：

```text
中国的首都是北京
```

编码结果是：

```text
Token IDs: [96328, 102357, 96878, 98116]
Token:     中国 | 的首 | 都是 | 北京
```

Tokenizer 没有为这次 `encode` 自动加入 BOS；完整 decode 可以无损还原原文。
网页中的 Position 按 tokenizer 位置编号，而不是按汉字编号。

## 3. 第 n 列表示什么

第 `n` 列以输入 token `x_n` 为标签，单元格使用第 `l` 层、第 `n` 个位置的
post-block residual hidden state：

```math
h_{l,n}\in\mathbb{R}^{2560}
```

由于 causal attention，`h_l,n` 可以包含前缀 `x_0..x_n` 的信息，但看不到
`x_{n+1}` 及之后的 token。它是当前位置对已见前缀的上下文化状态，不是
“每个历史 token 各贡献多少”的归因图。

中间层单元格计算：

```math
\operatorname{logits}_{l,n}
=W_U\operatorname{norm}(J_lh_{l,n})
```

它应读作“这个内部状态通常倾向被语言化成哪些 token 或概念”。只有最终层
Raw vocabulary 是该位置真实的 next-token logits；位置 `n` 的最终 logits
用于预测 `n+1`。

如果要知道哪个历史 token 造成了某个候选，需要额外使用 attribution、
activation patching 或消融，J-lens 单独不能回答。

## 4. 为什么中间层候选和最终输出不同

真实模型后续计算为非线性的 prompt-specific 映射：

```math
\operatorname{logits}_{final,n}=W_U\operatorname{norm}(h_{L,n})
```

J-lens 使用在大量文本和位置上平均的线性矩阵：

```math
\operatorname{logits}_{lens,l,n}
=W_U\operatorname{norm}(\bar J_lh_{l,n})
```

两者不同的主要原因是：

1. `J_l` 是线性近似，不能重演后续 attention、MLP 和归一化的全部非线性。
2. 它是约 1000 条样本上的全局平均，不是当前句子的精确 Jacobian。
3. 它汇总当前位置对当前及未来目标位置的影响，目标是读取可语言化概念，
   不是逐层复制真实 next-token 分布。
4. 当前权重路径表明拟合数据来自 Wikitext；中文叙事可能存在分布偏移。

校验实现是否正确时，应比较同一 Position 的 `Layer 31 + Raw vocabulary` 与
底部“最终层下一 token”。两者必须一致；中间层不要求一致。

## 5. Readable、Raw vocabulary 和红色排名是什么

### Readable

使用官方 `mask_display=True`，隐藏空白、标点、特殊 token 和不可读片段，
但不重新计算或归一化 logits。页面显示的是过滤后的首个可读候选。

例如原始排名为：

```text
#1  不可读 token
#2  不可读 token
...
#19 不可读 token
#20 car
```

页面显示 `car²⁰`，含义是 `car` 是第一个可读候选，但在未过滤完整词表中
真实排名为第 20。`20` 不是概率、token ID 或显示列表序号。

### Raw vocabulary

不应用显示过滤，直接展示完整词表真实 Top-10：

```text
#1, #2, ... #10
```

切换 Readable / Raw vocabulary 不重新运行模型；服务端在一次分析中同时返回
两组候选。已预计算 rank 轨迹的 token 可以固定查看；为控制内存与响应体积，
服务端每次最多跟踪 2,048 个候选，超过上限的低频候选仍可显示，但不能固定。

### 颜色

- 黄到绿、再到深蓝：候选在完整词表中的排名由高到低，对数刻度。
- 红色上标：未过滤完整词表的精确排名。
- 粉色框或背景：当前选择或 pinned 状态，不代表模型置信度。
- 顶部绿、黄、红圆点：运行时 ready、loading、error。

## 6. J-lens 是多层 MLP 吗

不是。学习到的部分是一组按源层索引的线性矩阵，没有隐藏层、激活函数或
bias：

```text
Layer 0  → J_0
Layer 1  → J_1
...
Layer 30 → J_30
Layer 31 → I（最终层，不保存 J）
```

每个矩阵尺寸为：

```math
J_l\in\mathbb{R}^{2560\times2560}
```

查看 Layer 16 时只执行 `h_16 @ J_16.T`，不会串联
`J_16 → J_17 → ... → J_30`。31 个 FP16 矩阵共有约 2.03 亿个元素，对应当前
约 406 MB 的 Lens 文件。

## 7. 每层的 J 是否不同

是。每个矩阵直接描述该源层到最终层的平均敏感度：

```math
J_l\approx\mathbb{E}
\left[\frac{\partial h_L}{\partial h_l}\right]
```

所有矩阵形状相同，但数值不同。早层的矩阵需要概括更多后续层的信息传输，
晚层只概括少量剩余层。只有最终层严格使用单位映射 `I`。

对单个输入而言，从某层到最终层的真实 Jacobian 可以由后续模块局部
Jacobian 的链式乘积理解；保存的 `J_l` 则是整个端到端导数在语料和位置上的
平均，并不是保存矩阵之间的乘积。

## 8. J 的输入是不是 hidden state

是，输入是 decoder block 输出后的 residual hidden state，而不是单独的
attention、MLP、attention 权重或 KV cache。

模型运行整句时，每层记录：

```text
[batch, sequence_length, hidden_size]
```

例如 42 个 token：

```text
[1, 42, 2560]
```

J-lens 对每个位置分别应用同一个 `J_l`：

```text
[42, 2560] @ [2560, 2560].T → [42, 2560]
```

页面 Layer 0 是第一个 Transformer block 执行后的状态。它通常对应 Hugging
Face `output_hidden_states=True` 中的 `hidden_states[1]`；
`hidden_states[0]` 通常是 block 之前的 embedding/residual。

## 9. 转到 vocabulary 使用哪个 lm_head

使用原始 Qwen3.5-4B 自己的 final RMSNorm、`lm_head`、tokenizer 和 vocabulary，
没有额外训练词表分类器：

```text
h_l,n
  → J_l
  → Qwen final RMSNorm
  → Qwen lm_head
  → vocabulary logits
```

公式是：

```math
\operatorname{logits}_{l,n}
=W_U\operatorname{norm}(J_lh_{l,n})
```

Lens 文件只保存各层 `J_l`。中间层和最终层的差异来自线性 transport 近似，
不是因为换了 `lm_head`。

## 10. J 具体怎样拟合出来

对源层 `l`、源位置 `p` 和最终层目标位置 `p'`，局部 Jacobian 是：

```math
\frac{\partial h_{L,p'}}{\partial h_{l,p}}
\in\mathbb{R}^{2560\times2560}
```

完整跨位置 Jacobian 太大，官方实现将它压缩为每层一个矩阵。对一条 prompt，
代码使用的估计量是：

```math
J_l^{(x)}=
\frac{1}{|P|}
\sum_{p\in P}
\sum_{\substack{p'\in P\\p'\ge p}}
\frac{\partial h_{L,p'}}{\partial h_{l,p}}
```

这里对当前及未来目标位置求和，再对有效源位置求平均。默认有效位置跳过前
16 个 attention-sink 位置和最后一个没有 next-token target 的位置。

实现流程：

1. 冻结 Qwen，运行一次保留计算图的前向传播。
2. 在最终 residual 的一个输出维度、所有有效目标位置放置 one-hot cotangent。
3. 使用 `torch.autograd.grad` 向所有源层同时反向传播。
4. 反向梯度自动汇总源位置对当前及未来目标位置的影响。
5. 对有效源位置平均，得到矩阵的一行。
6. 默认一次并行计算 8 行；`2560 / 8 = 320` 次 backward 得到完整矩阵。
7. 对每条 prompt 重复，最后对约 1000 条 prompt 的矩阵取平均。
8. 每个梯度行转为 FP32 并在 CPU 上累加，保存时默认转换为 FP16。

它没有监督标签、交叉熵或优化器。本质上是直接测量冻结模型的输入—输出
微分，再做统计平均。详细代码见 `jlens/fitting.py`。

## 11. 解释结果时应遵守什么边界

- 把 Top-K 当作相关概念集合，不要拼成模型的隐藏句子。
- 单个 token、单层信号不足以支撑结论；优先查看相邻层和同义词簇。
- J-lens 是相关性读出，不证明模型在下游计算中因果使用了该概念。
- 只有 activation patching、方向消融、steering 或概念交换等干预才能增强
  因果证据。
- 早层、最前面的位置以及与拟合语料差异较大的中文输入应更谨慎解释。
