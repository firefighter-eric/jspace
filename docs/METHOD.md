# Jacobian Lens 与 J-space 方法原理

## 1. 目标

Jacobian Lens（J-lens）读取一个中间层激活在大量上下文中通常会使模型现在或未来更倾向说出哪些 token。它不是隐藏 chain-of-thought 的逐字转录，也不能单凭一个高排名 token 证明模型相信或使用了该概念。

## 2. 平均 Jacobian

设第 `l` 层、位置 `t` 的 residual stream 为 `h_l,t`，最终层位置 `t'` 的 residual stream 为 `h_final,t'`。局部一阶影响由 Jacobian 描述：

```math
\Delta h_{final,t'} \approx
\frac{\partial h_{final,t'}}{\partial h_{l,t}}\Delta h_{l,t}
```

J-lens 对拟合语料、源位置和所有当前及未来目标位置求平均：

```math
J_l = \mathbb{E}_{prompt,t,t'\ge t}
\left[\frac{\partial h_{final,t'}}{\partial h_{l,t}}\right]
```

`J_l` 将中间层 residual stream 线性输送到最终层坐标系，再使用模型自己的 norm 和 unembedding `W_U` 解码：

```math
\operatorname{lens}(h_l) =
\operatorname{softmax}\left(W_U\operatorname{norm}(J_lh_l)\right)
```

与 logit lens 直接假设 `J_l = I` 相比，J-lens 显式修正了层间表征坐标的变化。

## 3. J-space

`W_U J_l` 中每个词表 token 对应一个 J-lens 方向。由于词表通常大于 hidden size，这些方向是过完备的；J-space 因而不是普通的低维线性子空间。

论文把 J-space 操作化为少量 J-lens 方向的稀疏非负组合，常用稀疏度约为 `k = 25`。一个激活的 J-space component 是最接近它的稀疏组合，其余部分称为 non-J-space component。

## 4. 三种使用方式

### 读取

- 对每个层和位置输出词表排名。
- 对预先指定的概念记录 rank、score 或 cosine similarity。
- 使用稀疏分解得到较少冗余的活跃概念集合。

### 消融与 steering

沿 token 方向增加激活：

```math
h' = h + \alpha v_{target}
```

使用负强度或移除该方向的投影，可以测试该概念是否对输出有因果影响。

### 概念交换

令 `V = [v_source, v_target]`，读取局部坐标 `c = V^\dagger h`，交换两个坐标后写回：

```math
h' = h + V(\sigma(c) - c)
```

这种操作保留与两个方向张成空间正交的激活分量。若把多跳推理中的 `spider` 换为 `ant` 后答案由 8 变为 6，才构成模型下游计算使用该中间表示的因果证据。

## 5. 解释规则

- 把 top-k 当作概念集合，不要读成一句话。
- 优先寻找同义词、相关词形成的主题，而非只看一个精确 token。
- 单层信号可能是噪声；跨多个相邻层或位置复现更可信。
- 中间层的桥接概念应与输入原词和最终输出区分开。
- 读取结果必须配合 logit lens、随机同范数方向和无干预基线。
- 因果结论需要消融或交换实验，不应只凭相关性。

## 6. 已知限制

- 当前方法主要识别可由单 token 命名的概念。
- 输出是概念袋，不能恢复概念之间完整的关系结构。
- 某些层和位置的读出不可解释。
- 早层缺少信号可能是模型机制，也可能是 lens 的限制。
- 中间 workspace 表征与晚层输出准备之间没有绝对边界。
- 在一个开放模型上的复现不能证明所有模型都有相同结构。
- 这些结果不能证明模型具有主观体验或现象意识。

## 7. 官方出处

- 论文 HTML：https://transformer-circuits.pub/2026/workspace/index.html
- 代码：https://github.com/anthropics/jacobian-lens
- 本地论文：[papers/verbalizable-representations-global-workspace.pdf](papers/verbalizable-representations-global-workspace.pdf)
