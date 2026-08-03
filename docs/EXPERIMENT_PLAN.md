# J-lens 实验执行计划

## 阶段 A：官方实现验收

目标是确认导入内容完整且本地运行环境可靠。

1. 运行官方单元测试。
2. 加载 `Qwen/Qwen3.5-4B` 与官方预拟合 lens。
3. 运行 walkthrough 的多跳 prompt。
4. 比较 J-lens 与 logit lens 的逐层 top-k。
5. 生成一个自包含的 slice HTML 页面。

验收条件：无测试失败、无 NaN、同一输入重复读出一致、输出中可定位中间桥接概念。

## 阶段 B：官方样本复现

按下列顺序运行内置样例：

1. `multihop`
2. `modulation-arithmetic`
3. `ascii-face`
4. `off-by-one`
5. `overdose-flag`

每次实验保存模型 revision、tokenizer revision、lens revision、设备、dtype、prompt、读取层、读取位置和 top-k。不要只保存截图。

## 阶段 C：定量评估

使用 `data/evaluations/` 中六类官方数据：

- multihop
- multilingual
- order of operations
- poetry
- typo
- association

核心指标：`pass@k`、MRR、对 `log(k)` 的 pass@k AUC、跨 prompt 稳定性，以及相对 logit lens 的增益。拟合语料和评估数据必须分离。

## 阶段 D：因果验证

对可控的多跳任务执行：

- source 方向消融；
- source 到 target 的概念交换；
- 随机同范数方向对照；
- logit-lens 方向对照；
- 多个干预强度 `alpha`。

指标包括输出 KL divergence、目标答案概率变化、top-1 翻转率和非目标能力退化。只有读出与干预结果共同支持假设时，才报告因果结论。

## 阶段 E：中文模型探查

在自定义中文实验开始前，先验证候选中间概念是否为单 token。对多 token 概念，应登记同义词集合或暂不纳入主要指标。

优先研究：

- 中文多跳事实推理；
- 中英文之间的共享概念；
- 代码 bug 的内部识别；
- 模型知道答案不可靠但仍输出的冲突场景；
- prompt injection 和评估意识。

所有结论都应包含无关 prompt 假阳性检查，并至少在多个 prompt 改写上复现。
