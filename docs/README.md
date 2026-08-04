# J-space 项目文档

本仓库以 Anthropic 发布的 Jacobian Lens 参考实现为基础，用于读取、可视化和拟合开放权重语言模型的 J-lens。

## 文档导航

- [快速开始](QUICKSTART.md)：直接使用 Anthropic walkthrough 指定的预拟合 Qwen J-lens。
- [预拟合 Lens](PREFITTED_LENS.md)：本地 Lens 文件、来源、元数据与校验值。
- [方法原理](METHOD.md)：Jacobian Lens 与 J-space 的数学含义、读取方式和结论边界。
- [J-lens 常见问题](JLENS_FAQ.md)：位置语义、候选排名、Readable/Raw、矩阵结构、hidden state、lm_head 与拟合过程。
- [实验计划](EXPERIMENT_PLAN.md)：从官方样例复现到因果干预的分阶段方案。
- [论文归档](papers/README.md)：官方论文 PDF、来源、校验值和引用格式。

## 其他文档入口

- [`../README.md`](../README.md)：仓库总览、安装、应用与拟合 API。
- [`../web/README.md`](../web/README.md)：本地 API、React 前端和 Readable / Raw vocabulary 说明。
- [`../walkthrough.ipynb`](../walkthrough.ipynb)：Anthropic 官方端到端 notebook。

## 官方资源

- 源码：https://github.com/anthropics/jacobian-lens
- 论文 HTML：https://transformer-circuits.pub/2026/workspace/index.html
- Anthropic 研究文章：https://www.anthropic.com/research/global-workspace
- 交互演示：https://www.neuronpedia.org/jlens

## 仓库边界

核心 `jlens/` 实现以及配套的 `data/`、`assets/`、`walkthrough.ipynb` 和
上游测试源自 Anthropic Apache-2.0 参考仓库。本项目新增了 `jspace_server/`、
`web/`、`docs/`、Observatory 测试与 Python 打包配置；部分上游入口文件也已
加入明确的派生项目说明。
