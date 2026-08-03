# J-space 项目文档

本仓库以 Anthropic 发布的 Jacobian Lens 参考实现为基础，用于读取、可视化和拟合开放权重语言模型的 J-lens。

## 文档导航

- [快速开始](QUICKSTART.md)：直接使用官方预拟合 Qwen J-lens。
- [预拟合 Lens](PREFITTED_LENS.md)：本地 Lens 文件、来源、元数据与校验值。
- [方法原理](METHOD.md)：Jacobian Lens 与 J-space 的数学含义、读取方式和结论边界。
- [实验计划](EXPERIMENT_PLAN.md)：从官方样例复现到因果干预的分阶段方案。
- [论文归档](papers/README.md)：官方论文 PDF、来源、校验值和引用格式。

## 官方资源

- 源码：https://github.com/anthropics/jacobian-lens
- 论文 HTML：https://transformer-circuits.pub/2026/workspace/index.html
- Anthropic 研究文章：https://www.anthropic.com/research/global-workspace
- 交互演示：https://www.neuronpedia.org/jlens

## 仓库边界

根目录中的 `jlens/`、`tests/`、`data/`、`assets/`、`walkthrough.ipynb`、`pyproject.toml` 和 `uv.lock` 来自 Anthropic 官方仓库。`docs/` 是本项目补充的中文说明与论文归档。
