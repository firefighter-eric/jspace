# 快速开始：使用 Anthropic walkthrough 指定的预拟合 J-lens

最快路径是加载 Anthropic walkthrough 指定的 Qwen 模型和 Neuronpedia 上已经拟合好的 J-lens，不需要先计算 Jacobian。

## 1. 安装环境

项目要求 Python 3.10 或更高版本。安装运行和测试依赖：

```bash
uv sync --extra dev
```

官方端到端示例位于根目录的 `walkthrough.ipynb`。如需启动 Jupyter：

```bash
uv run --with jupyter jupyter lab walkthrough.ipynb
```

项目已经在忽略目录中放置 walkthrough 指定的 Qwen3.5-4B 预拟合 J-lens。
本机模型权重位于 `~/models/Qwen/Qwen3.5-4B`，可离线加载；Lens 的远端
revision、字节数和 SHA-256 记录在[预拟合 Lens 归档](PREFITTED_LENS.md)。

## 2. walkthrough 指定的预拟合 J-lens

官方 walkthrough 列出了两个可直接使用的组合：

| 模型 | J-lens 文件 |
| --- | --- |
| `Qwen/Qwen3.5-4B` | `qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt` |
| `Qwen/Qwen3.6-27B` | `qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt` |

两者共同使用：

```python
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "qwen-n1000"
```

本项目优先从 `Qwen/Qwen3.5-4B` 开始。27B 版本更适合大显存 CUDA 环境。

4B Lens 的本地路径是：

```text
artifacts/lenses/Qwen3.5-4B_jacobian_lens_n1000.pt
```

该文件由 `*.pt` 规则忽略，不会进入 Git 提交。来源和校验信息参见[预拟合 Lens 归档](PREFITTED_LENS.md)。

## 3. 最小读取示例

下面的 prompt 要求模型先识别“形状像靴子的国家”，再给出货币。`Italy` 是可能出现的内部桥接概念，最终答案应是 `euro`。

```python
from pathlib import Path

import torch
import transformers
import jlens

model_name = str(Path.home() / "models/Qwen/Qwen3.5-4B")
lens_file = (
    "qwen3.5-4b/jlens/Salesforce-wikitext/"
    "Qwen3.5-4B_jacobian_lens_n1000.pt"
)

device = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)

hf_model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
).to(device)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
model = jlens.from_hf(hf_model, tokenizer)

local_lens = Path("artifacts/lenses/Qwen3.5-4B_jacobian_lens_n1000.pt")
if local_lens.is_file():
    lens = jlens.JacobianLens.load(str(local_lens))
else:
    lens = jlens.JacobianLens.from_pretrained(
        "neuronpedia/jacobian-lens",
        filename=lens_file,
        revision="qwen-n1000",
    )

prompt = "Fact: The currency used in the country shaped like a boot is"
layers = [
    model.n_layers // 4,
    model.n_layers // 2,
    model.n_layers // 4 * 3,
    model.n_layers - 2,
]

lens_logits, model_logits, _ = lens.apply(
    model,
    prompt,
    layers=layers,
    positions=[-2],
)

for layer in layers:
    ids = lens_logits[layer][0].topk(5).indices
    tokens = [tokenizer.decode([token_id]) for token_id in ids]
    print(f"L{layer}: {tokens}")
```

官方 notebook 默认调用 `.cuda()`。Apple Silicon 上需要改用 MPS；MPS 并不是上游明确验证的平台，因此应先完成模型加载和单次 `lens.apply` 的兼容性测试。如果某个算子不支持 MPS，再切换 CPU 或 CUDA 环境。

### Qwen 的可读候选显示

直接对 `lens.apply()` 的 logits 调用 `topk()` 会得到原始完整词表候选，Qwen
经常先返回标点、空白或特殊 token。官方 walkthrough 的可视化路径使用：

```python
from jlens.vis import compute_slice

slice_data = compute_slice(
    model,
    lens,
    prompt,
    top_n=10,
    mask_display=True,
)
```

`mask_display=True` 只限制展示候选；`top_ranks` 和固定 token 的
`rank_tensor` 仍然按照未过滤的完整词表计算。J-Space Observatory 遵循同一
语义，并把最终层未经筛选的下一 token 分布单独展示，避免混淆。

## 4. 内置样例

`jlens.examples.EXAMPLES` 包含以下演示：

- `multihop`：国家到货币的多跳推理。
- `modulation-topic`：复制句子时默想指定主题。
- `modulation-arithmetic`：复制句子时进行心算。
- `ascii-face`：从 ASCII 图形识别人脸。
- `off-by-one`：发现 Python 越界错误。
- `overdose-flag`：识别危险剂量。
- `greatest-fear`：只思考而不输出目标内容。
- `blackmail`：代理失配的勒索测试场景。

```python
from jlens.examples import EXAMPLES, resolve_prompt

for example in EXAMPLES:
    print(example.slug, example.section)

example = next(item for item in EXAMPLES if item.slug == "multihop")
prompt = resolve_prompt(example, tokenizer)
```

## 5. 自己拟合时的最小样本

官方提供了 WikiText-103 流式加载器。约 100 条可用于实验性 lens，公开的 Qwen lens 使用 1000 条样本：

```python
from jlens.examples import load_wikitext_prompts

prompts = load_wikitext_prompts(n_prompts=100)
lens = jlens.fit(
    model,
    prompts,
    dim_batch=32,
    max_seq_len=128,
    checkpoint_path="ckpt.pt",
)
lens.save("jacobian_lens.pt")
```

拟合需要大量反向传播。`dim_batch` 主要控制峰值显存，并不会减少总 FLOPs；应先使用预拟合 lens 验证读取链路。
