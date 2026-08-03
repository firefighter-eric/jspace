# 官方预拟合 J-lens

## 已下载文件

```text
artifacts/lenses/Qwen3.5-4B_jacobian_lens_n1000.pt
```

| 项目 | 值 |
| --- | --- |
| 对应模型 | `Qwen/Qwen3.5-4B` |
| Hugging Face 仓库 | `neuronpedia/jacobian-lens` |
| revision | `qwen-n1000` |
| 远端文件 | `qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt` |
| 本地字节数 | `406332644` |
| SHA-256 | `1f9a8f8fd593f0ffec1a9640993257ca4560f8ae3e5602315643d5cc6818534e` |
| 拟合样本数 | `1000` |
| hidden size | `2560` |
| source layers | `0..30`，共 31 层 |
| 下载日期 | `2026-08-03` |

文件已使用 `jlens.JacobianLens.load` 实际反序列化验证，结果为：

```text
JacobianLens(d_model=2560, n_prompts=1000, source_layers=[0..30] (31 layers))
```

## 加载本地文件

```python
import jlens

lens = jlens.JacobianLens.load(
    "artifacts/lenses/Qwen3.5-4B_jacobian_lens_n1000.pt"
)
print(lens)
```

## Git 策略

上游 `.gitignore` 已包含 `*.pt`，所以 Lens 保存在项目工作目录中，但不会被 Git 跟踪。这避免把约 406 MB 的模型派生文件放进普通源码提交。换电脑或清理工作区后，可以根据[快速开始](QUICKSTART.md)中的 Hugging Face 配置重新下载。

模型权重没有复制到仓库；首次加载 `Qwen/Qwen3.5-4B` 时由 Transformers 下载到 Hugging Face 缓存。
