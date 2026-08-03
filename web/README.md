# J-Space Observatory Web UI

交互前端使用本地 Qwen3.5-4B 和 Anthropic 官方预拟合 J-lens 展示真实
Layer × Position 读数。自定义输入不会继承官方样本的候选词或目标词。

先在仓库根目录启动后端：

```bash
.venv/bin/python -m jspace_server.app --host 127.0.0.1 --port 8765
```

默认模型目录为 `/Users/eric/models/Qwen/Qwen3.5-4B`，Lens 路径为
`artifacts/lenses/Qwen3.5-4B_jacobian_lens_n1000.pt`。后端会在第一次分析时
把模型和 Lens 惰性加载到 MPS；不支持 MPS 时回退到 CPU。

然后启动前端：

```bash
cd web
npm install
npm run dev
```

生产构建：

```bash
npm run build
```

默认开发地址为 <http://127.0.0.1:5174>。

Vite 将 `/api` 代理到 `127.0.0.1:8765`。浏览器展示的 tokenizer 位置、
word-like Top-10、完整词表 rank、softmax 概率、固定 token rank 轨迹和最终层
原始读出均由 `jspace_server.runtime.AnalysisRuntime` 返回。word-like
筛选与官方 `compute_slice(mask_display=True)` 一致，只改变展示候选，不修改
logits 或完整词表排名；没有分析结果时，候选区保持为空。
