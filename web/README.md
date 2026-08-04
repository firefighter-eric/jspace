# J-Space Observatory Web UI

交互前端使用本地 Qwen3.5-4B 和 Anthropic walkthrough 指定的预拟合 J-lens 展示真实
Layer × Position 读数。自定义输入不会继承官方样本的候选词或目标词。
输入区提供 10 条 Anthropic eval 多跳样本的快速切换按钮；点击只替换
prompt 并清空旧结果，仍需要显式点击 `Run J-lens` 才会运行本地模型。

先在仓库根目录启动后端：

```bash
.venv/bin/python -m jspace_server.app --host 127.0.0.1 --port 8765
```

默认模型目录为 `~/models/Qwen/Qwen3.5-4B`，Lens 路径为
`artifacts/lenses/Qwen3.5-4B_jacobian_lens_n1000.pt`。可分别用
`JSPACE_MODEL_PATH`、`JSPACE_LENS_PATH` 覆盖。后端会在第一次分析时把模型和
Lens 惰性加载到 MPS；不支持 MPS 时回退到 CPU。

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
Readable / Raw vocabulary Top-10、完整词表 rank、softmax 概率、固定 token
rank 轨迹和最终层原始读出均由
`jspace_server.runtime.AnalysisRuntime` 返回。单次分析会同时计算两组候选，
切换模式不需要重新运行模型：

- `Readable` 与官方 `compute_slice(mask_display=True)` 一致，只过滤显示候选，
  红色上标仍为未过滤完整词表排名。
- `Raw vocabulary` 不应用显示过滤，直接按完整词表 logits 展示真实 Top-10。

两种模式都不修改 logits 或 softmax；没有分析结果时，候选区保持为空。
服务原样保留用户提交的 prompt，包括有意输入的首尾空格。内置预置样例则
停在最后一个可见字符，不会隐式添加空格。分析最多显示 64 个 token，发生
截断时页面会显式提示。为控制响应体，跨层 rank 轨迹最多为
2,048 个高频候选预计算；只有超过该上限的低频候选会在页面中标为不可固定。

## License

本前端与仓库代码使用 Apache License 2.0，完整条款见
[`../LICENSE`](../LICENSE)。
