# agentorskill-cli（中文版说明）

用于评测深度学习框架**迁移库**（如 TorchAX、MindTorch 等）的命令行工具：先做环境探测，可选用 **LLM 阅读 README/文档** 生成结构化 **`AdapterSpec`**，再运行 **smoke / core / models / training / numeric / benchmark** 等测试套件，输出 **适配率汇总**、**与 CPU 基线对比的数值保真度**，以及 **JSON + Markdown** 报告。

## 安装

```bash
cd /path/to/-demo
pip install -e .
```

本地开发和运行单元测试可安装测试依赖：

```bash
pip install -e ".[test]"
pytest -q
```

默认安装不会拉取 PyTorch、MindSpore、JAX、CANN 或具体迁移库。请按目标评测环境自行安装这些框架和厂商依赖。

## LLM 配置（API 密钥、模型、Base URL）

只有在**不使用** `--adapter-file`、需要让工具从文档抽取 `AdapterSpec` 时，才需配置以下项（适用于 `extract` 与 `run`）。使用 `--adapter-file` 时会直接读取手写 adapter，不会调用 LLM。

### 环境变量

| 变量 | 是否必需 | 说明 |
|------|----------|------|
| `OPENAI_API_KEY` | LLM 抽取时 **必需** | OpenAI 兼容 Chat Completions 接口的 API Key。 |
| `OPENAI_MODEL` | 否 | 模型名；未设置时代码内默认使用 `gpt-4o-mini`。 |
| `OPENAI_BASE_URL` | 否 | 自定义 API 根地址；不设置则使用官方 OpenAI 默认端点。 |

若使用 Azure OpenAI、反向代理或其它兼容 OpenAI 协议的厂商，请按其文档设置 `OPENAI_BASE_URL`（注意路径是否要包含 `/v1`）。

### 命令行覆盖

下列选项可在单次命令中覆盖对应环境变量：

- `--model` — 模型名（亦读取 `OPENAI_MODEL`）。
- `--openai-base-url` — Base URL（亦读取 `OPENAI_BASE_URL`）。

示例（PowerShell）：

```powershell
$env:OPENAI_API_KEY = "sk-..."
$env:OPENAI_MODEL = "gpt-4o-mini"
$env:OPENAI_BASE_URL = "https://api.openai.com/v1"   # 可选；用默认可不设

eval-migration extract --local-path path\to\repo --out adapter.json
eval-migration run --local-path path\to\repo --suite smoke --yes
```

显式指定模型与 Base URL（不在环境中写模型与 URL）：

```powershell
$env:OPENAI_API_KEY = "sk-..."

eval-migration run `
  --local-path path\to\repo `
  --model gpt-4o-mini `
  --openai-base-url "https://your-compatible-endpoint.example/v1" `
  --suite smoke --yes
```

**说明：** 若使用 `--adapter-file` 直接提供 JSON，则**不会**调用 LLM，也**不需要**配置 API Key 与 Base URL。

## 测试套件（Suite）

| 套件 | 作用 |
|------|------|
| `smoke` | 最简：导入、张量、矩阵乘等冒烟检查。 |
| `core` | 较广的算子覆盖：张量形状类操作、逐元/规约、常见 NN 层、损失与**单步**优化器，以及部分卷积/LayerNorm/Dropout/Autograd 等。 |
| `models` | 小型 MLP、CNN、单层 Transformer Encoder（前向）。 |
| `training` | 多步 SGD / AdamW 训练环、梯度裁剪、train/eval 切换；可选 CUDA AMP（无 CUDA 时相关用例会 **skip**）。 |
| `numeric` | 同一小网络跑两次：基线**不**加迁移库 preamble，对比**加** preamble；对展平后的输出与梯度按 `torch.allclose` 风格比较（`--numeric-rtol`、`--numeric-atol`）。 |
| `benchmark` | 粗粒度计时类微基准。 |
| `all` | 按顺序执行：smoke → core → models → training → numeric → benchmark。若 **smoke 失败** 会**跳过** benchmark。 |

## 适配率与报告

每条用例带有 **category**（如 `tensor`、`math`、`nn`、`training`、`numeric_fidelity`、`perf`）。运行结束后，JSON / Markdown 中会包含 **`evaluation_summary`**：

- **`adaptation_rate_overall`**：在参与统计的套件内，**计入分母的用例**上，`通过数 / 总数`。**跳过（skip）** 不计入分母。
- **`by_category` / `by_suite`**：按类别或按套件拆开的通过率。
- **默认参与适配率统计的套件**：`core`、`models`、`training`、`numeric`（默认**不包含** `smoke` 与 `benchmark`，除非下文另行指定）。

相关 CLI：

- `--adaptation-suites core,models,training,numeric` — 自定义哪些套件计入适配率（逗号分隔）。
- `--include-smoke-in-adaptation` — 将 `smoke` 也计入适配率。
- `--honor-adapter-device` — 在 preamble 之后根据 `AdapterSpec.device.target_device` 调用 `torch.set_default_device(...)`（例如 TorchAX 场景下的 `jax`），使算子在对应后端上执行（取决于当前 PyTorch / 迁移库是否支持）。

报告中 **`numeric_fidelity`** 字段含 `within_tolerance_y`、`within_tolerance_g`、`max_abs_err_*`、`rmse_*` 等。

## 常用命令

```bash
# 查看帮助
eval-migration --help

# 探测本机硬件（可交互确认）
eval-migration probe

# 使用手写 Adapter JSON（不调 LLM）
eval-migration run --adapter-file examples/torchax_adapter.json --suite all --yes

# 全量测评且按 Adapter 指定设备（TorchAX + JAX 设备示例）
eval-migration run --adapter-file examples/torchax_adapter.json --suite all --yes --honor-adapter-device

# 从远程 README 抽取并运行（需要 OPENAI_API_KEY）
eval-migration run --readme-url https://raw.githubusercontent.com/google/torchax/main/README.md --suite smoke --yes
```

### 使用 LLM + 本地仓库跑全量测试示例

```powershell
$env:OPENAI_API_KEY = "你的密钥"
cd D:\codes\agentorskill

eval-migration run --local-path D:\path\to\target-repo --suite all --yes
# 若需在 jax 等设备上测 TorchAX：
eval-migration run --local-path D:\path\to\torchax --suite all --yes --honor-adapter-device
```

## Adapter JSON

`examples/` 目录下有最小的 **`AdapterSpec`** 示例，可作手写模板。

### 跳过（Skipped）用例

子测试可返回 `{"ok": true, "skipped": true, "reason": "..."}`（例如 SDPA 不可用、无 CUDA 无法跑 AMP）。**跳过不算失败**，且在计算适配率时**会从分母中排除**。

---

英文说明见 [README.md](README.md)。
