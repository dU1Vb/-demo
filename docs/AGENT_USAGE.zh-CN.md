# Agent 兼容性测试使用手册

本文说明如何使用 `agentorskill-cli` 的 Agent 兼容性测试功能，包括 API key 安全配置、三层 Agent 模式、常用命令和报告解读。

## 1. 当前 Agent 能做什么

当前实现的 Agent 是一个三层闭环：

1. 诊断 Agent：读取失败 case、traceback、AdapterSpec 和环境信息，调用 LLM 判断失败类别，并判断是否计入迁移库兼容性分数。
2. 复验 Agent：执行白名单复验动作，例如重跑同一 case、关闭 adapter 重跑、按 adapter 设备重跑，用来排除环境、配置、依赖、设备初始化等非本体影响因素。
3. 修复 Agent：只尝试修复 AdapterSpec，并重跑失败 case 验证。当前不会直接修改迁移库源码。

报告会记录：

- `failure_class`
- 诊断证据
- 是否计入 compatibility/adaptation 分数
- 复验动作和结果
- AdapterSpec 修复 diff
- ME 统计输入：LLM 调用数、prompt/completion 字符数、复验次数、修复尝试次数、patch 增删行、耗时
- `AR`
- `migrate@k`

## 2. 安全配置 API key

不要把 API key 写进仓库文件、README、adapter JSON 或命令行参数。

推荐使用 `/home/ma-user/work/load_deepseek_agent_env.sh`：

```bash
source /home/ma-user/work/load_deepseek_agent_env.sh
```

该脚本会从私有文件读取 key：

```text
/home/ma-user/work/.secrets/deepseek_api_key
```

推荐权限：

```bash
chmod 700 /home/ma-user/work/.secrets
chmod 600 /home/ma-user/work/.secrets/deepseek_api_key
chmod 700 /home/ma-user/work/load_deepseek_agent_env.sh
```

加载后会设置：

```bash
OPENAI_API_KEY
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-v4-flash
```

当前实现使用 OpenAI-compatible Chat Completions 协议。以后如果改用 Anthropic Messages 协议，不能只改 URL，还需要接入 Anthropic provider。

## 3. 基础运行方式

进入项目：

```bash
cd /home/ma-user/work/-demo
source /home/ma-user/work/activate_torch4ms_ms272_cann85.sh
source /home/ma-user/work/load_deepseek_agent_env.sh
```

先确认 CLI 可用：

```bash
eval-migration run --help
```

不用 Agent 的普通测试：

```bash
eval-migration run \
  --adapter-file examples/noop_adapter.json \
  --suite smoke \
  --single-process \
  --yes \
  --no-probe \
  --report-dir reports_noop_smoke
```

使用 `--adapter-file` 时不会调用 LLM，除非同时开启 `--agent-mode` 且有失败 case 需要 Agent 处理。

## 4. 三种 Agent 模式

### diagnose

只诊断，不复验、不修复。

```bash
eval-migration run \
  --adapter-file examples/torch4ms_adapter.json \
  --suite smoke \
  --single-process \
  --yes \
  --agent-mode diagnose \
  --agent-max-failures 5 \
  --report-dir reports_agent_diagnose
```

适合先观察 Agent 如何分类失败。

### revalidate

先诊断，再执行受控复验动作。

```bash
eval-migration run \
  --adapter-file examples/torch4ms_adapter.json \
  --suite all \
  --single-process \
  --yes \
  --agent-mode revalidate \
  --agent-max-failures 10 \
  --report-dir reports_agent_revalidate
```

如果复验后判断失败属于环境、配置、依赖、导入顺序或设备初始化问题，该 case 会被标记为不计入兼容性分数。

### repair

先诊断和复验，再尝试 AdapterSpec 修复。

```bash
eval-migration run \
  --adapter-file examples/torch4ms_adapter.json \
  --suite smoke \
  --single-process \
  --yes \
  --agent-mode repair \
  --agent-max-failures 5 \
  --agent-repair-attempts 2 \
  --report-dir reports_agent_repair
```

当前修复层只会让 LLM 输出新的 AdapterSpec，并验证失败 case 是否通过。它不会修改迁移库源码。

## 5. 从文档抽取 AdapterSpec 再测试

如果没有手写 adapter，可以让 LLM 从文档抽取：

```bash
eval-migration run \
  --local-path /path/to/migration-library \
  --suite smoke \
  --yes \
  --agent-mode diagnose \
  --report-dir reports_agent_doc_extract
```

这个命令会先调用 LLM 生成 AdapterSpec。如果测试失败，还会继续进入 Agent 诊断。

也可以只抽取 adapter：

```bash
eval-migration extract \
  --local-path /path/to/migration-library \
  --out adapter.generated.json
```

之后使用手写/生成的 adapter 文件测试：

```bash
eval-migration run \
  --adapter-file adapter.generated.json \
  --suite all \
  --single-process \
  --yes \
  --agent-mode revalidate
```

## 6. 常用参数

| 参数 | 作用 |
|------|------|
| `--agent-mode off` | 默认值，不启用 Agent。 |
| `--agent-mode diagnose` | 只做 LLM 诊断。 |
| `--agent-mode revalidate` | 诊断后执行复验。 |
| `--agent-mode repair` | 诊断、复验，再尝试 AdapterSpec 修复。 |
| `--agent-max-failures N` | 最多处理 N 个失败 case，避免一次调用过多 LLM。 |
| `--agent-repair-attempts K` | repair 模式下每个 case 最多尝试 K 次修复。 |
| `--single-process` | smoke/core/models/training 复用同一进程；numeric/benchmark 仍使用子进程。 |
| `--no-probe` | 跳过交互式硬件确认，但仍会收集硬件信息。 |
| `--honor-adapter-device` | 按 AdapterSpec 的目标设备设置 torch 默认设备。 |
| `--report-dir DIR` | 指定报告输出目录。 |

## 7. 报告怎么看

每次运行会生成 JSON 和 Markdown，例如：

```text
reports_agent_revalidate/<library_name>/report-YYYYMMDD-HHMMSS.json
reports_agent_revalidate/<library_name>/report-YYYYMMDD-HHMMSS.md
```

重点看 JSON 里的这些字段：

```text
evaluation_summary
suites
agent_report
```

`agent_report` 包含：

```text
enabled
mode
provider
protocol
model
temperature
tool_calls
diagnostics
revalidations
repair_attempts
migration_effort
repair_metrics
```

`migration_effort` 是 ME 的原始统计输入：

```text
agent_rounds
llm_calls
prompt_chars
completion_chars
revalidation_runs
repair_attempts
patches_generated
patch_lines_added
patch_lines_deleted
elapsed_s
```

`repair_metrics` 包含：

```text
attempted_cases
successful_repairs
AR
migrate_at_k
```

## 8. 退出码规则

普通模式下，只要有失败 case，命令会返回非 0。

Agent 模式下，如果失败 case 被 Agent 判定为不计入兼容性，或被 AdapterSpec 修复验证通过，则不再视为 unresolved failure。只有仍然计入兼容性且未被修复的失败会导致命令返回非 0。

## 9. API key 是否会被提交

推荐配置下不会。

原因：

- key 放在 `/home/ma-user/work/.secrets/deepseek_api_key`，不在 `-demo` git 仓库内。
- `.gitignore` 已忽略 `.env` 和 `.env.*`。
- 当前代码不会把 API key 写进报告。

仍需避免：

- 不要把真实 key 写进 README。
- 不要把真实 key 写进 adapter JSON。
- 不要把真实 key 写进 shell 脚本并提交到仓库。
- 不要用 `--api-key xxx` 这种命令行参数形式；当前 CLI 也没有提供这种参数。

## 10. 当前边界

- 当前 provider 是 OpenAI-compatible Chat Completions。
- Anthropic Messages Provider 还没有实现。
- 修复层目前只修 AdapterSpec，不修迁移库源码。
- Agent 不会自动安装依赖或修改系统环境。
- `--agent-max-failures` 建议先设小，例如 3 到 10，确认报告质量后再扩大。

