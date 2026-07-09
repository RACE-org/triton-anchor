# T2.5 Adapter 可观测性指标

## 使用方式

命令与 T2.1 诊断完全一致，无需额外参数，输出自动包含指标：

```bash
python -m triton_anchor.diagnose /path/to/input.mlir \
  --pipeline triton-linalg \
  --output-dir ./out
```

运行测试：

```bash
PYTHONPATH=python python -m pytest python/triton_anchor/tests/test_diagnostics.py -v
```

---

## 提供的指标

**逐 Pass 指标**（写入每条 `records[]`）：

| 字段 | 说明 |
|---|---|
| `duration_ms` | 该 pass 的墙钟耗时（毫秒） |
| `before_ir_bytes` | pass 执行前 IR 的 UTF-8 字节数 |
| `after_ir_bytes` | pass 执行后 IR 的 UTF-8 字节数（失败时为 0） |
| `ir_delta_bytes` | IR 大小变化量（失败时为 0） |
| `peak_rss_bytes` | pass 结束时采样的进程峰值 RSS（字节） |

**汇总指标**（写入 `summary.json` 顶层）：

| 字段 | 说明 |
|---|---|
| `total_duration_ms` | 所有已执行 pass 的耗时总和 |
| `input_ir_bytes` | 进入 pipeline 时的 IR 大小 |
| `output_ir_bytes` | 最后一个执行 pass 后的 IR 大小（失败时为失败前最后已知值） |
| `peak_rss_bytes` | 整个 pipeline 过程中的最高 RSS |
| `slowest_pass` | 耗时最长的 pass（含 `name` 和 `duration_ms`） |

**CLI 输出示例（成功）**：
```
OK: pipeline triton-linalg completed, 9/9 passes executed.
total duration: 342.56 ms
input IR: 1523 bytes
output IR: 2847 bytes
peak RSS: 87.34 MB
slowest pass: #7 triton_to_linalg (189.23 ms)
summary: ./out/summary.json
```

**CLI 输出示例（失败）**：
```
FAILED: pipeline triton-linalg failed at pass 4/9: pointer_strength_reduction
pass duration: 23.45 ms
before IR: 1523 bytes
total duration (up to failure): 98.76 ms
peak RSS: 65.21 MB
location: kernel.py:12:8
operation: tt.load
summary: ./out/summary.json
```

---

## 改动文件

**零新文件**，全部改动在 T2.1 已有文件里：

| 文件 | 改动内容 |
|---|---|
| `python/triton_anchor/diagnostics.py` | `PassRunRecord` 新增 5 个指标字段；`PassDiagnosticResult` 新增 4 个汇总字段和 `slowest_pass` 属性；`_diagnose_pipeline` 里对每个 pass 计时、计算 IR 大小、采样 RSS |
| `python/triton_anchor/diagnose.py` | `_print_result` 增加总耗时、peak RSS、输入/输出 IR 大小、最慢 pass 的展示 |
| `python/triton_anchor/tests/test_diagnostics.py` | 新增 2 个测试，覆盖成功路径和失败路径的指标采集 |

---

## 测试结果

新增 2 个测试，与 T2.1 原有 6 个合并后共 **8 项全部通过**：

| 测试项 | 覆盖点 |
|---|---|
| `test_pass_diagnostic_records_timing_and_ir_sizes` | 逐 pass 计时、IR 大小、汇总指标、`slowest_pass` |
| `test_pass_diagnostic_metrics_on_failure` | 失败时指标仍被采集、`output_ir_bytes` 为最后已知正确值 |
| T2.1 原有 6 项 | 失败定位、MLIR location 提取、summary.json、CLI |

---

## 构建细节

- **计时**：每个 pass 用 `time.monotonic()` 包裹，精度为毫秒。
- **IR 大小**：对 `str(module)` 做 UTF-8 编码后计字节数，pass 前后各采样一次。
- **RSS 采样**：调用 `resource.getrusage(RUSAGE_SELF).ru_maxrss`，采样整个 Python 进程峰值 RSS。对 pybind adapter 是进程级粗粒度监控，不能精确归因到单个 pass；macOS 单位为字节，Linux 为 KB（已做换算）。
- **失败时**：失败 pass 的 `duration_ms` 和 `before_ir_bytes` 仍会记录；`after_ir_bytes` 和 `ir_delta_bytes` 置 0；`output_ir_bytes` 保留失败前最后一个成功 pass 的 IR 大小。

---

## 设计决策：为什么不做独立 CLI

早期曾尝试 `triton-anchor-adapter-eval` 作为独立工具，但 todo 文档的依赖语义（T2.1→T2.5→T3.4）一贯意味着**复用/扩展**，而不是新增并列工具。独立 CLI 会导致与 diagnose 功能重叠、T3.4 CI 需对接两套输出，偏离整体设计脉络。

最终方案：**扩展 `PassDiagnostic`**，指标直接写入同一份 `summary.json`，与 T2.2/T3.7 的集成方式保持一致。

---

## 任务背景

- **任务**：T2.5 Adapter 层健壮性，优先级 P1，属于任务2（开发者工具链补全）
- **依赖链**：T2.1 诊断 CLI → T2.5 Adapter 健壮性 → T3.4 交付测试 CI

---

## 后续工作

1. **Subprocess adapter RSS 跟踪**：triton-shared 集成完成后，用 `Popen` + `psutil` 采样子进程内存。
2. **T3.4 CI 集成**：用 `summary.json` 新字段做质量卡点（如单 pass 耗时 >5s 报警）。
3. **T3.7 性能基准对接**：复用 `total_duration_ms` 和逐 pass 耗时，建立性能回归检测。
