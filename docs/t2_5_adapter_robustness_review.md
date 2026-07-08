# T2.5 Adapter 层健壮性评估

## 汇报定位

根据《共性前端项目补充优化工作 todo.md》，本项属于：

| 项目 | 内容 |
| --- | --- |
| 任务大类 | 任务2：开发者工具链补全 |
| 当前任务 | T2.5 Adapter 层健壮性 |
| 优先级 | P1 |
| 工作范围 | 可观测性（结构化日志、调试模式）|
| 我的任务定位 | 为 Adapter 层补齐可观测性基础能力，复用并扩展 T2.1 PassDiagnostic |

## 依赖关系与设计原则

**依赖链**（来自 todo line 756、781）：
```
T2.1 诊断 CLI ──> T2.5 Adapter 健壮性 ──> T3.4 交付测试 CI
```

T2.1 提供 `PassDiagnostic` 和 `triton-anchor-diagnose` CLI，已经能逐 Pass 跑 adapter pipeline（如 `--pipeline triton-linalg`）。T2.5 的任务是**在 T2.1 基础上扩展**，让 diagnose 跑 adapter 时自动产出健壮性/可观测性指标，直接写入同一份 `summary.json` 和 CLI 输出。

这与 T2.2、T3.7 的依赖模式一致（todo line 213、357）：*"在 PassDiagnostic 基础上扩展"* / *"复用 PassDiagnostic 的计时能力"*。

**核心原则：不新增独立工具，扩展已有的 `triton-anchor-diagnose`**。

## 改动清单

| 文件 | 改动类型 | 内容 |
| --- | --- | --- |
| `python/triton_anchor/diagnostics.py` | 扩展 | 给 `PassRunRecord` 添加 `duration_ms` / `before_ir_bytes` / `after_ir_bytes` / `ir_delta_bytes` / `peak_rss_bytes` 字段；给 `PassDiagnosticResult` 添加 `total_duration_ms` / `input_ir_bytes` / `output_ir_bytes` / `peak_rss_bytes` / `slowest_pass` 属性；在 `_diagnose_pipeline` 里对每个 pass 计时、计算 IR 大小、采样 RSS。 |
| `python/triton_anchor/diagnose.py` | 扩展 | `_print_result` 增加指标展示：总耗时、peak RSS、输入/输出 IR 大小、最慢 Pass。 |
| `python/triton_anchor/tests/test_diagnostics.py` | 扩展 | 新增 2 个测试覆盖指标采集：成功路径和失败路径。 |
| `docs/t2_5_adapter_robustness_review.md` | 更新 | 本文档（反映集成方案）。 |

**零新文件**。改动全部在 T2.1 已有文件里，完全符合"复用/扩展"的依赖语义。

## 使用方式

与之前完全一样——用户无需改命令，只是输出更丰富：

```bash
# 诊断 triton-linalg adapter pipeline（pybind，逐 Pass）
triton-anchor-diagnose --pipeline triton-linalg --python 'lambda: add_kernel_ttir()'
```

**输出示例（成功）**：
```
OK: pipeline triton-linalg completed, 9/9 passes executed.
total duration: 342.56 ms
input IR: 1523 bytes
output IR: 2847 bytes
peak RSS: 87.34 MB
slowest pass: #7 triton_to_linalg (189.23 ms)
diagnostic output: /tmp/triton-anchor-diagnose-xyz
summary: /tmp/triton-anchor-diagnose-xyz/summary.json
```

**输出示例（失败）**：
```
FAILED: pipeline triton-linalg failed at pass 4/9: pointer_strength_reduction
before IR: /tmp/.../04-pointer_strength_reduction.before.mlir
diagnostic detail: /tmp/.../04-pointer_strength_reduction.diagnostic.txt
pass duration: 23.45 ms
before IR: 1523 bytes
total duration (up to failure): 98.76 ms
peak RSS: 65.21 MB
location: 12:8
operation: tt.load
ir line: 12:   %0 = tt.load %ptr : !tt.ptr<f32>
error: failed to apply pointer strength reduction
diagnostic output: /tmp/triton-anchor-diagnose-xyz
summary: /tmp/triton-anchor-diagnose-xyz/summary.json
```

`summary.json` 新增字段：
```json
{
  "ok": true,
  "pipeline": "triton-linalg",
  "total_passes": 9,
  "executed_passes": 9,
  "total_duration_ms": 342.56,
  "input_ir_bytes": 1523,
  "output_ir_bytes": 2847,
  "peak_rss_bytes": 91582464,
  "records": [
    {
      "index": 1,
      "name": "triton_to_ppl",
      "ok": true,
      "duration_ms": 12.34,
      "before_ir_bytes": 1523,
      "after_ir_bytes": 1687,
      "ir_delta_bytes": 164,
      "peak_rss_bytes": 85123072,
      ...
    },
    ...
  ]
}
```

## 已覆盖的能力（本轮聚焦可观测性）

| 能力 | 状态 | 说明 |
| --- | --- | --- |
| 逐 Pass 耗时 | ✅ | 每个 pass 的墙钟时间（毫秒），识别热点 Pass（如 `triton_to_linalg` 通常最慢）。可直接被 T3.7 性能基准复用。 |
| IR 大小变化 | ✅ | 每个 Pass 前后 IR 的 UTF-8 字节数，以及增量（`ir_delta_bytes`）。汇总为 `input_ir_bytes` / `output_ir_bytes`。 |
| Pass 数量 + 总耗时 | ✅ | `executed_passes` / `total_duration_ms`，作为端到端汇总指标。 |
| RSS 内存采样 | ✅ | 用 `resource.getrusage(RUSAGE_SELF).ru_maxrss` 采样进程峰值 RSS。注意：pybind adapter 是进程内调用，采样的是整个 Python 进程的 RSS，**不是单个 Pass 的隔离内存**；subprocess adapter 的子进程需单独跟踪（本轮未实现，见"后续工作"）。 |
| 结构化日志 | ✅ | 所有指标写入 `summary.json`，机器可读。 |
| 调试模式 | ✅ | 沿用 T2.1 的 `TRITON_ANCHOR_DEBUG=1` / `--save-ir`，保存每个 Pass 的 before/after IR。 |

## 技术说明

### RSS 采样的局限

- **Pybind adapter**（如 triton-linalg）：Pass 在同一进程内调用 C++ 扩展，`ru_maxrss` 反映的是整个 Python 进程的峰值，不能隔离单个 Pass 的内存贡献。适合粗粒度监控，不适合精确归因。
- **Subprocess adapter**（如 triton-shared）：当前是 stub，未实际集成 `triton-shared-opt`。将来真正跑子进程时，需要单独 `Popen` + `psutil` 采样子进程 RSS，本轮未实现。

### 为什么不做独立 CLI

早期版本曾尝试 `triton-anchor-adapter-eval` 独立工具，但这违背了 todo 文档的依赖语义——T2.1→T2.5 的箭头在文档里一贯意味着**复用/扩展**（见 T2.2 line 213、T3.7 line 357），而不是"并列工具"。独立 CLI 会导致：
- 与 diagnose 功能重叠，用户困惑；
- T3.4 的 CI 集成要对接两套输出；
- 偏离"在 PassDiagnostic 基础上构建"的设计脉络。

所以最终采用**扩展 PassDiagnostic**的集成方案，和 T2.2/T3.7 保持一致。

## 测试结果

新增 2 个单元测试，覆盖成功/失败路径的指标采集。与 T2.1 原有 6 个测试合并后共 **8 项全部通过**：

```bash
PYTHONPATH=python python -m pytest python/triton_anchor/tests/test_diagnostics.py -v
```

| 测试项 | 覆盖点 | 结果 |
| --- | --- | --- |
| `test_pass_diagnostic_records_timing_and_ir_sizes` | 逐 Pass 计时、IR 大小、汇总指标、`slowest_pass` 属性 | PASS |
| `test_pass_diagnostic_metrics_on_failure` | 失败时指标仍被采集、`output_ir_bytes` 为最后已知正确值 | PASS |
| （T2.1 原有 6 项） | 失败定位、MLIR location 提取、summary.json、CLI | PASS |

## 后续工作项

1. **Subprocess adapter 的子进程 RSS 跟踪**：当 triton-shared 集成完成后，改造 `TritonSharedAdapter.convert()` 用 `Popen` + `psutil` 实时采样子进程内存。
2. **与 T3.4 CI 集成**：复用 `summary.json` 的新字段做质量卡点（如"单 Pass 耗时 >5s 报警"、"输出 IR 超过 10MB 报警"）。
3. **与 T3.7 性能基准对接**：直接复用 `total_duration_ms` 和逐 Pass 耗时，建立性能回归检测。

---

**本轮交付物**：在 `triton-anchor-diagnose` 已有流程上叠加可观测性指标，零新增独立工具，完全复用 T2.1 基础设施，符合 todo 文档依赖语义。
