# Triton Anchor 编译诊断工具

本文档说明 `triton-anchor-diagnose` 和 JIT 自动诊断 hook 的当前能力、使用方法和边界。

## 1. 能诊断什么

当前诊断能力覆盖两个阶段：

1. TTIR pipeline

   可以定位 TTIR 优化阶段中首个失败 pass，例如：

   ```text
   common.inliner
   ttir.combine
   common.canonicalizer
   ttir.reorder_broadcast
   common.cse
   common.licm
   common.symbol_dce
   ```

2. TTIR 之后的 `triton-linalg` adapter pipeline

   可以定位 `TTIR -> Linalg` lowering 阶段中首个失败 pass，例如：

   ```text
   triton_linalg.canonicalize_triton
   triton_linalg.pointer_strength_reduction
   triton_linalg.triton_to_linalg
   triton_linalg.arith_to_linalg
   triton_linalg.math_to_linalg
   common.cse
   common.licm
   ```

如果底层 MLIR/C++ diagnostic 包含位置信息，工具会进一步提取：

```text
source location: file:line:column
operation: tt.xxx
before IR: xxx.before.mlir
diagnostic detail: xxx.diagnostic.txt
summary: summary.json
```

## 2. 不能诊断什么

当前还未覆盖：

- Python/Triton 前端生成 TTIR 之前的错误。
- MLIR parse 阶段已经失败的非法 IR。
- PPLIR 阶段失败。
- `ppl-compile` 失败。
- CMake / `.so` 生成失败。
- kernel runtime crash、结果错误、精度错误。

这些阶段需要后续继续接入对应的诊断逻辑。

## 3. CLI 基本用法

开发环境中建议优先用源码模块方式运行，避免已安装 wheel 不是最新代码：

```bash
cd /workspace/triton-anchor
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/workspace/triton-anchor/python:/workspace/triton-anchor \
/opt/venv/bin/python -m triton_anchor.diagnose --help
```

安装后也可以直接使用：

```bash
triton-anchor-diagnose --help
```

如果 `triton-anchor-diagnose --help` 看不到最新参数，说明当前环境中的 console script 来自旧安装包。可以先使用上面的 `python -m triton_anchor.diagnose` 方式。

## 4. 诊断已有 IR 文件

已有 `.ttir/.mlir` 文件时，可以让工具从该 IR 开始重跑指定 pipeline，定位这份 IR 在后续 pass 中的失败点。

诊断 TTIR pipeline：

```bash
cd /workspace/triton-anchor
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/workspace/triton-anchor/python:/workspace/triton-anchor \
/opt/venv/bin/python -m triton_anchor.diagnose \
  /path/to/input.ttir \
  --pipeline ttir \
  --output-dir /workspace/triton-anchor/diagnose-output/ttir
```

诊断 `triton-linalg` pipeline：

```bash
cd /workspace/triton-anchor
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/workspace/triton-anchor/python:/workspace/triton-anchor \
/opt/venv/bin/python -m triton_anchor.diagnose \
  /path/to/input.mlir \
  --pipeline triton-linalg \
  --output-dir /workspace/triton-anchor/diagnose-output/triton-linalg
```

注意：文件输入模式不是“检查 IR 是否已经生成成功”，而是复现“这份中间 IR 在后续 pass 中为什么失败”。

## 5. 不提前 dump IR，直接诊断 Python kernel

如果没有现成 `.ttir` 文件，可以通过 `--python MODULE:FUNCTION` 让 CLI 在运行时生成 TTIR 后诊断：

```bash
cd /workspace/triton-anchor
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/workspace/triton-anchor/python:/workspace/triton-anchor \
/opt/venv/bin/python -m triton_anchor.diagnose \
  --python tests.test_smoke:_smoke_add_kernel \
  --signature '*fp32,*fp32,*fp32,i32' \
  --constant BLOCK=256 \
  --pipeline ttir \
  --output-dir /workspace/triton-anchor/diagnose-output/python-ttir
```

`--constant` 可以重复传入，key 支持参数名或参数下标：

```bash
--constant BLOCK=256
--constant 4=256
```

当前 `--python` 模式只支持 `--pipeline ttir`。

## 6. Sophgo JIT 编译失败自动诊断

如果希望 `python3 /workspace/triton-sophgo-backend/tests/test_jit.py` 或新增算子在编译失败时自动定位 pass/op，可以打开：

```bash
TRITON_ANCHOR_DIAGNOSE_ON_ERROR=1
```

推荐命令：

```bash
cd /workspace/triton-sophgo-backend
PYTHONPATH=/workspace/triton-anchor/python:/workspace/triton-anchor:$PYTHONPATH \
TRITON_ANCHOR_DIAGNOSE_ON_ERROR=1 \
TRITON_ANCHOR_DIAGNOSE_DIR=/workspace/triton-anchor/diagnose-output/jit-auto \
python3 tests/test_jit.py
```

该能力不是只针对 `test_jit.py` 里已有算子。任意新增算子只要通过 Sophgo 后端 `fn[grid](...)` 进入 JIT 编译，并在 `_make_ttir()` 或 `_make_linalg()` 阶段失败，都会自动触发同一套诊断。

自动诊断输出示例：

```text
[AnchorDiagnose] pipeline triton-linalg failed at pass 7/14: triton_linalg.triton_to_linalg
[AnchorDiagnose] before IR: ...
[AnchorDiagnose] diagnostic detail: ...
[AnchorDiagnose] location: file.py:line:column
[AnchorDiagnose] operation: tt.elementwise_inline_asm
[AnchorDiagnose] summary: ...
```

输出目录规则：

1. 设置 `TRITON_ANCHOR_DIAGNOSE_DIR` 时，写入该目录。
2. 否则设置 `TRITON_DUMP_DIR` 时，写入 `<TRITON_DUMP_DIR>/<kernel>/anchor-diagnose/<pipeline>/`。
3. 否则写入当前工作目录下的 `triton-anchor-diagnose/<kernel>/<pipeline>/`。

## 7. 输出文件说明

诊断输出目录通常包含：

```text
01-pass-name.before.mlir
01-pass-name.after.mlir
...
NN-failed-pass.before.mlir
NN-failed-pass.diagnostic.txt
summary.json
```

文件含义：

- `*.before.mlir`：某个 pass 执行前的 IR。
- `*.after.mlir`：某个成功 pass 执行后的 IR。
- `*.diagnostic.txt`：失败 pass 的错误、MLIR diagnostic、location/op 和 traceback。
- `summary.json`：机器可读的诊断摘要，包含 pipeline、失败 pass、失败序号、location/op 和各 pass 记录。

失败时 CLI 输出示例：

```text
FAILED: pipeline triton-linalg failed at pass 7/14: triton_linalg.triton_to_linalg
before IR: /path/to/07-triton_linalg.triton_to_linalg.before.mlir
diagnostic detail: /path/to/07-triton_linalg.triton_to_linalg.diagnostic.txt
location: /path/to/kernel.py:14:8
operation: tt.elementwise_inline_asm
error: PassManager::run failed
summary: /path/to/summary.json
```

## 8. 已验证的失败样例

仓库内保留了一个真实失败样例：

```text
ops-diagnose-cli/test/anchor_diag_bad_kernels.py
```

该样例使用 `tl.inline_asm_elementwise`，TTIR 阶段可以通过，但在 `triton-linalg` lowering 中失败。当前工具可以定位到：

```text
pipeline: triton-linalg
pass: triton_linalg.triton_to_linalg
operation: tt.elementwise_inline_asm
location: ops-diagnose-cli/test/anchor_diag_bad_kernels.py:14:8
```

相关诊断产物保存在：

```text
ops-diagnose-cli/test/bad-inline-ttir-diagnose/
ops-diagnose-cli/test/bad-inline-linalg-diagnose/
```

## 9. 常见问题

### 9.1 `triton-anchor-diagnose: command not found`

说明当前环境还没有安装包含 console script 的包。可以先用：

```bash
PYTHONPATH=/workspace/triton-anchor/python:/workspace/triton-anchor \
/opt/venv/bin/python -m triton_anchor.diagnose --help
```

或者重新构建安装当前仓库 wheel。

### 9.2 `triton-anchor-diagnose --help` 看不到 `--python`

说明命令来自旧安装包。优先使用源码模块方式运行，或重新安装当前 wheel。

### 9.3 诊断没有定位到 op

op/location 是 best-effort 解析，依赖底层 MLIR/C++ diagnostic 是否输出了相关信息。如果底层只报 `PassManager::run failed`，工具仍能稳定定位失败 pass 和失败前 IR，但不一定能定位到具体 op。
