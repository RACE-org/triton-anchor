# Triton Anchor 编译诊断工具

本文档说明 `triton-anchor-diagnose` 和 JIT 自动诊断 hook 的当前能力、使用方法和边界。

## 1. 能诊断什么

当前诊断能力覆盖四类失败：

1. Python/Triton 前端生成 TTIR 之前的错误

   使用 `--python MODULE:FUNCTION` 时，如果 Python kernel 导入、签名/constexpr 解析、Triton AST 到 TTIR 生成过程失败，工具会输出 pre-pass 诊断：

   ```text
   stage: python-frontend
   diagnostic detail: python-frontend.diagnostic.txt
   summary: summary.json
   ```

   这一阶段还没有进入 MLIR pass，因此不会有 failed pass，但会保留 traceback，并尽量从错误文本中提取源码 location 和 op。

2. MLIR parse 阶段已经失败的非法 IR

   使用文件输入时，如果 `.ttir/.mlir` 在 `ir.parse_mlir_module()` 阶段失败，工具会输出：

   ```text
   stage: input-parse
   diagnostic detail: input-parse.diagnostic.txt
   summary: summary.json
   ```

   这一阶段同样没有 failed pass，但会保留 parse diagnostic、traceback，并尽量根据 `file:line:column` 回查输入文件中的 IR 行。

3. TTIR pipeline

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

4. TTIR 之后的 `triton-linalg` adapter pipeline

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

- PPLIR 阶段失败。
- `ppl-compile` 失败。
- CMake / `.so` 生成失败。
- kernel runtime crash、结果错误、精度错误。

这些阶段需要后续继续接入对应的诊断逻辑。

## 3. 环境安装与命令确认

进入虚拟环境：

```bash
cd /workspace/triton-anchor
source /opt/venv/bin/activate
```

如果当前代码还没有安装到虚拟环境，先按仓库构建流程重新安装 wheel：

```bash
cd /workspace/triton-anchor
source envsetup.sh
uv build --wheel --no-build-isolation
uv pip install dist/triton_anchor-*.whl
```

确认命令已经来自当前虚拟环境：

```bash
which triton-anchor-diagnose
triton-anchor-diagnose --help
```

正常情况下 `which` 应输出：

```text
/opt/venv/bin/triton-anchor-diagnose
```

## 4. 诊断已有 IR 文件

已有 `.ttir/.mlir` 文件时，可以让工具从该 IR 开始重跑指定 pipeline，定位这份 IR 在后续 pass 中的失败点。

诊断 TTIR pipeline：

```bash
cd /workspace/triton-anchor
triton-anchor-diagnose \
  /path/to/input.ttir \
  --pipeline ttir \
  --output-dir /workspace/triton-anchor/diagnose-output/ttir
```

诊断 `triton-linalg` pipeline：

```bash
cd /workspace/triton-anchor
triton-anchor-diagnose \
  /path/to/input.mlir \
  --pipeline triton-linalg \
  --output-dir /workspace/triton-anchor/diagnose-output/triton-linalg
```

注意：文件输入模式不是“检查 IR 是否已经生成成功”，而是复现“这份中间 IR 在后续 pass 中为什么失败”。

## 5. 不提前 dump IR，直接诊断 Python kernel

如果没有现成 `.ttir` 文件，可以通过 `--python MODULE:FUNCTION` 让 CLI 在运行时生成 TTIR 后诊断：

```bash
cd /workspace/triton-anchor
triton-anchor-diagnose \
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

如果 TTIR 还没有生成成功，CLI 会在进入 pass 前停止并输出 `python-frontend` 诊断。例如：

```text
FAILED: python-frontend failed before TTIR generation.
python target: tests.test_smoke:_smoke_add_kernel
diagnostic detail: /path/to/python-frontend.diagnostic.txt
location: kernel.py:4:5
operation: tl.load
error: <frontend error>
diagnostic output: /path/to/output
summary: /path/to/output/summary.json
```

## 6. Sophgo JIT 编译失败自动诊断

如果希望 `python3 /workspace/triton-sophgo-backend/tests/test_jit.py` 或新增算子在编译失败时自动定位 pass/op，可以打开：

```bash
TRITON_ANCHOR_DIAGNOSE_ON_ERROR=1
```

推荐命令：

```bash
cd /workspace/triton-sophgo-backend
TRITON_ANCHOR_DIAGNOSE_ON_ERROR=1 \
TRITON_ANCHOR_DIAGNOSE_DIR=/workspace/triton-anchor/diagnose-output/jit-auto \
python3 tests/test_jit.py
```

该能力不是只针对 `test_jit.py` 里已有算子。任意新增算子只要通过 Sophgo 后端 `fn[grid](...)` 进入 JIT 编译，并在 `_make_ttir()` 或 `_make_linalg()` 阶段失败，都会自动触发同一套诊断。

仓库中的失败样例可以直接这样运行：

```bash
cd /workspace/triton-anchor
source /opt/venv/bin/activate

TRITON_ANCHOR_DIAGNOSE_ON_ERROR=1 \
python /workspace/triton-anchor/ops-diagnose-cli/test/anchor_diag_bad_kernels.py
```

该文件在 `__main__` 中不会依赖 `torch` / `torch_tpu`，而是直接从 `@triton.jit` 函数生成内存 TTIR，然后依次诊断 TTIR pipeline 和 `triton-linalg` pipeline。失败时会输出 pass/op/location，默认诊断产物写入：

```text
/workspace/triton-anchor/ops-diagnose-cli/test/auto-diagnose/
```

另一个样例用于验证 Python/Triton frontend 生成 TTIR 前失败：

```bash
cd /workspace/triton-anchor
source /opt/venv/bin/activate

TRITON_ANCHOR_DIAGNOSE_ON_ERROR=1 \
python /workspace/triton-anchor/ops-diagnose-cli/test/anchor_diag_bad_frontend_kernel.py
```

该样例使用不存在的 Triton language builtin：

```text
tl.this_builtin_does_not_exist(x)
```

因此会在 TTIR 生成前失败，预期输出：

```text
FAILED: python-frontend failed before TTIR generation.
diagnostic detail: .../python-frontend.diagnostic.txt
summary: .../summary.json
```

如果想通过 Sophgo JIT 后端自动 hook 诊断实际算子 launch，则 Python 文件必须调用 `kernel[grid](...)`。如果只是定义 `@triton.jit` 函数但没有任何编译或 launch 入口，直接运行它不会触发诊断。

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
input-parse.diagnostic.txt
python-frontend.diagnostic.txt
01-pass-name.before.mlir
01-pass-name.after.mlir
...
NN-failed-pass.before.mlir
NN-failed-pass.diagnostic.txt
summary.json
```

文件含义：

- `input-parse.diagnostic.txt`：输入 `.ttir/.mlir` 无法 parse 时的错误、location/op 和 traceback。
- `python-frontend.diagnostic.txt`：`--python` 模式下生成 TTIR 前失败的错误、location/op 和 traceback。
- `*.before.mlir`：某个 pass 执行前的 IR。
- `*.after.mlir`：某个成功 pass 执行后的 IR。
- `*.diagnostic.txt`：失败 pass 的错误、MLIR diagnostic、location/op 和 traceback。
- `summary.json`：机器可读的诊断摘要。pass 失败时包含失败 pass、失败序号、location/op 和各 pass 记录；pre-pass 失败时包含失败 stage、输入来源、location/op 和 traceback。

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
location: ops-diagnose-cli/test/anchor_diag_bad_kernels.py:23:8
```

相关诊断产物保存在：

```text
ops-diagnose-cli/test/bad-inline-ttir-diagnose/
ops-diagnose-cli/test/bad-inline-linalg-diagnose/
```

## 9. 常见问题

### 9.1 `triton-anchor-diagnose: command not found`

说明当前 shell 没有进入 `/opt/venv`，或当前环境还没有安装包含 console script 的 wheel。先确认：

```bash
cd /workspace/triton-anchor
source /opt/venv/bin/activate
which triton-anchor-diagnose
```

如果仍然找不到，重新构建安装当前仓库 wheel：

```bash
cd /workspace/triton-anchor
source envsetup.sh
uv build --wheel --no-build-isolation
uv pip install dist/triton_anchor-*.whl
```

### 9.2 `triton-anchor-diagnose --help` 看不到 `--python`

说明命令来自旧安装包。重新安装当前 wheel：

```bash
cd /workspace/triton-anchor
source /opt/venv/bin/activate
source envsetup.sh
uv build --wheel --no-build-isolation
uv pip install dist/triton_anchor-*.whl
```

### 9.3 诊断没有定位到 op

op/location 是 best-effort 解析，依赖底层 MLIR/C++ diagnostic 是否输出了相关信息。如果底层只报 `PassManager::run failed`，工具仍能稳定定位失败 pass 和失败前 IR，但不一定能定位到具体 op。
