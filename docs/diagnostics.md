# Triton Anchor 编译诊断工具

本文档说明 `triton-anchor-diagnose` 和 JIT 自动诊断 hook 的当前能力、使用方法和边界。

## 1. 能诊断什么

当前诊断能力覆盖以下失败：

1. Python/Triton 前端生成 TTIR 之前的错误

   推荐通过 `TRITON_ANCHOR_DIAGNOSE_ON_ERROR=1 python your_kernel.py` 走真实 JIT 编译路径触发诊断。辅助的 `--python MODULE:FUNCTION` 模式也可以单独调试 Python kernel 到 TTIR 的生成过程。如果 Python kernel 导入、签名/constexpr 解析、Triton AST 到 TTIR 生成过程失败，工具会输出 pre-pass 诊断：

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

5. Sophgo PPLIR lowering pipeline

   可以定位 Sophgo 后端 `Linalg -> PPLIR` 阶段中首个失败 pass：

   ```text
   sophgo.triton_to_ppl
   sophgo.linalg_to_ppl
   common.cse.pplir
   common.canonicalizer.pplir
   ```

   CLI 可用：

   ```bash
   triton-anchor-diagnose \
     /path/to/linalg.mlir \
     --pipeline sophgo-pplir \
     --output-dir /workspace/triton-anchor/diagnose-output/sophgo-pplir
   ```

   该 pipeline 需要当前 Python 环境能加载 `triton-sophgo-backend` 的 PPLIR pass 绑定，即 `triton_sophgo._C.passes`。

6. `ppl-compile` 失败

   Sophgo JIT 自动诊断开启后，`ppl-compile` 返回非 0 时会写入：

   ```text
   ppl-compile.diagnostic.txt
   summary.json
   ```

   诊断内容包括命令行、return code、压缩后的 `ppl-compile` 输出、输入 `.mlir` 路径、工作目录，以及从工具输出中 best-effort 提取的 MLIR location/op。

7. CMake / `.so` 生成失败

   CMake 配置、`make install` 或最终 `.so` 未生成时，会写入：

   ```text
   cmake.diagnostic.txt
   so-generation.diagnostic.txt
   summary.json
   ```

   诊断内容包括命令行、return code、构建目录、CMake template、检查过的 `.so` 路径和 traceback。

8. kernel runtime launch 失败

   Sophgo launcher 调用 `torch_tpu.CallCppDynLib(...)` 抛出 Python 异常时，会写入：

   ```text
   runtime-launch.diagnostic.txt
   summary.json
   ```

   诊断内容包括 `.so` 路径、entry name、grid、运行时参数的 shape/dtype/device/data_ptr 摘要和 traceback。

9. 结果错误 / 精度错误

   Sophgo 测试中的 `assert_close()` 在数值不匹配时会写入：

   ```text
   result-mismatch.diagnostic.txt
   summary.json
   ```

   诊断内容包括 `rtol/atol`、最大绝对误差、最大相对误差、不匹配元素数量、actual/expected tensor 摘要和最多 10 个 mismatch 样本。

如果底层 MLIR/C++ diagnostic 包含位置信息，工具会进一步提取：

```text
source location: file:line:column
operation: tt.xxx
before IR: xxx.before.mlir
diagnostic detail: xxx.diagnostic.txt
summary: summary.json
```

## 2. 当前边界

当前仍有以下边界：

- runtime 诊断能捕获 Python 层 launch 异常；如果底层 runtime 直接导致进程 abort/segfault，Python 代码没有机会写完整诊断，只能依赖崩溃前已经产生的日志和产物。
- 结果错误 / 精度错误必须有 expected output。当前已在 Sophgo 测试 `assert_close()` 中接入，业务代码如果不做结果比较，工具无法凭空判断数值是否正确。
- op/location 仍是 best-effort，依赖 MLIR / C++ / 外部工具输出中是否包含 location 或 operation 文本。

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

## 4. 推荐入口：运行 Python 算子时自动诊断

实际调试算子编译失败时，优先使用真实执行路径打开自动诊断：

```bash
TRITON_ANCHOR_DIAGNOSE_ON_ERROR=1 python your_kernel.py
```

如果希望固定诊断输出目录：

```bash
TRITON_ANCHOR_DIAGNOSE_ON_ERROR=1 \
TRITON_ANCHOR_DIAGNOSE_DIR=/workspace/triton-anchor/diagnose-output/jit-auto \
python your_kernel.py
```

`your_kernel.py` 必须实际调用 `kernel[grid](...)`，或通过测试入口触发 kernel 编译。只定义 `@triton.jit` 函数不会进入 Triton/Sophgo 编译流程，因此不会触发诊断。

运行 Sophgo 后端测试时可以这样用：

```bash
cd /workspace/triton-sophgo-backend
TRITON_ANCHOR_DIAGNOSE_ON_ERROR=1 \
TRITON_ANCHOR_DIAGNOSE_DIR=/workspace/triton-anchor/diagnose-output/jit-auto \
python3 tests/test_jit.py
```

该能力不是只针对 `test_jit.py` 里已有算子。任意新增算子只要通过 Sophgo 后端 `fn[grid](...)` 进入 JIT 编译，并在 `_make_ttir()`、`_make_linalg()`、`_make_pplir()`、`ppl-compile`、CMake / `.so` 生成或 runtime launch 阶段失败，都会自动触发对应诊断。

自动诊断输出示例：

```text
[AnchorDiagnose] pipeline triton-linalg failed at pass 7/14: triton_linalg.triton_to_linalg
[AnchorDiagnose] before IR: ...
[AnchorDiagnose] diagnostic detail: ...
[AnchorDiagnose] location: file.py:line:column
[AnchorDiagnose] operation: tt.elementwise_inline_asm
[AnchorDiagnose] summary: ...
```

非 pass 阶段诊断输出示例：

```text
[AnchorDiagnose] stage ppl-compile failed
[AnchorDiagnose] diagnostic detail: .../ppl-compile.diagnostic.txt
[AnchorDiagnose] command: ppl-compile ...
[AnchorDiagnose] returncode: 1
[AnchorDiagnose] summary: .../summary.json
```

输出目录规则：

1. 设置 `TRITON_ANCHOR_DIAGNOSE_DIR` 时，写入该目录。
2. 否则设置 `TRITON_DUMP_DIR` 时，写入 `<TRITON_DUMP_DIR>/<kernel>/anchor-diagnose/<pipeline>/`。
3. 否则写入当前工作目录下的 `triton-anchor-diagnose/<kernel>/<pipeline>/`。

仓库内自检样例：

```bash
cd /workspace/triton-anchor
source /opt/venv/bin/activate

TRITON_ANCHOR_DIAGNOSE_ON_ERROR=1 \
python /workspace/triton-anchor/ops-diagnose-cli/test/anchor_diag_bad_kernels.py
```

该文件在 `__main__` 中直接从 `@triton.jit` 函数生成内存 TTIR，然后依次诊断 TTIR pipeline 和 `triton-linalg` pipeline。失败时会输出 pass/op/location，默认诊断产物写入：

```text
/workspace/triton-anchor/ops-diagnose-cli/test/auto-diagnose/
```

Python/Triton frontend 生成 TTIR 前失败样例：

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

PPLIR 阶段失败样例：

```bash
cd /workspace/triton-anchor
source /opt/venv/bin/activate

TRITON_ANCHOR_DIAGNOSE_ON_ERROR=1 \
TRITON_ANCHOR_DIAGNOSE_DIR=/workspace/triton-anchor/ops-diagnose-cli/test/bad-pplir-diagnose-check \
python /workspace/triton-anchor/ops-diagnose-cli/test/anchor_diag_bad_pplir_kernel.py
```

该样例直接生成内存 TTIR 后依次诊断 TTIR、`triton-linalg`、`sophgo-pplir`。它构造了一个 2D 到 3D 的 broadcast，最终在 Sophgo `Linalg -> PPLIR` 的 C 维 broadcast 限制处失败。预期输出包含：

```text
OK: pipeline ttir completed, 7/7 passes executed.
OK: pipeline triton-linalg completed, 14/14 passes executed.
FAILED: pipeline sophgo-pplir failed at pass 2/4: sophgo.linalg_to_ppl
location: /workspace/triton-anchor/ops-diagnose-cli/test/anchor_diag_bad_pplir_kernel.py:28:24
operation: linalg.broadcast
ir line: 19: %broadcasted = linalg.broadcast ...
```

## 5. 诊断已有 IR 文件

已有 `.ttir/.mlir` 文件时，可以让工具从该 IR 开始重跑指定 pipeline，定位这份 IR 在后续 pass 中的失败点。这里不只支持 TTIR；已经生成的 `triton-linalg` MLIR / Linalg IR 也可以作为输入继续诊断后续 pipeline。

诊断 TTIR pipeline：

```bash
cd /workspace/triton-anchor
triton-anchor-diagnose \
  /path/to/input.ttir \
  --pipeline ttir \
  --output-dir /workspace/triton-anchor/diagnose-output/ttir
```

诊断 `triton-linalg` pipeline，即从 TTIR/MLIR 输入继续执行 Triton Anchor adapter lowering：

```bash
cd /workspace/triton-anchor
triton-anchor-diagnose \
  /path/to/input.mlir \
  --pipeline triton-linalg \
  --output-dir /workspace/triton-anchor/diagnose-output/triton-linalg
```

诊断 Sophgo PPLIR pipeline，即从已经生成的 linalg MLIR 继续执行 Sophgo `Linalg -> PPLIR` lowering：

```bash
cd /workspace/triton-anchor
triton-anchor-diagnose \
  /path/to/linalg.mlir \
  --pipeline sophgo-pplir \
  --output-dir /workspace/triton-anchor/diagnose-output/sophgo-pplir
```

注意：文件输入模式不是“检查 IR 是否已经生成成功”，而是复现“这份中间 IR 在后续 pass 中为什么失败”。

## 6. 辅助入口：CLI 从 Python Kernel 生成 TTIR

`triton-anchor-diagnose --python MODULE:FUNCTION` 是辅助调试入口，只负责从 Python `@triton.jit` kernel 生成 TTIR，然后诊断 TTIR pipeline。它不走完整 Sophgo JIT 编译链路，因此不作为实际算子编译失败的主推荐入口。

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

如果要诊断完整 Python 算子编译链路，使用第 4 节的 `TRITON_ANCHOR_DIAGNOSE_ON_ERROR=1 python your_kernel.py`。如果已经有 `.ttir/.mlir` 中间文件，使用第 5 节的文件输入模式，把该 IR 交给 `--pipeline triton-linalg` 或 `--pipeline sophgo-pplir` 继续诊断。

## 7. 输出文件说明

诊断输出目录通常包含：

```text
input-parse.diagnostic.txt
python-frontend.diagnostic.txt
ppl-compile.diagnostic.txt
cmake.diagnostic.txt
so-generation.diagnostic.txt
runtime-launch.diagnostic.txt
result-mismatch.diagnostic.txt
01-pass-name.before.mlir
01-pass-name.after.mlir
...
NN-failed-pass.before.mlir
NN-failed-pass.diagnostic.txt
summary.json
```

文件含义：

- `input-parse.diagnostic.txt`：输入 `.ttir/.mlir` 无法 parse 时的错误、location/op 和 traceback。
- `python-frontend.diagnostic.txt`：生成 TTIR 前失败的错误、location/op 和 traceback，可来自自动诊断 hook 或辅助 `--python` 模式。
- `ppl-compile.diagnostic.txt`：`ppl-compile` 命令、return code、工具输出、输入 IR 和 traceback。
- `cmake.diagnostic.txt`：CMake 配置阶段失败的命令、输出、构建目录和 traceback。
- `so-generation.diagnostic.txt`：`make install` 或最终 `.so` 查找失败的命令、输出、检查路径和 traceback。
- `runtime-launch.diagnostic.txt`：runtime launch 异常、`.so`、entry、grid 和参数摘要。
- `result-mismatch.diagnostic.txt`：数值不匹配的误差统计和样本。
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

## 8. 常见问题

### 8.1 `triton-anchor-diagnose: command not found`

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

### 8.2 `triton-anchor-diagnose --help` 看不到 `--python`

说明命令来自旧安装包。重新安装当前 wheel：

```bash
cd /workspace/triton-anchor
source /opt/venv/bin/activate
source envsetup.sh
uv build --wheel --no-build-isolation
uv pip install dist/triton_anchor-*.whl
```

### 8.3 诊断没有定位到 op

op/location 是 best-effort 解析，依赖底层 MLIR/C++ diagnostic 是否输出了相关信息。如果底层只报 `PassManager::run failed`，工具仍能稳定定位失败 pass 和失败前 IR，但不一定能定位到具体 op。
