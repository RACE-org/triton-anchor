# AnchorBackendBase 与 Sophgo 后端改造总结

## 1. 文档目的

本文记录 `triton-anchor_v3.0` 中 Anchor 后端基础设施、硬件能力校验和
`triton-sophgo-backend` 的本轮改造，说明以下内容：

- `AnchorBackendBase` 的职责、扩展接口和编译阶段所有权；
- `HWCapability` 校验及诊断能力的增强；
- Sophgo 后端从独立实现迁移到 `AnchorBackendBase` 的方式；
- PPL 编译辅助函数从运行时驱动迁回编译器的原因；
- Python 包初始化、CMake 构建和 LLVM 动态链接问题的修复；
- 单元测试、冒烟测试和 JIT 测试分别验证什么；
- Linux/TPU 环境中的构建、安装、测试和故障诊断方法。

涉及的主要目录为：

```text
python/triton_anchor/
triton-anchor_v3.0/triton-sophgo-backend/
```

## 2. 最终架构与职责边界

最终编译路径为：

```text
Triton JIT
  │
  ▼
AnchorBackendBase.add_stages()
  ├─ ttir       Anchor 负责
  ├─ linalg     Anchor 负责，形成 Linalg AnchorIR
  ├─ pplir      Sophgo compiler.py 负责
  └─ so         Sophgo compiler.py 负责
                 │
                 ▼
           Sophgo driver.py
           仅加载并启动 .so
```

核心边界如下：

| 模块 | 职责 |
| --- | --- |
| `triton_anchor/backend.py` | 管理 TTIR 和 AnchorIR 前端阶段，选择 Adapter，记录元数据，执行可选的 AnchorIR 校验 |
| `triton_anchor/hw_capability.py` | 描述硬件能力，严格校验配置，输出完整诊断报告 |
| `triton_sophgo/compiler.py` | 声明 Sophgo 硬件能力，追加 PPL IR 和 `.so` 厂商阶段，完成 PPL 编译 |
| `triton_sophgo/driver.py` | 运行时加载二进制、准备参数并启动内核，不执行编译或构建 |
| `triton_sophgo/__init__.py` | 导出、注册并激活 Sophgo 后端，处理共享库和 libdevice 注入 |
| 根 `CMakeLists.txt` | 构建 `_C.so`，安全定位 Triton/Anchor，并确保使用同一份 `libtriton.so` |

## 3. `backend.py`：Anchor 后端公共基础类

文件：

```text
python/triton_anchor/backend.py
```

### 3.1 `AnchorCompilationContext`

`AnchorCompilationContext` 保存一次编译所需的前端决策：

- `hw`：经过校验的 `HWCapability`；
- `track`：选择的 `AnchorIRTrack`；
- `adapter`：Linalg 路径使用的转换 Adapter；
- `validate_ir`：是否启用 AnchorIR 合法性校验。

它是编译配置上下文，不是 MLIR Context。具体 MLIR Context 仍由 Triton
创建，并从进入 stage 的 MLIR module 中获得。

### 3.2 `AnchorBackendBase`

`AnchorBackendBase` 继承 Triton 的 `BaseBackend`，统一实现硬件后端的
Anchor 前端流程。

厂商后端必须实现两个接口：

```python
def get_hw_capability(self, options) -> HWCapability:
    ...

def add_vendor_stages(self, stages, options, ctx) -> None:
    ...
```

- `get_hw_capability()` 声明当前目标硬件的能力；
- `add_vendor_stages()` 只能追加 AnchorIR 之后的厂商编译阶段。

可选扩展接口包括：

- `load_vendor_dialects(context)`：加载厂商专用 MLIR Dialect；
- `get_triton_gpu_conversion_target(options, hw)`：覆盖 TritonGPU
  转换目标字符串。

### 3.3 编译阶段所有权

`add_stages()` 的处理顺序为：

1. 调用 `get_hw_capability()`；
2. 严格校验 `HWCapability`；
3. 根据 `anchor_ir_track` 选择 Linalg 或 TritonGPU 路径；
4. 注册 Anchor 所有的 `ttir` 和 AnchorIR stage；
5. 调用厂商的 `add_vendor_stages()`；
6. 检查厂商没有覆盖 Anchor 已注册的 stage；
7. 检查厂商至少追加了一个 stage。

两条前端路径分别为：

```text
LINALG track:      ttir → linalg → vendor stages
TRITON_GPU track:  ttir → ttgir  → vendor stages
```

如果厂商覆盖 `ttir`、`linalg` 或 `ttgir`，会抛出 `RuntimeError`。如果厂商
没有追加任何 stage，也会直接报错。这保证公共前端逻辑只有一份实现。

### 3.4 Anchor 前端附加能力

`AnchorBackendBase` 还统一提供：

- 使用硬件能力构建 TTIR pipeline；
- 根据硬件能力从 `AdapterRegistry` 选择 Adapter；
- 为每个 specialization 生成稳定且带哈希后缀的 kernel symbol；
- 写入 `anchor_hw_name`、`anchor_arch_family`、`anchor_compute_paradigm`
  和 `anchor_ir_track` 等编译元数据；
- 将外部 Adapter 返回的文本重新解析为当前 Context 下的 MLIR module；
- 加载 Anchor Dialect，再通过 hook 加载厂商 Dialect；
- 通过编译选项 `validate_anchor_ir=True` 或环境变量
  `TRITON_ANCHOR_VALIDATE_IR=1` 启用 AnchorIR 校验。

### 3.5 包导出

`python/triton_anchor/__init__.py` 的版本更新为 `0.2.0`，并通过
`__getattr__()` 延迟导出：

```python
AnchorBackendBase
AnchorCompilationContext
```

延迟导出避免纯 Python 环境仅使用 `HWCapability` 或 AnchorIR 定义时，
过早要求 Triton 的已编译扩展存在。

## 4. `HWCapability` 校验增强

文件：

```text
python/triton_anchor/hw_capability.py
```

### 4.1 新增配置字段

为 spine 风格 CPU/Tensor 后端增加：

- `arch_id: Optional[str]`；
- `force_vector_interleave: int = 2`；
- `num_threads: Optional[int]`。

这些字段用于向 Adapter 或后续 lowering 传递架构和并行配置。

### 4.2 严格构造与统一校验

正常构造 `HWCapability` 时仍然是严格模式。`__post_init__()` 会调用
`validate()`，无效配置不能进入编译流水线。

校验项目包括：

1. 身份和策略字段：
   - `name`、`arch_family` 必须是非空字符串；
   - `compute_paradigm` 必须是 `ComputeParadigm`；
   - `anchor_ir_track` 必须是 `AnchorIRTrack`；
   - `ptr_model` 必须属于 `structured`、`axis_info`、`hybrid`、`gpu`。

2. 计算范式与能力对象：
   - `AME_MATRIX` 必须提供 `matrix_cap`；
   - `TENSOR_PROCESSOR` 必须提供 `tensor_cap`；
   - `GPGPU` 必须提供 `gpgpu_cap`；
   - 三种专用 capability 应保持互斥。

3. 数值和结构：
   - 核心数、warp、stage、CTA、DMA channel 等必须为正整数；
   - 内存容量允许为零，但不能为负数；
   - tile/cluster shape 必须是正整数 tuple；
   - `supported_dtypes` 必须是非空字符串集合；
   - 布尔字段必须是真正的 `bool`，不能用整数替代。

4. Adapter：
   - `preferred_adapter` 必须已注册；
   - Adapter 输出的 AnchorIR track 必须与 `anchor_ir_track` 一致；
   - 无法推断输出 track 时给出 warning，而不是伪装成成功。

`validate()` 和诊断报告共用同一组检查函数，避免“构造时的判断”和
“诊断命令的判断”发生偏差。

### 4.3 诊断接口

已经构造成功的实例可以输出完整报告：

```python
print(hw.diagnose())
```

需要一次收集所有配置错误时，可以使用不会在构造阶段立即抛错的入口：

```python
report = HWCapability.diagnose_config(
    name="",
    arch_family="tpu",
    compute_paradigm=ComputeParadigm.TENSOR_PROCESSOR,
    anchor_ir_track="invalid-track",
    ptr_model="invalid-pointer-model",
)
print(report)
```

报告包含：

- 总状态 `PASS` 或 `FAIL`；
- error/warning 数量；
- 当前完整配置；
- 每项检查的 `OK`、`WARN` 或 `ERROR` 结果。

`_validate_on_init` 仅用于 `diagnose_config()` 内部，外部传入会被拒绝。

## 5. Sophgo 编译器迁移

文件：

```text
triton-anchor_v3.0/triton-sophgo-backend/triton_sophgo/compiler.py
```

### 5.1 继承 `AnchorBackendBase`

`SophgoBackend` 从独立实现改为继承 `AnchorBackendBase`：

```python
class SophgoBackend(AnchorBackendBase):
    ...
```

后端不再重复实现 TTIR 和 Linalg 前端 lowering。

### 5.2 Sophgo 硬件能力

`get_hw_capability()` 声明：

```text
compute_paradigm = TENSOR_PROCESSOR
anchor_ir_track  = LINALG
ptr_model        = axis_info
preferred_adapter = triton-linalg
```

芯片名来自 `TRITON_CHIP_NAME`，核心数依次读取：

```text
TRITON_SOPHGO_NUM_CORES
TRITON_CORE_NUM
默认值 1
```

对应 stage 顺序必须为：

```text
ttir → linalg → pplir → so
```

### 5.3 厂商 stages

`add_vendor_stages()` 仅追加：

```python
stages["pplir"] = ...
stages["so"] = ...
```

- `_make_pplir()`：执行 TritonToPPL 和 LinalgToPPL pass；
- `_pplir_to_so()`：输出 MLIR、执行 `ppl-compile`、调用 CMake/Make，
  查找生成的共享库并以 `bytes` 返回给 Triton cache manager。

## 6. PPL 编译逻辑回归 `compiler.py`

此前 JIT 报错：

```text
AttributeError: module 'triton_sophgo.driver' has no attribute 'compile_to_so'
```

根因是编译阶段依赖了运行时驱动中的 `compile_to_so()`，而参考实现的职责
划分是编译器完成二进制生成，驱动只负责加载和执行。

本轮调整后：

- `_pplir_to_so()` 在 `compiler.py` 内直接完成编译；
- `subprocess`、`shutil`、PPL/CMake/Make 等构建逻辑只存在于
  `compiler.py`；
- `driver.py` 不再包含 `compile_to_so()`；
- `driver.py` 不再导入 `subprocess`、`shutil`；
- `driver.py` 只保留 Buffer、binary loader、launcher、设备接口和
  `SophgoDriver`。

因此，`.so` 生成属于编译缓存链路，而 `.so` 加载与启动属于运行时链路。

## 7. Python 包初始化与注册

文件：

```text
triton-anchor_v3.0/triton-sophgo-backend/triton_sophgo/__init__.py
```

### 7.1 修复模块路径

此前基础 import 报错：

```text
ModuleNotFoundError: No module named 'triton_sophgo.runtime'
```

实际驱动实现位于 `driver.py`，初始化代码改为从 `.driver` 导入：

```python
from .driver import SophgoDriver, StridedBuffer
SOPHGODriver = SophgoDriver
```

`SOPHGODriver` 作为旧名称兼容别名保留。

### 7.2 后端发现和主动注册

模块同时提供：

```python
compiler_cls = SophgoBackend
driver_cls = SophgoDriver
```

供 Triton 的 `triton.backends` entry point 发现机制使用。

`_register_sophgo_backend()` 还支持用户主动 `import triton_sophgo` 时：

- 将 `sophgo` 写入 Triton backend registry；
- 设置 `SophgoDriver` 为活跃驱动；
- 在 Triton entry point 循环初始化期间安全返回，由 Triton 完成注册；
- 将非致命的驱动激活问题输出到 `stderr`，避免污染构建系统捕获的
  `stdout`。

包初始化还保留：

- `extend_path()`；
- `libtriton.so` 和 Anchor `_C.so` 的全局符号加载；
- libdevice monkey patch 的 eager 注入。

## 8. CMake 与 LLVM 重复注册修复

文件：

```text
triton-anchor_v3.0/triton-sophgo-backend/CMakeLists.txt
```

### 8.1 Ninja `build.ninja` lexing error

原配置阶段通过：

```python
import triton
```

查找安装路径。导入 Triton 会触发 backend entry point 和设备初始化，并
可能向标准输出写入：

```text
Arch sg2260e specified by env var
[triton_sophgo] Warning: ...
```

CMake 将多行输出整体当作 `TRITON_ROOT`，最终生成含非法路径的
`build.ninja`，Ninja 报：

```text
ninja: error: build.ninja:435: lexing error
```

修复方式是使用无导入副作用的：

```python
importlib.util.find_spec("triton")
importlib.util.find_spec("triton_anchor")
```

同时检查命令返回值和空路径，发现失败时立即输出 `FATAL_ERROR`。

### 8.2 CMP0116

设置：

```cmake
if(POLICY CMP0116)
    cmake_policy(SET CMP0116 NEW)
endif()
```

用于消除 Ninja DEPFILE 路径转换的开发者警告。

### 8.3 LLVM CommandLine option 重复注册

替换后端 wheel 后曾发生：

```text
Option 'use-dereferenceable-at-point-semantics' registered more than once
LLVM ERROR: inconsistency in registered CommandLine options
```

该错误表示同一进程中加载了两份互不一致的 LLVM/MLIR 实现。目标后端的
`_C.so` 必须复用当前 Triton 安装中的 `libtriton.so`，不能在找不到该
共享库时静默构建出另一套 LLVM/MLIR 运行时。

CMake 现在执行：

```cmake
find_library(
    TRITON_SHARED_LIB
    triton
    PATHS ${TRITON_LIB_DIR}
    NO_DEFAULT_PATH
)
```

并且：

- 找不到 `libtriton.so` 时直接终止构建；
- `_C.so` 显式链接该 `libtriton.so`；
- `BUILD_RPATH` 和 `INSTALL_RPATH` 指向同一 `TRITON_LIB_DIR`。

构建日志必须出现类似：

```text
-- Auto-discovered triton at: /opt/venv/lib/python3.12/site-packages/triton
-- Found libtriton: /opt/venv/lib/python3.12/site-packages/triton/_C/libtriton.so
```

## 9. 测试文件说明

### 9.1 Anchor 基础类测试

文件：

```text
python/triton_anchor/tests/test_backend.py
```

覆盖：

- Linalg track 会选择 `triton-linalg` Adapter；
- Linalg stage 顺序为 `ttir → linalg → vendor binary`；
- TritonGPU track 不选择 Linalg Adapter；
- TritonGPU stage 顺序为 `ttir → ttgir → vendor binary`；
- 厂商后端覆盖 Anchor 自有 stage 时会被拒绝。

该文件使用：

```python
pytest.importorskip("triton._C.libtriton")
```

因此缺少 Triton C++ 扩展时结果是 `SKIPPED`，不能解释为测试通过。

### 9.2 硬件能力测试

文件：

```text
python/triton_anchor/tests/test_hw_capability.py
```

覆盖：

- 三种 `ComputeParadigm`；
- Sophgo、SpacemiT、GPU 三类有效配置；
- `to_gpu_target()` 兼容接口；
- 缺少范式专用 capability；
- Adapter 未注册；
- Adapter 输出 track 不匹配；
- Matrix/Tensor capability 非法数值；
- `diagnose()` 的 PASS 报告；
- `diagnose_config()` 一次收集多个错误；
- 空/不完整配置的非抛异常诊断。

### 9.3 Sophgo 冒烟测试

文件：

```text
triton-anchor_v3.0/triton-sophgo-backend/tests/test_smoke.py
```

包含五组检查：

1. 导入 `triton_sophgo` 和 `_C.passes`；
2. 确认 `SophgoBackend` 继承 `AnchorBackendBase`，并可实例化 Driver；
3. 确认硬件能力和 `ttir → linalg → pplir → so` stage 顺序；
4. 确认 wheel 的 `triton.backends` entry point；
5. 确认 LinalgToPPL pass 可以加入 MLIR PassManager。

### 9.4 Sophgo JIT 测试

文件：

```text
triton-anchor_v3.0/triton-sophgo-backend/tests/test_jit.py
```

通过标准 `@triton.jit` 和 `kernel[grid](...)` 路径验证：

- 向量加法；
- 逐元素乘法；
- ReLU；
- 融合乘加。

测试会将 CPU 计算结果与 TPU 输出比较，因此同时覆盖：

```text
entry point 发现
→ Anchor/Sophgo 编译 stages
→ PPL 编译
→ .so 加载
→ SophgoLauncher 启动
→ 数值正确性
```

`tests/conftest.py` 提供公共的测试执行、TPU tensor 转换、结果比较、dump
目录准备、环境信息和失败统计。

## 10. 构建与验证步骤

### 10.1 Anchor 单元测试

在仓库根目录执行：

```bash
python3 -m pytest \
  python/triton_anchor/tests/test_backend.py \
  python/triton_anchor/tests/test_hw_capability.py
```

### 10.2 Sophgo wheel 构建和安装

在 Linux 容器中执行：

```bash
cd /triton/anchor_sophgo_v3.0/triton-anchor_v3.0/triton-sophgo-backend

uv build --wheel --no-build-isolation

uv pip install --force-reinstall \
  dist/triton_sophgo_backend-0.1.0-*.whl
```

必须确认构建日志中定位到了预期的 Triton 和 `libtriton.so`。

### 10.3 冒烟和 JIT 测试

```bash
python3 tests/test_smoke.py
python3 tests/test_jit.py
```

建议先执行 smoke。只有 smoke 的 import、注册、pipeline 和 PassManager
全部通过后，才进入需要 PPL 工具链和 TPU 设备的 JIT 测试。

### 10.4 LLVM 重复注册仍存在时

如果干净重建和强制安装后仍发生 LLVM option 重复注册，收集：

```bash
ldd /opt/venv/lib/python3.12/site-packages/triton_sophgo/_C*.so

readelf -d \
  /opt/venv/lib/python3.12/site-packages/triton_sophgo/_C*.so |
  grep -E 'NEEDED|RPATH|RUNPATH'
```

重点检查：

- `NEEDED` 是否包含预期的 `libtriton.so`；
- RPATH/RUNPATH 是否指向当前 venv 的 `triton/_C`；
- 是否额外加载了另一目录下的 LLVM/MLIR 动态库。

## 11. 验收标准

本轮改造的完整验收条件为：

- Anchor 单元测试通过；没有 Triton C++ 扩展造成的 skip 需单独说明；
- wheel 构建日志明确找到同一 venv 中的 `libtriton.so`；
- `test_smoke.py` 五项全部通过；
- `test_jit.py` 四个 kernel 编译、启动和数值比较全部通过；
- `driver.py` 中不存在 PPL/CMake/Make 编译逻辑；
- 实际 stage 顺序为 `ttir → linalg → pplir → so`；
- 导入和执行期间不再出现：
  - `No module named 'triton_sophgo.runtime'`；
  - `build.ninja: lexing error`；
  - `driver has no attribute 'compile_to_so'`；
  - LLVM CommandLine option 重复注册。

当前 Windows 工作区可以完成代码和静态结构核对；wheel 链接、PPL 编译及
TPU 数值测试必须以 Linux/TPU 容器的实际执行结果作为最终结论。
