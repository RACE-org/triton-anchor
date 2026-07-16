# 自定义 Adapter 开发指南

Adapter 位于 TTIR 公共优化管线与硬件后端之间，负责把优化后的 TTIR 转换为 Linalg Track 的 AnchorIR。它解决的是“如何从 TTIR 得到统一的硬件无关 IR”，而后端插件解决的是“如何把 AnchorIR 编译并运行在具体硬件上”。后端开发请参阅 [custom_backend.md](custom_backend.md)。

```text
优化后的 TTIR
    |
    v
Adapter.convert()                 Pybind 或外部 opt 子进程
    |
    v
Linalg Track AnchorIR
    |
    +--> validate_pre_hook()      只允许基础白名单
    +--> on_anchor_ir_ready()     后端 Hook 可注入扩展 Op
    +--> validate_post_hook()     基础白名单 + 后端扩展白名单
    |
    v
硬件后端
```

> [!NOTE]
> TritonGPU Track 由 GPU 后端直接处理，不经过本指南所述的 TTIR-to-Linalg Adapter。

## 1. Adapter 接口层

接口定义位于 [`python/triton_anchor/adapters/base.py`](../python/triton_anchor/adapters/base.py)。继承关系如下：

```text
ITritonToLinalgAdapter
├── ILinalgPybindAdapter   # 进程内调用 pybind pass
└── ILinalgOptAdapter      # 子进程调用外部 MLIR opt 工具
```

### 1.1 `ITritonToLinalgAdapter`

`ITritonToLinalgAdapter` 是所有 TTIR-to-Linalg Adapter 的统一抽象接口。它定义行为契约，但不限定转换发生在当前进程还是外部进程。

子类必须实现：

| 方法 | 契约 |
|---|---|
| `name() -> str` | 返回全局唯一且稳定的注册名，例如 `"triton-shared"` |
| `convert(ttir_module, metadata, context=None) -> Any` | 把优化后的 TTIR 转换为符合 Linalg Track AnchorIR 的结果；失败时抛出 `AdapterConversionError` |

可按需覆盖：

| 方法 | 默认行为 |
|---|---|
| `validate_output(linalg_ir) -> bool` | 使用默认的 Linalg Track `AnchorIRValidator` 做单阶段布尔校验 |
| `get_required_passes() -> List[str]` | 返回空列表；该信息用于文档和诊断，不会自动执行 pass |
| `get_output_dialects() -> List[str]` | 返回常见 Linalg 方言列表；该信息用于声明和诊断，不会修改 AnchorIR 白名单 |

`convert()` 的输入输出约定取决于具体运行方式：

- 进程内 Adapter 通常接收 `ir.Module`，原地修改并返回同一个对象。
- 子进程 Adapter 通常把输入转成 MLIR 文本，并返回转换后的 MLIR 字符串。
- `metadata` 是编译元数据字典，Adapter 可以补充 kernel 名等字段。
- `context` 是可选的 MLIR Context；不需要时可以忽略。

### 1.2 `ILinalgPybindAdapter`

`ILinalgPybindAdapter` 是 `ITritonToLinalgAdapter` 的进程内特化。它没有新增抽象方法，主要表达一项 ABI 约束：调用的 pass 必须与宿主 Triton 链接到同一套兼容的 LLVM/MLIR 和 `libtriton`。

因此，“`ITritonToLinalgAdapter` 还是 `ILinalgPybindAdapter`”并不是并列选型：

- 所有 Adapter 都满足 `ITritonToLinalgAdapter`。
- 只有进程内调用 pybind pass 的实现才继承 `ILinalgPybindAdapter`。
- 调用外部 `opt` 工具的实现应继承对应的 `ILinalgOptAdapter`。

仓库内的 [`TritonLinalgAdapter`](../python/triton_anchor/adapters/triton_linalg_adapter.py) 是 pybind 示例；[`TritonSharedAdapter`](../python/triton_anchor/adapters/triton_shared_adapter.py) 是 subprocess/opt 示例。

## 2. Pybind Adapter 与 Subprocess Adapter 的选型

| 维度 | Pybind Adapter | Subprocess Adapter |
|---|---|---|
| 基类 | `ILinalgPybindAdapter` | `ILinalgOptAdapter` |
| 调用方式 | 当前 Python 进程内直接构建并运行 PassManager | 通过 `subprocess` 调用外部 `*-opt` |
| IR 传递 | `ir.Module` 对象，通常原地修改 | MLIR 文本或临时文件 |
| 启动开销 | 低 | 有进程启动和文本序列化开销 |
| ABI 要求 | 必须与宿主 `libtriton`/LLVM/MLIR ABI 一致 | 工具进程隔离，可使用自己的 LLVM/MLIR 构建 |
| 故障隔离 | pass 崩溃可能终止宿主编译进程 | 外部工具崩溃通常可转换为受控异常 |
| 部署 | pass 必须在构建期链接并暴露 pybind API | 需要安装并定位外部可执行文件 |
| 升级 | 与 triton-anchor 构建版本绑定较紧 | 工具可独立发布，但需维护文本 IR 兼容性 |
| 调试 | 可直接使用 Python/C++ 调试链路 | 容易复现和记录完整命令行、stdin/stdout/stderr |

按以下顺序决策：

1. 如果 pass 已经编译进当前 `libtriton`，并且能保证 LLVM/MLIR ABI 完全一致，优先使用 Pybind Adapter。
2. 如果转换器来自独立项目、使用不同 LLVM/MLIR 版本，或者只能提供 `opt` 工具，使用 Subprocess Adapter。
3. 如果编译吞吐和单次延迟是主要约束，且 ABI 可控，使用 Pybind Adapter。
4. 如果隔离性、独立升级和失败恢复更重要，使用 Subprocess Adapter。

> [!CAUTION]
> 不要为了省去子进程开销而在宿主进程中加载一个使用不同 MLIR 构建的 pybind 动态库。C++ 符号、RTTI 和 MLIR 对象布局冲突可能表现为随机崩溃，而不是可捕获的 Python 异常。

## 3. 实现一个最小 Adapter

下面以 `TritonSharedAdapter` 的结构为参考，实现一个外部 `my-linalg-opt` Adapter。仓库中的 `TritonSharedAdapter` 当前仍标记为 stub，因此应参考它的进程隔离、工具发现和临时文件结构，而不是把它视为已经完成的端到端集成。

### 3.1 创建包结构

建议把第三方 Adapter 做成独立 Python 包：

```text
triton-my-adapter/
├── pyproject.toml
├── src/
│   └── triton_my_adapter/
│       ├── __init__.py
│       └── adapter.py
└── tests/
    └── test_adapter.py
```

### 3.2 实现接口

```python
# src/triton_my_adapter/adapter.py
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, List, Optional

from triton_anchor.adapters import AdapterConversionError, ILinalgOptAdapter


class MyLinalgAdapter(ILinalgOptAdapter):
    def __init__(self, opt_path: Optional[str] = None):
        self._opt_path = opt_path

    def name(self) -> str:
        return "my-linalg"

    def _find_opt_tool(self) -> str:
        if self._opt_path:
            return self._opt_path
        env_path = os.environ.get("MY_LINALG_OPT_PATH")
        if env_path and os.path.isfile(env_path):
            return env_path
        return shutil.which("my-linalg-opt") or ""

    def convert(
        self,
        ttir_module: Any,
        metadata: dict,
        context: Any = None,
    ) -> str:
        opt_path = self._find_opt_tool()
        if not opt_path:
            raise AdapterConversionError(
                self.name(),
                kernel_name=metadata.get("name", ""),
                detail=(
                    "my-linalg-opt not found; set MY_LINALG_OPT_PATH "
                    "or add it to PATH"
                ),
            )

        ttir_text = (
            ttir_module if isinstance(ttir_module, str) else str(ttir_module)
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "input.mlir"
            dst = Path(tmpdir) / "output.mlir"
            src.write_text(ttir_text, encoding="utf-8")

            cmd = [
                opt_path,
                str(src),
                "--convert-triton-to-linalg",  # 替换为工具的真实 pipeline flag
                "-o",
                str(dst),
            ]
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise AdapterConversionError(
                    self.name(),
                    kernel_name=metadata.get("name", ""),
                    detail=str(exc),
                ) from exc

            if result.returncode != 0:
                raise AdapterConversionError(
                    self.name(),
                    kernel_name=metadata.get("name", ""),
                    detail=(
                        f"my-linalg-opt exited with {result.returncode}: "
                        f"{result.stderr.strip()}"
                    ),
                )
            if not dst.is_file():
                raise AdapterConversionError(
                    self.name(),
                    kernel_name=metadata.get("name", ""),
                    detail="my-linalg-opt did not produce output.mlir",
                )

            return dst.read_text(encoding="utf-8")

    def get_required_passes(self) -> List[str]:
        return ["convert-triton-to-linalg"]

    def get_output_dialects(self) -> List[str]:
        return [
            "linalg",
            "tensor",
            "memref",
            "arith",
            "math",
            "scf",
            "func",
        ]
```

这个最小实现需要满足以下约束：

- `name()` 必须稳定且唯一；`AdapterRegistry` 实际使用这个返回值作为 key。
- 入口点自动发现会无参数执行 `MyLinalgAdapter()`，因此构造函数不能要求必填参数。路径等配置应提供默认值、环境变量或 `PATH` 查找。
- 调用外部工具时传递参数列表，不使用 `shell=True`；为调用设置超时并把 stderr 放进 `AdapterConversionError`。
- Adapter 必须输出最终 AnchorIR，不能把 `tt`、`tts`、`tptr` 等过渡方言留给后端继续清理。
- `get_output_dialects()` 应描述所有可能出现的输出方言，但声明某个方言并不代表 AnchorIR 会允许它。

### 3.3 改写为 Pybind Adapter

如果转换 pass 已链接到宿主 `libtriton`，保留相同的 `name()`、错误类型和输出约束，把基类及 `convert()` 改为进程内形式：

```python
from typing import Any

from triton_anchor.adapters import AdapterConversionError, ILinalgPybindAdapter


class MyLinalgAdapter(ILinalgPybindAdapter):
    def name(self) -> str:
        return "my-linalg"

    def convert(
        self,
        ttir_module: Any,
        metadata: dict,
        context: Any = None,
    ) -> Any:
        try:
            from triton._C.libtriton import ir
            from triton._C.libtriton.anchor import anchor_passes

            pm = ir.pass_manager(ttir_module.context)
            anchor_passes.my_pipeline.add_convert_triton_to_linalg(pm)
            pm.run(ttir_module)
            return ttir_module
        except Exception as exc:
            raise AdapterConversionError(
                self.name(),
                kernel_name=metadata.get("name", ""),
                detail=str(exc),
            ) from exc
```

这里的 `anchor_passes.my_pipeline` 是占位名称。实际实现必须先在 C++ 构建和 pybind 层注册对应 pass，并保证它与当前 `ttir_module` 来自同一个 MLIR Context/ABI。

### 3.4 最小测试范围

至少覆盖以下行为：

1. `name()` 返回预期且不与已有 Adapter 重名。
2. 工具不存在、超时和非零退出码都会抛出 `AdapterConversionError`。
3. 一个最小 TTIR 输入能够转换，返回类型符合所选 Adapter 形式。
4. 输出能够通过 `AnchorIRValidator(track=AnchorIRTrack.LINALG).validate_pre_hook()`。
5. 输出中没有残留 `tt`、`tts`、`tptr` 或目标硬件专用方言。

## 4. 在 AdapterRegistry 中注册

注册表实现位于 [`python/triton_anchor/adapters/registry.py`](../python/triton_anchor/adapters/registry.py)，支持显式注册和 Python entry point 自动发现。

### 4.1 显式注册

显式注册适合测试、应用内定制，或者需要向构造函数传递配置的场景：

```python
from triton_anchor.adapters import AdapterRegistry
from triton_my_adapter.adapter import MyLinalgAdapter

# 先完成 entry point 发现，再用带配置的实例显式覆盖同名 Adapter。
AdapterRegistry.discover()
AdapterRegistry.register(MyLinalgAdapter(opt_path="/opt/my/bin/my-linalg-opt"))

adapter = AdapterRegistry.get("my-linalg")
assert adapter is not None
```

`register()` 接收的是 Adapter 实例，不是类。重复注册同名 Adapter 会记录 warning，并用新实例覆盖旧实例。若同一个 Adapter 也声明了 entry point，应像上例一样先调用 `discover()`；否则后续 `get()` 触发的首次发现可能用无参数实例覆盖此前的定制实例。

### 4.2 通过 entry point 自动发现

可分发的 out-of-tree Adapter 应在 `pyproject.toml` 中注册到 `triton.adapters` 分组：

```toml
[project.entry-points."triton.adapters"]
my-linalg = "triton_my_adapter.adapter:MyLinalgAdapter"
```

安装包后，`AdapterRegistry.discover()` 会加载入口点、无参数实例化类并调用 `register()`：

```python
from triton_anchor.adapters import AdapterRegistry

AdapterRegistry.discover()
print(AdapterRegistry.list_adapters())
```

入口点左侧的 `my-linalg` 用于包元数据和发现日志；注册表的实际 key 仍由 `adapter.name()` 决定。建议两者保持一致，避免排查时产生歧义。

`discover()` 在一个进程内只执行一次。测试中修改入口点或隔离全局状态后，可调用 `AdapterRegistry.reset()`；正常应用应在安装新包后重启进程。

### 4.3 让硬件能力选择新 Adapter

`AdapterRegistry.get_adapter(hw)` 的选择顺序是：

1. 如果 `hw.preferred_adapter` 非空，按该名称精确选择。
2. 否则按 `hw.ptr_model` 使用内置映射。
3. 没有匹配项时，当前实现会 warning 并返回第一个已注册 Adapter；没有任何 Adapter 时抛出 `AdapterNotFoundError`。

内置映射为：

| `HWCapability.ptr_model` | Adapter 名称 |
|---|---|
| `structured` | `triton-shared` |
| `axis_info` | `triton-linalg` |
| `hybrid` | `hybrid` |

自定义 Adapter 不会因为注册了新的名称而自动进入这张映射。后端应通过 `preferred_adapter` 显式选择：

```python
from triton_anchor.adapters import AdapterRegistry
from triton_anchor.anchor_ir import AnchorIRTrack
from triton_anchor.hw_capability import (
    ComputeParadigm,
    HWCapability,
    TensorCapability,
)

hw = HWCapability(
    name="my-npu-v1",
    arch_family="tpu",
    compute_paradigm=ComputeParadigm.TENSOR_PROCESSOR,
    anchor_ir_track=AnchorIRTrack.LINALG,
    ptr_model="structured",
    preferred_adapter="my-linalg",
    tensor_cap=TensorCapability(num_cores=8),
)

adapter = AdapterRegistry.get_adapter(hw)
```

对于 `ptr_model="gpu"` / TritonGPU Track，调用方应直接绕过 Linalg Adapter。不要依赖注册表 fallback，因为它可能返回一个与 GPU Track 无关的已注册 Adapter。

## 5. AnchorIR 验证与 Adapter 输出

AnchorIR 是 Adapter 与后端之间的输出契约。它约束的是 `convert()` 的实际结果，而不是 Adapter 的类名、pass 名或 `get_output_dialects()` 声明。

### 5.1 输出表示与验证内容

无论 Adapter 返回 `ir.Module` 还是字符串，验证前都应得到 MLIR 文本：

```python
ir_text = output if isinstance(output, str) else str(output)
```

`AnchorIRValidator` 当前按文本扫描 `dialect.operation`，检查 Track 对应的允许和禁止方言。它不是 MLIR 语法、类型或 SSA 结构验证器。因此 Adapter 还必须依靠 MLIR parser、PassManager 和端到端测试保证 IR 本身结构合法。

`get_output_dialects()` 只用于声明、文档和诊断：

- 它不会触发验证。
- 它不会把方言加入 AnchorIR 白名单。
- 它与真实输出不一致时，Validator 仍以真实 IR 为准。

例如，当前 stub `TritonSharedAdapter.get_output_dialects()` 中仍列有 `tptr` 和 `triton_structured`。这不能豁免它们：`tptr` 在 Linalg Track 中明确禁止，未知的 `triton_structured` 也会在 pre-hook 校验中被拒绝。完整实现必须在输出前继续 lowering，或调整正式 AnchorIR 规范，而不能只修改该方法的返回值。

### 5.2 两阶段验证时机

推荐的集成顺序是：

```python
from triton_anchor.adapters import AdapterRegistry
from triton_anchor.anchor_ir import AnchorIRError, AnchorIRValidator


def raise_on_violations(phase, violations):
    if violations:
        details = "\n".join(str(item) for item in violations)
        raise AnchorIRError(f"AnchorIR {phase} validation failed:\n{details}")


adapter = AdapterRegistry.get_adapter(hw)
output = adapter.convert(optimized_ttir, metadata)

validator = AnchorIRValidator(track=hw.anchor_ir_track)
ir_text = output if isinstance(output, str) else str(output)
raise_on_violations("pre-hook", validator.validate_pre_hook(ir_text))

# 由后端实现；兼容原地修改和返回新结果两种方式。
hook_output = backend.on_anchor_ir_ready(output)
if hook_output is not None:
    output = hook_output
ir_text = output if isinstance(output, str) else str(output)
extensions = set(backend.get_allowed_dialects() or ())
raise_on_violations(
    "post-hook",
    validator.validate_post_hook(ir_text, ext_allowed=extensions),
)
```

两个阶段的责任边界是：

| 阶段 | 输入来源 | 允许的方言 |
|---|---|---|
| `validate_pre_hook()` | Adapter 的直接输出 | 当前 Track 的基础白名单；未知方言和禁止方言都会失败 |
| `validate_post_hook()` | 后端 Hook 处理后的 IR | 基础白名单 + `get_allowed_dialects()` 声明的扩展；禁止方言仍然失败 |

后端声明的扩展白名单只作用于 post-hook。Adapter 不能提前输出自定义扩展方言并期待它通过 pre-hook。

> [!IMPORTANT]
> `AdapterRegistry` 只负责发现和选择，不会调用 `convert()` 或执行 AnchorIR 验证；`convert()` 本身也不会自动调用 `validate_output()`。编译管线的集成方必须在 Adapter 输出后显式执行验证。

`adapter.validate_output(output)` 可用于简单的 Linalg Track 布尔检查，但它使用兼容旧接口的单阶段默认 Validator，不包含后端 Hook 和扩展白名单。正式编译链路应使用 `HWCapability.anchor_ir_track` 构造 Validator，并执行上述 pre-hook/post-hook 两阶段流程。

## 6. 完成检查清单

- Adapter 继承了与运行方式匹配的基类。
- `name()` 唯一、稳定，并与 entry point 名称一致。
- `convert()` 统一包装失败为 `AdapterConversionError`，错误中包含 kernel 和工具诊断信息。
- entry point 可无参数实例化；需要定制参数时支持显式注册。
- `get_required_passes()` 和 `get_output_dialects()` 与真实实现保持同步。
- Adapter 的直接输出通过 Linalg Track pre-hook 验证，不含过渡方言。
- 后端 Hook 之后的输出通过 post-hook 验证，扩展方言已显式声明。
- Pybind 实现验证了 LLVM/MLIR ABI 一致性；subprocess 实现设置了超时并保留 stderr。
- 单元测试覆盖成功、工具缺失、超时、转换失败、注册发现和 AnchorIR 拒绝路径。
