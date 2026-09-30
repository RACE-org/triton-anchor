# 自定义 Adapter 开发指南

本文面向需要把一条自定义 **TTIR -> Linalg** 转换管线接入
`triton-anchor` 的开发者。读完后，你应该能够：

1. 判断应该使用外部 `opt` 进程还是进程内 Pybind pass。
2. 实现一个可安装、可发现的 Adapter 包。
3. 让硬件能力通过 `preferred_adapter` 选择它。
4. 验证 Adapter 输出满足 AnchorIR 契约，并定位常见接入失败。

如果你的目标是实现设备编译、Kernel Launcher 或运行时 Driver，请改看
[自定义硬件后端指南](./custom_backend.md)。Adapter 只负责
**把优化后的 TTIR 转换为 Linalg Track AnchorIR**，不负责生成硬件二进制。

> [!NOTE]
> TritonGPU Track 不经过本文所述的 TTIR-to-Linalg Adapter。

## 1. 先理解接入边界

Adapter 位于公共 TTIR 优化管线和硬件后端之间：

```text
Triton kernel
    |
    v
公共 TTIR 优化
    |
    |  输入：优化后的 TTIR
    v
Adapter.convert()
    |
    |  输出：Linalg Track AnchorIR
    v
validate_pre_hook()        Adapter 输出只能包含基础白名单方言
    |
    v
backend.on_anchor_ir_ready()   后端可在这里注入扩展 Op
    |
    v
validate_post_hook()       基础白名单 + 后端声明的扩展方言
    |
    v
硬件后端 lowering / codegen / runtime
```

这里最重要的契约是：

- 输入已经经过公共 TTIR 优化，Adapter 不应重新实现 Triton 前端。
- 输出必须是 Linalg Track AnchorIR，不能残留 `tt`、`tts`、`tptr`、
  `triton_gpu` 等禁止方言。
- Adapter 输出不能提前包含后端私有方言。私有方言应由后端 Hook 注入，并在
  post-hook 校验时通过 `get_allowed_dialects()` 声明。
- `AdapterRegistry` 只负责发现和选择 Adapter，不会调用 `convert()`，也不会
  自动执行 AnchorIR 校验。编译管线的集成代码必须显式完成这两步。

### 当前仓库的实际状态

开始开发前，先区分“已经实现”和“设计中的能力”：

| 能力 | 当前状态 |
|---|---|
| `ITritonToLinalgAdapter`、`ILinalgOptAdapter`、`ILinalgPybindAdapter` | 已实现接口 |
| `AdapterRegistry` 显式注册与 `triton.adapters` entry point 发现 | 已实现 |
| `TritonLinalgAdapter` | 已实现进程内 pass 调用，并默认注册 |
| `TritonSharedAdapter` | 有外部工具调用代码，但文件仍标记为 stub，未作为默认 entry point 注册 |
| `HybridAdapter` | stub；当前直接委托给 `TritonLinalgAdapter` |
| 自动调用 Adapter 和两阶段校验的统一编译 stage | 当前仓库中没有；需要后端/集成层接线 |

相关实现：

- [base.py](../python/triton_anchor/adapters/base.py)
- [registry.py](../python/triton_anchor/adapters/registry.py)
- [triton_linalg_adapter.py](../python/triton_anchor/adapters/triton_linalg_adapter.py)
- [triton_shared_adapter.py](../python/triton_anchor/adapters/triton_shared_adapter.py)
- [anchor_ir.py](../python/triton_anchor/anchor_ir.py)

## 2. 选择 Subprocess 还是 Pybind

`ITritonToLinalgAdapter` 是统一接口。实际实现应继承下面两个标记基类之一：

| 维度 | `ILinalgOptAdapter` | `ILinalgPybindAdapter` |
|---|---|---|
| 转换位置 | 外部 `*-opt` 子进程 | 当前 Python 进程 |
| 输入/输出 | MLIR 文本或临时文件 | `ir.Module`，通常原地修改 |
| LLVM/MLIR ABI | 与宿主隔离 | 必须与宿主 `libtriton` 完全兼容 |
| 故障影响 | 通常可转成 Python 异常 | C++ crash 可能终止宿主进程 |
| 部署 | 安装并定位外部可执行文件 | 构建时链接 pass 并提供 Pybind API |
| 调试 | 可保存命令、输入和 stderr，容易复现 | 可直接调试 Python/C++ 调用栈 |
| 性能 | 有进程启动和文本序列化开销 | 无子进程启动开销 |

按下面的规则选择：

- 转换器来自独立项目、LLVM/MLIR 版本不同，或已经提供 `*-opt` 工具：使用
  `ILinalgOptAdapter`。
- pass 已经链接进当前 Triton 的 `libtriton`，并确认使用同一套 LLVM/MLIR：
  使用 `ILinalgPybindAdapter`。
- 不确定 ABI 是否兼容时，先用 subprocess 方式完成接入。

> [!CAUTION]
> 不要在宿主进程中加载使用另一套 LLVM/MLIR 构建的 Pybind 动态库。符号、
> RTTI 或 MLIR 对象布局冲突可能直接导致进程崩溃，无法由
> `AdapterConversionError` 捕获。

## 3. 从一个可运行的 Subprocess Adapter 开始

下面创建一个独立包 `triton-acme-adapter`。示例中的转换器叫
`acme-linalg-opt`；你只需要把工具名、环境变量和 pass 参数替换成自己项目的
真实值。

### 3.1 项目结构

```text
triton-acme-adapter/
├── pyproject.toml
├── src/
│   └── triton_acme_adapter/
│       ├── __init__.py
│       └── adapter.py
└── tests/
    └── test_adapter.py
```

`pyproject.toml`：

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "triton-acme-adapter"
version = "0.1.0"
requires-python = ">=3.8"
dependencies = ["triton-anchor"]

[project.entry-points."triton.adapters"]
acme-linalg = "triton_acme_adapter.adapter:AcmeLinalgAdapter"

[tool.setuptools.packages.find]
where = ["src"]
```

`src/triton_acme_adapter/__init__.py`：

```python
from .adapter import AcmeLinalgAdapter

__all__ = ["AcmeLinalgAdapter"]
```

### 3.2 实现 Adapter

`src/triton_acme_adapter/adapter.py`：

```python
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, List, Optional

from triton_anchor.adapters import AdapterConversionError, ILinalgOptAdapter
from triton_anchor.anchor_ir import AnchorIRTrack, AnchorIRValidator


class AcmeLinalgAdapter(ILinalgOptAdapter):
    """Run acme-linalg-opt out of process and return AnchorIR text."""

    TOOL_NAME = "acme-linalg-opt"
    TOOL_ENV = "ACME_LINALG_OPT_PATH"

    def __init__(
        self,
        opt_path: Optional[str] = None,
        timeout_s: float = 60.0,
    ) -> None:
        # Both arguments have defaults because entry-point discovery calls
        # AcmeLinalgAdapter() without arguments.
        self._opt_path = opt_path
        self._timeout_s = timeout_s

    def name(self) -> str:
        # This is the actual AdapterRegistry key.
        return "acme-linalg"

    def _find_tool(self) -> str:
        if self._opt_path:
            return self._opt_path

        env_path = os.environ.get(self.TOOL_ENV)
        if env_path:
            return env_path

        return shutil.which(self.TOOL_NAME) or ""

    def _command(self, tool: str, src: Path, dst: Path) -> List[str]:
        # Replace this flag with the real pipeline accepted by your tool.
        return [
            tool,
            str(src),
            "--convert-triton-to-linalg",
            "-o",
            str(dst),
        ]

    def convert(
        self,
        ttir_module: Any,
        metadata: dict,
        context: Any = None,
    ) -> str:
        del context  # The external process owns its MLIR context.
        kernel_name = str(metadata.get("name", ""))
        tool = self._find_tool()
        if not tool:
            raise AdapterConversionError(
                self.name(),
                kernel_name=kernel_name,
                detail=(
                    f"{self.TOOL_NAME} not found; set {self.TOOL_ENV} "
                    "or add the tool to PATH"
                ),
            )

        ttir_text = (
            ttir_module if isinstance(ttir_module, str) else str(ttir_module)
        )

        try:
            with tempfile.TemporaryDirectory(prefix="acme-adapter-") as tmpdir:
                src = Path(tmpdir) / "input.mlir"
                dst = Path(tmpdir) / "output.mlir"
                src.write_text(ttir_text, encoding="utf-8")

                result = subprocess.run(
                    self._command(tool, src, dst),
                    capture_output=True,
                    text=True,
                    timeout=self._timeout_s,
                    check=False,
                )

                if result.returncode != 0:
                    diagnostic = (
                        result.stderr.strip()
                        or result.stdout.strip()
                        or "no diagnostic output"
                    )
                    raise AdapterConversionError(
                        self.name(),
                        kernel_name=kernel_name,
                        detail=(
                            f"{self.TOOL_NAME} exited with "
                            f"{result.returncode}: {diagnostic}"
                        ),
                    )

                if not dst.is_file():
                    raise AdapterConversionError(
                        self.name(),
                        kernel_name=kernel_name,
                        detail=f"{self.TOOL_NAME} did not create {dst.name}",
                    )

                output = dst.read_text(encoding="utf-8")
        except subprocess.TimeoutExpired as exc:
            raise AdapterConversionError(
                self.name(),
                kernel_name=kernel_name,
                detail=f"{self.TOOL_NAME} timed out after {self._timeout_s}s",
            ) from exc
        except OSError as exc:
            raise AdapterConversionError(
                self.name(),
                kernel_name=kernel_name,
                detail=f"failed to execute {tool}: {exc}",
            ) from exc

        self._validate_anchor_ir(output, kernel_name)
        return output

    def _validate_anchor_ir(self, output: str, kernel_name: str) -> None:
        validator = AnchorIRValidator(track=AnchorIRTrack.LINALG)
        violations = validator.validate_pre_hook(output)
        if not violations:
            return

        shown = "\n".join(str(item) for item in violations[:20])
        omitted = len(violations) - 20
        if omitted > 0:
            shown += f"\n  ... {omitted} more violation(s)"
        raise AdapterConversionError(
            self.name(),
            kernel_name=kernel_name,
            detail=f"output is not Linalg Track AnchorIR:\n{shown}",
        )

    def get_required_passes(self) -> List[str]:
        # Diagnostic metadata only. The framework does not run these names.
        return ["convert-triton-to-linalg"]

    def get_output_dialects(self) -> List[str]:
        # Diagnostic metadata only. This does not extend the AnchorIR whitelist.
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

这个实现有几个不能省略的细节：

- `name()` 返回的字符串必须稳定、唯一，最好和 entry point 左侧名称一致。
- 构造函数必须支持无参数调用，因为发现逻辑执行的是
  `adapter_cls()`。
- 子进程参数使用列表传递，不使用 `shell=True`。
- 超时、启动失败、非零退出码和缺失输出都统一包装为
  `AdapterConversionError`。
- 工具的 stderr 被保留在异常中，否则真实 pass 错误很难定位。
- 返回前执行 pre-hook 校验，确保错误停在 Adapter 边界，而不是拖到后端。

`context` 对 subprocess Adapter 通常没有作用。输入被序列化为文本，外部工具
会创建自己的 MLIR Context。

### 3.3 安装并确认自动发现

在 `triton-acme-adapter/` 目录运行：

```bash
python3 -m pip install -e .

python3 - <<'PY'
from triton_anchor.adapters import AdapterRegistry

AdapterRegistry.reset()
adapters = AdapterRegistry.list_adapters()
print(adapters)
assert adapters["acme-linalg"] == "AcmeLinalgAdapter"
PY
```

期望输出中至少包含：

```text
{'acme-linalg': 'AcmeLinalgAdapter', ...}
```

如果断言失败，先不要调试转换 pass，直接查看
[8. 常见问题](#8-常见问题)。发现失败和转换失败是两类问题。

### 3.4 让硬件能力选择这个 Adapter

注册成功不等于会被选中。内置 `ptr_model` 映射只认识：

| `ptr_model` | 默认 Adapter 名称 |
|---|---|
| `structured` | `triton-shared` |
| `axis_info` | `triton-linalg` |
| `hybrid` | `hybrid` |

自定义 Adapter 应通过 `preferred_adapter` 精确选择：

```python
from triton_anchor.anchor_ir import AnchorIRTrack
from triton_anchor.hw_capability import (
    ComputeParadigm,
    HWCapability,
    TensorCapability,
)

hw = HWCapability(
    name="acme-npu-v1",
    arch_family="tpu",
    compute_paradigm=ComputeParadigm.TENSOR_PROCESSOR,
    anchor_ir_track=AnchorIRTrack.LINALG,
    ptr_model="structured",
    preferred_adapter="acme-linalg",
    tensor_cap=TensorCapability(num_cores=8),
)

from triton_anchor.adapters import AdapterRegistry

adapter = AdapterRegistry.get_adapter(hw)
assert adapter.name() == "acme-linalg"
```

不要依赖注册表的 fallback 行为。没有匹配项时，当前实现可能返回第一个已注册的
Adapter，这个结果与安装顺序有关。生产后端应始终设置
`preferred_adapter`。

## 4. Adapter 接口契约

### 4.1 必须实现的方法

| 方法 | 输入 | 返回 | 失败方式 |
|---|---|---|---|
| `name()` | 无 | 全局唯一的稳定名称 | 不应失败 |
| `convert(ttir_module, metadata, context=None)` | 优化后的 TTIR、可变元数据、可选 Context | Pybind 返回 `ir.Module`；subprocess 返回 MLIR 字符串 | 抛出 `AdapterConversionError` |

`metadata` 是编译元数据字典，Adapter 可以使用 `setdefault()` 补充 kernel 名等
信息，但不应删除调用方已有字段。

### 4.2 可选方法

| 方法 | 默认值 | 当前真实作用 |
|---|---|---|
| `validate_output(linalg_ir)` | Linalg Track 单阶段布尔校验 | 需要调用方显式调用 |
| `get_required_passes()` | `[]` | 文档和诊断元数据，不会自动调度 pass |
| `get_output_dialects()` | 常见 Linalg 方言 | 文档和诊断元数据，不会修改白名单 |

特别注意：把 `my_dialect` 放进 `get_output_dialects()` **不会**让它通过
AnchorIR 校验。

### 4.3 注册表的行为

`AdapterRegistry` 支持两种注册方式：

```python
# 方式一：安装包通过 entry point 自动发现，适合发布。
from triton_anchor.adapters import AdapterRegistry

adapter = AdapterRegistry.get("acme-linalg")
```

```python
# 方式二：显式注册实例，适合测试或带运行时配置的应用。
from triton_anchor.adapters import AdapterRegistry
from triton_acme_adapter import AcmeLinalgAdapter

# 先发现，再覆盖，避免稍后的首次发现覆盖定制实例。
AdapterRegistry.discover()
AdapterRegistry.register(
    AcmeLinalgAdapter(opt_path="/opt/acme/bin/acme-linalg-opt")
)
```

注册表使用 `adapter.name()` 作为 key，不使用 entry point 的名称作为 key。重复
注册会记录 warning，并由新实例覆盖旧实例。`discover()` 在一个进程中只执行
一次；测试修改安装元数据后应调用 `AdapterRegistry.reset()`，应用安装新插件后
应重启进程。

## 5. AnchorIR 输出要求

### 5.1 Linalg Track 基础白名单

当前基础白名单定义在
[anchor_ir.py](../python/triton_anchor/anchor_ir.py)，包括：

```text
linalg, linalg_ext, tensor, memref, arith, math, math_ext,
scf, func, cf, affine, aux, index, bufferization, vector
```

明确禁止：

```text
tt, triton, tts, tptr, smt, triton_gpu, nvidia_gpu
```

未知方言也会失败。因此 Adapter 的直接输出必须只包含基础白名单方言；后端
扩展方言不能出现在 pre-hook 阶段。

### 5.2 两阶段校验

推荐的集成顺序如下。它属于编译管线/后端集成代码，不属于 Adapter 注册表：

```python
from triton_anchor.adapters import AdapterRegistry
from triton_anchor.anchor_ir import AnchorIRError, AnchorIRValidator


def raise_on_violations(phase, violations):
    if violations:
        details = "\n".join(str(item) for item in violations)
        raise AnchorIRError(
            f"AnchorIR {phase} validation failed:\n{details}"
        )


adapter = AdapterRegistry.get_adapter(hw)
output = adapter.convert(optimized_ttir, metadata)

validator = AnchorIRValidator(track=hw.anchor_ir_track)
ir_text = output if isinstance(output, str) else str(output)
raise_on_violations("pre-hook", validator.validate_pre_hook(ir_text))

# Backend hook may mutate output in place or return a replacement.
hook_result = backend.on_anchor_ir_ready(output)
if hook_result is not None:
    output = hook_result

ir_text = output if isinstance(output, str) else str(output)
extensions = set(backend.get_allowed_dialects() or ())
raise_on_violations(
    "post-hook",
    validator.validate_post_hook(ir_text, ext_allowed=extensions),
)
```

当前 Validator 是基于 MLIR 文本的 operation 扫描器，不是完整 MLIR parser 或
verifier。它主要检查 `dialect.operation`：

- 不能证明 SSA、类型和 region 结构正确。
- 不能替代外部 `opt` 的 parser/verifier。
- 类型文本中出现某个方言但没有对应 operation 时，可能不会被扫描到。

所以真实测试必须同时覆盖：外部工具成功解析输入、输出能被下一阶段 MLIR 工具
解析，以及 AnchorIR 方言校验通过。

## 6. 测试 Adapter

至少分三层测试：

| 层级 | 验证内容 | 是否需要真实工具 |
|---|---|---|
| 单元测试 | 命令拼装、超时、退出码、stderr、输出校验 | 否，可用临时假工具 |
| 注册测试 | wheel/editable install 后能被 entry point 发现 | 否 |
| 集成测试 | 真实 TTIR 经真实 pass 转换并被下一阶段消费 | 是 |

下面的单元测试会创建临时可执行文件，不依赖真实 `acme-linalg-opt`。

`tests/test_adapter.py`：

```python
import stat

import pytest

from triton_anchor.adapters import AdapterConversionError
from triton_acme_adapter import AcmeLinalgAdapter


def make_tool(tmp_path, body):
    tool = tmp_path / "fake-opt"
    tool.write_text(
        "#!/usr/bin/env python3\n" + body,
        encoding="utf-8",
    )
    tool.chmod(tool.stat().st_mode | stat.S_IXUSR)
    return str(tool)


def test_convert_success(tmp_path):
    tool = make_tool(
        tmp_path,
        """
import pathlib
import sys

dst = pathlib.Path(sys.argv[sys.argv.index("-o") + 1])
dst.write_text(
    "module { func.func @kernel() { return } }",
    encoding="utf-8",
)
""",
    )

    adapter = AcmeLinalgAdapter(opt_path=tool)
    output = adapter.convert("module {}", {"name": "kernel"})
    assert "func.func @kernel" in output


def test_nonzero_exit_keeps_stderr(tmp_path):
    tool = make_tool(
        tmp_path,
        """
import sys
sys.stderr.write("unsupported tt.dot")
raise SystemExit(7)
""",
    )

    adapter = AcmeLinalgAdapter(opt_path=tool)
    with pytest.raises(AdapterConversionError, match="unsupported tt.dot"):
        adapter.convert("module {}", {"name": "kernel"})


def test_timeout(tmp_path):
    tool = make_tool(
        tmp_path,
        """
import time
time.sleep(1)
""",
    )

    adapter = AcmeLinalgAdapter(opt_path=tool, timeout_s=0.01)
    with pytest.raises(AdapterConversionError, match="timed out"):
        adapter.convert("module {}", {"name": "kernel"})


def test_rejects_non_anchor_ir(tmp_path):
    tool = make_tool(
        tmp_path,
        """
import pathlib
import sys

dst = pathlib.Path(sys.argv[sys.argv.index("-o") + 1])
dst.write_text(
    '''module {
  tt.func @kernel() {
    tt.return
  }
}
''',
    encoding="utf-8",
)
""",
    )

    adapter = AcmeLinalgAdapter(opt_path=tool)
    with pytest.raises(AdapterConversionError, match="not Linalg Track"):
        adapter.convert("module {}", {"name": "kernel"})
```

运行：

```bash
python3 -m pip install -e '.[test]'
pytest -q tests/test_adapter.py
```

如果你使用上述命令，需要在 `pyproject.toml` 增加：

```toml
[project.optional-dependencies]
test = ["pytest>=7"]
```

真实工具的集成测试至少应使用一个 pointwise kernel 和一个包含
`tt.load`/`tt.store` 的 kernel，并检查：

1. 输出能被下一阶段的 MLIR parser 读取。
2. pre-hook 违规列表为空。
3. 输出不存在 `tt`、`tts`、`tptr` operation。
4. 不支持的输入会产生带 kernel 名和 pass 诊断的受控错误。

## 7. 改为 Pybind Adapter

只有在转换 pass 已经链接到宿主 `libtriton` 时才使用该方式。Python 类结构与
subprocess Adapter 相同，主要区别是 `convert()` 直接操作 `ir.Module`：

```python
from typing import Any, List

from triton_anchor.adapters import (
    AdapterConversionError,
    ILinalgPybindAdapter,
)
from triton_anchor.anchor_ir import AnchorIRTrack, AnchorIRValidator


class AcmeLinalgAdapter(ILinalgPybindAdapter):
    def name(self) -> str:
        return "acme-linalg"

    def convert(
        self,
        ttir_module: Any,
        metadata: dict,
        context: Any = None,
    ) -> Any:
        del context  # Use ttir_module.context to avoid context mixing.
        kernel_name = str(metadata.get("name", ""))

        try:
            from triton._C.libtriton import ir
            from triton._C.libtriton.anchor import anchor_passes

            pm = ir.pass_manager(ttir_module.context)

            # Replace acme_pipeline and this function with your real binding.
            anchor_passes.acme_pipeline.add_convert_triton_to_linalg(pm)
            pm.run(ttir_module)
        except Exception as exc:
            raise AdapterConversionError(
                self.name(),
                kernel_name=kernel_name,
                detail=str(exc),
            ) from exc

        violations = AnchorIRValidator(
            track=AnchorIRTrack.LINALG
        ).validate_pre_hook(str(ttir_module))
        if violations:
            details = "\n".join(str(item) for item in violations)
            raise AdapterConversionError(
                self.name(),
                kernel_name=kernel_name,
                detail=f"output is not Linalg Track AnchorIR:\n{details}",
            )

        return ttir_module

    def get_required_passes(self) -> List[str]:
        return ["convert-triton-to-linalg"]
```

要让这段代码实际工作，C++ 侧必须已经完成：

1. 定义并注册 conversion pass。
2. 把 pass 和所需 dialect 链接进当前 Triton 构建使用的 `libtriton`。
3. 在 `triton._C.libtriton.anchor.anchor_passes` 下暴露 Pybind 函数。
4. 确保 pass、`ttir_module` 和 `ttir_module.context` 来自同一套 LLVM/MLIR。

`anchor_passes.acme_pipeline` 是明确的占位符，不是当前仓库已有 API。仓库内可
运行的调用方式请参考
[TritonLinalgAdapter._add_passes()](../python/triton_anchor/adapters/triton_linalg_adapter.py)。

## 8. 常见问题

### `AdapterRegistry.list_adapters()` 中没有我的 Adapter

依次检查：

```bash
# 1. 当前 Python 是否就是安装插件的解释器
python3 -m pip show triton-acme-adapter

# 2. entry point 是否写入安装元数据
python3 - <<'PY'
from importlib.metadata import entry_points

try:
    adapters = entry_points(group="triton.adapters")
except TypeError:  # Python 3.8/3.9 importlib.metadata API
    adapters = entry_points().get("triton.adapters", [])

print(list(adapters))
PY

# 3. 模块和类能否直接导入、无参数实例化
python3 - <<'PY'
from triton_acme_adapter.adapter import AcmeLinalgAdapter
print(AcmeLinalgAdapter().name())
PY
```

常见原因包括 entry point group 误写、editable install 使用了另一个虚拟环境、
模块导入失败，以及构造函数要求必填参数。发现异常会被注册表捕获并记录 warning，
不会直接终止进程。

### 已注册，但没有被硬件选择

确认 `HWCapability.preferred_adapter` 与 `name()` 完全一致。entry point 左侧名称
相同并不能弥补 `name()` 拼写不一致。

### 显式注册的配置丢失了

如果先 `register()`、后首次触发 `discover()`，同名 entry point 实例可能覆盖
你的实例。先调用 `AdapterRegistry.discover()`，再显式注册带配置实例。

### 外部命令在终端能运行，在 Adapter 中找不到

确认运行编译的 Python 进程拥有相同 `PATH`。更稳定的方式是通过专用环境变量
传递绝对路径，例如：

```bash
export ACME_LINALG_OPT_PATH=/opt/acme/bin/acme-linalg-opt
```

不要依赖 shell alias；`subprocess.run()` 不会加载交互式 shell 配置。

### 输出报 `Unknown dialect`

如果该方言由 Adapter 产生，它必须在 pre-hook 基础白名单内，否则需要继续
lowering。不要通过 `get_output_dialects()` 试图放行它。

如果该方言属于硬件后端，应在 `on_anchor_ir_ready()` 阶段注入，并由后端的
`get_allowed_dialects()` 声明，使其只在 post-hook 阶段放行。

### 输出仍有 `tt`、`tts` 或 `tptr`

转换管线没有完成。检查 pass 顺序和 dialect conversion target，不要把这些
过渡 operation 交给后端清理。MLIR 的 conversion target 应把不允许残留的
dialect/operation 标记为 illegal，并让转换失败发生在 pass 内部。

### Pybind 方式随机崩溃

首先核对 LLVM commit、MLIR 构建参数、C++ ABI、RTTI/exception 设置以及
`libtriton` 来源。如果无法证明 ABI 一致，改为外部 `opt` 子进程隔离。

### 如何保留失败现场

`TemporaryDirectory` 会在失败后删除输入和输出。开发阶段可以增加一个明确的
debug 选项，把 `input.mlir`、命令行和 stderr 复制到指定诊断目录；不要默认把
含用户模型信息的 IR 写入共享目录。

## 9. 发布前检查清单

- [ ] Adapter 继承了正确的 `ILinalgOptAdapter` 或 `ILinalgPybindAdapter`。
- [ ] `name()` 唯一、稳定，并与 entry point 名称一致。
- [ ] entry point 类可以无参数实例化。
- [ ] `convert()` 的所有受控失败都包装为 `AdapterConversionError`。
- [ ] subprocess 设置超时、保留 stderr，且不使用 `shell=True`。
- [ ] Pybind pass 与宿主 LLVM/MLIR/`libtriton` ABI 一致。
- [ ] 直接输出通过 Linalg Track pre-hook 校验。
- [ ] 输出能被下一阶段 MLIR parser/verifier 消费。
- [ ] 输出不残留 `tt`、`tts`、`tptr` 或硬件私有方言。
- [ ] `get_required_passes()` 和 `get_output_dialects()` 与真实实现一致。
- [ ] 硬件能力显式设置 `preferred_adapter`。
- [ ] 单元测试覆盖成功、工具缺失、超时、非零退出码和非法输出。
- [ ] 安装后的 wheel/editable package 通过 entry point 发现测试。
- [ ] 至少一个真实 Triton kernel 完成端到端转换。
