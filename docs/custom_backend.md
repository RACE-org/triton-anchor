# 自定义硬件后端指南

`triton-anchor` 采用"前端核心 + 后端插件"架构。后端以独立的 Python 包形式存在，通过 `entry_points` 机制被 Triton 自动发现和加载，无需修改 Triton 或 triton-anchor 源码。

## 1. 架构总览

```
@triton.jit 装饰的函数
    │
    ▼  首次调用时触发编译
compiler.py — MyDeviceBackend(BaseBackend)
    │  add_stages() 定义编译管线
    │  TTIR → Linalg → 硬件 IR → .so/.elf
    │
    ▼  编译完成后加载执行
driver.py — MyDeviceDriver(DriverBase)
    │  launcher_cls → MyDeviceLauncher
    │  设备管理接口 (target, stream, device)
    │
    ▼  每次 kernel 调用时
MyDeviceLauncher.__call__()
    └── 加载 .so 并调用硬件运行时 API 执行
```

### 项目结构

```text
triton-mydevice-backend/
├── pyproject.toml             # 依赖与 entry_points 注册
├── CMakeLists.txt             # C++ 构建（如有自定义 Dialect/Pass）
├── csrc/                      # C++ 源码（MLIR Dialect、Conversion Pass、Pybind11）
├── triton_mydevice/           # Python 包（后端核心只需三个文件）
│   ├── __init__.py            # 导出 compiler_cls / driver_cls + 后端注册
│   ├── compiler.py            # 继承 BaseBackend，定义编译管线
│   └── driver.py              # 继承 DriverBase，定义 Launcher + Driver
└── tests/
    └── test_smoke.py
```

## 2. 插件注册

通过 `pyproject.toml` 注册到 `triton.backends` 分组：

```toml
[project.entry-points."triton.backends"]
my_device = "triton_mydevice"
```

## 3. 导出核心类

```python
# triton_mydevice/__init__.py
from .compiler import MyDeviceBackend
from .driver import MyDeviceDriver

compiler_cls = MyDeviceBackend   # Triton 的 _discover_backends() 读取
driver_cls = MyDeviceDriver
```

如果需要 `import triton_mydevice` 即激活后端，可在 `__init__.py` 中主动注入：

```python
from triton.backends import backends, Backend
from triton.runtime.driver import driver as _driver_config

backends["my_device"] = Backend(compiler=MyDeviceBackend, driver=MyDeviceDriver)
_driver_config.set_active(MyDeviceDriver())
```

## 4. 实现编译器后端 (`compiler.py`)

继承 `BaseBackend`，核心是 `add_stages()` 定义编译管线：

```python
# triton_mydevice/compiler.py
from triton.backends.compiler import BaseBackend, GPUTarget

class MyDeviceBackend(BaseBackend):
    binary_ext = 'so'

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == 'my_device'

    def parse_options(self, opts):
        # 解析 @triton.jit 的 kwargs，返回 Options 对象
        ...

    def pack_metadata(self, metadata):
        # 将编译元数据打包为 tuple，传递给 Launcher
        ...

    def load_dialects(self, ctx):
        # 加载自定义 MLIR 方言（如有 C++ Dialect）
        ...

    def add_stages(self, stages, options):
        stages["ttir"]   = lambda src, metadata: _make_ttir(src, metadata, options)
        stages["so"]     = lambda src, metadata: _make_anchor_binary(
            src, metadata, options
        )
```

### 编译阶段

每个 stage 签名为 `(module, metadata) → module`，最后一个 stage 必须返回 `bytes`。

```python
def _make_ttir(mod, metadata, options):
    """Stage 1: 标准 TTIR 优化 (inliner, combine, cse, licm 等)。"""
    from triton._C.libtriton import ir, passes
    pm = ir.pass_manager(mod.context)
    passes.common.add_inliner(pm)
    passes.ttir.add_combine(pm)
    passes.common.add_canonicalizer(pm)
    # ... 其他标准 passes
    pm.run(mod)
    return mod


def _make_anchor_binary(ttir, metadata, options) -> bytes:
    """TTIR → AnchorIR 强验证 → 后端 Hook → 强验证 → 二进制。"""
    from triton_anchor.adapters.triton_linalg_adapter import TritonLinalgAdapter

    report = TritonLinalgAdapter().compile(
        ttir,
        metadata,
        hook=None,  # 或实现 on_anchor_ir_ready()/get_allowed_dialects() 的后端 Hook
        backend_lowering=lambda anchor_ir: _lower_anchor_ir(
            anchor_ir, metadata
        ),
        context=ttir.context,
    )
    return report.lowered_output


def _lower_anchor_ir(mod, metadata) -> bytes:
    """post-hook 验证通过后的 Linalg → 硬件二进制。

    1. Dump IR 到文件
    2. 调用硬件编译工具链（你的编译器）
    3. 读取产物 .so 为 bytes 返回
    """
    metadata.setdefault("shared", 0)
    function_name = metadata["name"]
    # ... dump mod 到 .mlir 文件
    # ... 调用硬件编译器
    # ... 读取 .so
    metadata["so_path"] = so_path          # Launcher 通过此字段获取路径
    with open(so_path, "rb") as f:
        return f.read()                    # 必须返回 bytes
```

> [!IMPORTANT]
> 生产后端不能直接调用 `adapter.convert()`、手工执行
> `add_triton_to_linalg()` 后立即进入 `_lower_anchor_ir()`，否则会绕过
> pre-hook/Hook/post-hook 强验证。必须调用 Adapter 的 `compile()`，或者对
> TritonGPU/其他 Track 使用
> `AnchorIRLifecycleOrchestrator.run_module_or_raise()`。
> 仓库内通用 `triton.compiler.compile()` 只执行后端登记的 stages，不会自动插入
> AnchorIR 生命周期；因此后端必须像上例一样把 fail-closed 入口放入自己的 stage。

T6.5 的结构化规则、生命周期、Golden 语义和一键验收命令见
[《AnchorIR 结构化强验证与 Golden 回归》](anchor_ir_validation.md)。

> [!CAUTION]
> 最后一个 stage **必须返回 `bytes`**，不要返回文件路径字符串。Triton 的缓存系统会接管这串字节流并写入 `~/.triton/cache`。

后端的 `hash()` 还必须包含 `ANCHOR_IR_SPEC_VERSION`、规范化版本和 Hook
实现版本。否则规则或 Hook 改变后可能复用旧缓存，跳过本次编译应产生的验证和
Golden Stage。端到端 Golden/corpus 验收应由 T10.3 在
`TRITON_ALWAYS_COMPILE=1` 下采集真实阶段输出。

`triton-anchor-validate --allow-extension` 只扩展 post-hook policy，不会动态加载
厂商 Dialect 或自定义 parser。厂商自定义语法应在后端已注册 Dialect 的显式
MLIR Context 中解析，并把真实 `ModuleOp` 交给 Python lifecycle；不能假设 CLI
仅凭 namespace 名称就能理解 out-of-tree custom syntax。

`get_allowed_dialects()` 只能声明当前 Track 的非核心 namespace。声明当前
Track 的核心 Forbidden namespace（例如两条 Track 的 `smt`、Linalg Track 的
`tt`）会在 Hook 执行前以 `ValueError` 拒绝，Hook 和 lowering 均不会执行。
生命周期报告的 `post_hook_snapshot` 是 post-hook 验证后、lowering 前的
不可变规范化文本与哈希，T10.3 应以它采集 `boundary.post_hook`；ModuleOp lowering
得到的是 clone，因此常规原地 lowering 不会改写 `report.output` 这个已验证边界。

## 5. 实现运行时驱动 (`driver.py`)

driver.py 包含四个组件：

| 组件 | 职责 |
|------|------|
| `MyDeviceUtils` | 设备属性查询、`load_binary()` 透传编译产物 |
| `MyDeviceLauncher` | 解析 JIT 参数，调用硬件 API 执行 kernel |
| `MyDeviceInterface` | 模拟 CUDA 的 Event/Stream 语义（autotuner 需要） |
| `MyDeviceDriver` | 设备管理入口，继承 `DriverBase` |

```python
# triton_mydevice/driver.py
from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase


class MyDeviceUtils:
    """设备属性 + load_binary (透传编译产物给 Launcher)。"""
    @staticmethod
    def load_binary(name, kernel_obj, shared, device):
        return (None, kernel_obj, None, None)


class MyDeviceLauncher:
    """内核启动器。

    __init__(src, metadata): 从 src 提取 constants/signature，
                             计算非常量参数的位置映射。
    __call__(*args):         前 9 个是 triton 框架参数 (grid, stream, hooks...),
                             第 10 个起是用户 kernel 参数。
    """
    def __init__(self, src, metadata):
        # 解析 src.constants, src.signature → 参数位置映射
        self.kernel_name = getattr(metadata, "name", "kernel")
        self.so_path = getattr(metadata, "so_path", "")

    def __call__(self, *args, **kwargs):
        gridX, gridY, gridZ = args[0], args[1], args[2]
        kernel_user_args = args[9:]
        # 调用你的硬件运行时 API
        # my_runtime.launch(self.so_path, self.kernel_name, kernel_user_args, grid=...)


class MyDeviceInterface:
    """模拟 CUDA device/stream/event（triton benchmarking 需要）。"""
    # 实现 current_device(), synchronize(), Event(), Stream() 等


class MyDeviceDriver(DriverBase):
    def __init__(self):
        super().__init__()
        self.utils = MyDeviceUtils()
        self.launcher_cls = MyDeviceLauncher

    @staticmethod
    def is_active():
        return True

    def get_current_target(self):
        return GPUTarget("my_device", 0, 0)

    def get_device_interface(self):
        return MyDeviceInterface

    # 还需实现: get_current_device(), get_current_stream(),
    #           get_device_capability(), set_current_device() 等
```

## 6. 测试与验证

```python
# tests/test_smoke.py
import triton_mydevice
from triton.backends.compiler import GPUTarget
from triton_mydevice.compiler import MyDeviceBackend
from triton_mydevice.driver import MyDeviceDriver

target = GPUTarget("my_device", 0, 32)
assert MyDeviceBackend.supports_target(target)

driver = MyDeviceDriver()
assert driver.is_active()
print(f"current_target: {driver.get_current_target()}")
```

完成 backend stage 接入、cache key、T10.3 真实语料采集和设备运行门禁后，才可以
把 `@triton.jit` 路径视为完整端到端接入。
