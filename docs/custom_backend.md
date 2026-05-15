# 自定义硬件后端指南

`triton-anchor` 采用了与硬件彻底解耦的“前端核心 + 后端插件”架构。如果你想为一款全新的硬件（如自主研发的 NPU、TPU 或带有扩展矩阵指令的 DSP）适配 Triton 生态，你需要按照本文档的指导，创建一个独立的 **out-of-tree** 硬件后端包。

## 1. 架构总览

在全新的架构下，后端无需去直接修改 Triton 或 `triton-anchor` 的源代码，而是以一个独立的 Python 包的形式存在。通过标准的 Python `entry_points` 机制，Triton 能够在运行时自动发现并加载你的后端。

一个典型的硬件后端目录结构如下（包含 C++ 扩展与 Python 封装）：

```text
triton-mydevice-backend/
├── pyproject.toml             # 依赖与 entry_points 注册
├── setup.py                   # 如果需要编译 C++ 扩展 (CMakeExtension)
├── CMakeLists.txt             # C++ 库的 CMake 构建逻辑
├── include/                   # C++ 头文件 (如自定义 Dialect、Passes)
├── lib/                       # C++ 源码实现 (如 Conversion/LinalgToMyDevice)
├── src/
│   └── triton_mydevice/
│       ├── __init__.py        # 导出 compiler_cls 和 driver_cls
│       ├── compiler.py        # 继承 BaseBackend，定义编译管线 (Passes)
│       ├── runtime.py         # 继承 DriverBase，定义运行时加载与执行逻辑
│       └── _C/                # 编译生成的 C++ Pybind 扩展模块存放处
└── tests/
    └── test_smoke.py          # 冒烟测试
```

## 2. 插件注册

你的后端需要通过 `pyproject.toml` (或 `setup.py`) 注册到 `triton.backends` 分组下。

```toml
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"

[project]
name = "triton-mydevice-backend"
version = "0.1.0"
dependencies = [
    "triton>=3.0.0",
]

[project.entry-points."triton.backends"]
my_device = "triton_my_device"
```

这告诉 Triton：当用户通过 `triton.compile(..., target=GPUTarget(backend="my_device", ...))` 请求编译时，请去加载 `triton_my_device` 包。

## 3. 导出核心类

在你的后端包的 `__init__.py` 中，**必须**在模块级别导出以下两个类：

```python
# triton_my_device/__init__.py
from .compiler import MyDeviceBackend
from .runtime import MyDeviceDriver

# 供 Triton 自动发现拉取 (Pull)
compiler_cls = MyDeviceBackend
driver_cls = MyDeviceDriver
```

## 4. 实现编译器后端 (`BaseBackend`)

编译器负责将 `triton-anchor` 生成的标准 IR（通常是 TTIR 或者是转换后的 Linalg IR）一步步 Lowering 成最终的硬件可执行文件（`.so`，`.elf` 或其他格式）。

你需要继承 `triton.backends.compiler.BaseBackend`：

```python
# triton_my_device/compiler.py
from triton.backends.compiler import BaseBackend, GPUTarget
from triton_anchor import HWCapability, ComputeParadigm, AnchorIRTrack

class MyDeviceBackend(BaseBackend):
    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        
        # 必须声明硬件能力，供 triton-anchor 在前端进行验证
        self.hw_capability = HWCapability(
            name="my_device",
            arch_family=target.arch,
            compute_paradigm=ComputeParadigm.TENSOR_PROCESSOR, # 或 gpGPU
            anchor_ir_track=AnchorIRTrack.LINALG,              # 使用 Linalg 轨道
            ptr_model="axis_info",
        )

    def parse_options(self, opts: dict) -> dict:
        """解析用户在 @triton.jit 时传入的 kwargs"""
        parsed = {"debug": opts.get("debug", False)}
        return parsed

    def add_stages(self, architecture: str, options: dict) -> dict:
        """
        定义编译的各个阶段 (Stages)。
        返回一个有序字典，Key 是该阶段产物的名字（如 "ttir", "linalg", "so"），
        Value 是处理该阶段的函数。
        """
        from triton.backends.compiler import CompileBase
        
        def _make_ttir(ast, metadata, opts):
            # 将 AST 转换为 TTIR
            ...
            
        def _make_linalg(ttir, metadata, opts):
            # 调用 triton-anchor 的 Adapter 将 TTIR 转为 Linalg
            ...

        def _make_binary(linalg_ir, metadata, opts) -> bytes:
            # 【重要】最后一步必须返回包含二进制内容的 bytes 对象。
            # Triton 的 CacheManager 会自动将这串 bytes 写出到磁盘缓存并生成真实的路径。
            ...

        stages = dict()
        stages["ttir"] = _make_ttir
        stages["linalg"] = _make_linalg
        stages["binary"] = _make_binary
        return stages
```

> **⚠️ 重要陷阱警告**：
> `add_stages` 中定义的最后一个函数（即生成硬件二进制的步骤），必须返回纯粹的 **`bytes` 对象**（包含二进制内容），绝对**不要返回文件路径的字符串**！Triton 的核心缓存管理机制会接管这串字节流，将其保存至 `~/.triton/cache` 中，并在后续流程中将实际的磁盘路径传递给 Launcher。

## 5. 实现运行时驱动 (`DriverBase`)

运行时驱动负责与底层硬件交互（查询设备数量、分配内存等），并负责加载编译器生成的二进制文件执行。

你需要继承 `triton.backends.driver.DriverBase`：

```python
# triton_my_device/runtime.py
from triton.backends.driver import DriverBase

class MyDeviceLauncher:
    """真实的 Kernel 启动器"""
    def __init__(self, src, metadata):
        # src 是一个元组，Triton CacheManager 会传递形如：
        # ("<生成二进制的文件绝对路径>", ) 或者 (bytes, path) 的格式
        
        # 在标准的 JIT 流程中，我们从 src 中提取写好的缓存文件路径
        self.so_path = src[1] if isinstance(src, tuple) else src
        self.metadata = metadata

    def __call__(self, *args, **kwargs):
        # 使用你底层的 C++ 运行时 API 加载并执行 self.so_path
        # 例如: my_runtime_lib.launch(self.so_path, self.metadata.name, *args)
        pass

class MyDeviceDriver(DriverBase):
    def __init__(self):
        super().__init__()
        # 指向你的启动器类
        self.launcher_cls = MyDeviceLauncher

    def get_current_device(self):
        # 返回当前选中的设备 ID
        return 0

    def get_current_stream(self, device):
        # 返回当前的计算流 ID (Stream)
        return 0

    def get_device_capability(self):
        # 对于非 Nvidia 显卡，Triton 要求返回 Tuple，通常返回类似 ("my_arch", 0) 即可
        return ("my_arch", 0)
    
    ... # 其他你需要实现的抽象方法
```

## 6. 测试与验证

完成上述开发后，你可以使用以下脚本冒烟验证你的后端是否能被 `triton-anchor` 正确识别：

```python
import triton
from triton.backends.compiler import GPUTarget

# 1. 验证目标对象实例化
target = GPUTarget(backend="my_device", arch="arch_v1", warp_size=32)

# 2. 验证 Backend 注册发现
backend_cls = triton.compiler.compiler.make_backend(target)
print("发现 Backend:", backend_cls)

# 3. 验证硬件能力校验
assert hasattr(backend_cls, "hw_capability"), "必须声明 hw_capability"
print("使用的 AnchorIR 轨道:", backend_cls.hw_capability.anchor_ir_track)
```

**一切就绪后，你就可以像往常一样，通过 `@triton.jit` 并指定 `target=target` 或者配置对应的环境变量（如 `TRITON_PRINT_AUTOTUNING` 等），让标准的 Triton 前端通过 `triton-anchor` 平滑过度到你自定义的芯片后端上执行了！**
