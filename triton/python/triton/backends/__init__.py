from triton._no_ttgir import NO_TTGIR as _NO_TTGIR

if not _NO_TTGIR:
    import os
    import importlib.util
    import inspect
else:
    import sys
    import importlib.metadata
from dataclasses import dataclass
from .driver import DriverBase
from .compiler import BaseBackend
if not _NO_TTGIR:


    def _load_module(name, path):
        spec = importlib.util.spec_from_file_location(name[:-3], path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


    def _find_concrete_subclasses(module, base_class):
        ret = []
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, base_class) and not inspect.isabstract(attr):
                ret.append(attr)
        if len(ret) == 0:
            raise RuntimeError(f"Found 0 concrete subclasses of {base_class} in {module}: {ret}")
        if len(ret) > 1:
            raise RuntimeError(f"Found >1 concrete subclasses of {base_class} in {module}: {ret}")
        return ret[0]


@dataclass(frozen=True)
class Backend:
    compiler: BaseBackend = None
    driver: DriverBase = None


if not _NO_TTGIR:
    def _discover_backends():
        backends = dict()
        root = os.path.dirname(__file__)
        # Discover in-tree backends
        for name in os.listdir(root):
            if not os.path.isdir(os.path.join(root, name)):
                continue
            if name.startswith('__'):
                continue
            compiler = _load_module(name, os.path.join(root, name, 'compiler.py'))
            driver = _load_module(name, os.path.join(root, name, 'driver.py'))
            backends[name] = Backend(_find_concrete_subclasses(compiler, BaseBackend),
                                     _find_concrete_subclasses(driver, DriverBase))

        # Discover out-of-tree backends via python entry_points (pull 模式)
        # 使用 ep.load() 直接加载插件类，从中读取 compiler/driver 信息，
        # 避免导入整个顶层包（会触发循环导入：plugin 尝试 import backends，但 backends 还未赋值）。
        import importlib.metadata
        try:
            eps = importlib.metadata.entry_points(group="triton.backends")
        except TypeError:
            # Python 3.8/3.9 compatibility
            eps = importlib.metadata.entry_points().get("triton.backends", [])

        for ep in eps:
            try:
                # 抑制 stdout/stderr，防止 CMake execute_process 捕获到杂乱输出
                import io
                import contextlib
                with contextlib.redirect_stdout(io.StringIO()), \
                     contextlib.redirect_stderr(io.StringIO()):
                    plugin_obj = ep.load()

                # 支持两种 entry_point 格式:
                #   1. 类: ep.load() 返回一个类，实例化后读取 compiler_cls / driver_cls
                #   2. 模块: ep.load() 返回模块，读取模块级 compiler_cls / driver_cls
                if isinstance(plugin_obj, type):
                    plugin = plugin_obj()
                else:
                    plugin = plugin_obj

                compiler_cls = getattr(plugin, 'compiler_cls', None)
                driver_cls = getattr(plugin, 'driver_cls', None)

                if compiler_cls and driver_cls:
                    backends[ep.name] = Backend(compiler=compiler_cls, driver=driver_cls)
            except Exception as e:
                import sys
                print(f"Warning: Failed to load out-of-tree backend '{ep.name}': {e}",
                      file=sys.stderr)

        return backends
else:
    def _discover_backends():
        """通过 entry_points 发现已安装的 out-of-tree 后端插件。

    后端插件（如 triton-sophgo-backend）在 pyproject.toml 中声明:
        [project.entry-points."triton.backends"]
        sophgo = "triton_sophgo"
    这样 `import triton` 时就能自动发现并注册该后端。
    """
        backends = dict()

        try:
            eps = importlib.metadata.entry_points(group="triton.backends")
        except TypeError:
            # Python 3.9 兼容: entry_points() 不支持 group 参数
            eps = importlib.metadata.entry_points().get("triton.backends", [])

        for ep in eps:
            try:
                plugin_obj = ep.load()

                # 支持两种 entry_point 指向格式:
                #   1. 类:   实例化后读取 compiler_cls / driver_cls
                #   2. 模块: 直接读取模块级 compiler_cls / driver_cls
                plugin = plugin_obj() if isinstance(plugin_obj, type) else plugin_obj

                compiler_cls = getattr(plugin, 'compiler_cls', None)
                driver_cls = getattr(plugin, 'driver_cls', None)

                if compiler_cls and driver_cls:
                    backends[ep.name] = Backend(compiler=compiler_cls, driver=driver_cls)
            except Exception as e:
                print(f"Warning: Failed to load out-of-tree backend '{ep.name}': {e}",
                      file=sys.stderr)

        return backends


backends = _discover_backends()
