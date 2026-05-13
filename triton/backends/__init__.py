import os
import importlib.util
import inspect
from dataclasses import dataclass
from .driver import DriverBase
from .compiler import BaseBackend


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

    # Discover out-of-tree backends via python entry_points
    import importlib.metadata
    try:
        eps = importlib.metadata.entry_points(group="triton.backends")
    except TypeError:
        # Python 3.8/3.9 compatibility
        eps = importlib.metadata.entry_points().get("triton.backends", [])

    for ep in eps:
        try:
            # 导入 out-of-tree 后端模块以触发其自注册逻辑。
            # 必须抑制 stdout/stderr，因为 CMake 的 execute_process 会捕获 stdout，
            # 而后端模块的 import 可能打印警告信息，导致 CMake 变量被污染。
            top_module_name = ep.module.split('.')[0]
            import importlib
            import io
            import contextlib
            with contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                importlib.import_module(top_module_name)
        except Exception as e:
            import sys
            print(f"Warning: Failed to load out-of-tree backend '{ep.name}': {e}",
                  file=sys.stderr)

    return backends


backends = _discover_backends()
