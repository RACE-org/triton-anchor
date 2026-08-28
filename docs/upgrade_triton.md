# 升级上游 Triton 版本指南

本文档描述如何将 `triton-anchor/triton/` 中 vendor 的上游 Triton 源码升级到新版本。

## 前提知识

triton-anchor 将上游 `triton-lang/triton` 的源码以 vendor 方式放在 `triton/` 子目录中，
并做了**极少量的定制修改**。升级时只需要：

1. 替换 `triton/` 目录为新版上游源码
2. 裁剪掉不需要的部分
3. 重新应用定制修改（仅 2 个文件）

当前基线版本记录在 `triton/TRITON_VERSION` 中。

---

## 步骤零：升级前检查清单

升级必须在独立分支中进行。开始替换源码前，由升级负责人完成以下清单，并将结果写入升级 PR；任何标记为“阻塞”的项目未确认前，不得进入合并和发布阶段。

### 0.1 确定升级范围和回退点

- [ ] 明确发起厂商、当前/目标 Triton commit、目标版本分支（`triton_v<X.Y>`）和升级原因。
- [ ] 对照 [Triton / LLVM 版本兼容性矩阵](compatibility_matrix.md)，列出目标版本分支的**全部**使用厂商。同一分支的其他厂商也属于受影响范围。
- [ ] 记录升级前 `triton-anchor` commit、`triton/TRITON_VERSION`、`triton/cmake/llvm-hash.txt`、各受影响后端 commit 和已发布 wheel 版本，作为回退点。
- [ ] 保存升级前回归结果、已知失败列表和性能基线；没有可比较的基线时，先在旧版本上补跑一次。
- [ ] 确认升级窗口、负责人、各厂商验证人和硬件测试资源。

建议在 PR 描述中使用以下记录表：

| 项目 | 升级前 | 升级后目标 | 确认人/证据 |
|------|--------|------------|-------------|
| triton-anchor 分支与 commit | `<branch>@<commit>` | `<branch>@<commit>` | `<PR/CI 链接>` |
| Triton | `<version>@<commit>` | `<version>@<commit>` | `TRITON_VERSION` |
| LLVM/MLIR | `<version>@<commit>` | `<version>@<commit>` | `llvm-hash.txt` / 厂商确认 |
| 厂商后端 | `<repo>@<commit>` | `<repo>@<commit>` | `<厂商验证人>` |
| FlagGems | `<version>@<commit>` | `<version>@<commit>` | `<回归报告>` |
| PyTorch/Python | `<versions>` | `<versions>` | `<环境清单>` |
| 回退点 | `<anchor/backend commits + wheel>` | - | `<制品地址>` |

### 0.2 Triton / LLVM / 厂商后端兼容性确认（阻塞）

- [ ] 从目标 Triton commit 的 `cmake/llvm-hash.txt` 读取 LLVM commit，不得只根据 LLVM 主版本号推断兼容。
- [ ] 对每个受影响厂商确认其 LLVM/MLIR 是目标 commit、目标 commit 的可验证派生版本，或已有明确的适配补丁；把差异和补丁链接写入 PR。
- [ ] 使用厂商实际工具链完成 `triton-anchor` 全量构建和链接，确认 MLIR C++ API、Pass 注册、Dialect/Target 依赖及 ABI 均兼容。
- [ ] 检查 Triton 插件接口变化，至少覆盖 `BaseBackend`、`DriverBase`、`GPUTarget`、`triton.backends` entry point、JIT/cache/launcher 接口。
- [ ] 检查 TTIR、TritonGPU IR、Python DSL 和 Pybind 接口变化；重点审阅本仓库 `.github/scripts/upstream_watch.py` 标记为高/中影响的路径。
- [ ] 确认各受影响后端能构建、安装、被自动发现，并能完成至少一个 JIT 编译和硬件/仿真执行。
- [ ] 更新 `docs/compatibility_matrix.md` 的目标版本信息；矩阵与代码必须在同一升级 PR（或相互关联的阻塞 PR）中更新。

兼容性结论只能填写“兼容”“需同步补丁”或“不兼容”。“需同步补丁”必须关联厂商后端 PR；“不兼容”时停止升级，不能通过跳过测试放行。

### 0.3 FlagGems 兼容性确认（阻塞）

- [ ] 固定厂商当前支持的 FlagGems commit、PyTorch/Python 版本、测试配置、精度阈值和设备环境，避免只记录模糊的发布版本号。
- [ ] 检查目标 Triton 对 FlagGems 所用 DSL/API 的变化，至少包括 JIT 装饰器、`tl.*` 语义、dtype/type promotion、block pointer、reduce/scan、atomic 和 autotune/cache。
- [ ] 在升级前版本上保存厂商维护的核心算子集和全量支持算子集结果；已知失败、波动用例和硬件限制必须有 issue 或历史报告作为依据。
- [ ] 在升级候选版本上用**同一 FlagGems commit 和同一测试配置**做差分回归。不得同时升级 Triton 和 FlagGems 后直接把差异归因于其中一方。
- [ ] 若目标升级本身要求新版 FlagGems，先做四组合定位：旧 Triton/旧 FlagGems、旧 Triton/新 FlagGems、新 Triton/旧 FlagGems、新 Triton/新 FlagGems，并在 PR 中说明不兼容边界。
- [ ] 取得每个受影响厂商的 FlagGems 验证结论；缺少硬件时可继续开发，但不得标记升级完成或发布稳定版本。

通过标准是：原来通过的算子不得新增失败、崩溃或卡死；历史失败不得扩大；性能满足升级前约定的厂商阈值。任何新增回归都必须修复，或由厂商验证人和 `triton-anchor` 维护者共同批准带到期时间的例外。

---

## 步骤一：获取新版上游源码

```bash
# 克隆上游仓库（如果还没有的话）
git clone https://github.com/triton-lang/triton.git /tmp/triton-upstream
cd /tmp/triton-upstream

# 切到目标版本
git checkout <TARGET_COMMIT_HASH>

# 记录 commit 信息，后面更新 TRITON_VERSION 需要
git log -1 --format="%H %ci"
```

---

## 步骤二：替换 triton/ 目录

```bash
cd <triton-anchor 项目根目录>

# 备份当前的 triton/ 目录（可选）
mv triton triton.bak

# 从上游复制需要的目录
mkdir -p triton
cp -r /tmp/triton-upstream/include triton/
cp -r /tmp/triton-upstream/lib triton/
cp -r /tmp/triton-upstream/cmake triton/
mkdir -p triton/python
cp -r /tmp/triton-upstream/python/src triton/python/
cp -r /tmp/triton-upstream/python/triton triton/python/
mkdir -p triton/third_party
cp -r /tmp/triton-upstream/third_party/f2reduce triton/third_party/
```

> **注意**: 只复制上面列出的目录。不要复制 `third_party/nvidia/`、`third_party/amd/`、
> `third_party/proton/` 等 NVIDIA/AMD 专属内容。

---

## 步骤三：裁剪

### 3.1 删除不需要的目录和文件

```bash
# 删除 NVIDIA 专属算子
rm -rf triton/python/triton/ops/

# 删除上游的构建文件（triton-anchor 有自己的顶层 setup.py 和 pyproject.toml）
rm -f triton/python/setup.py
rm -f triton/python/pyproject.toml
rm -f triton/python/MANIFEST.in
rm -rf triton/python/examples/
rm -rf triton/python/test/
rm -rf triton/python/tutorials/

# 删除不需要的 cmake 文件
rm -f triton/cmake/nvidia-toolchain-version.txt
rm -f triton/cmake/pybind11-version.txt
```

### 3.2 从 `__init__.py` 中移除 `ops` 导出

编辑 `triton/python/triton/__init__.py`，从 `__all__` 列表中删除 `"ops"` 一行：

```diff
  __all__ = [
      ...
      "next_power_of_2",
-     "ops",
      "OutOfResources",
      ...
  ]
```

### 3.3 从 CMakeLists.txt 中移除 NVGPUIR 依赖

编辑 `triton/lib/Conversion/TritonGPUToLLVM/CMakeLists.txt`，
从 `LINK_LIBS PUBLIC` 列表末尾删除 `NVGPUIR`：

```diff
      TritonGPUTransforms
      TritonNvidiaGPUTransforms
-     NVGPUIR
  )
```

> **为什么?** `NVGPUIR` 是 NVIDIA 后端的 CMake target，裁掉 `third_party/nvidia/` 后该 target 不存在。
> `TritonNvidiaGPUTransforms` 虽然名字带 nvidia，但它定义在 triton 核心代码中（非 third_party），
> 所以需要保留。如果新版上游移除了该依赖或改了名字，请根据实际编译错误调整。

---

## 步骤四：应用定制修改

需要修改的文件**只有 2 个**。以下给出完整的目标代码。

### 4.1 修改 `python/src/main.cc`

对上游的 `main.cc` 做 3 处修改：

**修改 A: 删除 in-tree 后端宏定义块**

删除 `namespace py = pybind11;` 之后、函数声明之前的**整块宏代码**
（从 `#define FOR_EACH_1` 到 `FOR_EACH_P(DECLARE_BACKEND, TRITON_BACKENDS_TUPLE)` 的所有行）：

```diff
  namespace py = pybind11;

- #define FOR_EACH_1(MACRO, X) MACRO(X)
- #define FOR_EACH_2(MACRO, X, ...) MACRO(X) FOR_EACH_1(MACRO, __VA_ARGS__)
- ...（删除全部 FOR_EACH_* / CONCATENATE / REMOVE_PARENS / DECLARE_BACKEND / INIT_BACKEND 宏定义）...
- FOR_EACH_P(DECLARE_BACKEND, TRITON_BACKENDS_TUPLE)
-
  void init_triton_env_vars(pybind11::module &m);
```

**修改 B: 新增 `init_triton_anchor` 声明，删除宏展开的后端声明**

在函数声明区域，删除 `FOR_EACH_P(DECLARE_BACKEND, ...)` 并添加 `init_triton_anchor`：

```diff
  void init_triton_passes(pybind11::module &&m);
- FOR_EACH_P(DECLARE_BACKEND, TRITON_BACKENDS_TUPLE)
+ void init_triton_anchor(pybind11::module &&m);
```

> 如果上游此处有其他 `init_triton_xxx` 声明，保留它们即可。

**修改 C: 在 `PYBIND11_MODULE` 中添加 anchor 调用，删除后端宏调用**

```diff
  PYBIND11_MODULE(libtriton, m) {
    m.doc() = "Python bindings to the C++ Triton API";
    ...（保留上游已有的 init_triton_xxx 调用）...
    init_triton_llvm(m.def_submodule("llvm"));
+   init_triton_anchor(m.def_submodule("anchor"));
-   FOR_EACH_P(INIT_BACKEND, TRITON_BACKENDS_TUPLE)
  }
```

> **注意**: 保留上游已有的所有 `init_triton_xxx` 调用，只需新增 `init_triton_anchor`
> 并删除 `FOR_EACH_P(INIT_BACKEND, ...)` 一行。

---

### 4.2 修改 `python/triton/backends/__init__.py`

triton-anchor 不使用 in-tree 后端，将文件简化为只保留 entry_points 发现机制。

**修改 A: 删除不需要的 import 和辅助函数**

删除以下内容（triton-anchor 无 in-tree 后端，不需要文件系统扫描逻辑）：

```diff
- import os
- import importlib.util
- import inspect
+ import sys
+ import importlib.metadata
  from dataclasses import dataclass
  from .driver import DriverBase
  from .compiler import BaseBackend
-
-
- def _load_module(name, path):
-     ...
-
-
- def _find_concrete_subclasses(module, base_class):
-     ...
```

**修改 B: 替换 `_discover_backends()` 函数体**

删除 in-tree 的 `os.listdir` 发现循环，替换为 entry_points 发现：

```diff
  def _discover_backends():
      backends = dict()
-     root = os.path.dirname(__file__)
-     for name in os.listdir(root):
-         ...（删除整个 in-tree 发现循环）...
+
+     try:
+         eps = importlib.metadata.entry_points(group="triton.backends")
+     except TypeError:
+         # Python 3.9 兼容
+         eps = importlib.metadata.entry_points().get("triton.backends", [])
+
+     for ep in eps:
+         try:
+             plugin_obj = ep.load()
+             plugin = plugin_obj() if isinstance(plugin_obj, type) else plugin_obj
+
+             compiler_cls = getattr(plugin, 'compiler_cls', None)
+             driver_cls = getattr(plugin, 'driver_cls', None)
+
+             if compiler_cls and driver_cls:
+                 backends[ep.name] = Backend(compiler=compiler_cls, driver=driver_cls)
+         except Exception as e:
+             print(f"Warning: Failed to load out-of-tree backend '{ep.name}': {e}",
+                   file=sys.stderr)
+
      return backends
```

---

## 步骤五：更新版本记录

编辑 `triton/TRITON_VERSION`，更新 commit 和日期：

```
# Vendored from: https://github.com/openai/triton
# Commit: <新的 COMMIT_HASH>
# Date: <YYYY-MM-DD>
# 升级指南见 docs/upgrade_triton.md
```

---

## 步骤六：编译与回归测试 SOP

参考 [build.md](build.md) 准备环境。以下测试按顺序执行，前一层失败时先停止并定位，不要继续用后续失败覆盖根因。

### 6.1 triton-anchor 自身回归（所有升级必需）

| 测试 | 命令 | 通过标准 |
|------|------|----------|
| 纯 Python 单元测试 | `PYTHONPATH=python python3 -m pytest python/triton_anchor/tests/ -v --tb=short` | 全部通过 |
| wheel 构建 | `uv build --wheel --no-build-isolation` | 使用目标 LLVM 完成编译和链接 |
| 安装后冒烟 | `python3 tests/test_smoke.py` | 导入、C++ 绑定、Dialect、AnchorIR、TTIR pipeline 和 AST → TTIR 全部通过 |
| CI/等价检查 | 运行 `.github/workflows/ci_basic.yml` 定义的检查 | `lint` 和 Python 3.9/3.10/3.11/3.12 `unit-test` 全部通过 |

测试必须在干净虚拟环境中安装本次新构建的 wheel，不能复用旧的可编辑安装。日志中记录以下版本：

> 若目标 `triton_v<X.Y>` 分支尚未配置为自动触发 CI，升级负责人必须执行同等命令并附完整日志，不能把“未触发”视为通过。

```bash
python3 -c "import sys, triton, triton_anchor; print(sys.version); print(triton.__version__); print(triton.__file__); print(triton_anchor.__file__)"
llvm-config --version
```

### 6.2 厂商后端回归（每个受影响厂商必需）

先安装升级候选 `triton-anchor` wheel，再构建并安装对应厂商的后端 wheel。所有厂商统一从 `triton-anchor` 根目录运行 `tests/test_backend_smoke.py`：

```bash
# 环境中只安装了一个厂商后端时
python3 tests/test_backend_smoke.py

# 环境中安装了多个厂商后端时，必须指定 entry point 名称
python3 tests/test_backend_smoke.py --backend sophgo
python3 tests/test_backend_smoke.py --backend tsingmicro
python3 tests/test_backend_smoke.py --backend fantasy
python3 tests/test_backend_smoke.py --backend spine_triton
```

该测试统一验证以下四项：

1. 厂商后端 wheel 的 `triton.backends` entry point 能被发现，且不存在重复注册。
2. 后端已注册到 `triton.backends.backends`。
3. 注册项包含接口完整的 Compiler 和 Driver。
4. Driver 能成功初始化并返回非空 Target。

四项检查必须全部显示 `PASS`，脚本退出码必须为 `0`。如果因驱动、运行库或设备环境缺失而无法初始化 Driver，只能记为“待厂商验证”，不能记为通过。

`test_backend_smoke.py` 不编译或运行 kernel；升级后的实际算子编译、执行和精度由 6.3 的 FlagGems 回归覆盖。

### 6.3 FlagGems 回归（每个受影响厂商必需）

1. 先运行厂商维护的核心算子集，至少覆盖 pointwise/type promotion、reduction、matmul/BLAS、softmax/attention、normalization、tensor constructor、随机/atomic（硬件支持时）。
2. 核心算子集通过后，运行升级前清单中固定的全量支持算子集。
3. 使用相同 FlagGems commit、随机种子、精度阈值、设备数和测试配置，对比升级前后结果。
4. 输出算子级报告：通过/失败/崩溃/超时数量、新增回归列表、已知失败列表和性能差异。报告和原始日志作为升级 PR 的制品保存。

FlagGems 回归按差分判定：允许升级前已有且证据充分的已知失败继续存在；不允许新增失败、崩溃、超时或未获批准的性能退化。

### 6.4 发布前验收标准

升级仅在以下条件全部满足后才算完成：

- [ ] `triton-anchor` 自身回归全部通过，CI 绿色。
- [ ] 所有受影响厂商的后端构建成功，且 `tests/test_backend_smoke.py` 四项检查全部通过。
- [ ] 所有受影响厂商的 FlagGems 核心与全量差分回归通过。
- [ ] `TRITON_VERSION`、LLVM hash、兼容性矩阵、后端适配 PR 和测试报告可互相追溯。
- [ ] 回退制品可用，且“步骤八”的升级通知内容已准备完毕。

---

## 步骤七：升级失败时的快速回退

### 7.1 未合并或灰度阶段

1. 停止安装/发布候选 wheel，保留失败日志、环境清单和候选制品用于定位。
2. 切回步骤 0.1 记录的旧 `triton-anchor` 版本分支和 commit；不要覆盖或强推稳定分支。
3. 在干净虚拟环境中重新安装已归档的旧 `triton-anchor` wheel 和匹配的厂商后端 wheel。
4. 为回退验证使用新的 `TRITON_CACHE_DIR`，避免候选版本缓存污染旧版本结果。
5. 运行 `tests/test_smoke.py`、受影响后端 smoke/JIT，以及最小 FlagGems 核心算子集，确认服务恢复。

需要从源码重建旧版本时，可从记录的 commit 创建临时回退分支，不改动当前失败分支：

```bash
git switch -c rollback/<vendor>-<date> <LAST_KNOWN_GOOD_ANCHOR_COMMIT>
uv build --wheel --no-build-isolation
```

### 7.2 已合并或已发布

1. 立即通知受影响厂商暂停更新，并给出上一个可用的 anchor/backend commit 和 wheel 地址。
2. 为升级 commit 创建 `git revert` PR；若升级通过合并 commit 引入，先确认主线父提交后使用 `git revert -m 1 <MERGE_COMMIT>`。禁止通过改写公共分支历史回退。
3. 如果某厂商刚迁移到新的 `triton_v<X.Y>` 分支，将该厂商依赖重新固定到旧版本分支/commit；保留新分支供问题修复。
4. 重新发布带明确版本号的回退制品，并重复 7.1 第 5 步的最小验证。
5. 在兼容性矩阵和原升级 PR 中标记回退状态、原因、影响范围和后续修复 issue。

回退完成的判定是“旧版本最小回归恢复且厂商确认”，不是仅完成代码 revert。

---

## 步骤八：升级后通知流程

升级负责人在发布前验收通过后发送通知，收件人包括：发起厂商、同一 `triton_v<X.Y>` 分支的其他厂商、相关后端维护者和 FlagGems 对接人。通知应关联升级 PR/commit 和测试报告，并给出明确的后端更新截止时间；各厂商需在跟踪 issue 中回复“已更新并验证”或说明阻塞项。

标准模板：

```text
主题：【triton-anchor 升级通知】【<版本分支>】【<厂商/影响范围>】

升级已完成：
- triton-anchor：<branch>@<commit>，wheel：<version/artifact>
- Triton：<old version@commit> -> <new version@commit>
- LLVM/MLIR：<version@commit>
- 兼容 FlagGems：<version@commit>
- 生效时间：<date/timezone>

影响与不兼容项：
- <DSL/IR/Python/C++/运行时接口变化；无则写“无已知不兼容项”>
- <厂商后端需要同步的补丁或配置>

验证结果：
- triton-anchor：<CI/report link>
- 厂商后端：<build + test_backend_smoke.py 4/4 PASS result link>
- FlagGems：<核心/全量通过数、已知失败、新增回归=0、report link>

请各后端维护者在 <deadline> 前完成：
1. 将 triton-anchor 依赖固定到上述 branch/commit 或正式版本；
2. 合并对应适配补丁并构建新后端 wheel；
3. 运行 `tests/test_backend_smoke.py` + FlagGems 回归；
4. 在 <tracking issue> 回复验证结果。

回退信息：
- 上一稳定 triton-anchor：<branch>@<commit>，wheel：<artifact>
- 上一稳定后端：<repo>@<commit>，wheel：<artifact>
- 回退 SOP：docs/upgrade_triton.md#步骤七升级失败时的快速回退

负责人：<name/contact>
跟踪 issue：<url>
```

若升级后来发生回退，沿用同一收件范围和跟踪 issue，主题改为“升级回退通知”，明确停止使用的版本、恢复版本和验证状态。

---

## 步骤九：多版本场景——厂商升级 Triton 基线时的响应流程

`triton-anchor` 按 Triton 主/次版本维护 `triton_v<X.Y>` 分支。当前路由为 Sophgo、Fantasy 使用 `triton_v3.0`，Tsingmicro 使用 `triton_v3.3`，SpacemiT 使用 `triton_v3.6`；准确的 Triton/LLVM commit 以 [兼容性矩阵](compatibility_matrix.md) 为准。不得因为一家厂商升级就直接抬升其他厂商的基线，也不得把版本专用适配未经验证合入所有版本分支。

### 9.1 受理与分支决策

1. 厂商提交基线升级需求，提供当前/目标 Triton commit、LLVM commit、厂商 fork 差异、后端 commit、FlagGems commit、目标时间和验证资源。
2. `triton-anchor` 维护者评估上游变更报告，列出 DSL/TTIR/TritonGPU/LLVM/Python 插件接口影响和需要重放的本地定制。
3. 若目标 `triton_v<X.Y>` 已存在，厂商优先适配该分支；该分支的现有厂商全部加入受影响列表。
4. 若目标分支不存在，从该厂商当前稳定版本分支创建 `triton_v<X.Y>` 的升级工作分支，按本文完成源码替换和适配。旧版本分支继续维护，不在升级验收前迁移厂商绑定。
5. 同一主/次版本内的 patch/commit 升级在现有 `triton_v<X.Y>` 分支通过 PR 完成；所有使用该分支的厂商必须共同回归。

### 9.2 双线适配与验收

1. Anchor 侧先保证新分支的 TTIR pipeline、AnchorIR 双轨契约、Pybind 和 entry point 机制通过自身回归。
2. 厂商侧在独立后端分支适配 API/IR/LLVM 变化；Anchor PR 与后端 PR 相互链接，任何一侧不得声称单独完成端到端升级。
3. 对目标分支的所有现有厂商执行步骤六；若通用修复会反向同步到旧版本分支，则旧分支也必须跑对应回归。
4. 版本无关的修复可按需 cherry-pick 到其他版本分支；版本相关补丁留在目标分支，并在提交/PR 中注明适用版本。避免直接合并不同 `triton_v<X.Y>` 分支。
5. 全部验收通过后，更新兼容性矩阵中的厂商 → anchor 分支 → Triton/LLVM/backend/FlagGems 对应关系，再通知厂商修改依赖固定点。

### 9.3 迁移与旧分支生命周期

- 厂商完成新版本真机和 FlagGems 回归并确认后，才将其默认依赖从旧分支切到新分支。
- 至少保留旧分支的最后稳定 commit、wheel 和测试报告，作为快速回退点；何时停止维护由单独的弃用通知决定。
- 同一厂商在迁移期可并行维护新旧两个组合，但制品和报告必须包含 anchor 分支、Triton commit、LLVM commit 和 backend commit，禁止使用无法区分基线的“latest”制品。
- 新版本出现阻塞时按步骤七退回旧组合，不把未验证修复强行同步给其他厂商。

---

## 附录：定制修改速查表

升级时只需关注以下文件，其余所有文件均可直接从上游复制：

| 文件 | 修改类型 | 说明 |
|------|---------|------|
| `python/src/main.cc` | 替换 | 删除后端宏，新增 `init_triton_anchor` |
| `python/triton/backends/__init__.py` | 替换 | 删除 in-tree 发现，替换为 entry_points 发现 |
| `python/triton/__init__.py` | 删一行 | 从 `__all__` 中移除 `"ops"` |
| `lib/Conversion/TritonGPUToLLVM/CMakeLists.txt` | 删一行 | 移除 `NVGPUIR` 依赖 |
| `python/triton/ops/` | 删目录 | 删除 NVIDIA 专属算子 |
| `third_party/nvidia,amd,proton/` | 不复制 | 不需要的上游后端 |
