# ADR-0002：T6.3 / Triton 3.6 发布制品与版本专属门禁治理

- 总体状态：**Approved design — implementation binding pending；不是 Accepted**
- 决策状态：`APPROVED`
- 实现绑定：`UNBOUND`
- 日期：2026-08-24
- 适用对象：triton-anchor Core 0.2.0 的 Triton 3.6 发布候选
- 实现基线：`609c3c7d1f4e6032e8b1399531d1c998b970fdb8`
- upstream 基线：`9ceba3e222fd2c9af9cbaf6a440beb23911c197b`
- 分支：`fix/t6.3-v3.6-release-gates`
- 前置规范：`docs/adr/0001-t63-backend-plugin-protocol-1x-native-abi.md`
- 前置追踪：`docs/adr/0001-t63-common-governance-traceability.md`
- 验收基线报告：`/home/dingbl/race_workspace/reports/t6.3-triton-v3.6-acceptance.md`
  （18,172 bytes；SHA-256
  `1dc1a4e86a9fbc7127178f01115b97d6ef3bed3a711274aae5858f10055b398c`）
- 明确排除：T10.2 Conformance Kit

> [!IMPORTANT]
> D-01～D-12 与全文规则已获第一次维护者批准，允许进入实施阶段；实现绑定仍为
> `UNBOUND`。在 exact candidate、fresh wheel、全部新鲜证据和第二次维护者确认完成前，
> 本 ADR 不是 release Accepted 合同，`AUTH-SHA` 仍是 SPEC GAP。

本 ADR 使用两阶段状态，避免让批准文档循环绑定自身。第一次确认已经把 `决策状态` 改为
`APPROVED`，但 `实现绑定` 仍为 `UNBOUND`，总体状态不得写成 Accepted。只有最终生产
候选、全新 wheel 和新鲜证据完成，并由发布责任人第二次确认，
`决策状态=APPROVED` 且 `实现绑定=BOUND` 时，总体状态才可改为 **Accepted**。

## 1. 决策背景

ADR-0001 已冻结跨 Triton 版本的 Common 合同：INV-01～10、F1 完整兼容诊断、
F2 Python lifecycle、F3 Schema/parser structural domain，以及 Protocol/Schema 1.0
对 native/subprocess 的 explicit rejection。它同时明确把 exact wheel payload、Legacy、
版本专属 compiler/driver abstract surface 和真实硬件 JIT 排除在 Common 范围之外。

Triton 3.6 后续实现增加了 F4、F5 和 F6 可执行测试，但没有版本专属 Accepted ADR。
2026-08-24 的独立验收得到：

- 既有软件 pytest 530 个唯一 identity 全部 PASS，0 skip；
- Common 409 与 F5/F6 118 的合并 statement coverage 为 76%；
- 新 wheel 的 ZIP、RECORD、build-info、fresh install 和 10/10 软件 smoke 通过；
- `triton/bin/triton-shared-opt` 的 `DT_RUNPATH` 泄漏旧 evidence/toolchain 绝对路径；
- 当前分支未 pin Ruff 且没有 Ruff 配置；同一 Ruff 0.16.4 下，T6.3 相对 upstream
  新增 490 条 full-check 诊断和 9 个 format 不合规文件；
- wheel 包含 `triton_anchor.tests`，但发布策略未定义是否允许；
- Accepted 材料未冻结最终生产候选 commit；
- 当前环境无可用真实加速器，硬件 JIT 无法执行。

因此仍存在三个规范缺口：最终候选身份、F4/F5/F6 的权威合同、tests-in-wheel 策略；
另有 ART-WHEEL-001 与 QA-CI-001 两个可执行门禁失败。

## 2. 权威层级与不可变边界

1. ADR-0001 继续是 Common Protocol/Schema/Registry/Native 合同的唯一权威来源。
2. 本 ADR 经确认后，只为 Triton 3.6 冻结 F4/F5/F6、发布制品、Ruff、硬件分类和
   证据规则；不得改变 ADR-0001 的 error order、lifecycle、state、版本域或
   explicit-rejection wire contract。
3. Core `0.2.0`、Protocol `1.0`、Manifest Schema `1.0`、Triton `3.6.0`、vendored
   commit `6cc4505027d7b39fe18a44a7f89085b8babb7400`、LLVM/MLIR commit
   `a992f29451b9e140424f35ac5e20177db4afbdc0` 全部保持不变。
4. 除记录其 scope exclusion 外，不运行、不实现、不导入、不依赖 T10.2；其 Kit、fixture、
   artifact、测试与 PASS 结果永远不能作为本门禁证据，任何依赖命中均为 scope violation。
5. 不允许用其他 Triton 版本的 wheel、CMake cache、LLVM build-info、JUnit、coverage
   或 PASS 结论替代 Triton 3.6 当前候选证据。
6. 在各自范围内，权威顺序为 Accepted ADR-0001 与 `决策状态=APPROVED` 的本 ADR >
   被 ADR 以完整身份 pin 的 Schema、Triton 3.6 ABC 与 fixture > 非规范
   traceability/README/tests/evidence。
   测试负责执行合同，不能反向创造或放宽合同；旧 PASS 不能充当当前候选证据。
7. 若本 ADR 与 ADR-0001 在 Common 范围出现语义冲突，必须停止并报告 SPEC GAP；不得
   以“后文覆盖前文”自动解释。

### 2.1 三阶段效力

| 状态 | 规范效力 | 是否可实施 | 是否可发布 |
|---|---|---:|---:|
| `PROPOSED + UNBOUND` | 仅供审阅，不是合同 | 否 | 否 |
| `APPROVED + UNBOUND` | D-01～D-12 与 F4/F5/F6 是规范性设计授权 | 是 | 否 |
| `APPROVED + BOUND`（总体 Accepted） | 设计合同与 exact candidate/artifact/evidence 均冻结 | 是 | 是，仍受硬件结论上限约束 |

第一次确认只授予实施本设计的权限。它不批准尚不存在的生产候选或制品；第二次确认才是
release acceptance。只有 `决策状态=APPROVED` 时，本 ADR 在其版本专属范围高于
traceability/README/tests/evidence。

### 2.2 缺口与失败的关闭条件

| 项目 | 第一次批准后的状态 | 唯一关闭条件 |
|---|---|---|
| `AUTH-F4/F5/F6` | CLOSED（规范性设计已批准） | 可识别发布责任人批准完整 D bundle 与本 ADR requirement；任何未决占位符都会使其保持 SPEC GAP。 |
| `PKG-TESTS` | CLOSED（禁止 tests-in-wheel 的规则已批准） | D-02 获批；旧 wheel 随即按规则成为 FAIL，绝不能转成 PASS。 |
| `AUTH-SHA` | SPEC GAP | final SHA、fresh artifact 与 fresh evidence 经第二次确认并 `BOUND`。 |
| `ART-WHEEL-001` | FAIL | 新 wheel 的所有 F4/ELF gate 实际 PASS。 |
| `QA-CI-001` | FAIL | 候选的本地与 CI 共用 Ruff gates 实际 PASS。 |

策略获批本身不会关闭两个 FAIL。最终只有上述三项 SPEC GAP 均按各自条件关闭、两个 FAIL
均由新鲜证据转为 PASS，软件结论才可写 `FAIL=0, SPEC_GAP=0`。

## 3. 术语与候选身份

- **修复起点**：本分支创建点 `609c3c7…`，仅用于 release-gate 修复本身的回归 diff
  与审计身份，不得作为 T6.3 Ruff changed-files baseline。
- **T6.3 changed-files baseline**：唯一基线是 upstream
  `9ceba3e222fd2c9af9cbaf6a440beb23911c197b`；因此既有 T6.3 实现与本轮门禁修复都在
  changed-files 集合内。
- **生产候选**：完成测试、构建、packaging、CI 与 Ruff 修复后，被 fresh `git archive`
  实际构建和测试的 exact commit，记为 `PRODUCTION_CANDIDATE_SHA`。
- **候选提交 `I`**：`PRODUCTION_CANDIDATE_SHA`，最后一个执行性提交。
- **证据提交 `E`**：以 `I` 为祖先，只新增 Proposed conformance annex、日志索引或 hash，
  不改变生产、测试、CI 或构建逻辑；用户第二次确认时必须明确审阅这个已知 SHA。
- **绑定提交 `A`**：`E` 的直接后继，只写批准 identity/statement、
  `EVIDENCE_COMMIT_SHA=E` 与 `BOUND/Accepted` 状态。`A` 不在自身内容中记录自己的 SHA；
  它由 Git commit identity 及可选外部签名/tag 标识，避免无限自引用。
- **候选失效**：`PRODUCTION_CANDIDATE_SHA` 之后若生产、测试 oracle、CI、build/packaging
  文件或版本 pin 有任何变化，旧 wheel/JUnit/coverage 全部失效，必须冻结新候选并重跑。

最终 conformance annex 必须写入完整 40 位 `PRODUCTION_CANDIDATE_SHA`。在该字段仍未知、
指向非祖先或测试输入不是该 commit 的 archive 时，`AUTH-SHA` 保持 SPEC GAP，禁止发布。
最终必须满足 `I` 是 `E` 的祖先、`A^=E`，且 `git diff I..A` 只能是 ADR、annex 与证据
索引等非执行文档。候选之后若改变任何生产、测试、构建、packaging 或 CI 文件，绑定
自动失效，必须选取新候选并完整重验。

## 4. 推荐决策 bundle

本草案推荐一次确认以下不可拆分的规范选择：

| ID | 推荐决策 |
|---|---|
| D-01 | 将本 ADR 设为 Triton 3.6 F4/F5/F6 与发布门禁的规范来源；ADR-0001 保持不变。 |
| D-02 | production wheel 排除 `triton_anchor.tests` 及所有 descendants；测试保留在源码 checkout，sdist policy 不在本 ADR 范围。 |
| D-03 | 所有 wheel ELF 的 RPATH/RUNPATH 动态库搜索路径必须可重定位；`triton-shared-opt` 使用 install RUNPATH 构建。该规则不把正常的 ELF `PT_INTERP` 绝对路径误作 RUNPATH。 |
| D-04 | Ruff 精确 pin `0.15.22`，CI 不再安装未约束的最新版本。 |
| D-05 | 所有 tracked `.py`/`.pyi` 文件执行 `E9,F63,F7,F82` hard gate。 |
| D-06 | 所有相对 upstream 基线新增或修改的 `.py`/`.pyi` 文件执行 Ruff 0.15.22 默认规则集的完整 check（不是 `--select ALL`）和 format check。 |
| D-07 | 禁止 unsafe fixes、file-level/无 code 的 blanket noqa，以及通过 ignore 配置绕过 changed-files full gate。 |
| D-08 | 既有 530 个唯一 pytest identity 全部保留且 PASS；新增 release-gate tests 另计，不以删除/改名维持“530”。 |
| D-09 | wheel 必须从生产候选的 fresh `git archive` 和全新 CMake 目录重建；禁止复用 SHA `a6785a37…` 的旧 wheel。 |
| D-10 | 缺真实硬件时 hardware 项为 BLOCKED，软件全通过且 0 SPEC GAP 时最高只能 CONDITIONAL PASS。 |
| D-11 | conformance annex 冻结候选 SHA、wheel SHA、工具链、命令、退出码、JUnit、coverage 与硬件分类。 |
| D-12 | 各修改类别独立提交，不 squash；不 push、不创建 PR，除非另获用户明确授权。 |

## 5. F4：Triton 3.6 Production Wheel Package and ELF Integrity Gate

### 5.1 发布 package domain

F4 使用三个 package 集合：

- `A_release`：当前 candidate source 中可发布的 Python package；
- `B_setup`：`setup.py` 实际声明的 package；
- `C_wheel`：从新 wheel package-root file members 机械导出的 Python package。

必须满足 `A_release == B_setup == C_wheel`。`A_release` 的机械定义为：

1. source roots 仅为 `triton/python/triton -> triton` 与
   `python/triton_anchor -> triton_anchor`；
2. 仅枚举这些 roots 下由候选 commit tracked 的 regular blob `.py`、`.pyi`、`.json` 文件；
   roots 下任何 symlink/submodule 或指向 root 外的 payload 都是 F4 FAIL，不得静默忽略；
3. 在导出 package identifier 及 ancestors **之前**，精确排除 source member
   `python/triton_anchor/tests` 或以 `python/triton_anchor/tests/` 开头的路径；
4. 再从映射后的 payload parent 中，仅以每段都是 Python identifier 的 ancestors 导出
   package 集合。

不得把任意名为 `tests` 的 path component 全局列为禁项；这会误伤合法 vendored payload。
`B_setup` 必须排除 dotted package `triton_anchor.tests` 及 descendants；`C_wheel` 必须
精确排除 archive member 等于 `triton_anchor/tests` 或以 `triton_anchor/tests/` 开头。
`C_wheel` 只从 `triton/`、`triton_anchor/` 下允许的 `.py/.pyi/.json/.so` regular-file
members，按与 A 相同的 identifier-ancestor 算法导出。无 suffix 的
`triton/bin/triton-shared-opt` 不贡献 package identity；`triton/bin` 是 intentional
non-package binary container，即使某些 import machinery 把该目录视为 namespace，也不得
因此加入 `C_wheel`。

除 package 集合外，定义 `A_payload` 为上述两个 source roots 映射后的全部 tracked
`.py/.pyi/.json` payload，应用同一 tests exclusion。除生成型
`triton_anchor/_build_info.json` 外，wheel 对每个 `A_payload` member 必须存在且逐字节等于
candidate archive；wheel 不得额外加入无来源的 `.py/.pyi/.json` member。生成型与二进制
allowlist 仅为 `_build_info.json`、`triton/_C/libtriton.so`、
`triton/bin/triton-shared-opt`，并分别受下述 build-info/ELF 合同约束。

Wheel regular-file member set 必须精确等于：`A_payload`（其中 source build-info template
由 generated build-info 同路径替换）+ 上述两个 binary + 以下六个 dist-info members；
directory entries 只允许是这些文件的 parents。这样未知 extension/data file 也不能绕过
suffix inventory 成为额外 payload。

排除只断言 production wheel，不得删除 checkout 中的源码测试或削弱 CI/unit gate。
当前推荐实现通过通用 `setup.py` discovery exclude 完成，可能同时使 sdist 不含该 package；
本 ADR 允许该副作用，但不对 sdist payload 作 PASS 声明。
`triton_anchor.language.ext` 必须继续存在于三个集合中。

### 5.2 Wheel 完整性

| ID | 强制要求 |
|---|---|
| F4-01 | 输入必须是 `PRODUCTION_CANDIDATE_SHA` 的 fresh `git archive`；archive SHA 写入 annex。 |
| F4-02 | 使用工作树外、从未配置过的 CMake/build/staging/wheel 目录；不得复用旧 wheel 或旧 CMakeCache。 |
| F4-03 | 如复用 LLVM 工具链，必须在构建前验证 `llvm-config` version、LLVMConfig、MLIRConfig、`VCSRevision.h` 与 commit `a992f294…`。 |
| F4-04 | ZIP 成员唯一且安全，无 traversal、symlink、encryption、pyc/cache、build tree 或 VCS；允许顶层恰为 `triton/`、`triton_anchor/` 与单一 `triton_anchor-0.2.0.dist-info/`，不得出现其他顶层。 |
| F4-05 | RECORD 覆盖全部成员；除 RECORD 自身外每项 SHA-256 与 size 必须独立重算一致。 |
| F4-06 | filename 必须为 `triton_anchor-0.2.0-cp312-cp312-linux_x86_64.whl`；METADATA/WHEEL 与之精确一致：Name `triton-anchor`、Version `0.2.0`、Requires-Python `>=3.10`、唯一 `packaging>=21`、`Root-Is-Purelib: false`、Tag `cp312-cp312-linux_x86_64`。 |
| F4-07 | `_build_info.json` 必须 `generated=true`，关键字段非 null，版本/commit 精确匹配，`libtriton.so` SHA 与 Core fingerprint 可独立重算。 |
| F4-08 | fresh venv 清除 `PYTHONPATH/PYTHONHOME/VIRTUAL_ENV`，使用 `python -I`；`pip check` 为 0，模块只能来自 fresh site-packages。 |
| F4-09 | entry point 必须且只能包含 tuple `(group="triton.adapters", name="triton-shared", value="triton_anchor.adapters.triton_shared_adapter:TritonSharedAdapter")`。 |
| F4-10 | archive member set 必须满足 §5.1 exact payload equation，并包含下列 required semantic payload；任何额外 regular file 都是 FAIL。 |
| F4-11 | fresh process 软件 smoke 必须按 §5.3 的固定十项得到 `10 PASS / 0 FAIL / 0 SKIP`。 |
| F4-12 | production wheel/RECORD 中 tests prefix member 计数必须为 0。 |
| F4-13 | fresh `python -I` 中 `importlib.util.find_spec("triton_anchor.tests")` 必须返回 `None`。 |

F4-10 的最小 required member 集合精确为：

- `triton_anchor/_build_info.json`；
- `triton_anchor/backends/registry.py`；
- `triton_anchor/backends/schemas/backend_manifest.schema.json`；
- `triton_anchor/backends/examples/triton_anchor_backend.example.json`；
- `triton_anchor/adapters/triton_shared_adapter.py`；
- `triton_anchor/language/ext/__init__.py`；
- `triton/_C/libtriton.so`；
- `triton/bin/triton-shared-opt`；
- 单一规范化 dist-info 下的 `licenses/LICENSE`、`METADATA`、`WHEEL`、`entry_points.txt`、
  `top_level.txt` 与 `RECORD`。其中 LICENSE 必须逐字节等于 candidate root `LICENSE`，
  `top_level.txt` 必须精确为 `triton\ntriton_anchor\n`。

### 5.3 Fresh-install 10/10 smoke

十个 identity 与最低断言固定为：

1. import `triton`，版本为 `3.6.0`，路径属于 fresh site-packages；
2. import `triton_anchor`，版本为 `0.2.0`，公共 AnchorIR/HW/pipeline API 存在；
3. import `triton._C.libtriton`，`ir` 与 `passes` 存在；
4. `libtriton.triton_shared.load_dialects` 存在；
5. 创建 MLIR context 并加载 Triton 与 triton-shared dialect；
6. 构造 Tensor Processor、AME、GPGPU 三类 `HWCapability`，错误组合 fail closed，且
   `to_gpu_target()` 可用；
7. `AnchorIRValidator` 接受合法 Linalg IR、拒绝含 `tt` 的非法 IR，并通过 pre/post hook；
8. 在 `hw=None` 下构造固定 7-pass TTIR pipeline；
9. fresh metadata discovery 找到唯一 `triton-shared` adapter；
10. 固定 add-kernel ASTSource 生成非空 TTIR，包含函数定义与 `tt.load`。

这些 identity/最低断言是合同，候选中的 smoke 实现只能增加断言，不能改名、删除或用
skip 替代。所有 import 和脚本必须来自 fresh site-packages；checkout 仅可作为经 hash
验证的测试驱动输入，不得出现在 `sys.path` 或被 import。

## 6. ELF RPATH/RUNPATH 合同

F4 必须按 ELF magic 自动枚举 wheel 内**全部** ELF 成员，不能只写死已知文件名。每个
ELF 使用 `readelf -dW` 或等价静态工具检查，不得通过 import/dlopen 来完成该检查。

| ID | 强制要求 |
|---|---|
| ELF-00 | 按 ELF magic 得到的成员集合必须精确为 `triton/_C/libtriton.so` 与 `triton/bin/triton-shared-opt`；额外 ELF 必须先修订本 ADR。 |
| ELF-01 | `DT_RPATH` 禁止；发布制品只允许现代 `DT_RUNPATH` 或无动态搜索路径。 |
| ELF-02 | 每个 RUNPATH component 必须恰为 `$ORIGIN`/`${ORIGIN}`，或 token 后紧跟 `/`；以该 ELF member 的 parent 作为 ORIGIN 做 POSIX lexical normalization 后必须仍位于 wheel-owned `triton/` tree。禁止 `$ORIGINevil`、绝对/空/cwd component、反斜杠、NUL，以及经 `..` 解析后逃出该 tree。 |
| ELF-03 | component 不得含源码、build、cache、staging、evidence 或工具链绝对目录。 |
| ELF-04 | `triton-shared-opt` 的 RUNPATH **必须精确为**单一 `$ORIGIN/../lib`；`libtriton.so` 的 RPATH/RUNPATH 必须均 absent。允许语法中的 `..`，因为从 `triton/bin` 规范化后得到 wheel-owned `triton/lib`。 |
| ELF-05 | `triton-shared-opt` CMake target 必须设置 `BUILD_WITH_INSTALL_RPATH ON`，使被 `setup.py` 复制的 build-tree binary 已使用 install RUNPATH。 |
| ELF-06 | 每个 `DT_NEEDED` 必须是不含 `/`、反斜杠或 NUL 的 basename；两个 ELF 的 NEEDED set 必须精确为 `{libz.so.1, libstdc++.so.6, libm.so.6, libgcc_s.so.1, libc.so.6, ld-linux-x86-64.so.2}`，不得隐含 LLVM/MLIR 或其他未打包 DSO。 |
| ELF-07 | fresh install 后再次检查安装文件 dynamic tags；从已验证的 fresh `site-packages` 解析绝对路径 `triton/bin/triton-shared-opt`，清除 `LD_LIBRARY_PATH`、`LD_PRELOAD`、`LD_AUDIT` 后以 `--version` 执行，必须 exit 0 且报告 LLVM `22.0.0git`。不得调用 PATH 上同名文件。 |

`readelf` 不可用、ELF 解析不完整或任一动态 tag 无法确定时，该门禁为 BLOCKED/FAIL（取决于
是环境缺失还是制品非法），绝不能 skip 后汇总为 PASS。`PT_INTERP` 单独记录，但不属于
RPATH/RUNPATH component；这避免把 Linux 正常 loader path 错判成 ELF-02 失败。

普通编译断言/诊断字符串中的上游源路径不等同于 loader search path；本 ADR 不把它们
自动判为 ELF-00～07 失败。若发布需要 reproducible-build path redaction，另立决策，
不得与 RUNPATH 合规性混为一谈。

## 7. F5：Triton 3.6 Governed Lazy Legacy Bridge

F5 只适用于没有 Manifest 的真实 Triton 3.6 layout Legacy distribution；Manifest plugin
继续由 ADR-0001 治理。
F5 的 load/interface/selection/conflict/lifecycle failures 必须使用 ADR-0001 已冻结的
结构化 record keys、identity、total order 与 primary-error 规则；不得以裸 ImportError、
TypeError 或插件 `repr` 作为公开错误。

| ID | 强制要求 |
|---|---|
| F5-01 | discovery/list/inspect/conflict/validate/diagnostics 保持 metadata-only，不 import Legacy entry point。 |
| F5-02 | Manifest-first；仅首次 compiler/runtime consumption 才可把 Legacy entry-point value 当作 package root，分别从 `<root>.compiler` 与 `<root>.driver` materialize exact pair；不要求 root module re-export class alias。 |
| F5-03 | F6 必须在 constructor、probe、initialize、public mapping 或 selection publication 前通过。 |
| F5-04 | compiler/runtime 消费同一 exact record 与 reset-bound lease，不得产生分裂 mapping/cache/state。 |
| F5-05 | Legacy 状态始终为 `LEGACY_UNVERIFIED`，不得伪装成 Manifest-verified。 |
| F5-06 | rejected/ambiguous Manifest、explicit selector error、non-empty kernel capability requirement 均禁止 Legacy fallback。 |
| F5-07 | 0 或多个 compiler/driver class、active ambiguity、probe/constructor/target/commit failure 必须结构化且全原子失败。 |
| F5-08 | 8 callers、共享失败波次、同/不同 target、reentry 和 reset-stage races 使用确定性同步与有界 timeout；旧 epoch 不得发布。 |
| F5-09 | real no-Manifest v3.6-layout fixture 内容 SHA-256 固定为 `97402abf825cf56d800edf5f4cfa25a8513be585b2b067391d0766664d18a455`；绝对路径不是合同，module path 与 import-once sentinel 写入 annex。 |
| F5-10 | explicit exact selector、public compiler/runtime adapter、manual `backends` entry 与 default/active resolution 均不得绕开同一 governed record、F6 gate 或 lease commit。 |
| F5-11 | reset 必须清除该 generation 的 mapping/default/active/selection/cache 与未提交 lease；下一次消费重新 metadata discovery/materialization，Legacy entry-point load 每 record/generation 至多一次且 reset 后可再次 load。pair materialization、F6 validation、lease/publication 与 runtime default/active 的 `is_active`/driver construction/current-target 是 single-flight；但每个 public compiler `make_backend` caller 必须独立执行 `supports_target` 与 compiler constructor（8 callers 即各 8 次）。旧 generation 完成不得污染新 generation。 |

F5-09 的 companion snapshot 是
`tests/acceptance_t63/triton_v36_legacy_baseline.json`（candidate-base Git blob
`6b08d790891eab97588e79e28b5c796bf1ecc3dd`，文件 SHA-256
`8eadf0d184da2bb01b6b8a32398c5d8ccc87ff79388c0ec82ae1efefeb492d79`）。该 JSON 中
`observed` 只记录 upstream 行为，规范性 delta 以本 ADR 与其 `governed_port_delta` 一致部分
为准。fixture 必须按内容 SHA 查找并在 fresh venv 安装，不能把本机绝对路径写入合同。

## 8. F6：Triton 3.6 Runtime-Pair Interface Publication Gate

F6 的 validator 实现可从固定身份的 Triton 3.6 `BaseBackend`、`DriverBase` 与
`GPUTarget` 做只读 introspection，但规范 oracle 是本节冻结的 source identity 与 surface
snapshot；不得复制 Triton 3.3/其他版本的方法列表，也不得跟随 candidate 漂移。

Canonical source identity 同时冻结为：

- `triton/python/triton/backends/compiler.py`，Git blob
  `4b560876e393eefa10212e9cba862f9ea1ebdad1`；
- `triton/python/triton/backends/driver.py`，Git blob
  `0d9db7884ba01ea3f7681bda073d13ebf21369c1`。

以本 ADR 固定的 vendored commit 为准，当前 surface snapshot 是：

- `BaseBackend`：abstract staticmethod `supports_target(target)`；abstract instance methods
  `hash(self)`、`parse_options(self, options)`、`add_stages(self, stages, options)`、
  `load_dialects(self, context)`、`get_module_map(self)`；
- `DriverBase`：abstract classmethod `is_active(cls)`；abstract instance methods
  `map_python_to_cpp_type(self, ty)`、`get_current_target(self)`、
  `get_active_torch_device(self)`、`get_benchmarker(self)`；
- frozen dataclass `GPUTarget(backend: str, arch: int | str, warp_size: int)`，field order 与
  frozen identity 都属于合同。

动态派生只是一种安全读取/验证实现，必须与上述 blob 和 snapshot 相等；它不能把 candidate
自身修改后的 surface 反向变成新 oracle。若 pinned vendored commit、blob 或上述 surface
变化，现有 F6 binding 自动失效，必须先修订并重新批准本 ADR。

| ID | 强制要求 |
|---|---|
| F6-01 | `compiler_cls`/`driver_cls` 必须是 type；conforming ABC subclasses 与 conforming structural classes 均允许。 |
| F6-02 | 每个 required abstract member、descriptor kind 和 compatible call shape 必须完整；额外 unresolved abstract 拒绝。 |
| F6-03 | 验证发生在 initialize/plugin callback/class construction/public mapping/selection/default-active 前。 |
| F6-04 | 所有独立错误以 `backend_plugin_interface_error` 聚合并稳定排序，继承 ADR-0001 固定字段/identity/order，不执行 hostile property/metaclass/descriptor/`repr`。 |
| F6-05 | validator 对 same object 幂等；不同 object 重用 identity 或 late registration 拒绝。reset 保留已注册 validator contract/object identity，但使旧 generation validation/result 失效；新 generation 第一次消费必须 single-flight 重新验证。 |
| F6-06 | 8 invalid callers 共享一次 validation 与同一错误；reset 胜过 blocking validation，旧 generation 不发布。 |
| F6-07 | 失败插件不得污染合法插件、compiler/runtime adapter、mapping、selection、default 或 active state。 |
| F6-08 | Registry validate/select、compiler `make_backend`、runtime driver/default-active 等所有公开路径必须观察同一 gate 与同一结构化错误；不得存在旁路。 |

## 9. Ruff 与 CI 合同

### 9.1 固定工具与文件集合

- Ruff 必须精确安装 `ruff==0.15.22`；不得使用浮动版本范围。
- CI checkout 必须设置 `fetch-depth: 0` 并含完整候选祖先历史，使固定 upstream SHA 可解析；
  禁止依赖 runner 恰好存在的 `upstream/triton_v3.6` remote ref。执行前必须用
  `git merge-base --is-ancestor 9ceba3e222fd2c9af9cbaf6a440beb23911c197b "$T63_GATE_SHA"`
  hard-fail 验证谱系。
- CI pull request 必须显式 checkout event 的 head SHA，不得把 GitHub synthetic merge commit
  当成候选。resolved `T63_GATE_SHA` 必须等于 `git rev-parse HEAD`；`git write-tree` 必须等于
  `HEAD^{tree}`，tracked worktree/cached diff 必须为空。否则文件清单可能来自非候选 index，
  门禁必须在调用 Ruff 前失败。
- `ALL_PYTHON_FILES` 定义为 `git ls-files -z -- '*.py' '*.pyi' | LC_ALL=C sort -z`
  的全部结果，包括
  vendored Triton 与测试源码；Git submodule 内部内容不属于当前 superproject 的 tracked
  文件集合。
- `T63_CHANGED_PYTHON_FILES` 定义为：

```bash
T63_GATE_SHA="${PRODUCTION_CANDIDATE_SHA:-HEAD}"
git diff --diff-filter=ACM --name-only -z \
  --no-renames \
  9ceba3e222fd2c9af9cbaf6a440beb23911c197b "$T63_GATE_SHA" \
  -- '*.py' '*.pyi' | LC_ALL=C sort -z
```

CI 中 `T63_GATE_SHA` 必须等于 checkout 的 `HEAD`；最终证据中必须等于完整
`PRODUCTION_CANDIDATE_SHA`。上述 diff 的规范 `--diff-filter` 为 `ACM`；配合
`--no-renames`，rename 的新路径按 added 处理。两个文件集合都必须以 NUL-safe Bash
array 传给 Ruff，清单按 `LC_ALL=C sort -z` 排序、写入证据并做 hash。若 changed 集合
为空，脚本必须显式记录 `count=0` 并跳过该调用，禁止 Ruff 无路径参数退化为扫描 `.`。
另以相同 baseline/candidate 和 `--no-renames --diff-filter=D` 生成
`T63_DELETED_PYTHON_FILES`；默认必须为空。删除或 rename-away 任一 `.py/.pyi` 都必须先由
发布责任人修改/批准 D bundle，不能用“文件已不存在所以无需 lint”规避门禁。

该定义包括 `setup.py`、`python/`、`tests/` 以及被 T6.3 修改的 vendored `triton/`
Python 文件；不允许为减少修复量而静默排除其中任一文件。

### 9.2 Hard gates

| ID | 强制要求 |
|---|---|
| RUFF-01 | 对 `ALL_PYTHON_FILES` 执行 `ruff check --isolated --no-cache --no-fix --target-version py310 --select E9,F63,F7,F82`，退出码必须为 0。 |
| RUFF-02 | 对 `T63_CHANGED_PYTHON_FILES` 执行 `ruff check --isolated --no-cache --no-fix --target-version py310`，不传 `--select/--ignore`；“完整”指 Ruff 0.15.22 默认规则集，不是 `--select ALL`，退出码必须为 0。 |
| RUFF-03 | 对 `T63_CHANGED_PYTHON_FILES` 执行 `ruff format --isolated --no-cache --target-version py310 --check`，退出码必须为 0。 |
| RUFF-04 | CI 不得把 full check/format 转为 warning、`continue-on-error` 或 `|| true`。 |
| RUFF-05 | acceptance/CI 命令禁止 `--fix` 与 `--unsafe-fixes`；开发修复也禁止 unsafe fixes。安全自动修复如用于开发，必须审阅 diff 并重跑全部行为门禁。 |
| RUFF-06 | 禁止新增 file-level `# ruff: noqa`/`# flake8: noqa`、无具体 code 的 bare `# noqa`、`ALL`/wildcard blanket per-file-ignore，或通过全局 ignore 绕过 changed-files gate。 |
| RUFF-07 | 精确 code 的局部 suppression 只有在语义上无法消除、带相邻理由并获单独审批时允许；本 release-gate 修复默认不新增。 |
| RUFF-08 | 独立 policy scanner 必须以 Python tokenize comment 语义审计 changed `.py/.pyi`，并审计 tracked Ruff config、gate script 与 workflow；拒绝 RUFF-06 模式、`--unsafe-fixes`、gate `--ignore`/per-file-ignore。精确 suppression 清单必须写入 annex；未列例外为 0。 |

为了让 v3.6 PR 可执行相同规则，门禁逻辑应进入仓库脚本并由 CI 调用；本地与 CI 必须
复用同一脚本、基线 SHA 和 Ruff pin，避免两套命令漂移。三个 gate 分别保存退出码后
再与 policy scanner 合并判定，任何一个非零都必须让 job 失败，不得因顺序或 shell 条件
吞掉失败。Ruff 文件参数必须位于显式 `--` 之后；changed 集合为空时只允许记录空集 PASS，
不得无参数调用 Ruff。

最终 tree 无法证明开发历史中从未执行过 unsafe fixer，因此证据分两层：CI/acceptance
完整 argv 与 scanner 机械证明门禁未使用/配置 unsafe fix；发布责任人在 annex 对开发过程
作 attestation，并审阅本轮每个 style commit diff。不得把无法机械证明的历史写成工具 PASS。

## 10. 测试 identity、coverage 与失败处理

### 10.1 冻结的 530 identity baseline

修复起点 `609c3c7…` 的三份 fresh JUnit 只用于冻结 identity，不得复用其 PASS 结论：

| Suite | count | baseline JUnit SHA-256 |
|---|---:|---|
| Common | 409 | `cb16e2e524028279975e14ac5605f466618b20c2e048b16d7e8caec603bb4cbb` |
| F5/F6 | 118 | `d945e20f54747cb85a84dda1b4d440db8b505c4f6c36db11e33c996fdc640cf3` |
| F4 | 3 | `c5521ec99e7793cfb723f175017cf4a76bbb38d0aca1ee47371d6994c220c038` |

Canonical identity manifest 的 SHA-256 为
`f6a7af60a2bf8bbf7a822c1f855c19045b975169e6c67d1f8846c5f66a307235`
（76,376 bytes）。重算算法固定为：

1. 依次用 XML parser 选择三份 JUnit 的所有 `<testcase>` descendants；每项读取缺省为空串的
   `classname`、`name`，组成对象；
2. 不去重，并单独断言 aggregate count 与 unique tuple count 都精确为 530；
3. 按 Python Unicode code-point 的 `(classname, name)` 二元组升序；
4. 使用 `json.dumps(items, ensure_ascii=False, sort_keys=True, separators=(",", ":"))`
   序列化，追加且只追加一个 LF，以 UTF-8 编码后计算 SHA-256。

最终 candidate 的 fresh JUnit 必须包含这 530 个 tuple 且每个实际 PASS，
`fail/error/skip/xfail=0`。RUNPATH、package 与 Ruff policy 新测试必须在生产修复前提交并
形成 additional identity 集合；最终总集为 baseline 530 与 additional N 的不相交并集，
不得用改名/替换新节点维持表面数字。Common 409、F5/F6 118、F4 3 与新增测试均须有
fresh JUnit，跨文件 identity 不得重复。

### 10.2 Coverage gate

Coverage population 精确为 Common 409 + F5/F6 118 = 527；F4 3 与新增 release-gate tests
仍必须 PASS/JUnit，但不进入 coverage floor。执行合同为：

- candidate fresh archive、同一个全新 Python 3.12.3 venv；pytest `9.1.1`、pytest-cov
  `7.1.0`、coverage.py `7.15.4`；
- source domain 精确为 `python/triton_anchor` 的全部 statement，包括其源码 tests；不启用
  branch coverage，不得使用 omit/include 或隐式发现 `.coveragerc`/pyproject 配置；
- Common 与 F5/F6 各使用预先不存在、互不相同的 `COVERAGE_FILE` 和 fresh JUnit/cache/tmp；
  在新的空 combine 目录中只 combine 这两份 data，并断言输入/成功合并数恰为 2；
- 两份 coverage JUnit 仍须分别通过 409/118 identity 与状态 audit；节点不足或 pytest
  非零时，任何 XML rate 都不能 PASS；
- `coverage report --precision=2 --fail-under=76.00` 必须 exit 0，再生成 XML；独立解析并
  记录 `lines-covered`、`lines-valid` 与 exact ratio。

规范 floor 有意取 `76.00%`；观测 baseline 是 `3369/4418 = 76.25622454%`、XML
`line-rate=0.7626`。这是明确容许最多约 0.2563 个百分点波动的 release floor，不宣称
zero-regression。用户若不接受该容差，应在第一次批准时把 D bundle 改为 exact-baseline
ratio，而不能在实施后临时改变分母或 rounding。

### 10.3 失败处理

任一失败必须继续定位；不得改 oracle 迎合实现、删除节点、skip/xfail、blanket noqa、
broad catch、retry 或 fallback 掩盖失败。修复后必须重新执行受影响定向测试、全部软件
套件、Ruff、format、wheel、fresh install 与 coverage。所有 timeout 是失败诊断边界，
不是把未完成工作改为 PASS 的机制。

## 11. 硬件门禁与最终分类

硬件探针至少记录 device nodes、驱动工具、PyTorch build、CUDA/HIP/XPU/MPS availability、
device count 和实际 backend entry point。仅有 `/dev/dxg` 或驱动库文件不能证明设备可用。

| ID | 强制要求 |
|---|---|
| HW-01 | 只接受从 `PRODUCTION_CANDIDATE_SHA` fresh archive 构建并安装的同一 wheel、匹配 Triton/LLVM pins、fresh venv/fresh `TRITON_CACHE_DIR`，以及真实受支持 backend/device；禁止 source checkout import、旧 cache、其他版本 wheel、mock 或 CPU 替代。 |
| HW-02 | fresh process 记录 `triton.backends` entry-point distribution/name/value、governed record identity、driver target、device name/count、驱动/runtime 版本与 availability probe；任何必要项缺失即 BLOCKED，不得 pytest-skip→PASS。 |
| HW-03 | 执行本节固定的非 T10.2 vector-add：host CPU 生成 `N=1024`、float32 `x=arange(N)`、`y=2*arange(N)` 与 reference `3*arange(N)`，复制输入到真实 device；`BLOCK=256`，通过 `triton.jit` 在实际 target 上 cold compile+launch，再 warm launch，显式 synchronize，两次结果复制回 host 后都逐元素 bit-exact 等于 reference。 |
| HW-04 | cold 与 warm 各设 300 秒 hard timeout，记录 compile/launch/sync duration、stdout/stderr/exit；在已满足 HW-01/02 后发生 compile、launch、sync 或 result mismatch 均为 FAIL。 |
| HW-05 | 运行前后快照 Registry generation/record/selection/mapping/default/active；失败不得发布不完整 record 或 stale generation。硬件/driver 自身建立的合法 active state 可存在，但必须可归因且 reset 后按合同清理。 |

固定 kernel 只使用 `tl.program_id`、`tl.arange`、masked `tl.load`、float32 add 与 masked
`tl.store`，grid 为 `(ceil_div(N, BLOCK),)`；不得导入或调用 T10.2 Kit/fixture/helper。若某个
真实 backend 无法执行这组 Triton 3.6 公共语义，则是该候选/后端的 FAIL，而不是改 kernel
迎合实现。hardware probe、kernel source 与预期输出的 SHA 必须进入 annex。

| 条件 | 硬件分类 | 最终结论上限 |
|---|---|---|
| 有匹配真实硬件与 backend，JIT/launch/result 全通过 | PASS | 全部其他必需项通过时可 FULL PASS |
| 无匹配硬件/backend，软件与 Mock 全通过且 0 SPEC GAP | BLOCKED | **CONDITIONAL PASS**，不得 FULL PASS |
| 硬件测试实际执行并违反明确合同 | FAIL | FAIL |
| 软件核心构建/验收无法开展 | BLOCKED | BLOCKED |

硬件缺失不能导致软件 test skip 被计为 PASS；软件 TTIR smoke 也不能替代真实 launch。

## 12. 构建与最终执行顺序

用户第一次确认本 ADR 后按以下顺序实施：

1. 将决策状态改为 `APPROVED`，保持实现绑定 `UNBOUND`，记录确认主体、角色、日期和
   明确批准语句；此时总体状态仍不是 Accepted；
2. 在任何 test 修改前，从修复起点重新生成 §10.1 canonical 530 manifest，核对 count、
   uniqueness、三份 JUnit SHA 与 pinned manifest SHA，并保存机器可读 baseline；
3. 添加 RUNPATH、package inventory 和 Ruff policy 回归测试，确认它们能捕获旧行为；
4. 在 `triton-shared-opt` target 设置 `BUILD_WITH_INSTALL_RPATH ON`；
5. 在 `setup.py` production package discovery 排除 `triton_anchor.tests`；
6. pin Ruff、实现本地/CI 共用 gates，并修复全部 T6.3 changed Python files；
7. 逐提交运行定向测试；随后冻结候选提交 `I = PRODUCTION_CANDIDATE_SHA`；
8. 从 `I` 的 fresh `git archive`、全新 CMake 目录重建 wheel；禁止复用
   `a6785a37fa471e0b407da252638900c2772c7c50ff14d7a77e0fd66bba8e5ac8`；
9. 运行既有 530 identities、新增 tests、全 ELF audit、RECORD、fresh `python -I`、
   `pip check`、10/10 smoke、Ruff、format 与 coverage；
10. 生成并提交 Proposed conformance annex，形成证据提交 `E`；验证 `git diff I..E` 仅有
    docs/evidence、所有 hash 与候选身份一致；
11. 将 `I`、`E`、wheel filename/size/SHA、fixture SHA 与全部 evidence hashes 提交给发布
    责任人第二次确认；确认后以 `A` 记录 `EVIDENCE_COMMIT_SHA=E`，把实现绑定改为
    `BOUND`、总体状态改为 Accepted。若第 7 步后的任何执行文件有变化，返回第 7 步。

## 13. 提交边界

禁止 squash。推荐独立提交序列：

1. `docs: propose T6.3 Triton 3.6 release gates`（本草案）；
2. `docs: approve T6.3 Triton 3.6 release gate decisions`（第一次用户确认，仍 UNBOUND）；
3. `test: add T6.3 Triton 3.6 release gate regressions`；
4. `fix: make triton-shared-opt wheel RUNPATH relocatable`；
5. `fix: exclude triton-anchor tests from production wheel`；
6. `ci: pin Ruff and enforce T6.3 release gates`；
7. `style: satisfy T6.3 changed-file Ruff gates`；
8. `docs: add proposed T6.3 Triton 3.6 conformance evidence`（形成 `E`，仍 UNBOUND）；
9. `docs: bind T6.3 Triton 3.6 conformance evidence`（第二次用户确认后形成 `A`，转为 Accepted）。

若某类同时需要生产与测试修改，仍以测试先行提交；不得把版本 pin、T10.2 或无关重构
混入上述提交。未经用户另行明确授权，不 push、不创建 PR。

## 14. Conformance annex 必备字段

annex 文件建议为 `docs/adr/0002-t63-triton-v36-release-gates-conformance.md`。`/docs/`
当前被通用 ignore 规则覆盖，因此 ADR 与 annex 都必须显式成为 tracked Git blob；仅存在于
本机 ignored worktree 不构成证据。annex 至少记录：

- ADR decision approval 与 final binding 的 identity/role/date/statement；
- branch、upstream SHA、修复起点、`PRODUCTION_CANDIDATE_SHA=I`、
  `EVIDENCE_COMMIT_SHA=E`；其中 `E` 在 Proposed annex 中暂为待绑定字段，只能由其后继
  binding commit `A` 写入，annex 不要求也不得自写 `A` 的 SHA；
- candidate archive SHA 与生成命令；
- Core/Protocol/Schema/Triton/vendored/LLVM/MLIR pin；
- Python、CMake、Ninja、compiler、OS、libc 与 exact LLVM toolchain 证据；
- 新 wheel filename/size/SHA，明确旧 wheel SHA 未复用；
- ZIP、package inventory、tests absence、RECORD、METADATA、build-info、全 ELF dynamic
  tags、fresh venv module paths、pip check、10/10 smoke；
- 既有 530 canonical manifest schema/count/SHA、最终 missing/additional identity 集合、
  新增 node count、每个 JUnit SHA、退出码和 duration；
- Ruff distribution/version、resolved base/candidate/tree SHA、ancestor/clean-index assertions、
  policy script/workflow SHA、effective settings、all/changed NUL manifest count/SHA、每个 gate
  的完整 argv/raw+JSON output SHA/diagnostic count/exit/timeout、suppression scanner 与例外表；
- coverage population/source/config、Python/pytest/pytest-cov/coverage versions、两份 data SHA、
  combine input count、covered/valid/exact rate、report/XML SHA；
- hardware PASS/BLOCKED 证据与最终结论；
- 所有命令的 cwd、环境、开始/结束 UTC、timeout、原始输出与 exit code；
- 最终 `git status --short --branch`、`git diff --check`，以及无 T10.2 依赖的证明。

只有 annex 中所有必需软件项均 PASS、`FAIL=0`、`SPEC GAP=0` 时才可签署。

## 15. 第一次维护者批准记录

- 审批角色：**T6.3 发布责任人/维护者；姓名或公开 handle 未提供**
- 审批日期：**2026-08-24**
- 批准范围：**D-01～D-12 不可拆分 bundle 与本 ADR 全文**
- 明确接受：**RUNPATH、Ruff、coverage 与 hardware 分类/执行规则**
- 授权状态：**进入 `APPROVED + UNBOUND` 实施阶段**
- 偏离：**无**
- 本任务会话收到的书面确认：

```text
护者身份确认 ADR-0002 D-01～D-12 及全文规则，接受 RUNPATH、Ruff、coverage 和
hardware 分类方案，授权进入 APPROVED + UNBOUND 实施阶段。
```

回复开头在传输文本中呈截断形式，但其确认对象、接受的四类方案及实施授权均明确，并且是
对前一版 §15 确认请求的直接回复；本记录不据此补写未提供的姓名或公开 handle。

自本记录起，`AUTH-F4/F5/F6` 与 `PKG-TESTS` 的规范决策缺口按 §2.2 关闭；旧 wheel 仍因
tests-in-wheel 与 RUNPATH 规则为 FAIL，`QA-CI-001` 仍为 FAIL，`AUTH-SHA` 仍为 SPEC GAP。

## 16. 第二次绑定确认（尚未发生）

完成候选提交 `I`、证据提交 `E`、fresh wheel 与全部软件门禁后，必须向发布责任人提供：

- 完整 `PRODUCTION_CANDIDATE_SHA=I` 与 `EVIDENCE_COMMIT_SHA=E`；
- 新 wheel filename/size/SHA-256；
- 530 baseline + additional identity、JUnit、Ruff、coverage、F4/ELF/fresh-install 证据；
- hardware PASS/BLOCKED 证据及最终结论上限；
- `git diff I..E` 仅包含 docs/evidence 的证明。

只有发布责任人第二次明确确认这些 exact identities 后，才可形成绑定提交 `A` 并把状态改为
`APPROVED + BOUND / Accepted`。
