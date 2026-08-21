# T6.3 第一阶段需求—不变量—实现—测试追踪矩阵

- 状态：Accepted，ADR-0001 推荐 bundle 已于 2026-08-21 获维护者批准
- 验收 baseline commit：`fe8f2253c365f9d8799cade6428b7efebbc9b90c`
- 生产代码 baseline：`444ead1ac0b314d971dfef6f2592e5f091d38690`
- 当前完整套件：177 nodes = 163 pytest PASS / 13 FAIL / 1 SKIP；语义分类为
  160 PASS / 13 FAIL / 1 BLOCKED / 3 SPEC GAP
- 目标：baseline JUnit 中 177 个唯一 `(classname, name)` testcase identity 全部保留。
  若新增参数化测试后总数为
  `N >= 177`，则 `N - 7` PASS / 6 FAIL / 1 BLOCKED / 0 SPEC GAP / 0 NOT RUN；
  仅当节点数不变时是 170 / 6 / 1 / 0。

> 本表随 ADR-0001 构成已批准的实现与验证边界；生产实现须按独立提交顺序推进。
> 完整 baseline node 集合来自已哈希的 `phase1_baseline.xml`，不是 manifest 中只列出的
> failure/gap 子集；最终门禁必须直接比较新旧 JUnit 的 `(classname, name)` tuples，
> 不通过字符串拼接猜测路径式 pytest nodeid。

## 1. F3：Schema/parser structural domain

| ID | 需求/合同 | 不变量 | 获批后拟实现 | 冻结/新增测试 | 当前 |
|---|---|---|---|---|---|
| F3-01 | shipped Schema 与 `parse_manifest` 同域 | INV-01 | `manifest.py` 共享 lightweight predicates；semantic checks 分层，不引入 runtime jsonschema | `test_public_parser_rejects_value_forbidden_by_shipped_schema` | 1 FAIL |
| F3-02 | 所有 `nonEmptyString` 引用一致 | INV-01 | required/optional/version strings、string tuples 共用首尾空白及 CR/LF 规则 | display/vendor、Protocol/Core/Triton/LLVM/MLIR versions、targets/capabilities/requires_capabilities 的空、空白、LF、CRLF、Unicode、合法值参数化双 oracle | 待新增 |
| F3-02A | 所有 anchored patterns 拒绝 trailing CR/LF | INV-01 | Schema 显式排除 CR/LF，不依赖 `$` 行尾；parser 同 predicate | schema_version/plugin_id/entry_point/gitCommit/nonEmptyString/native path 的 trailing LF/CR/CRLF differential | 已知差异/待 D-02 |
| F3-03 | native path predicate 一致 | INV-01、06 | parser 使用 Schema path predicate；containment/RECORD 留在 semantic/static 层 | `./x.so`、absolute、dot/dotdot、backslash、double separator、合法 path differential | 已知差异/待新增 |
| F3-04 | JSON Schema integer 语义一致 | INV-01 | 接受 finite integral JSON numbers并 canonicalize `int`；boolean/非整数拒绝 | `1`、`1.0`、`1e0`、`1.5`、bool、极值 differential | 已知差异/待新增 |
| F3-05 | structural string 不混入 PEP 440 semantic check | INV-01、02 | parser 按 Schema 接受 non-empty；pure semantic validator 在 import 前验证 SpecifierSet | `backend_protocol="banana"` Schema/parser ACCEPT + semantic REJECT；现有 malformed protocol node迁移但保留 | 已知差异/待新增 |
| F3-06 | cross-record uniqueness 属 semantic 层 | INV-01、02 | duplicate plugin/entry point 从 parser 移入 import-free semantic validator | 现有 duplicate plugin_id/entry_point nodeid 保留，新增 Schema/parser ACCEPT assertion与 semantic REJECT | 已知差异/待新增 |
| F3-07 | JSON text duplicate member 仍 fail-closed | INV-02、07 | `object_pairs_hook` 在 object 构造前拒绝；与 in-memory Schema domain 分开说明 | `test_registry_rejects_duplicate_json_member_names` | PASS |

批准依赖：ADR D-01～D-04。若 D-01 为 published/unknown 且维护者不接受 wire
erratum，本阶段完全停在协议门，F3 也不得开始。只有用户另行授权“F3-only partial
remediation”时，才可仅把 parser 扩展到既有 Schema，且不得宣称本阶段完成。

## 2. F1：完整兼容诊断

| ID | 需求/合同 | 不变量 | 获批后拟实现 | 冻结/新增测试 | 当前 |
|---|---|---|---|---|---|
| F1-01 | Protocol/Core/Triton/LLVM 独立错误一次报告 | INV-02、03、07、08 | `compatibility.py` pure evaluator 返回 checks/errors tuple；Registry 批量记录 | `test_registry_reports_every_simultaneous_compatibility_failure` | 1 FAIL |
| F1-02 | stable total order 与完整结构 | INV-03、07 | ADR §8 rank/field order；所有 stable keys 始终存在 | 待新增：恰好四项、重复运行同序、remediation/non-null identity、无随机 repr | coverage待补 |
| F1-03 | 单错误 public behavior 兼容 | INV-03、07 | 需抛 API 仍抛排序第一项；record 保存 tuple | 现有 Protocol/Core/Triton/LLVM 单维测试、strict/identifier paths | PASS/回归 |
| F1-04 | 所有独立检查 pre-load | INV-02、05、08 | distribution-dependent failure 不阻止 version checks；不捕获宽泛 Exception | simultaneous test `load_calls == 0`；新增 compile/init/public-state counters | 部分 PASS/待补 |
| F1-05 | native/capability 不回归 | INV-02、03、06 | unsupported isolation 在 structural Schema/parser 阶段即停止，不进入 compatibility evaluator；capability 仍走 semantic 末位 | native rejection/capability pre-import tests | 待 ADR |
| F1-06 | multi-record conflict identity 稳定 | INV-05、07 | 每个受影响 record 一条自身 identity error；sorted `related_plugin_ids/record_ids`；root summary 才可 null | duplicate SONAME/symbol/plugin_id/entry-point tests新增 per-record identity与 canonical sort assertions | behavior缺口/待补 |

批准依赖：ADR §6.2、§8（D-04、D-06），不依赖无关的 native init 顺序。

## 3. F2：Python lifecycle contracts

| ID | 需求/合同 | 不变量 | 获批后拟实现 | 冻结/新增测试 | 当前 |
|---|---|---|---|---|---|
| F2-01 | optional hooks 可全部缺失 | INV-04 | sentinel；init/shutdown no-op；diagnostics 返回独立 `{}` | optional-hooks node当前断言 `None`，获批后同 nodeid 迁移到 public contract；Base default node PASS | current PASS / oracle待冻 |
| F2-02 | initialize callable、同步、返回 None | INV-04、05、09 | signature bind、awaitable/非 None reject、最多一次 best-effort shutdown | noncallable PASS；non-None FAIL；exception cleanup PASS；新增 wrong signature/awaitable | 1 FAIL/coverage待补 |
| F2-03 | shutdown 错误结构化、继续清理、幂等 | INV-04、07、09 | sentinel、signature/result gate、attempt ledger、继续其他 plugins | noncallable/wrong-return 2 FAIL；reset idempotence/race PASS；新增 exception/multi-plugin | 2 FAIL/coverage待补 |
| F2-04 | diagnostics Mapping snapshot 与生命周期隔离 | INV-04、07 | 仅检查 `Mapping`，执行顶层 `dict(value)` copy；不新增 JSON-value 限制；error-only result | noncallable/wrong-return 2 FAIL；Base default PASS；新增 missing/exception/top-level mutation isolation | 2 FAIL/coverage待补 |
| F2-05 | initialize once、reset race、hook re-entry | INV-05、09 | hook 在 lock 外；generation token；同 hook once | concurrent initialize/reset nodes PASS；新增 init/shutdown/diagnostics re-entry | PASS/coverage待补 |
| F2-06 | state/public mapping publish gate | INV-05、08、09 | REGISTERED 仅在 successful None result 后；reject detach mapping/cache | initialize exception/non-None + integration mapping assertions | 部分 PASS/待补 |

批准依赖：ADR §7（D-05、D-10）。Python lifecycle 不依赖未来 native loader 方案。

## 4. Native/Protocol SPEC GAP closure

推荐 ADR disposition 是 Protocol/Schema 1.0 explicit rejection，而非 native support。

| ID | 规范问题 | 不变量 | 推荐合同/实现 | 冻结测试/证据 | 当前 |
|---|---|---|---|---|---|
| ABI-01 | 必需 symbol/name/namespace/version/signature | INV-02、06、10 | 1.0 无 native symbol；Schema/parser 先拒绝，测试不得发明名称 | `test_native_required_symbol_name_is_not_defined_by_manifest_1_0` 改为同 nodeid rejection oracle | SPEC GAP |
| ABI-02 | plugin compiler/stdlib/C++/C++11 ABI 证明 | INV-02、06、10 | 1.0 native 不支持，故不适用；Core fingerprint 不能冒充 plugin proof | `test_native_plugin_cxx_abi_is_not_attested_by_manifest_1_0` 改为同 nodeid no-load rejection | SPEC GAP |
| ABI-03 | `dlopen` timing/flags/init/failure | INV-02、05、06、08、09 | 1.0 禁止 helper/host dlopen，无 unsafe flag；future loader 另 ADR | `test_native_static_inspection_does_not_prove_dlopen_success` 改为同 nodeid no-dlopen oracle | SPEC GAP |
| ABI-04 | Core fingerprint 边界 | INV-06、10 | v1 canonical material不变；只测试 Core material integrity；plugin native fingerprint 在 operational path 不可达 | Core build-info fingerprint tamper与 wheel independent recomputation；原 plugin mismatch node迁移为 isolation-first rejection | behavior PASS / oracle待迁移 |
| ABI-05 | static RECORD/hash/ELF 防御 | INV-06、07 | 只有 `inspect_native_artifacts()` 作为 evidence-only 低层工具；`validate_backend_plugin()` 与所有 operational APIs必须拒绝手工构造 native record | valid/tamper/undeclared/wheel-tag/SONAME/symbol低层 tests；新增 public validator rejection及 conflict per-record identity/related arrays | PASS/coverage待补 |
| ABI-05A | `python_only` owning distribution 不含 native artifacts | INV-02、06 | pre-import inventory structured reject；不声称拦截依赖或 trusted Python 自行 loader | 新增 `.so/.dylib/.dll/.pyd` inventory negative test、state/counters/public mapping assertions | behavior存在/coverage待补 |
| ABI-06 | capability interface mapping | INV-02、10 | 1.0 opaque exact token；具体保证另建 catalog | opaque/missing capability tests | requirement gap |
| ABI-07 | priority range/tie | INV-03、05、07 | unbounded 1.0；higher wins；top tie stable error | integer extremes、unique/equal priority/order independence | requirement gap / behavior PASS |
| ABI-08 | subprocess IPC/IR | INV-01、02、10 | 1.0 Schema/parser explicit reject；支持另建 IPC/IR ADR | isolation parser/semantic/no-import tests | requirement gap |
| ABI-09 | reader/writer window | INV-01、02、10 | rolling 1.x optional-only；required/narrow/removal 走 major或显式 erratum | 现有 5 个 protocol-field nodes 保留 nodeid，但不得继续把 requirement range 当 producer actual version；按 ADR §12.2 迁移 oracle | requirement gap / oracle待迁移 |
| ABI-10 | diagnostics timeline | INV-04、07、10 | since 1.0、missing `{}`、无 replacement 不弃用 | 同一批 field-policy nodes 与 lifecycle diagnostics tests在 F2 提交重写/移除错误 1.1/1.2/2.0 policy | requirement gap / oracle待迁移 |

批准依赖：ADR D-01～D-12。若选择 native implementation，ADR §11 明确仍不具备可
实施 header/loader contract，必须返回协议门，不能开始代码。

## 5. Explicit rejection 的 oracle 迁移与状态门

批准 `unpublished-freeze-1.0-reject` 或显式 security erratum 后：

- `test_registry_accepts_all_schema_isolation_modes` 保留同一函数 nodeid，改验
  `python_only` ACCEPT、另外两种 mode Schema/parser REJECT；
- illegal-isolation 参数 cases 保留证据，按新 Schema 预期更新并显式冻结 ids；
- native static tests 的 public-parser fixtures 改用低层 immutable inspector fixture；
  static defense tests 不删除；
- 三个 SPEC GAP witness nodeid 分别证明 symbol/attestation/dlopen 在 1.0 均不可达；
- `validate_backend_plugin()` 及 Registry/Triton operational APIs 对正常解析或手工构造
  native/subprocess record 都拒绝；只有低层 inspector 可返回 evidence report；
- 所有拒绝路径断言 `load_calls == initialize_calls == compile_calls == 0`；
- static preflight 失败或 unsupported mode 都不调用 helper/host `dlopen`；
- record 为 REJECTED，不进入 REGISTERED/SELECTED/ACTIVE/public backends mapping；
- 不 xfail、不 skip、不删除 baseline nodeid；新增参数化 cases 可以增加总数。

若 D-01 为 published 且不批准 erratum，上述 oracle 迁移和生产实现均不得发生，本阶段
保持 blocked。

## 6. 范围外回归护栏

| Defect | 完整节点 | 期望终态 |
|---|---|---|
| F4 wheel payload | `tests/acceptance_t63/test_packaging_wheel.py::test_built_wheel_payload_and_provenance` | FAIL |
| F4 package declaration | `tests/acceptance_t63/test_packaging_wheel.py::test_every_source_python_package_is_declared_for_the_wheel` | FAIL |
| F5 public mapping | `tests/acceptance_t63/test_triton_integration_acceptance.py::test_triton_integration_legacy_entry_point_remains_in_public_mapping` | FAIL |
| F5 compiler | `tests/acceptance_t63/test_triton_integration_acceptance.py::test_triton_integration_legacy_compiler_make_backend_regression` | FAIL |
| F5 runtime/reset | `tests/acceptance_t63/test_triton_integration_acceptance.py::test_triton_integration_legacy_runtime_driver_and_reset_regression` | FAIL |
| F6 abstract surface | `tests/acceptance_t63/test_triton_integration_acceptance.py::test_triton_integration_rejects_v30_abstract_classes_before_selection` | FAIL |
| hardware | `tests/acceptance_t63/test_triton_integration_acceptance.py::test_triton_integration_actual_hardware_jit_launch` | BLOCKED |

Triton v3.0/v3.6、T10.2、version pin、Legacy、abstract surface、wheel 漏包均不修改。

## 7. 获批后的验证门禁

每个生产提交先跑其定向节点，然后运行：

```bash
export PYTHONPATH="$PWD/python:$PWD/triton/python"
export PYTHONDONTWRITEBYTECODE=1
export T63_PYTHON=/home/dingbl/race_workspace/t63-acceptance-venv/bin/python

$T63_PYTHON -m pytest -q \
  python/triton_anchor/tests \
  tests/acceptance_t63/test_registry_acceptance.py \
  tests/acceptance_t63/test_schema_oracle_acceptance.py \
  tests/acceptance_t63/test_native_abi_acceptance.py \
  -ra \
  -o cache_dir=/home/dingbl/race_workspace/reports/t6.3-evidence/pytest-cache-common-fixes
```

修复完成时 common-fixes 命令必须 exit 0、全部 PASS。完整复跑使用 baseline manifest
记录的 exact Core wheel、验收 Python、工作树外 cache/JUnit/log，并满足：

- 从 hashed baseline JUnit 提取的 177 个 `(classname, name)` identities 全部仍存在；
- 总数 `N` 下 `N - 7` PASS；
- 仅 F4/F5/F6 共 6 FAIL；
- 仅真实 hardware 1 BLOCKED；
- 0 SPEC GAP / 0 NOT RUN；
- 无 xfail、无放宽、无新增非硬件 skip。

## 8. 提交边界

| 顺序 | 提交信息 | 允许内容 |
|---|---|---|
| 1 | `test: add Triton 3.3 T6.3 acceptance baseline` | **已完成**：仅 acceptance baseline |
| 2 | `docs: freeze T6.3 protocol and native ABI rules` | 仅获批 ADR/矩阵；批准后独立创建 |
| 3 | `fix: align backend manifest parser with schema` | F3 structural/semantic layering + differential tests |
| 4 | `fix: aggregate backend compatibility diagnostics` | F1 evaluator/record + order/schema tests |
| 5 | `fix: enforce backend plugin lifecycle contracts` | F2 hooks/state/concurrency tests |
| 6 | `feat: enforce native backend ABI and load contract` | 推荐路径仅 explicit rejection/no-load contract；不加 ABI sketch |

维护者已确认 1.0 未对外发布并批准推荐 bundle。先将两份 ignored 文档精确 force-add
为第 2 个独立提交，再依次开始第 3～6 个生产提交。
