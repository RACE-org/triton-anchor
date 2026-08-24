# T6.3 Triton 3.6 第一阶段需求—不变量—实现—测试追踪矩阵

- 状态：Accepted；ADR-0001 推荐 bundle 已于 2026-08-21 获维护者批准
- 公共规范来源：`b7449445461406a9381091d389cd7d960b1cd5cb`、`c9ab004e9bbaec34e018600c81f458fe11b3df6f`
- Triton 3.6 验收 baseline：`c4a98fd835cd46fa369e5b5c6ce4fbe200d48cf3`
- Triton 3.6 生产代码 baseline：`735f9ee357e1e698e81785b002c1b39c925008af`
- upstream baseline：`9ceba3e222fd2c9af9cbaf6a440beb23911c197b`
- baseline：157 nodes = 150 pytest PASS / 7 FAIL；语义分类为
  147 PASS / 7 FAIL / 0 BLOCKED / 3 SPEC GAP / 0 NOT RUN
- 最终门：保留 baseline JUnit 全部 157 个唯一 `(classname, name)` identity；新增
  参数化节点后公共总数为 `N` 时，必须为 `N` PASS / 0 FAIL / 0 BLOCKED /
  0 SPEC GAP / 0 NOT RUN

> 本表是 Triton 3.6 环境附录；规范性合同仍由 ADR-0001 唯一定义。baseline identity
> 来自独立 v3.6 JUnit，不使用 v3.3 wheel、LLVM build、JUnit 或 PASS 结论。

## 1. F3：Schema/parser structural domain

| ID | 需求/合同 | 不变量 | 实现边界 | 冻结/新增测试 | baseline |
|---|---|---|---|---|---|
| F3-01 | shipped Schema 与 `parse_manifest` 同域 | INV-01 | `manifest.py` 共享 lightweight predicates；semantic checks 分层；不引入 runtime jsonschema | `test_public_parser_rejects_value_forbidden_by_shipped_schema` | 1 FAIL |
| F3-02 | 所有 `nonEmptyString` 引用一致 | INV-01 | required/optional/version strings、string tuples 共用首尾空白与 CR/LF 规则 | display/vendor、Protocol/Core/Triton/LLVM/MLIR strings、targets/capabilities/requires_capabilities 的空、空白、LF、CRLF、Unicode、普通值 differential | coverage 待补 |
| F3-03 | 所有 anchored patterns 拒绝 trailing CR/LF | INV-01 | Schema 显式排除 CR/LF，不依赖 `$` 行尾；parser 同 predicate | schema_version/plugin_id/entry_point/gitCommit/nonEmptyString/native path differential | coverage 待补 |
| F3-04 | native path predicate 同域 | INV-01、06 | parser 使用 Schema path predicate；containment/RECORD 留在 semantic/static 层 | `./x.so`、absolute、dot/dotdot、backslash、double separator、合法 path | coverage 待补 |
| F3-05 | JSON Schema integer 语义一致 | INV-01 | finite integral JSON number canonicalize 为 `int`；bool/非整数拒绝 | `1`、`1.0`、`1e0`、`1.5`、bool、极值 | coverage 待补 |
| F3-06 | structural string 与 PEP 440 semantic 分层 | INV-01、02 | parser 按 Schema 接受 non-empty；pure semantic validator 在 import 前验证 SpecifierSet | malformed Protocol/Core/Triton/LLVM/MLIR requirements | 部分覆盖 |
| F3-07 | cross-record uniqueness 属 semantic 层 | INV-01、02 | duplicate plugin/entry point 从 parser 移入 import-free semantic validator | Schema/parser ACCEPT + Registry semantic REJECT | coverage 待补 |
| F3-08 | JSON 文本 duplicate member fail closed | INV-02、07 | `object_pairs_hook` 在 object 构造前拒绝 | `test_registry_rejects_duplicate_json_member_names` | PASS |

## 2. F1：完整兼容诊断

| ID | 需求/合同 | 不变量 | 实现边界 | 冻结/新增测试 | baseline |
|---|---|---|---|---|---|
| F1-01 | Protocol/Core/Triton/LLVM 独立错误一次报告 | INV-02、03、07、08 | pure evaluator 返回 checks/errors tuple；Registry 批量记录 | `test_registry_reports_every_simultaneous_compatibility_failure` | 1 FAIL |
| F1-02 | Protocol/Core/Triton version+commit/LLVM version+commit/MLIR version+commit/fingerprint 全覆盖 | INV-03、07 | 每个预期 `BackendPluginError` 独立捕获；不捕获宽泛 `Exception` | 单维与多维矩阵 | 部分覆盖 |
| F1-03 | stable total order 与稳定 record shape | INV-03、07 | ADR §8 rank；所有 keys 永远存在；无随机 repr | 恰好错误集合、重复运行同序、identity/remediation 完整 | coverage 待补 |
| F1-04 | 旧 throwing API 保持首错 | INV-03、07 | identifier/strict/load/register 抛排序第一项；record 保存全部 | strict、identifier、diagnostics 路径 | 部分覆盖 |
| F1-05 | 所有独立检查 pre-load | INV-02、05、08 | distribution-dependent failure 不阻止版本 checks | `load_calls == initialize_calls == compile_calls == 0` 与 public-state gate | 部分覆盖 |
| F1-06 | multi-record conflict identity 稳定 | INV-05、07 | 每个受影响 record 有自身 identity 与排序后的 related ids | duplicate SONAME/symbol/plugin_id/entry-point | 部分覆盖 |

## 3. F2：Python lifecycle contracts

| ID | 需求/合同 | 不变量 | 实现边界 | 冻结/新增测试 | baseline |
|---|---|---|---|---|---|
| F2-01 | optional hooks 可全部缺失 | INV-04 | sentinel；init/shutdown no-op；diagnostics 返回独立 `{}` | 迁移现有 optional-hooks node；验证两次默认值不共享 | oracle 待迁移 |
| F2-02 | initialize callable、同步、返回 None | INV-04、05、09 | signature bind；awaitable/非 None reject；最多一次 best-effort shutdown | noncallable、wrong signature、awaitable、non-None、exception+cleanup | 1 FAIL / coverage 待补 |
| F2-03 | initialize 失败不发布 | INV-05、09 | primary error 在 cleanup error 前；attempt ledger 防重复 cleanup | REGISTERED/SELECTED/ACTIVE/public mapping 均不可见 | 部分覆盖 |
| F2-04 | shutdown 合同、继续清理、幂等 | INV-04、07、09 | sentinel、signature/result gate；反向执行；稳定返回错误 | noncallable、non-None、exception、多插件、重复 reset | 2 FAIL / coverage 待补 |
| F2-05 | diagnostics Mapping snapshot 与生命周期隔离 | INV-04、07 | 缺失 `{}`；`dict(value)` 顶层复制；合同错误只进 diagnostics | noncallable、non-Mapping、exception、自定义 Mapping、mutation snapshot | 2 FAIL / coverage 待补 |
| F2-06 | initialize once、reset race、hook re-entry | INV-05、09 | hook 在 lock 外；generation token；同 hook once | 并发 initialize、reset race、fresh rediscovery、三类 hook re-entry | 部分 PASS / coverage 待补 |
| F2-07 | diagnostics 自 1.0 起存在且无弃用计划 | INV-04、10 | 移除错误的 1.1/1.2/2.0 producer-policy oracle | 保留五个 field-policy node identity 并迁移期望 | oracle 待迁移 |

## 4. Native/Protocol SPEC GAP closure

批准 disposition 是 Protocol/Schema 1.0 explicit rejection，不实现 native loader。

| ID | 规范问题 | 不变量 | 批准合同/实现 | 冻结/新增测试 | baseline |
|---|---|---|---|---|---|
| ABI-01 | 必需 symbol/name/namespace/version/signature | INV-02、06、10 | 1.0 无 native symbol；Schema/parser 先拒绝，测试不得发明名称 | 原 symbol-gap node 改为 unsupported/no-load oracle | SPEC GAP |
| ABI-02 | plugin compiler/stdlib/C++/C++11 ABI 证明 | INV-02、06、10 | 1.0 native 不支持，故不适用；Core fingerprint 不冒充 plugin proof | 原 attestation-gap node 改为 unsupported/no-load oracle | SPEC GAP |
| ABI-03 | `dlopen` timing/flags/init/failure | INV-02、05、06、08、09 | 1.0 禁止 Registry/helper/host dlopen；future loader 另 ADR | 原 dlopen-gap node 改为 constructor/call sentinel | SPEC GAP |
| ABI-04 | Core fingerprint 边界 | INV-06、10 | v1 canonical material不变；只能证明 Core material | build-info tamper；native operational path 不可达 | PASS / oracle 待迁移 |
| ABI-05 | static RECORD/hash/ELF 工具 | INV-06、07 | `inspect_native_artifacts()` 仅 evidence-only；不产生 load authorization | valid/tamper/undeclared/tag/SONAME/symbol 低层 tests | PASS |
| ABI-06 | `python_only` distribution 不含 native artifact | INV-02、06 | pre-import inventory structured reject | `.so/.dylib/.dll/.pyd` 负向矩阵 | coverage 待补 |
| ABI-07 | capability 语义 | INV-02、10 | opaque、大小写敏感、exact token | satisfied/missing/opaque tests | PASS |
| ABI-08 | priority 语义 | INV-03、05、07 | unbounded integer；higher wins；top tie error | integer extremes、unique/equal priority、order independence | 部分 PASS |
| ABI-09 | subprocess IPC/IR | INV-01、02、10 | 1.0 Schema/parser explicit reject；未来另 ADR | parser/Schema/no-import tests | coverage 待补 |
| ABI-10 | reader/writer window | INV-01、02、10 | rolling 1.x optional-only；required/narrow/removal 走 major或 erratum | field evolution oracle | coverage 待补 |

## 5. Explicit rejection 状态门

- `python_only` Schema/parser ACCEPT；`native_in_process`、`subprocess` 同域 REJECT。
- 手工构造 unsupported dataclass 也必须被 `validate_backend_plugin()` 与 Registry 所有
  operational APIs 拒绝；只有低层 inspector 可返回静态 evidence。
- 所有 unsupported 路径断言 `load_calls == initialize_calls == compile_calls == 0`，
  Registry/helper/host `dlopen` sentinel 未触发。
- record 为 REJECTED，不进入 REGISTERED/SELECTED/ACTIVE/public mapping。
- 三个 baseline SPEC GAP node identity 保留并迁移为上述正式 PASS oracle。
- 静态 defense tests 不删除、不跳过、不 xfail。

## 6. 范围外 v3.6 项目

| 项目 | 本阶段分类 | 处理 |
|---|---|---|
| compiler/driver abstract surface、`GPUTarget` | NOT RUN（范围外） | 不修改；不得把 v3.3 surface 带入 v3.6 |
| public backends adapter 与 DriverConfig reset adapter | NOT RUN（范围外） | 只做共性回归，保持 v3.6 实现 |
| Legacy automatic compatibility | NOT RUN（范围外） | 不移植两个 Legacy baseline nodes，不修改生产路径 |
| exact Core wheel payload/generated full profile | NOT RUN（范围外） | 不使用 v3.3 wheel；若本阶段不需 full profile则不构建 |
| hardware JIT | NOT RUN（范围外） | 不运行真实硬件 |
| Triton v3.0/v3.3 生产代码、T10.2 | NOT RUN（禁止范围） | 完全不修改/不导入 |

## 7. 验证门禁

```bash
cd /home/dingbl/race_workspace/triton-anchor-t63-v3.6
export PYTHONPATH="$PWD/python:$PWD/triton/python"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export T63_V36_PYTHON=/home/dingbl/race_workspace/t63-v36-acceptance-venv/bin/python
unset T63_CORE_WHEEL TRITON_PASS_PLUGIN_PATH LLVM_PASS_PLUGIN_PATH

"$T63_V36_PYTHON" -m pytest -q \
  python/triton_anchor/tests \
  tests/acceptance_t63/test_registry_acceptance.py \
  tests/acceptance_t63/test_schema_oracle_acceptance.py \
  tests/acceptance_t63/test_native_abi_acceptance.py \
  -ra \
  -o cache_dir=/home/dingbl/race_workspace/reports/t6.3-v3.6-evidence/pytest-cache-common-fixes
```

每个生产提交先跑对应定向节点，再跑上述 Registry/Schema/Native common gate。最终再加
真实 entry-point wheel sentinel 文件，生成新的 JUnit/log。最终 JUnit 必须包含 baseline
全部 157 identity；新增节点允许增加总数，但公共范围必须全部 PASS，且无 skip/xfail。

## 8. 提交边界

| 顺序 | 提交信息 | v3.6 状态/允许内容 |
|---|---|---|
| 1 | `test: add Triton 3.6 T6.3 common acceptance baseline` | 已完成：`c4a98fd`，仅 A 类 baseline |
| 2 | `docs: freeze T6.3 protocol and native ABI rules` | 从 `b744944`/`c9ab004` 以 `-x` 复用，并仅补 v3.6 环境附录 |
| 3 | `fix: align backend manifest parser with schema` | F3 structural/semantic layering + differential tests |
| 4 | `fix: aggregate backend compatibility diagnostics` | F1 evaluator/record/order/error-schema tests |
| 5 | `fix: enforce backend plugin lifecycle contracts` | F2 hooks/state/concurrency tests |
| 6 | `feat: enforce native backend ABI and load contract` | explicit rejection/no-load contract；不加 ABI sketch |

Triton 3.6 的 Core/Protocol/Schema/Triton/LLVM pin、GPUTarget、compiler、driver 与 build
结构不得由公共提交覆盖。
