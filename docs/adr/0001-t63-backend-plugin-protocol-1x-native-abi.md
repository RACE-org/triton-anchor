# ADR-0001：T6.3 Backend Plugin Protocol 1.0 与 Native 边界治理

- 状态：**Accepted — 推荐 bundle 已获维护者批准**
- 日期：2026-08-20
- 适用对象：triton-anchor Core 0.2.0 的 Triton 3.3/3.6 分支；规范条款不依赖 Triton 专属 Python abstract surface
- 当前版本 pin：Backend Plugin Protocol 1.0、Manifest Schema 1.0
- 公共规范来源提交：`b7449445461406a9381091d389cd7d960b1cd5cb`、`c9ab004e9bbaec34e018600c81f458fe11b3df6f`
- Triton 3.6 验收基线提交：`c4a98fd835cd46fa369e5b5c6ce4fbe200d48cf3`
- Triton 3.6 被测生产代码：`735f9ee357e1e698e81785b002c1b39c925008af`
- 关联追踪矩阵：`docs/adr/0001-t63-common-governance-traceability.md`

> [!IMPORTANT]
> 本文是已冻结合同。维护者已确认 Protocol/Manifest Schema 1.0 未对外发布，并批准
> D-01～D-12 推荐 bundle。后续生产代码和测试必须遵守本文，不能自行发明 native ABI。

## 1. 已批准结论

本阶段不允许修改版本 pin，而 Manifest 1.0 当前接受 `native_in_process` 和
`subprocess`，却没有可实施的 native C ABI 或 subprocess IPC/IR 合同。仓库也不能
证明 1.0 是否已对外发布。

维护者已确认 1.0 **从未对外发布**，并批准：

1. 首次冻结的 Schema/Protocol 1.0 只支持 `python_only`；
2. Schema 与 `parse_manifest` 对 `native_in_process`、`subprocess` 同域拒绝；
3. 拒绝发生在 entry-point import、initialize、任何 helper/host `dlopen` 和 compile 前；
4. Protocol 1.0 没有 native 必需符号、native init/shutdown 或插件工具链 attestation；
5. Core ABI fingerprint v1 保持原 canonical material，但不能被解释为 native 许可；
6. 真正 native 支持退回单独、完整、版本化的 C header/loader ADR，不在本阶段半实现。

这就是本阶段的 **explicit rejection contract**：它通过公开、可测试的“不支持”关闭
Native SPEC GAP，而不是声称现有静态检查已经证明动态 ABI 安全。

若 1.0 已发布，收窄 enum/parser 接受域是 breaking change。维护者只能明确批准
security erratum，或将本阶段标为协议阻塞。若维护者希望本阶段实现 native 1.0，
本文必须先补成可编译的 normative C header 并重新审批；本 ADR 不能授权该路径。

## 2. 当前事实与问题陈述

### 2.1 已存在的公开 surface

- `BackendPlugin` 1.0 的必需 Python surface 是 `compiler_cls`、`driver_cls`；
- `BackendPluginBase` 已公开 `initialize(context) -> None`、`shutdown() -> None`、
  `diagnostics() -> Dict[str, Any]`；
- Schema `$id` 为 `backend-manifest-1.0.json`，`schema_version` 接受所有 `1.x`；
- `backend_protocol` 是插件对 Core consumer version 的 **SpecifierSet requirement**，
  不是插件 producer 的实际 conformance version；
- Schema 1.0 enum 接受 `python_only`、`subprocess`、`native_in_process`；
- native static inspector 验证 distribution/RECORD/hash、ELF、SONAME、依赖和导出
  清单，但没有必需入口符号，也不执行 `dlopen`/`dlsym`；
- compatibility 到运行时才拒绝 `subprocess`；
- Core ABI fingerprint v1 包含所列 Core ABI material 与精确 `libtriton.so` hash；
- `DIAGNOSTICS_FIELD_POLICY` 的“1.1 引入、1.2 弃用、2.0 删除”与 1.0 public base
  method 冲突。

### 2.2 Schema/parser 已知差异

F3 不只是一条 `display_name` newline 特例。当前至少存在：

| 输入 | Shipped Schema | `parse_manifest` | 根因 |
|---|---|---|---|
| `display_name: "a\nb"` | REJECT | ACCEPT | `nonEmptyString` regex 与 `strip()` 不等价 |
| anchored field 为 `"value\n"` | ACCEPT | REJECT | 多个 Schema pattern 使用 `$`，会在最终 LF 前匹配；parser 使用 `fullmatch()`/`strip()` |
| `priority: 1.0` | ACCEPT | REJECT | JSON Schema integer 是数学整数；Python 实现只接受 `int` |
| `backend_protocol: "banana"` | ACCEPT | REJECT | Schema 只声明 non-empty；parser 混入 PEP 440 semantic check |
| `native_libraries: ["./x.so"]` | REJECT | ACCEPT | Schema path regex 与 `PurePosixPath` 检查不等价 |
| 重复 `plugin_id`/`entry_point` | ACCEPT | REJECT | 跨 record 唯一性无法由现有标准 Schema 表达，却在 parser 内执行 |

因此 INV-01 的修复必须先划清层次：`parse_manifest` 与 shipped Schema 负责同一个
**structural acceptance domain**；PEP 440、跨 record identity、distribution coverage
等 semantic checks 在独立、仍然 import-free 的 pre-load validator 中完成。不能通过
给 runtime 引入 jsonschema 依赖来掩盖两个 oracle 的差异。

## 3. 信任模型与非目标

`native_in_process` 一经加载便获得 Core 进程权限。Manifest、RECORD、ELF 检查和
self-attestation 只能证明包内一致性与兼容性线索，不能把恶意 DSO 变成沙箱。签名
provenance 才能把声明绑定到可信 builder；即便有签名，进程内代码仍是 trusted code。

本阶段非目标：

- 不承诺 Triton/LLVM/MLIR C++ ABI 永久稳定；
- 不实现 native 或 subprocess backend；
- 不修改 Protocol/Schema/Core/Triton/LLVM 版本 pin；
- 不处理 wheel payload、Legacy 行为或任一 Triton 版本的 compiler/driver abstract surface；
- 不处理真实硬件 JIT、其他 Triton 分支生产代码或 T10.2。

## 4. 强制系统不变量

| ID | 不变量 |
|---|---|
| INV-01 | `parse_manifest` 的 structural acceptance domain 必须与随包发布的对应 JSON Schema 完全一致；semantic validator 不得改变这一事实。 |
| INV-02 | 不兼容 Manifest 插件必须在 Python import、initialize、任何 Registry 发起的 `dlopen` 和 compile 前拒绝。 |
| INV-03 | 对 structural-valid plugin record，同一插件所有相互独立的兼容错误必须一次完整收集；只有真正依赖失败前置的检查可标为不可执行。 |
| INV-04 | optional hook 可以缺失；存在时必须同步 callable，参数与返回值符合公开合同。 |
| INV-05 | 失败插件不得进入 `REGISTERED`、`SELECTED`、`ACTIVE` 或公开 backends mapping。 |
| INV-06 | 若未来支持 native，所有候选必须先完成静态 Manifest、RECORD、hash、ELF、符号和 ABI metadata 检查，任何 DSO 才能动态加载。 |
| INV-07 | 所有 error 必须结构化、总排序，并始终含 `plugin_id`、`field`、`dimension`、`expected`、`actual`、`remediation` keys。 |
| INV-08 | discovery、structural/semantic validation、conflict inspection 和 diagnostics 不得隐式 import entry point。 |
| INV-09 | reset 必须继续清理其他插件；每个 hook 每个 generation 至多一次；initialize/reset 竞争不得发布过期实例。 |
| INV-10 | Manifest Schema、Backend Protocol、Native ABI 与 Core fingerprint 是四个独立版本域。 |

## 5. 版本模型与 1.0 发布决策

### 5.1 四个版本域

1. `schema_version` 标识 JSON document contract。当前 1.x 采用一个 rolling major
   Schema：所有 1.x 共享同一接受域，只允许 optional/additive extension。
2. Core 的 `BACKEND_PLUGIN_PROTOCOL_VERSION` 是 consumer 实际版本。
3. Manifest 的 `backend_protocol` 是该 consumer version 的 compatibility range，
   不能充当 producer actual version，也不能驱动字段 introduced/deprecated 判断。
4. Native ABI 若未来存在，必须有独立实际版本与规范 header。
5. Core ABI fingerprint schema 只版本化 digest material，不等于上述任一协议版本。

本阶段选择 rolling 1.x。D-02 先通过“首次发布冻结”或显式 erratum 确定新的 1.0
baseline shape（包括 isolation enum 与 pattern anchor 修正）；**从该获批 shape 生效后**，
1.x 不得再增加 required field、收窄字段、改变 enum 语义或增加旧 reader 无法安全
理解的新行为。此类后续修改走新 Schema major。若未来改用 exact-minor dispatch，
必须另建迁移 ADR；不能一边让 1.0 parser 接受 1.99，一边悄悄发布不同 1.1 shape。

### 5.2 无法从仓库推出发布事实

版本常量、文件名和本地 wheel 不足以证明第三方是否消费过 1.0；该事实只能由发布
责任人确认。维护者已在 D-01 明确记录 `unpublished`；`unknown` 不是可以破坏兼容性的
答案。

### 5.3 本阶段可选 disposition

| 选择 | 前提 | 行为 | 是否可进入代码阶段 |
|---|---|---|---|
| `unpublished-freeze-1.0-reject` | 维护者确认从未发布 | 首次冻结为 python-only；同步 Schema/parser | **推荐，可以** |
| `unpublished-return-for-native-1.0-contract` | 从未发布但要求 native | 先补全 C header、toolchain/loader/lifecycle 合同并重新审批 | **不可以，继续停门** |
| `published-security-erratum-reject` | 已发布且维护者接受 breaking security correction | release note/迁移诊断；同步收窄 | **明确批准后可以** |
| `published-immutable-blocked` | 已发布且不可变或状态 unknown | 不改 1.0，另开 major migration ADR | **不可以，本阶段 blocked** |

不存在“只要批准就把 pin 改成 1.1”的本阶段选项；用户未授权修改版本 pin。新的
Schema minor 也不能修复旧 1.0 已经接受的 document domain。

## 6. Manifest 1.x structural 与 semantic 合同

### 6.1 Structural oracle

Shipped Draft 2020-12 Schema 是 structural oracle；`parse_manifest` 必须对相同 JSON
instance 得出相同 ACCEPT/REJECT。实现可以共享轻量 predicate，但运行时不依赖
jsonschema。

依据已批准的 D-02，冻结后的 `nonEmptyString` 规范语义为：

- 必须是非空 string；
- 首尾不得是 Schema regex 认定的 whitespace；
- 任意位置的 LF/CR（包括 CRLF）均拒绝；
- 普通内部空格合法；
- 同一 predicate 用于 required/optional strings、version strings、targets、
  capabilities、requires_capabilities 与 native path 的 string envelope。

这不是对当前 Schema 文本的错误描述：现有 `$` anchor 会接受某些 trailing LF。
获批实现必须在所有 anchored patterns（至少 schema_version、plugin_id、entry_point、
gitCommit、nonEmptyString、nativeLibraryPath）显式排除 CR/LF，不能依赖 `$` 的行尾
行为；parser 使用同一规则。若 1.0 已发布，这也是接受域收窄，必须由 D-02 的
security/correctness erratum 覆盖；immutable/unknown disposition 下不得实施。

F3 differential tests 必须覆盖所有 `$defs/nonEmptyString` 引用、native path、Unicode
边界、空/首尾空白/LF/CRLF/普通值，逐项断言 Schema 与 parser 同结果。

其他已知差异按 Schema oracle 对齐：

- JSON Schema 判定为 integer 的有限数学整数（包括 JSON `1.0`）由 parser 接受并
  canonicalize 为 Python `int`；boolean 继续拒绝；
- version requirement 在 parser 只执行 non-empty structural check；PEP 440
  `SpecifierSet` validation 移入 semantic validator，并仍在 import 前拒绝；
- native path parser 使用与 Schema 相同的 path predicate；额外 path containment、
  symlink 和 RECORD 规则属于 semantic/static-native 层；
- 跨 plugin `plugin_id`/`entry_point` uniqueness 移入 semantic validator，因为
  当前标准 Schema 不表达 unique-by-key。

未知 root/plugin/triton-requirement properties 继续接受并保存在 `extensions`。JSON
文本层 duplicate member names仍由 loader 在构造 object 前拒绝；它不属于已经丢失
member identity 的 in-memory JSON Schema instance。

### 6.2 Semantic validator

Semantic validator 是纯函数、import-free，并返回确定排序的 errors tuple。它负责：

- PEP 440 requirements；
- plugin/entry-point 唯一性与 owning distribution coverage；
- owning distribution 的 pure-Python/native inventory 等 Schema 无法表达的规则；
- Core/Protocol/Triton/LLVM/MLIR/SOABI/platform/capability constraints；
- RECORD/path/ELF checks（仅 defense-in-depth；1.0 operational mode 仍拒绝 native）。

unsupported `isolation_mode` 属 structural Schema/parser rejection，不进入 compatibility
evaluator。手工构造 dataclass 绕过 parser 时，所有 operational public APIs 仍必须先
执行同一 unsupported guard。

需要抛异常的 public API 可以抛排序后的第一项；Registry record 必须保存全部项。

## 7. Backend Plugin Protocol 1.0 Python 合同

本节只适用于 Manifest plugin；不改变范围外的 Legacy bypass 行为。

### 7.1 Required runtime surface

- `compiler_cls: type`
- `driver_cls: type`

两者在 entry point load 后、initialize 前验证。具体 Triton 分支的 abstract-method
完整性属于版本专属验收，不在本阶段改变。

### 7.2 Optional synchronous hooks

- `initialize(context: Mapping[str, Any]) -> None`
- `shutdown() -> None`
- `diagnostics() -> Mapping[str, Any]`

合同如下：

1. 使用私有 sentinel 区分属性缺失与存在值；缺失 initialize/shutdown 是 no-op，
   缺失 diagnostics 返回独立空 `dict`。
2. 存在但非 callable 是 structured lifecycle error。
3. hooks 是同步合同；返回 coroutine/awaitable 不会被 await，按错误返回处理。
4. 对可 introspect callable，调用前用 signature binding 验证 `initialize(context)` 或
   zero-argument hook；无法 introspect 时以实际调用为准，`TypeError` 转 lifecycle error。
5. initialize context 是只读 Mapping，固定 keys 为 `environment`、`manifest`、
   `record_id`；前两者是 immutable snapshot，生命周期至少覆盖该 Registry generation。
6. initialize 返回值必须是 `None`。异常或非 None 后插件不得发布；最多一次
   best-effort Python shutdown，primary error 排在 cleanup error 前。
7. shutdown 返回值必须是 `None`。非 callable、异常、非 None 都产生 reset error；
   一个插件失败不阻止其余插件；同 generation 不重复调用。
8. diagnostics 返回值只要求是 `Mapping`。Core 立即执行 `dict(value)` 得到新的普通
   `dict`，使插件随后对原 mapping 的顶层增删改不污染快照；1.0 不对任意 values
   静默增加 JSON-serializable 限制。异常/合同错误只影响 diagnostics，不改变
   lifecycle/selection。
9. hook 调用在 Registry internal lock 外进行；同 generation initialize once；reset
   通过 generation token 阻止 stale publish；hook 内 Registry re-entry 不死锁或重入
   同一 hook。

### 7.3 State 与 selection

首次成功发布路径是：

`DISCOVERED -> VALIDATED -> LOADED -> REGISTERED`

纯静态 selection decision 可以在 load 前决定 winner，但公开 `SELECTED` 只能在该
winner 成功 REGISTERED 后发布；`ACTIVE` 只能来自 SELECTED。取消选择/reset 允许
`SELECTED -> REGISTERED`、`ACTIVE -> REGISTERED`。`REJECTED` 对当前 generation
终止；rediscovery 产生新 generation。任何失败先从 public mapping/runtime cache 脱离。

## 8. 完整兼容诊断合同

### 8.1 Stable record shape

本阶段沿用并冻结以下 code catalog：

- `backend_plugin_manifest_error`
- `backend_plugin_discovery_error`
- `backend_plugin_protocol_error`
- `backend_plugin_compatibility_error`
- `backend_plugin_capability_error`
- `backend_plugin_conflict_error`
- `backend_plugin_load_error`
- `backend_plugin_interface_error`
- `backend_plugin_selection_error`
- `backend_plugin_lifecycle_error`

每条 `to_dict()` 始终包含：

| key | type/rule |
|---|---|
| `code` | 上述 stable string |
| `message` | stable human summary，不参与机器排序 |
| `plugin_id` | string；root/discovery 阶段无法得知时为 `null`，不得猜测 |
| `entry_point` | string 或 `null` |
| `field` | Manifest/Python field string 或 `null` |
| `dimension` | compatibility dimension string 或 `null` |
| `expected` | deterministic string 或 `null` |
| `actual` | deterministic string 或 `null`；禁止包含地址、随机 repr、时间戳 |
| `remediation` | actionable string 或 `null` |
| `detail` | deterministic string 或 `null` |

plugin record 已读出 `plugin_id` 后，其所有后续 errors 必须非 null。INV-07 的“包含”
指 key 永远存在；无法安全归属的 root error 使用 JSON null，而不是虚构 identity。
`field` 与 `dimension` 至少一个非 null。Manifest values 用 canonical JSON rendering，
runtime objects 用稳定 qualified type name；不得把默认 `repr(object)` 地址写入 `actual`。

多插件 conflict 不得只返回一条 `plugin_id=null` 的模糊错误：每个受影响 record 获得
一条带自身 `plugin_id`/`entry_point` 的 specialized error，并以排序后的
`related_plugin_ids`、`related_record_ids` arrays 描述其他参与者。只有不归属任何
record 的 Registry/root summary 才允许 null identity；summary 不替代逐 record errors。

### 8.2 Total order

单 plugin compatibility errors 按以下 `(rank, field)` 总排序：

| rank | field/dimension |
|---|---|
| 10 | distribution ownership/provenance/platform tag |
| 20 | RECORD/path/hash inventory |
| 30 | native ELF metadata（未来适用；1.0 reject path 不进入） |
| 40 | `backend_protocol` |
| 50 | `requires_core` |
| 60 | `requires_triton.version` |
| 61 | `requires_triton.commit` |
| 70 | `requires_llvm_version` |
| 71 | `requires_llvm_commit` |
| 80 | `requires_mlir_version` |
| 81 | `requires_mlir_commit` |
| 90 | built/runtime Python SOABI |
| 91 | Core build/runtime platform |
| 100 | Core ABI fingerprint（未来 native） |
| 110 | `requires_capabilities` |
| 111 | kernel capability requirements |

完整 sort key 为 `(rank, canonical_field_or_dimension, code)`；rank 10/20 内也按已公开
canonical string 排序，不按 message/`repr()`。缺失 distribution 不阻止
Protocol/Core/Triton/LLVM 等独立 checks；只有确实需要 distribution bytes 的 checks
标为 unavailable。F1 必须验证恰好四个指定错误、重复运行顺序一致、所有 stable keys
与 remediation 完整、`load_calls == 0`。

批量 reset cleanup 的**执行顺序**按成功初始化的反序；返回的 cleanup errors 为稳定
consumer view，按 `plugin_id`、hook rank（initialize cleanup、shutdown）排序。
initialize primary error 始终先于该插件 cleanup errors。批量 diagnostics record 则按
`registry_key` 排序；diagnostics error 不进入 reset cleanup 序列。

## 9. Capability、priority、subprocess 与 diagnostics

### 9.1 Capability

Protocol 1.0 capability 是 opaque、大小写敏感、精确匹配 token：

- 不推导 alias、层级、蕴含或 version range；
- `requires_capabilities` 只由 Core-provided set 满足；
- kernel requirements 由 `Core ∪ selected plugin` 满足；
- 缺失在 import 前拒绝；
- capability 不替代 `compiler_cls`/`driver_cls` interface checks，也不等同
  `HWCapability`。

Core-owned capability 若要保证具体接口，必须另发 versioned catalog 与 executable
validator；在此之前 vendor capability 是 contractual self-claim。

### 9.2 Priority

1.0 保留 Schema 的无上下界 JSON integer：默认 `0`，boolean 非 integer；在 target、
compatibility、capability 过滤后，较大值优先。Python explicit selector > environment
selector > priority。最高值并列必须稳定报 ambiguity，禁止 lexical、安装顺序或
discovery-order fallback。跨 vendor priority 是 operator-overridable hint，不是安全证明。

### 9.3 Subprocess

1.0 明确拒绝 subprocess。未来支持需独立 ADR 至少定义 executable/RECORD binding、
handshake、IPC framing、IR/dialect versions、diagnostics、exit/crash/timeout/cancel、
process-group cleanup、environment/cwd/FD/resource policy。`native_libraries` 不是 process
endpoint 或 IR contract。

### 9.4 Diagnostics evolution

`diagnostics` 自 Protocol 1.0 起是 optional hook，缺失默认空 mapping；没有 replacement
前不弃用。未来 deprecation 必须先有 replacement/迁移 ADR，在至少一个完整受支持
minor 发 warning，并只在下一个 major 删除。当前 1.1/1.2/2.0 policy 不获冻结。

## 10. Protocol 1.0 Native explicit rejection contract

若批准推荐 disposition，本节是完整 normative native 规则：

| 问题 | Protocol/Schema 1.0 决定 |
|---|---|
| 支持的 isolation | 仅 `python_only` |
| Native 必需 symbol/name/namespace/version/signature | **不存在**；1.0 不允许 native DSO，因此测试不得发明名称 |
| plugin compiler/stdlib/C++ standard/C++11 ABI 声明 | **不存在且不适用**；不允许用自报 Core fingerprint 代替 |
| Core ABI fingerprint | 保持 v1 digest material；不能使被拒绝 mode 变成可加载 |
| static native inspection | 可作为 defense-in-depth 工具单测，但不产生 operational compatibility |
| helper/host `dlopen` | **禁止**；没有 RTLD flags 或 unsafe escape hatch |
| native init/shutdown | **不得调用**；无返回值或 cleanup contract |
| Python import/initialize/compile | unsupported error 后均不得发生 |
| Registry/public state | record 为 REJECTED；不得 REGISTERED/SELECTED/ACTIVE/public mapping |

`python_only` 的静态含义限定为：拥有 Manifest/entry point 的 distribution 本身不得
包含 `.so/.dylib/.dll/.pyd` native artifacts，Registry 在 import 前检查。它不是进程
沙箱；受信任 Python 代码或第三方 dependency 仍可能自行调用系统 loader。该行为若被
用来隐藏 plugin-owned native ABI 即违反插件合同，但只有进程隔离才能对恶意插件强制
阻断。本文的 no-`dlopen` 断言指 Registry 在 unsupported rejection 路径没有发起任何
helper/host load，不声称拦截任意 Python 代码的系统调用。

Schema 与 parser 的 unsupported error 使用 `backend_plugin_manifest_error`，
`field="isolation_mode"`，`expected="python_only"`，`actual` 为原 mode，remediation 指向
python-only 或未来有版本合同的 release。plugin_id 已可解析时必须包含。相同输入的
Schema/parser ACCEPT/REJECT 必须一致；Registry 不得采用“parser 接受、compatibility
晚拒绝”的模式。

公开导出的 `validate_backend_plugin()` 以及 Registry validate/load/register/select/
activate、Triton compiler/runtime adapter 都属于 operational APIs。即使调用者手工构造
`BackendPluginManifest` 绕过 parser，它们也必须先返回同一 unsupported error；不能让
native record 获得 `compatible=True`。只有 `inspect_native_artifacts()` 是
evidence-only 低层工具，可以返回静态事实；它的 report 永远不代表 operational
compatibility 或 load authorization。

## 11. Future Native ABI：非规范性设计约束

本节只记录未来 ADR 的入口条件，**不能批准 native implementation**。

若以后选择最小版本化 C handshake，候选 query 名可以是
`triton_anchor_backend_query_v1`，但在以下内容形成已安装、可编译的 normative C
header 前，该名称没有公开合同效力：

- 所有 struct 的固定 layout/alignment/calling convention；
- string encoding、length、ownership、lifetime；
- Core API、plugin descriptor、error buffer 的完整定义；
- Native major/minor negotiation、minimum `struct_size`、status code namespace；
- process singleton 或 instance/context ownership；
- initialize/shutdown 线程、重入、失败回滚和 exactly-once 规则；
- GNU symbol version 是否禁止/要求、其他 exports namespace、SONAME 唯一范围；
- RPATH/RUNPATH/`DT_NEEDED` allowlist；
- per-DSO attestation inventory、canonical JSON、`attestation_sha256`、ELF note byte
  layout 与 descriptor cross-check；
- compiler ID/version/target、stdlib ID/version/ABI、C++ standard/C++11 ABI、RTTI/
  exceptions、data model/endian、header/SDK digests 与 ABI definitions；
- Core fingerprint v2 material和 v1/v2 migration。

Loader ADR 还必须解决：

1. 所有候选先 static validate，纯静态选 winner，只有 winner 可进入动态阶段；
2. `dlopen` 会执行 constructors/IFUNC，故 helper load 已是插件代码执行，不能宣称
   “所有动态错误在任何 dlopen 前已证明”；
3. `RTLD_NOW | RTLD_LOCAL` 只是候选策略，须定义 constructor/IFUNC 禁止策略；
4. helper 后 re-stat/hash 不能完全消除 TOCTOU，需 FD pinning/immutable package strategy；
5. Python entry point 可能自行加载 extension DSO；Registry 若不是所有声明 DSO 的唯一
   loader，就无法保证 flags/handle ownership；
6. no-`dlclose`、crash/timeout、host reload 与 reset semantics。

由于当前 Triton/MLIR integration 共享广泛 C++ types，即使 query 是 C，边界也仍是
exact-build C++ coupled，不是永久稳定 C ABI。RECORD + self-attestation 不是签名证明。

## 12. Field evolution 与 reader/writer 窗口

### 12.1 Manifest rolling 1.x

| 变化 | 1.x 规则 |
|---|---|
| 新 optional property | 允许；旧 reader 忽略并保存在 extensions |
| 新 required property | 禁止；走新 major |
| 接受域收窄/enum 语义改变 | 禁止；走新 major，或显式 security erratum |
| property deprecation | 新 writer 停发；所有 1.x reader 继续接受并 warning |
| property removal | 只在下一个 major，且至少一个完整受支持 minor warning window |
| major mismatch | import 前结构化拒绝 |

当前 `schema_version` minor 不做 feature dispatch；所有 1.x 共用相同 shape/semantics。
因此未来 native 不得以“Schema 1.1 给既有 mode 加 conditional required fields”方式
进入 rolling 1.x。

### 12.2 Backend Protocol

Manifest `backend_protocol` 只是 consumer range。1.0 没有 producer actual-version 字段，
所以 optional hook 按 structural presence/default 消费，不能用 requirement range决定
introduced/deprecated。未来若需要 producer field negotiation，必须先新增明确的 actual
conformance version，并按 Schema 兼容规则发布。

### 12.3 Native 与 Core fingerprint

Native ABI future major/minor 由其 header 定义，append-only minor 规则不能由本文的
不完整 sketch 推导。`triton-anchor-core-abi-v1` 是对下列 canonical material 与精确
library bytes 的 digest：Core/Triton/LLVM/MLIR versions/commits、Core compiler/C++
facts、Python SOABI/platform、TTGPU 与 `libtriton.so` hash。它不是整个 build 的证明，
不覆盖 header tree、stdlib/target/RTTI 等未列材料，不证明 plugin facts 或来源。
不得在同名 v1 material 中追加字段。

## 13. 决策矩阵

下表记录维护者批准值。

| ID | 决策 | 可选值 | 建议 | 批准值 |
|---|---|---|---|---|
| D-01 | 1.0 发布事实 | `unpublished` / `published` / `unknown` | 必须给事实；unknown 时 blocked | `unpublished` |
| D-02 | 本阶段 disposition | §5.3 的四个值 | `unpublished-freeze-1.0-reject`；若 published 则优先 immutable-blocked | `unpublished-freeze-1.0-reject` |
| D-03 | Manifest 1.x 模型 | `rolling-major` / `exact-minor-dispatch` | 冻结当前 rolling-major；required/narrow 走 major | `rolling-major` |
| D-04 | Schema/parser layering | `schema-structural-oracle+semantic-preload` / 其他 | 前者；修复 §2.2 全部已知差异并做 schema-driven parameterized/property differential coverage | `schema-structural-oracle+semantic-preload` |
| D-05 | Python lifecycle | `approve-§7` / 修订 | 批准同步 hooks、空 diagnostics snapshot、generation rules | `approve-§7` |
| D-06 | Error schema/order | `approve-§8` / 修订 | 批准 stable keys/catalog/total order | `approve-§8` |
| D-07 | Capability | `opaque-1.0` / `catalog-bound` | opaque；接口保证另建 catalog | `opaque-1.0` |
| D-08 | Priority | `unbounded-1.0,tie-error` / 收窄 erratum | 保留 unbounded 与 tie error | `unbounded-1.0,tie-error` |
| D-09 | Subprocess | `reject-1.0;future-separate-ipc-adr` / return-to-design | 现在 reject；未来支持必须另建 IPC/IR ADR | `reject-1.0;future-separate-ipc-adr` |
| D-10 | Diagnostics | `since-1.0,missing-empty-map,no-removal` / 修订 | 前者 | `since-1.0,missing-empty-map,no-removal` |
| D-11 | Native unsafe escape hatch | `none` / explicit opt-in | none | `none` |
| D-12 | Future Native status | `deferred-separate-ADR` / return-to-expand-this-ADR | deferred；本阶段只实现 explicit rejection | `deferred-separate-ADR` |

### 推荐批准 bundle

仅当 D-01 的事实为 `unpublished` 时，可用一句“批准推荐 bundle”表示：

```text
D-01=unpublished
D-02=unpublished-freeze-1.0-reject
D-03=rolling-major
D-04=schema-structural-oracle+semantic-preload
D-05=approve-§7
D-06=approve-§8
D-07=opaque-1.0
D-08=unbounded-1.0,tie-error
D-09=reject-1.0;future-separate-ipc-adr
D-10=since-1.0,missing-empty-map,no-removal
D-11=none
D-12=deferred-separate-ADR
```

若 D-01 为 published/unknown，不能使用该快捷批准。

除 §5.3 已完整描述的 blocked/security-erratum disposition 外，选择任一非推荐值都表示
本 ADR 缺少相应规范，必须退回修订并重新审批，不能仅填写该值后直接开始生产代码。

## 14. 批准后的提交与验证边界

批准后仍按独立提交推进：

1. `docs: freeze T6.3 protocol and native ABI rules`
2. `fix: align backend manifest parser with schema`
3. `fix: aggregate backend compatibility diagnostics`
4. `fix: enforce backend plugin lifecycle contracts`
5. `feat: enforce native backend ABI and load contract`

第 5 个提交在推荐 disposition 下只实现 Schema/parser 同域 unsupported rejection、
structured error 与 no-import/no-dlopen/no-init/no-compile 状态门；不得加入 query symbol、
loader helper 或版本 pin 修改。

Triton 3.6 公共 baseline JUnit 中的 157 个 testcase identity 必须全部保留。门禁直接
比较 JUnit 的 `(classname, name)` tuple，不从 XML 猜测路径式 pytest nodeid。参数化
differential/lifecycle 测试可增加节点；最终公共套件总数为 `N` 时必须为 `N` PASS、
0 FAIL、0 BLOCKED、0 SPEC GAP、0 NOT RUN。版本专属、Legacy、wheel payload 和硬件
节点不在公共分母中，另行记录，不得借其状态稀释公共结果。

## 15. 维护者批准记录

- 审批角色：**维护者；姓名或公开 handle 未提供**
- 审批日期：**2026-08-21**
- D-01～D-12 批准值：**§13 推荐 bundle，逐项记录于决策矩阵**
- 1.0 发布事实的证据/责任人：**维护者在本任务会话书面确认“确认上述发布状态并批准推荐 bundle”；发布事实由维护者负责**
- 是否接受 security erratum：**不适用；D-01 为 unpublished**
- 备注/偏离：**无**

本 ADR 自上述批准记录起生效。后续实现必须按 §14 的独立提交和验证边界推进。

## 16. Triton 3.6 环境附录（非规范性）

Triton 3.6 的独立适配事实为：Triton `3.6.0`，vendored commit
`6cc4505027d7b39fe18a44a7f89085b8babb7400`，LLVM/MLIR commit
`a992f29451b9e140424f35ac5e20177db4afbdc0`（fixture version `22.0.0git`）。这些 pin
只进入 v3.6 fixture/environment，不改变 Protocol 1.0、Manifest Schema 1.0、错误顺序、
lifecycle 或 explicit-rejection wire contract。

v3.6 源码树未提供匹配的已构建 `triton._C.libtriton`，因此直接 `import triton` 的
源码环境探针失败；公共 Registry/Schema/Native 测试不导入 Triton。该 wheel/build
环境事实不授权复用 v3.3 wheel、LLVM build-info 或 JUnit，也不把版本专属 abstract
surface 引入公共规范。
