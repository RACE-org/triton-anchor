# ADR-0002 Conformance Annex：T6.3 / Triton 3.6 发布门禁候选证据

- Annex 状态：**Second fresh rerun proposed evidence — pending new evidence commit and binding review**
- ADR 决策状态：`APPROVED`
- 实现绑定：`UNBOUND`
- Release 状态：**NOT ACCEPTED**
- 证据日期：2026-08-24
- 适用 ADR：`0002-t63-triton-v36-release-gates.md`
- `PRODUCTION_CANDIDATE_SHA`：`b06daeb4042aa61e4c729bf05198e523ae368622`
- `PRODUCTION_CANDIDATE_TREE`：`74d8b6b8e863cc0474618e556ce78b0b3adfa10b`
- `EVIDENCE_COMMIT_SHA`：`PENDING_BINDING`
- 上一轮 docs-only evidence commit：`6bbc65d6933633b0ba5ae4822dfbe1e487e261b9`
- 硬件分类：`BLOCKED`
- 当前结论：**所有必需软件执行门禁 PASS；等待第二次绑定确认**

> 当前 tracked annex 来自上一轮 docs-only `E=6bbc65d…`；该提交不是第二轮 fresh evidence
> commit，不能被本轮绑定。本文更新后仍须形成满足 ADR topology 的新 docs-only `E`，且不
> 自写 `E` 或 `A` 的 commit SHA。发布责任人确认 exact candidate、artifact、新 `E` 与本轮
> fresh evidence 后，才允许由 `A` 写入 `EVIDENCE_COMMIT_SHA=E`，把 ADR 改为
> `APPROVED + BOUND / Accepted`。

## 1. 结论与状态边界

| 范围 | 结果 | 说明 |
|---|---|---|
| 必需软件执行门禁 | PASS | 638/638 PASS，0 FAIL、0 ERROR、0 SKIP、0 XFAIL |
| `ART-WHEEL-001` | PASS | fresh wheel 的 F4/ELF/RECORD/install/smoke 全部通过 |
| `QA-CI-001` | PASS | Ruff 0.15.22 真实四门、CI policy 与 100 个回归节点全部通过 |
| `AUTH-F4/F5/F6` | CLOSED | 第一次批准已关闭规范设计缺口，当前实现测试通过 |
| `PKG-TESTS` | CLOSED | production wheel 与 RECORD 中 tests member 为 0 |
| `AUTH-SHA` | **SPEC GAP** | 仍需发布责任人第二次确认 `I`、`E`、wheel 与新鲜证据 |
| 真实硬件 | **BLOCKED** | 无匹配设备与 candidate hardware backend，未执行 vector-add |
| 当前 release 结论 | **NOT ACCEPTED** | `APPROVED + UNBOUND`；不能写 FULL/CONDITIONAL PASS |

因此，本 annex 证明可执行软件范围 `FAIL=0` 且没有未决的软件 oracle/实现缺口；正式
release authority 仍为 `SPEC_GAP=1 (AUTH-SHA)`。若第二次确认完成并形成绑定提交 `A`，
由于硬件为 BLOCKED，最终结论上限是 **CONDITIONAL PASS**，不得是 FULL PASS。

第一次批准主体记录为“T6.3 发布责任人/维护者；姓名或公开 handle 未提供”，日期为
2026-08-24，批准范围为 D-01～D-12 与 ADR 全文；原始确认语句保存在 ADR-0002 §15。
本 annex 不补写未提供的身份，也不把第一次设计授权扩大为第二次 release acceptance。

## 2. 候选身份、范围与版本 pin

| 字段 | 值 |
|---|---|
| branch | `fix/t6.3-v3.6-release-gates` |
| upstream | `9ceba3e222fd2c9af9cbaf6a440beb23911c197b` (`upstream/triton_v3.6`) |
| 修复起点 | `609c3c7d1f4e6032e8b1399531d1c998b970fdb8` |
| candidate `I` | `b06daeb4042aa61e4c729bf05198e523ae368622` |
| candidate tree | `74d8b6b8e863cc0474618e556ce78b0b3adfa10b` |
| 上一轮 docs-only `E` | `6bbc65d6933633b0ba5ae4822dfbe1e487e261b9`；`E^=I`，只新增本 annex |
| Core | `0.2.0` |
| Protocol / Schema | `1.0` / `1.0` |
| Triton | `3.6.0` |
| vendored Triton commit | `6cc4505027d7b39fe18a44a7f89085b8babb7400` |
| LLVM/MLIR commit | `a992f29451b9e140424f35ac5e20177db4afbdc0` |
| LLVM observed version | `22.0.0git` |
| MLIR observed version | `22.0.0`；commit 与 LLVM pin 相同 |
| F5 real Legacy fixture | SHA-256 `97402abf825cf56d800edf5f4cfa25a8513be585b2b067391d0766664d18a455` |

`upstream` 是 `I` 的祖先。第二轮所有构建、pytest、coverage、Ruff 与 F4 输入均为 `I` 的
fresh archive 或 clean candidate clone，绝不以 shared `HEAD=6bbc65d…` 作为候选。上一轮
`E` 的 tree 为 `d71e0befa9d21a02c111b18b350add87e1fc47c3`，与 candidate tree 不同，仅因其新增
本 annex；它在这里仅作 topology/deny-reference 记录。`609c3c7..I` 未修改
Core/Triton/vendored/LLVM/Protocol/Schema pin，且没有 T10.2 文件。

### 2.1 非 squash 提交序列

| 类别 | commit |
|---|---|
| Proposed ADR | `36ba960835fa80427b9b65e3d2c8a24cde0c5d4c` |
| 第一次批准记录 | `eeb1e997db1492f4e630a342d4c095f3640d3e80` |
| 530 identity baseline | `cc030c7e20e65a083f6c74cd6159c1229e96b710` |
| F4/Ruff tests-first | `ea44ec4743bb210d5929095f3f9966419d5c5083` |
| CMake oracle hardening | `a6efaf5f63e712749b71e4c005a370240f1de056` |
| RUNPATH production fix | `c583a1366d4eba9173f9ae7464c9146bc8d676be` |
| wheel package production fix | `b9fff5f187b2facb65bd46e9c25f4793e2437bb9` |
| Ruff threat-model tests | `17f29769acc7b7d0feb7109015cb8760be730bab` |
| skipped-gate regression | `f2f65834c2f59b90b8ee5389ffe21fb28a981cb0` |
| fail-closed regression | `01a6b6ee32ad02f4e989c9c1c7cbd842693d978e` |
| canonical-step binding regression | `bd4dcb347d93b22dcaccc007543ba67523eaa42e` |
| Ruff/CI implementation | `1b369defa42a80999b6c550f76beb488a275f53c` |
| changed-file Ruff/style | `b06daeb4042aa61e4c729bf05198e523ae368622` |

未 squash，未 push，未创建 PR。

## 3. Fresh archive、工具链与构建

构建证据根：
`/home/dingbl/race_workspace/reports/t6.3-v3.6-reacceptance-20260824/t6.3-v3.6-rerun-20260824-EZb11mPt`

验收证据根：
`/home/dingbl/race_workspace/reports/t6.3-v3.6-full-reacceptance-I-b06daeb-20260824/final`

这两个目录均为第二轮重新创建的 evidence root。上一轮 build/test root、旧 wheel 与 PASS
材料只进入 denylist 或对照，不为下文任何 PASS 提供证据。

### 3.1 两份 archive 的身份

| 用途 | 格式 | bytes | SHA-256 | embedded commit |
|---|---:|---:|---|---|
| wheel build | `git archive --prefix=source/` | 6,604,800 | `31c8b0cdc35bbc87f126ef868fa7d85f3ba0126aaf17f204137a74cef3bee0d0` | `b06daeb4042aa61e4c729bf05198e523ae368622` |
| pytest/coverage | `git archive`，无 prefix | 6,604,800 | `34e22436c51e01203f41e946d4fa40c1cabdc63f2548cb5c8e892a07f9331174` | `b06daeb4042aa61e4c729bf05198e523ae368622` |

两份 archive 在第二轮独立重新生成；其路径前缀不同，但 embedded commit/tree 均精确指向
`I`，561 个逻辑 regular source file 的逐文件 SHA-256 完全相同。两者的确定性 SHA 与上一轮
相同不代表文件复用；本轮各自的创建路径、提取目录和 post-check 均独立。archive 内均无
`.git`、旧 build、dist、wheel 或 CMakeCache。验收 archive 的 561-file source manifest
SHA-256 为 `4c4d7553284bc33ac0ed445092b24285bc84824cf3ff5e6982fcaae745ea1983`；build
archive 的 561-blob/mode tree-validation JSON SHA-256 为
`e3bbc371c2dc5d9b034dca179c79836b20c0853906f87367569cd9c59874fe00`。

### 3.2 实际构建环境

| 组件 | 实际值 |
|---|---|
| OS / libc | Linux WSL2 x86_64 / glibc 2.39 |
| Python / SOABI | 3.12.3 / `cpython-312-x86_64-linux-gnu` |
| CMake / Ninja | 3.31.6 / 1.13.0 |
| GCC / G++ | 13.3.0 / 13.3.0 |
| pip | 26.1.2 |
| setuptools / wheel | 82.0.1 / 0.47.0 |
| pybind11 / packaging | 3.0.4 / 25.0 |
| build type / C++ | Release / C++17, CXX11 ABI 1 |

构建前端的 7 个 wheel 均被 filename、size 与 SHA-256 锁定在 build ledger 中；最终
安装使用 `--no-index`。LLVM root 为：

`/home/dingbl/race_workspace/reports/t6.3-v3.6-evidence/f4-wheel/toolchain/llvm-a992f294-ubuntu-x64`

构建前后对 `VCSRevision.h`、`LLVMConfig.cmake`、`MLIRConfig.cmake` 复算一致，revision
精确为 `a992f29451b9e140424f35ac5e20177db4afbdc0`。CMakeCache 在构建前不存在，最终记录：

- `CMAKE_HOME_DIRECTORY` 指向本次 build archive source；
- `CMAKE_CACHEFILE_DIR` 指向本次唯一 fresh CMake 目录；
- `LLVM_DIR` 与 `MLIR_DIR` 指向上述同一 LLVM root；
- compiler、Python、Ninja 与上表一致；
- wheel output 在构建前为 0 个文件。

唯一 wheel build 是 `raw/build-015-wheel-build.txt`，使用 `env -i`、`PIP_NO_INDEX=1`、
`--no-build-isolation`、`--no-cache-dir`，完整 argv 与 23 项环境 allowlist 位于 build
ledger。构建从 `2026-08-24T07:27:29.940254Z` 到 `2026-08-24T07:35:05.583926Z`，
耗时 455.644 秒，exit 0，实际 wheel build invocation 精确为 1。该 build command
**未配置外层 hard timeout**（`timeout=null / not imposed`）；本 annex 不把 §10 的 26 条
有 timeout 验收命令误称为覆盖 wheel build，也不虚构一个未使用的 deadline。

Build provenance：

- `provenance/ledger.json` SHA-256
  `b86cf1f8d9abc657885153508b8ee124c34f61317f1f40f7e7e9a49b5ad10785`；
- `provenance/attestation.json` SHA-256
  `a1402704b6164bcb85fde5375ad102cdcd9afc5790633b97f80a6268d07460f3`；
- `provenance/FINAL-SHA256SUMS` 62/62 OK、覆盖 364,102,584 bytes，manifest SHA-256
  `16257fa47b846c2189050e7cc872474cd853ebce0ff715ebfba1ce4cb12acbef`；
- `provenance/FINAL-SEAL-SHA256SUMS` 7/7 OK，manifest SHA-256
  `9f132b4d7118090c7ca59ed42dbc58f828b03c6bdb522a8c2abd3cd3aa497514`；
- `provenance/final-seal.json` SHA-256
  `0a94500b33fed7b243240bd3bbffc8da725199401e9d66fa501b87457c07670c`；
- build log SHA-256
  `876d76071a16d257d3e64511361b10b9ac4c28ef3541e4877f7b59df61e536c9`。

### 3.3 透明保留的非 build 脚本纠正

三个 rc=1 均完整保留在原始证据和 machine ledger。它们不是 wheel build invocation，未改
candidate、CMake output 或 wheel，也没有触发制品复用：

1. `raw/build-014-build-preflight.txt` 在构建前的只读检查中误写 archive 路径为不存在的
   `archive/candidate-I.tar`，rc=1；`raw/build-014b-build-preflight.txt` 改用实际
   `archive/I-b06daeb.tar`，并在 `015` 前重新证明 CMake、wheel、tmp、home 与四个 cache
   目录全部为空、archive/LLVM identity 正确。
2. `raw/build-022-ledger-attestation.txt` 在唯一 build 完成后因 parser 假设空输出记录仍有
   分隔行而 rc=1；失败发生在 ledger/attestation 写入前。
3. `raw/build-022b-ledger-attestation.txt` 随后因 frontend-wheelhouse manifest path 被重复
   resolve 而 rc=1；`raw/build-022c-ledger-attestation.txt` 修正这两个只读聚合问题并证明
   `build_invocation_count=1`、raw nonzero 精确为上述三项、最终 ledger/attestation PASS。

`raw/build-023-final-checksums.txt` 中 `covered_bytes=0` 是 awk 取错显示列；同一 62 项 manifest
的 hash 校验本身为 62/62 PASS，`final-seal.json` 从相同路径独立重算正确值
364,102,584 bytes 并透明绑定该显示纠正。唯一实际 wheel build 始终是 `015`，首次即 exit 0。
关键 correction log SHA-256 为：`014=08aaf456280e1719ce3798636851ccbdb2b1afb3ebee2ec399e17a1ab1b36c0f`，
`014b=ee5b00e64cbaa44e60ea809b0da4ee005f1082f93ca8666c7f9d6a764e15224b`，
`022=d98fae4718dfa40e49ad2cfa7bc463a20a6fc31c541defdde7e01fe039c72026`，
`022b=3c591118b23e2b3bdb810d59e64b8821a13c7e7017c01b5bc9aa58419e5d7a7b`，
`022c=92a088d643b0bcfd4cb15f3752570c7a6c60314ee44ffe28f80098da981e2edc`。

## 4. Wheel、package、RECORD 与 build-info

| 字段 | 值 |
|---|---|
| filename | `triton_anchor-0.2.0-cp312-cp312-linux_x86_64.whl` |
| size | 105,674,752 bytes |
| SHA-256 | `6a639fe5b77c48a2fbe631e7961cd1d3e50ea2253cdf15152e6e3979a338d8ed` |
| 禁止复用的旧 SHA | `a6785a37fa471e0b407da252638900c2772c7c50ff14d7a77e0fd66bba8e5ac8` |
| ZIP regular members | 133 unique，0 unsafe/duplicate/encrypted/symlink/cache/VCS |
| A/B/C package identities | 37 / 37 / 37，差集均为空 |
| `triton_anchor/tests` members | 0 |
| RECORD | 133 rows；全成员覆盖，非自身行 hash/size 独立复算一致 |
| metadata | Name `triton-anchor`；Version `0.2.0`；Python `>=3.10`；唯一依赖 `packaging>=21` |
| wheel tag | `cp312-cp312-linux_x86_64`；`Root-Is-Purelib: false` |
| entry point | `triton.adapters / triton-shared / triton_anchor.adapters.triton_shared_adapter:TritonSharedAdapter` |

本轮 wheel 与上一轮 fresh wheel 的 SHA/bytes 确定性相同，但并未复用文件：第二轮 preflight
证明 output/CMake/cache 目录为空，新文件的独立 inode/mtime/ctime、455.644 秒的唯一实际
编译以及 `raw/build-015-wheel-build.txt` 共同绑定其 fresh provenance。它与明确禁止的
`a6785a37…` 不同；上一轮 `6a639…` 仅作为 reproducibility 对照，不作为本轮 PASS 输入。

Generated `_build_info.json` 为 `generated=true`，Triton/Core/Protocol/Schema/LLVM/MLIR/
vendored commit 均匹配 §2；关键 identity 为：

- Core ABI fingerprint：
  `sha256:9a440b323ba82ff0ec99e8d60e839910be06a6bdc324843b67d04e033a13e9b9`；
- `libtriton.so` SHA-256：
  `eaca69f8cd4db4ea2204b0577307f0ae74821a0f9861516445a44e1cc8dcbfa3`；
- `triton-shared-opt` SHA-256：
  `eb3cbee28ee390477068ccd59d6d7ee27fa7a0a8f20f8d55e2bfa4dec12baa27`；
- build-info member SHA-256：
  `63a4033d020ad459ba178e77c45c0ee835c1b677cd34e08f072d2c025a7dd8d0`。

## 5. 全 ELF、fresh install 与 smoke

按 ELF magic 得到且只得到两项：

| member | RPATH | RUNPATH | PT_INTERP | DT_NEEDED |
|---|---|---|---|---|
| `triton/_C/libtriton.so` | absent | absent | none | 合同规定的 6 项 exact set |
| `triton/bin/triton-shared-opt` | absent | `$ORIGIN/../lib` | `/lib64/ld-linux-x86-64.so.2` | 合同规定的 6 项 exact set |

两项的 `DT_NEEDED` exact set 均为
`{libz.so.1, libstdc++.so.6, libm.so.6, libgcc_s.so.1, libc.so.6, ld-linux-x86-64.so.2}`，
无 LLVM/MLIR 或带 `/` 的依赖。wheel 与 fresh site-packages 内 ELF bytes 相同；installed
ELF 再审结果相同。CMake source oracle 确认 `triton-shared-opt` target 仅有一个生效的
`BUILD_WITH_INSTALL_RPATH ON`。

Fresh venv 使用 `python -I`，清除 `PYTHONPATH`、`PYTHONHOME`、`VIRTUAL_ENV`；所有模块
来自该 venv 的 site-packages。结果为：

- `triton==3.6.0`、`triton_anchor==0.2.0`；
- `find_spec("triton_anchor.tests") is None`；
- `pip install` exit 0，`pip check` exit 0；
- 清除 `LD_LIBRARY_PATH`、`LD_PRELOAD`、`LD_AUDIT` 后，以 fresh site-packages 内的绝对
  `triton/bin/triton-shared-opt --version` 执行，exit 0，报告 LLVM `22.0.0git`；
- 固定十项 smoke 为 `10 PASS / 0 FAIL / 0 SKIP`；driver SHA-256
  `9bdc7e79caa6c8021c0c032db1e703aa95053335bd24b73a23a548edac0623a9`。

F4 总结果为 11/11 PASS；包含全部 `F4_*_AUDIT=` JSON 的 log SHA-256 为
`f3c77bb6e08dbcf1b76889c5a1b3b3245c9c47e7ebae23c474e8cb3319fecb62`。

## 6. Pytest identity 与结果

| suite | nodes | result | duration | JUnit SHA-256 | raw log SHA-256 |
|---|---:|---|---:|---|---|
| Common | 409 | PASS | 7.002 s | `730d1dd52ebb11e7d11182073c33273c307fd21bfa6d23a4c30248ce7bff6377` | `241682d7a3171473486871eec28d4bb1cc91b7f8ab3bbcb3e8a62aaca3391406` |
| F5/F6 | 118 | PASS | 5.491 s | `c31230ccacf9645b0b8886a41ba4c404e3d8d26116ce0d649c45d7cf99d6de5a` | `8a18a1b96278432fcf323c6b4dfa6c6877ae70be8186074cb09e2106b910d7fe` |
| F4 | 11 | PASS | 18.608 s | `37ff6d7888b883eb2f40a4391efad5dac81ae79e89d49ad19242c723085667ad` | `f3c77bb6e08dbcf1b76889c5a1b3b3245c9c47e7ebae23c474e8cb3319fecb62` |
| Ruff policy | 100 | PASS | 7.877 s | `3f3140c870d4c9e0aff5154889dc0887451353c05ba70bbc4f9b8950f8b5c66d` | `dc04c34a60b23d3bdd3cdaa7e459a4697f0a3ceb86cbdbf4486c682da3da5b92` |

机械 identity audit：

- frozen baseline：530 / unique 530 / 76,376 bytes / SHA-256
  `f6a7af60a2bf8bbf7a822c1f855c19045b975169e6c67d1f8846c5f66a307235`；
- current：638 / unique 638 / 92,792 bytes / SHA-256
  `37a52dd2ec8ea2501606853b7255cca5b9404a587874673f929397b502448672`；
- additional：108 / unique 108 / 16,418 bytes / SHA-256
  `44ca79d027ce55524962e66871c3d17956e9614ab508be1969a3008099b8126b`；
- baseline missing 0、baseline/additional intersection 0、duplicates 0；
- 638 个 testcase 全为 PASS，failure/error/skip/xfail 均为 0。

Machine audit `manifests/final-junit-audit.json` SHA-256 为
`3d8b079e5ded1b2619ca7d2dd9e1768a8861d806e255dfbb7ab5b13078639d25`。

### 6.1 F5 real Legacy fixture

F5 使用的 real no-Manifest wheel 位于外部 evidence fixture 目录，绝对路径不构成合同；
其内容 SHA-256 精确为
`97402abf825cf56d800edf5f4cfa25a8513be585b2b067391d0766664d18a455`。
Companion snapshot `tests/acceptance_t63/triton_v36_legacy_baseline.json` 的 Git blob 为
`6b08d790891eab97588e79e28b5c796bf1ecc3dd`，文件 SHA-256 为
`8eadf0d184da2bb01b6b8a32398c5d8ccc87ff79388c0ec82ae1efefeb492d79`。

冻结 layout 为 entry-point/package root `t63_v36_legacy_oracle`，compiler
`t63_v36_legacy_oracle.compiler.LegacyCompiler`，driver
`t63_v36_legacy_oracle.driver.LegacyDriver`；不要求 root alias。Fresh F5 run 证明：首次
compiler/runtime 消费前 event log 不存在；首次 materialization 后
`package_import`、`compiler_module_import`、`driver_module_import` 各精确出现 1 次；reset
清除 publication 后允许新的 generation 重新 import，旧 generation 不得发布。相关 real
installed/reset/import-once identity 包含在本轮 118/118 PASS 中。

## 7. Ruff 0.15.22 与 CI gate

候选上的实际 shared gate 使用 Ruff `0.15.22`，executable SHA-256
`64aae5e444938e33121c3b940dff9b3d8ef8fc2a88c477e7f3a4fae2584a8fe8`。

| domain | count | NUL manifest SHA-256 |
|---|---:|---|
| all tracked `.py/.pyi` | 141 | `335c053f279343abd06a484775c83cd242c92f05400d411c51719b0f2a54af5e` |
| changed ACM vs `9ceba3e…` | 32 | `e7bd840de5d35cc8b0172d43da737e6a76e5a21a84ff384cb1e52cf3a73238ba` |
| deleted | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

四个实际 stage 均 exit 0：

1. critical：`ruff check --isolated --no-cache --no-fix --target-version py310
   --select E9,F63,F7,F82 -- <141-file manifest>`；
2. changed full：`ruff check --isolated --no-cache --no-fix --target-version py310
   -- <32-file manifest>`；
3. format：`ruff format --isolated --no-cache --target-version py310 --check --
   <32-file manifest>`；
4. policy scanner：exact repo/changed manifest/gate/workflow/output paths；findings `[]`。

逐门机器证据为：

| stage | argv0 SHA-256 | raw output SHA-256 | diagnostics/findings | exit |
|---|---|---|---:|---:|
| critical | `ead15fc78600cb7ec8640fc77647551e2a539b5fb6a854baaa630718e8e87363` | `82b3e6a6c090a57601d22943bd23fca9218d1031dbe5a7b754092f9a156b4f18` | 0 | 0 |
| changed full | `e09bdbfe8b8956713ad40161e8b07bf25aadd7993ceb142d3d0c09bd585c0316` | `82b3e6a6c090a57601d22943bd23fca9218d1031dbe5a7b754092f9a156b4f18` | 0 | 0 |
| format | `b1bbb53b177e5590348d41d3457e3934b2a04099c2d554b96351c7dddd74a17c` | `6ed8d1cb172b51dddf192664f6e4b2a6c56db357a990dcce6f028de11201bd61` | 0 | 0 |
| policy | `713435372b7ccbd3a2b05e97a76e20cef284fa445a1e4e23d8b6121909efd25a` | `819d0760769e4401166c2891ff293c3efd70eac1b2ae0a61e83e43865b179b9a` | 0 | 0 |

四门由同一次 shared-gate command 聚合，外层 hard timeout 为 300 秒；实际总 duration
0.432 秒、aggregate exit 0。逐门 exit 的 canonical JSON 是下述 four-exit summary。

完整 NUL argv、raw output 与 JSON 分别由 `ruff-gate/SHA256SUMS` 绑定；该 manifest
14/14 OK，SHA-256
`6844d2aa22d397dfc867c8f3283be854afd7cdb8504b272de5edcaba90dff792`。
关键 evidence：

- context SHA-256 `f4fc11e3fe85c2d01d0db04f1c6d7073ce5093d60eaf03239d042ebf66b99ee2`；
- four-exit summary SHA-256 `dadae38e7401afd9dc10dfc06753ed119f21902c9c054a79c91ff7d0aa9afaff`；
- scanner JSON SHA-256 `2a8d20252266b97b1664804ccc158c59f7efe450e3cbd29a4b997a84c0178d81`；
- actual gate log SHA-256 `aacc41406465f707d9adb9414d1948433e0237e7739fd2dd47f73496451f180d`。

Source identities：workflow `8a50db4535153b2b549a27d8f775efbe6502ca72af74c6f20afd0fd39e6bd727`，
gate script `96066b930effc3c0997389b5a91bc4fe73af691083ef1b147f4359391c8026d1`，
scanner `99df64ef5be6d8e7e9ddf1dabeee341ecbb84c161f558ebfd1ed377ee14e2ecf`，
policy tests `96110a7e988bbc89847a62faecbeaf4e6ffa026434181ecfb8e10ba62c1190c0`。

Suppression exception table 为空（0 项）。本实现/验收过程未调用 `ruff --fix`、
`--unsafe-fixes` 或 blanket noqa；style commit 使用显式人工修复与 `ruff format 0.15.22`，
10 个纯格式文件 AST 不变，`setup.py` 与 compiler 的两项语义保持修订已单独审查。

## 8. Coverage

Coverage population 精确为 Common 409 + F5/F6 118 = 527；F4/Ruff 不进入分母。

- Python 3.12.3；pytest 9.1.1；pytest-cov 7.1.0；coverage.py 7.15.4；
- source `python/triton_anchor`，statement-only，branch off，`--rcfile=/dev/null`，无
  omit/include/ambient config；
- 32/32 source `.py` 在 domain 中，无 missing/extra；
- 两份预先不存在的 data：Common SHA-256
  `e304fb7b1c7365153d1e4d04f80bf2b1a0661c90c1f38b279e349bb8a2afb475`，F5/F6 SHA-256
  `327bafb627bac64adbf0285922484065cadd8c47e3045bbc6631b27ebce969ac`；
- `coverage combine` 精确报告 `Combined 2 files`，combined data SHA-256
  `e7bf86221f0caea83ebd5cddaac0a91218e759656dfe873f1826f831177bbe07`；
- covered/valid = `3368/4418`，exact rate
  `76.23358985966500679040289724%`；
- `coverage report --precision=2 --fail-under=76.00` exit 0；
- report raw log SHA-256
  `64efd0318be48a26058b708a6fa98ccbec2145454389707de449646ed2f569de`；
- XML SHA-256 `3ef4bef6113cbb87e24991ce0ee12a89f383225052f15012f14a02ee6e42ba14`；
- JSON SHA-256 `ffd58a84eaf4e76686ca729e1a826ad57c98ff7eb3d9d79ea9bc110737e3642b`；
- machine audit SHA-256
  `35bf3efc3754d1c62a02bdc6d7e31a6937e5d42d4db4ce7f33a7a24547a60a8e`。

Coverage 两份 JUnit 仍分别为 409/118 全 PASS；其 SHA-256 为
`5f0a24fcc225c92366e9de36975cba6a733e4bdaf6ca9395b79b22d870398794` 与
`5f5a4bbb48266d4997a011c974624d1f214b8ac6cbbcdc4b714db8c43c0db58f`。

## 9. Hardware 分类

Candidate precondition evidence 为 PASS，但 hardware execution classification 为 BLOCKED：

- wheel filename/size/SHA、Triton 3.6.0、Anchor 0.2.0、build-info 与 fresh `python -I`
  module paths 全部绑定正确；
- candidate 提供唯一 `triton.adapters/triton-shared` entry point，但没有
  `triton.backends` hardware entry point；
- host 为 WSL2，仅发现 `/dev/dxg`；`nvidia-smi`、`rocminfo`、`rocm-smi`、
  `xpu-smi`、`sycl-ls`、`clinfo` 均不存在；
- PyTorch `2.13.0+cpu`，CUDA/HIP build 均为 null，CUDA/XPU device count 为 0，
  CUDA/HIP/XPU/MPS availability 全为 false。

其中 host-capability probe 使用宿主 Python，只采集设备/driver/PyTorch 事实；它观察到的
ambient Triton 3.8 entry point 明确不是候选证据。Candidate precondition 则使用已经通过
F4 的 fresh wheel venv 与 `python -I`，只以 Triton 3.6.0/Anchor 0.2.0 的 fresh paths 和
candidate entry points 作 HW-01/HW-02 判定；两种环境没有混用。

因此 HW-01/HW-02 前置条件不满足，固定 vector-add 未执行，不能把 CPU/mock 或软件 smoke
替代为 hardware PASS。最终使用的 candidate precondition JSON SHA-256 为
`116d38b7495a91c553abde08c8f495bfc2726387ff1930f107bc517a0d420fb3`；对应 host probe
JSON SHA-256 为 `6d9e37a9ab2f1afa53c8691245228ba54884e328a0f7211d7d1fd1cdaf921160`。

未执行的固定非 T10.2 oracle 仍已冻结：

- kernel source：3,115 bytes，SHA-256
  `63f81723bbae3d3a1da53883697b774df13b93e6b0246252dbe046cfabc1d1f4`；
- expected output：1,024 个 little-endian float32 / 4,096 bytes，SHA-256
  `1faf7ed7002b42761b557cbcfb72b035d36a4d50e724a2df7e3cdb1d2c12a96b`。

## 10. 命令、环境、总证据与范围排除

Final command ledger `command-evidence-ledger.json` 状态 PASS，包含 26 条命令的 cwd、
完整 argv、明确 set/unset 环境、UTC start/end、duration、timeout、exit、raw output hash 与
产物 hash；26/26 exit 0。其 SHA-256 为
`5837bd08ee27f01ede092c2c37c4569b8c07c7bf4e9a857c318488ab810dae53`；ledger 生成 log
SHA-256 为 `6a971b2bcf7cf5bd9bc0d491659be6f363694cf4e6887ebed2c885a5defe2f3f`。

Final environment manifest 状态 PASS，SHA-256 为
`0489b879b88e17dcb174153459490b7b1b9a5ca5667d53725f6418cc4eab8dbf`。

Final `SHA256SUMS` 包含 84 个证据文件，84/84 独立复算 OK；manifest 自身 SHA-256 为
`3312f932932f6d7c8b8a328f571caaf6cc67d42078ca28051f1694c637fff0f7`。
`SHA256SUMS.sha256` 的内容绑定该值；sidecar 文件自身 SHA-256 为
`a820d6586597be204148120d1c95541d483388a9078f8bdc9690c2bb5bda7fc0`。

T10.2 仅作为排除范围名称出现：四套 collect manifest 与 26 条实际命令的 T10.2/T102
匹配均为 0；未收集、未导入、未执行，未使用其 Kit、fixture、helper、artifact 或结果，
也不进入 node/coverage denominator。它不是 PASS、SKIP 或 BLOCKED 项。

## 11. 第二次确认前的最终完整性

第二轮开始与 build/test evidence 封口时（本 annex 更新之前）：

- shared worktree：`## fix/t6.3-v3.6-release-gates`，无 tracked/untracked/cached 修改；
- shared `HEAD=6bbc65d6933633b0ba5ae4822dfbe1e487e261b9`，`HEAD^=I`；
- shared `HEAD^{tree}=d71e0befa9d21a02c111b18b350add87e1fc47c3`；
- `git diff --name-status I..HEAD` 精确为新增本 annex；
- 所有执行输入仍精确为 `I`，candidate clone 的 `HEAD/tree/index/worktree` 为
  `b06daeb4042aa61e4c729bf05198e523ae368622` /
  `74d8b6b8e863cc0474618e556ce78b0b3adfa10b`，clean；
- `git diff --check` 与 cached diff check 均 exit 0；
- 无 version-pin 或 T10.2 diff；
- 没有 push 或 PR side effect。

`6bbc65d…` 只封存上一轮 annex，不能作为第二轮 `E`。本次修改目前只更新同一个 tracked
annex；形成第二轮 evidence commit 前，`EVIDENCE_COMMIT_SHA` 保持 `PENDING_BINDING`。
新 `E` 提交后必须机械验证：

1. `E^` 精确等于 `I`；
2. `I` 是 `E` 的祖先；
3. `git diff --name-only I..E` 精确只有本 annex；
4. 本 annex 是 tracked Git blob，`git diff --check I..E` 为 0；
5. annex 不含 `E` 自身 SHA；
6. worktree 再次 clean。

满足这些 topology 条件后，应向发布责任人提供 `I`、`E`、wheel identity、上述 evidence
hash 与 hardware BLOCKED 分类，请求第二次 exact-identity 确认。在确认发生前，
`EVIDENCE_COMMIT_SHA` 必须保持 `PENDING_BINDING`，`AUTH-SHA` 必须保持 SPEC GAP，本 ADR
不得标记 Accepted。
