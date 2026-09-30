# Local CI tools

工具可由 shell、Python 或 AI 调用。Codex 根据 PR 意图、实际改动和项目最低规则，
自主组织构建、测试与审查；工具负责执行具体工作，不负责调度 Codex。

| 工具 | 准备条件 | 行为 |
| --- | --- | --- |
| environment | 已准备的运行环境 | 检查命令、Python 依赖、LLVM 与 pip 依赖一致性 |
| control_plane | 被测 checkout 与 base SHA | 检查改动文件并运行相关控制面回归 |
| frontend_build | 前端构建依赖 | 构建 frontend wheel |
| frontend_install | frontend wheel | 无依赖更新地安装 wheel，验证实际 import 来源 |
| frontend_tests / frontend_smoke | 已安装 frontend | 选测 / 前端 smoke，可独立调用 |
| backend_build | 后端构建依赖、已安装并验证导入的 frontend wheel | 构建 backend wheel |
| backend_install | backend wheel、frontend | 安装 wheel 并验证后端发现 |
| backend_tests / backend_smoke | 已安装 frontend、backend | 后端选测 / 真实 JIT |
| flaggems | 可用后端与预置 FlagGems | 按影响或全量清单逐算子测试 |
| compile_time / pass_profile / ir_serialization | 可用后端 | 正确性及有效测量检查，再与同条件基线比较 |

准备条件用于安排工作，工具不会自动递归执行构建、安装或其他阶段。
目前仅 Triton 3.0 环境支持后端构建与测试；其他版本的后端相关工具返回 `not_applicable`，
只执行适用的前端验证。后端能力以各 variant 的可信 context 为准。
启用后端时，同一 variant 必须按 `frontend_build → frontend_install → backend_build → backend_install → backend_tests/backend_smoke`
的阶段依赖执行：前端 wheel 生成、安装及导入验证成功后，才能开始后端构建。
前端构建/安装与后端构建/测试不能并行；前置阶段失败或仍在运行时不得启动后续阶段。
该依赖也适用于原生命令，base 与 candidate 分别满足各自的前置条件。
路径分类生成的工具与参数是建议，Codex 根据实际 diff 选择，也可直接对源码运行定向检查。
`frontend_tests` / `frontend_smoke` 本身需要已安装 frontend；不需要安装包的独立源码测试
使用原生命令执行。`change_validation` 是结果中的影响、选测理由与证据汇总，不是工具。
编译可调整 `jobs`，并选择 `fresh` 或 `incremental`；共享环境依赖由 `prepare/` 管理，
生产源码链路经 Gitee。FlagGems 使用服务器预置目录，测试缓存与日志写入任务产物目录。

## 调用

```bash
"$PYTHON_BIN" /opt/local-ci/control/scripts/local_ci/tools/basic_tools/runner.py frontend_build \
  --context /task/artifacts/candidate-context.json --parameters '{"jobs":2,"build_mode":"incremental"}' --execute

"$PYTHON_BIN" /opt/local-ci/control/scripts/local_ci/tools/basic_tools/runner.py frontend_tests \
  --context /task/artifacts/candidate-context.json --parameters '{"paths":["tests/test_unit.py::test_add"]}' --execute
```

`context` 中 `source_dir`、`artifact_dir` 必填；`target_sha` 默认读取 checkout HEAD。
`python_bin`、`profile`、`base_sha` 等指定环境和比较对象。profile 的 tools 配置包含
`env_scripts`、`backend_env_scripts`、`llvm_dir`、`backend_dir`、`backend_test_paths`、
`backend_smoke_argv`、`expected_backend` 与 `flaggems_dir`。

不带 `--execute` 输出命令计划；Python 调用 `plan(tool_id, context, parameters)` 或
`execute(tool_id, context, parameters)` 使用同一实现。执行结果写到
`artifact_dir/<tool_id>/result.json`，命令输出在同目录的 `command.log`。
工具报告记录状态、参数、耗时和产物，可用于 AI 最终的 `agent-result.json` 汇总。
CI 流程检查没有可运行回归套件时返回 `limited`，保存已完成的契约检查和缺失范围；
Codex 应继续生成并执行定向测试，不能将工具缺少套件误判为环境故障或直接声称行为已通过。

文件契约检查安全解析 YAML 中的全部文档，支持 MLIR 配置的本地标签（如 `!LinalgOpConfig`），
标签只作为普通数据处理，不执行对象构造；各文档仍检查重复键和语法错误。
GitHub 工作流须为单文档且不接受自定义标签，并检查触发条件、作业依赖和执行步骤。
符号链接按最终解析目标判断 checkout 边界：仓库内的文件或目录链接记录链接摘要与目标路径，
不递归检查链接目录，也不将链接检查视为目标源码语法验证；越界、断链和循环链接拒绝通过。

build 参数为 `jobs`、`build_mode`；install 可用 `wheel` 指定现有 wheel，
默认读取 `artifact_dir/<build_tool>/wheel.json`。若构建产物在另一目录，
可用 `dependency_artifacts[build_tool]` 指定该目录。
pytest 参数为 `paths`（相对路径或 node ID）、`keyword`；FlagGems 为
`mode`（impact/full）、`ops`、`categories`；性能参数为 `kernels`、`repeat`、`warmup`。

先判断 PR 是否需要 FlagGems。非 full 最多 6 个不同算子；显式 ops / categories 展开后
超过上限直接提示使用 full，不自动截断。空 impact 固定使用 `abs`、`maximum`、`mm`、
`arange`、`exponential_`、`embedding` 六个样本。需要更广覆盖时说明原因并显式使用 full，
不要拆批规避上限。full 保留完整算子目录，每个算子仍可包含多个参数化用例。
full 的 runner 输出沿用 `triton-anchor-local-ci/flaggems-v1`：`summary` 保存总数及
通过/失败/超时计数，`results` 保存逐算子记录。Worker 会把该 JSON 独立发布到
`runs/ci_full_flaggems/<tested_sha>/<run_id>/flaggems-summary.json`，不并入任务
`result.json`，也不占普通证据附件预算。

普通 pytest 输出简单计数，失败、空收集和全部跳过都不会显示为通过。
后端选测使用 pytest 的 prepend 导入模式，以兼容测试对同目录 conftest 的导入；
前端选测保留 importlib 模式。两者都使用已选任务解释器和隔离启动，不读取用户 Python 环境。
Worker 为 base/candidate 分别提供完整 `runtime_env`、LLVM、profile、能力与环境指纹。
启用后端的任务环境同时设置 `TRITON_SOURCE_DIR` 为该侧 checkout 下的 `triton`，
`TRITON_ANCHOR_SOURCE_DIR` 为该侧 checkout；base 与 candidate 不共享源码目录变量。
基础工具只继承调用端的网络代理／证书设置，构建环境由选中的 context 提供；不能借用另一侧
的后端、LLVM 或 Python 搜索路径。原生 shell 使用同一环境入口：

```bash
"$LOCAL_CI_SEED_PYTHON" /opt/local-ci/control/scripts/local_ci/tools/basic_tools/variant_exec.py \
  --context /task/artifacts/base-context.json -- bash -c '"$PYTHON_BIN" -m pytest tests/test_example.py'
```

后端原生命令加 `--backend`，会校验该 variant 的能力并加载其后端脚本。只读依赖可共享，
两边 venv、构建、缓存与产物目录始终独立，Worker 提供的 context 包含完整运行环境。

性能基线由 `performance_baselines[tool_id]` 指定文件、提交与环境信息；比较时核对
后端、算子和采样条件。没有可比基线时报告 `not_comparable`；有效性能回退只报告，
测量无效或正确性失败则返回失败。
Codex 决定是否调用这三项固定 runner；Worker 直接读取并校验 runner 的结果文件，
使用四个固定 kernel、默认采样次数和 warmup 封存 Dashboard 数据，不接受手工填写的数值。
IR 比较覆盖页面展示的 `serialize`、`write_text`、`read_text`、`deserialize`、`roundtrip`。

## 审查与补充验证

`ai_review_tools/` 提供架构及专项审查说明，`ai_custom_tools/` 说明任务内脚本的用途。
专项方向依据描述和实际 diff 选择，标签仅供参考。
Codex 可以直接运行已有测试、编写定向复现或使用其他命令，自主选择范围和顺序。
真实编译器、运行时及后端接口改动仍需对应构建与 smoke/JIT；文档和普通注释可轻量验证。
显式 full 任务要求 Codex 完成 FlagGems full 及其必要的构建、安装和后端 smoke；
性能测试仍由 Codex 根据实际改动决定是否执行。
最终摘要列出实际改动影响、选测理由、已完成检查、审查结论、未完成项与所选重要证据。
