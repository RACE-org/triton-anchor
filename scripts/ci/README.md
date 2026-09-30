# Local CI Gateway 与接收器

Gateway 在 GitHub 冻结任务、执行前置检查并投递到 Gitee；Worker 从 Gitee 取得源码，
执行后发布结果，接收器回写 GitHub 状态、PR 评论和 Dashboard。

安全扫描将新增网络导入、请求及远程 Git 操作列为审查提示，不单凭网络能力阻塞升级。
凭据泄露、下载内容直接交给 shell、危险执行、受保护路径和依赖来源规则仍保持阻塞。
网络提示不代表来源已经验证可信，仍需结合实际改动审查。

## 工作流分工

| 入口 | 所在分支 | 职责 |
| --- | --- | --- |
| `ci-request.yml` / CI Request | `main` 接收 PR 与手动请求；各源码分支接收 push | 将任务路由到控制分支 |
| `ci-gateway.yml` / CI Gateway | `CI_dev` | 冻结任务、前置检查、审批与投递 |
| `ci-gateway.yml` / CI Gateway | `main` | 接收结果、回写状态与评论、发布 Dashboard |

需要 push 自动触发的分支须包含 `ci-request.yml`；PR 入口在默认分支运行。
路由不限制目标分支，实际环境由冻结源码中的 Triton 版本和 LLVM SHA 决定。

PR 任务冻结目标 base、贡献者 head 和精确 merge/tested SHA。
分支任务使用指定分支 HEAD 作为 tested，其第一父提交作为 base。
源码与随任务检出的子模块固定 refs 全部推送到 Gitee 后，才发布不可变 `tasks/`、
当前任务 `current` 与取消标记 `cancel`。源码 refs 包含完整 task ID。

FlagGems 使用服务器 profile 的固定只读目录；其子模块指针不切换服务器依赖。
其他子模块固定到被测 Git 对象。

### 新一轮验证与原任务恢复

PR 同步、重开、转为可审查、正文或标签变化，分支创建或强推，以及 rerun、手动验证时，
以入口的 `run_id:run_attempt` 生成 `trigger_id`，参与 task ID 计算。
即使 head、merge 未变化，也会投递新任务并重新执行服务器测试。
该编号写入冻结的 `task.json`，审批、派发、结果接收只传递这个文件及其 task ID。
其他自动路由按源码、基线、PR 审查信息和 full 范围计算任务身份。

Gateway 的 Re-run all jobs 和 Re-run failed jobs 都派发完整的新验证，
重新执行前置检查及所需审批；可从重跑作业的摘要进入新运行。

同一轮重复投递、执行恢复及结果补传仍沿用原 task ID 与重试预算；
receive 续接、单独补收及页面发布也不创建任务。
接收器只等指定 task ID 的结果，同 SHA 的上一轮结果不会结束新一轮。
历史任务不带 `trigger_id` 时保持原 ID，历史结果不删除。
Worker 的恢复和重试规则见 [Local CI 概览](../local_ci/README.md#执行与恢复)。

## 前置检查与审批

检查按成功依赖推进，状态回写可与下一检查并行：

1. CI Request 校验请求后写入 pending Summary，再派发 Gateway。
2. Gateway 冻结任务并取得执行归属，创建 Basic 并接管 Summary；PR 任务校验信息后执行 Basic CI。
3. Basic 通过后执行 API Compatibility，API 通过后执行 Security Gate。
4. 同仓库任务在前置检查通过后投递；外部 fork 在检查及回写完成后生成审批卡。
5. 外部 fork 经 `local-ci-fork-approval` environment 审批，复查冻结身份后投递。
6. 源码与任务成功发布到 Gitee 后更新 Summary 的等待说明，并启动接收器。

前置失败仅结束已到达的阶段。外部审批拒绝或校验失败终止审批阶段，
取消记录为错误状态，不创建 Dispatch。
PR 关闭、转草稿、目标变化或新增提交时结束失效任务的待定状态，并通知服务器停止执行。

审批卡同时写入 PR 与工作流摘要，展示前置结论、贡献者与来源、冻结 SHA、
改动数量和按路径归类的范围、diff 链接及审批入口。
PR 正文提供概述、影响范围和验证情况，创建 PR 时自动使用默认分支上的
[PR 模板](https://github.com/RACE-org/triton-anchor/blob/main/.github/PULL_REQUEST_TEMPLATE.md)；push/manual 不要求这些字段。

只有 environment 明确记录人工拒绝，才向仍匹配的 PR 通知审批未通过；
审批校验错误或取消按对应状态处理。任务取消仅更新状态和 Gitee 停止标记。

## 检查状态

Basic、API、Security、Approve、Dispatch 和 Summary 均使用 Commit Status。
PR 状态写入 `head_sha`，实际测试及证据对应冻结的 `tested_sha`；
分支任务状态写入 `tested_sha`。控制分支自身 push 也发布前置阶段状态。

`Summary` 表示整轮验证的结论，阶段检查只在执行到该阶段时创建。
Summary 从 CI Request 确认有效请求后为 `pending`，覆盖 Gateway 排队与初始化、前置检查、外部审批、
任务投递、服务器执行与结果接收；阶段切换不会使整轮验证短暂显示为全部成功。

| 情况 | Summary |
| --- | --- |
| 验证进行中、等待审批或结果 | `pending` |
| 最终结果及必需证据校验通过，验证成功 | `success` |
| PR 信息不完整、前置检查或服务器验证失败、人工拒绝审批 | `failure` |
| 取消、任务失效、审批配置异常、投递失败或接收超时 | `error` |

前置失败不创建后续未执行项，Summary 指出失败原因并链接对应运行。
请求阶段按 CI Request run ID 与 attempt 标记归属；Gateway 使用传入的 `request_id`
接管后绑定冻结 task ID。新请求会阻止旧任务回写，旧请求及其失败收尾不能覆盖新请求或已接管的任务。
派发失败或 Gateway 初始化中断时，尚未接管的 Summary 收尾为 `error`。
关闭或转为草稿的 PR 只执行取消流程，不创建新的等待状态。
直接派发 Gateway 时不填写 `request_id`，Summary 在可信任务初始化时创建。
`trigger_id` 由入口传递；手动运行时留空，由 Prepare 确定，勿复制上一轮的值。
新任务及完整重跑重新初始化 Summary；重复初始化不会重新打开已结束的阶段。
`Local CI Approve` 只用于外部 fork，同仓库任务不创建该项。
这些状态报告验证进度与结果，不配置强制合并规则，也不保证 Merge 按钮置灰。

阶段与工作流绑定 task ID、workflow run ID 和 attempt。
Basic 状态标识当前执行归属，接收器还要求对应 Dispatch 已成功；
迟到的运行不能覆盖当前状态。相同回写去重，状态变化保留历史，以最新记录为准。
最终测试结论来自 `Summary`，各 Actions 作业记录对应阶段的调度或交付结果。

检查名称、状态与失败说明使用英文，审批卡、Agent 解释和 PR 结果使用中文。
PR 结果按 `result + task_id + run_id + result_digest` 去重，每次运行追加一条结果评论。
评论区分 PR 提交与合并后验证提交，列出实际执行的检查、阻塞项及报告链接；
完整选测记录在 Dashboard 展示，日志及复现证据见完整执行报告。
每条发现分别列出结论、分析和代码位置，代码链接指向被测提交；保留风险等级，缺失时显示“未标注”。
失败检查不重复列为缺陷；没有阻塞 findings 时使用阻塞原因兜底。
环境、工具和证据不足单列“限制说明”，按范围未选择的检查不自动构成限制。
正文使用可读检查名称，自定义检查可提供 `display_name`；版本比较使用“base”和“候选”。
结果字段及表述要求见 [Agent 文档](../local_ci/AI_CI_PROGRAM.md#最终结果)。

## 手动派发与接收

### 分支验证

在 Actions → CI Request → Run workflow 选择 `main`，填写：

| 字段 | 内容 |
| --- | --- |
| `source_branch` | 实际被测分支 |
| `requested_sha` | 可选的完整 HEAD SHA，用于拒绝分支漂移 |

发起者需拥有 write、maintain 或 admin 权限。
路由统一派发到 `CI_dev`，控制提交由路由确定。

### 完整验证

在 Actions → CI Gateway → Run workflow 选择 `CI_dev`：

| 字段 | 内容 |
| --- | --- |
| `mode` | `run` |
| `worker_revision_sha` | Gateway 工作流提交的完整 SHA，记录任务生成来源 |
| `full` | `true` |
| `pr_number` | PR 编号；分支任务填 `0` |
| `source_branch` | 分支任务填写被测分支；PR 从自身信息确定 |
| `requested_sha` | PR head 或分支 HEAD 的完整 SHA |
| `action` | 手动分支任务填 `manual`；PR 任务留空 |

`full=true` 仍由 Codex 组织执行：完成必要的构建、安装及后端 smoke 后，
使用固定工具运行 FlagGems full。性能测试由 Codex 根据任务意图和 diff 决定，
full 不自动要求性能测试，也不允许用 impact 结果代替 full。外部 fork 使用同一审批流程。
最低验证要求见 [Agent 工作流程](../local_ci/AI_CI_PROGRAM.md#自主工作流程)，
base/candidate 的 profile 与能力见 [Local CI 环境](../local_ci/README.md#环境与生命周期)。

### 更新入口与任务协议

任务身份协议变化时，先同步 Gitee 并[更新服务器控制代码](../local_ci/prepare/README.md#更新已有服务器)，
再更新 `main` 和相关源码分支的 `ci-request.yml`，以及 `main` 的接收工作流。
旧 Worker 可能拒绝新协议任务，需显式更新。历史运行的 rerun 使用原工作流代码，
更新入口后应通过 Run workflow 发起新验证。

### 结果接收

任务投递成功后，在 `main` 启动 `mode=receive`，按 task ID 每分钟查询一次；
每轮最多 5 小时 40 分钟，共最多 3 轮。收到有效结果或任务失效时结束，
轮次耗尽报错，服务器任务按自身状态继续处理。

补收时在 CI Gateway 选择 `main`、`mode=receive`，
填写原 `task_id`，`receiver_round` 为 `1`。补收使用已有结果，不重新投递构建。

每轮解析控制分支 SHA 并加载接收器。结果就绪后，串行 publish 作业复查任务身份、
回写 PR 并生成页面；失效任务只结束自身的 pending Summary 并刷新 Dashboard。
`collect --task-id <ID>` 回写指定任务；不带 ID 时只汇总页面，加 `--reconcile-pending` 可补写当前有效任务的已有结果。

## 结果与页面发布

结果按事件、目标分支及源码提交保存：

- PR：`runs/pr/branch-<目标分支>/pr-<PR号>/<head_sha>/<run_id>/result.json`。
- push/manual：`runs/push/branch-<目标分支>/<head_sha>/<run_id>/result.json`。
- full 算子业务结果：`runs/ci_full_flaggems/<tested_sha>/<run_id>/flaggems-summary.json`。

分支名中的 `/` 使用 URL 编码，head SHA 为完整 40 位。
接收器按结果中的 task ID 匹配，不混用同一 SHA 的不同任务。
`result.json`、重要文件和同次 full 算子业务结果一次提交，标题为
`local-ci: <status> <head_sha前12位> <run_id>`。

`result.json`、`change_validation` 报告及 `checks.evidence` 文件必传，
`artifacts` 按剩余预算选择补充文件。
必传证据无法发布时保留检查状态，整体通过改为 `infra_error`；
选传文件超限保留在主机并注明。数量和大小限制见
[结果与证据](../local_ci/README.md#结果与证据)。
上传响应丢失后重试相同内容，不重新运行 Agent。

页面发布比较 schema、tasks 和静态资源 `site_digest`，仅内容变化才部署 Pages。
部署由独立 `deploy-dashboard` 作业完成，`github-pages` environment 允许 `main`。
手动发布使用 `main` 的 CI Gateway、`mode=publish`，不带 task ID；同时补写当前有效任务遗留的 pending Summary。
PR 任务以 head SHA 为展示标识，tested SHA 保留在验证详情中。

PR 评论发送失败记录为 `receiver_error`，发布失败在接收轮次预算内重试。
页面发布与测试结果独立处理；Worker 通过 Gitee 通信，健康告警由
[Cloudflare](../local_ci/maintenance/cloudflare/README.md) 读取服务器快照。

## 配置

以下变量和 Secret 配置在 GitHub 仓库中，取值须与服务器配置一致：

| 配置 | 用途与要求 |
| --- | --- |
| 变量 `GITEE_RESULTS_REPO_URL` | 可选覆盖；默认 `https://gitee.com/race-org/triton-anchor-local-ci-results.git`。承载被测源码 `ci/*` refs、任务控制数据与结果；对应服务器的 `gitee_repo_url` |
| 变量 `GITEE_SUBMODULE_MIRRORS` | 非预装子模块路径到 Gitee 镜像 URL 的 JSON 映射；没有这类子模块时可省略或设为 `{}` |
| 变量 `GITEE_USERNAME` | 可选覆盖；默认 `likehupochuan`，应为 `GITEE_TOKEN` 所属的 Gitee 用户名；不要求与仓库组织名或服务器用户名相同 |
| Secret `GITEE_TOKEN` | 对结果中转仓库具有 Git 读写权限的 Token |
| `local-ci-fork-approval` environment | 外部 fork 审批，配置非空 required reviewers |

服务器的 `control_repo_url` 指向控制代码镜像，Worker 从其控制分支更新执行代码；
它与 `GITEE_RESULTS_REPO_URL` 的用途不同，配置时分别指定。
具体服务器字段和认证方式见 [服务器准备](../local_ci/prepare/README.md)。

工作流按作业配置权限：状态回写使用 `statuses: write`，读取 Actions 与审批使用
`actions: read`，派发接续使用 `actions: write`，PR 评论使用 `pull-requests: write`。
接收和发布还按 YAML 配置检查读取权限，Pages 部署使用 `pages: write` 与 `id-token: write`。

新任务使用 `control_policy=worker`，不绑定服务器控制版本；Worker 按 Gitee 控制分支同步。
控制更新和 profile 配置见 [服务器准备](../local_ci/prepare/README.md)。

本地回归：

```bash
python3 -m pytest scripts/local_ci/tests -q
```

测试使用本地 bare Git 和模拟 API，检查冻结身份、回写归属、审批及交付行为。
