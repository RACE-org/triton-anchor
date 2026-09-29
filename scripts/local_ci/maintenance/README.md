# 运行维护

Worker 和执行器负责任务恢复，systemd 管理服务器进程；`maintenance/` 负责健康采集
与本地保留清理，Cloudflare 负责外部监测和 Gitee Issue 告警。

## 任务状态与恢复

Worker 最多有一个任务执行线程。主循环持续维护心跳、取消和恢复核对，
独立上传线程处理其他运行的补传。
任务依次处于 `preparing`、`running`、`sealing`、`publish_pending`、`published`；
恢复状态独立记录，动作、原因、次数、截止时间和结论写入 Journal。

| 当前证据 | 恢复动作 |
| --- | --- |
| 已封存结果 | 校验并补传原结果；Docker 不可用不触发重测 |
| 完整有效的执行报告 | 确认原执行停止后继续封存，pass 与 fail 同样处理 |
| 无可靠报告 | 确认原执行停止后，在原预算内重建隔离环境 |
| 确定的测试失败 | 封存并发布 fail，不作为临时环境问题重测 |
| 服务停机 | 保存可恢复状态；与 PR 关闭、取消、任务替代区分 |

宿主 checkpoint 保存可信封存上下文及 base/candidate 的 `environment.variants`，
候选源码不能替换该上下文。续封存使用原 checkpoint，重建重新校验两侧冻结源码；
已封存结果保持原字节。有效且可恢复的未封存任务暂缓控制更新，只有封存上传等待不阻止升级。
依赖与隔离规则见 [只读依赖](../prepare/DEPENDENCY_MOUNTS.md)。

### 预算

| 项目 | 默认上限或等待 |
| --- | --- |
| Codex 启动 | 10 次，首次计入 |
| Codex 执行 | 首次启动起共享 6 小时 |
| 普通重试 | 间隔 30 秒 |
| 新 session | 最多一次，计入启动次数；session 无效或连续两次 resume 退出后无有效进展时切换 |
| 环境创建 | 3 次，首次计入；等待 Docker/磁盘等依赖不消耗次数 |
| 准备恢复 | 首次基础设施故障起 6 小时，与已有执行截止时间取更早者 |
| 封存瞬态 I/O | 共 3 次，间隔 30、60 秒；无效报告不反复封存 |
| 上传 | 前 5 轮间隔 60、120、300、300 秒；此后每小时补传并告警 |
| 无有效进展 | 30 分钟提示、60 分钟复查；进程存活但静默时不自动终止 |

次数与截止时间持久化，重启、重建、更换 session 和同任务手动恢复均不刷新。
额度耗尽或无法确认剩余额度时停止自动重跑，保留证据并提示重新派发。
配置中的显式预算值优先。

认证错误等待凭据变化或明确恢复；限流按预算退避，不通过更换 session 规避。
分类依据 CLI 顶层错误、进程退出与实际容器状态，不扫描测试输出关键词。
CLI 事件只表明会话活跃，不重置恢复中的无进展计数。每轮退出后比较报告校验缺项是否
被修正，以及是否新增匹配该侧源码与环境的工具结果；重复回复、改写摘要和更新时间
不算有效进展。已观察到的完成项累计记录，来回增删同一项不能刷新计数。
报告不存在、格式/记录错误、必检缺失与参数不匹配会具体反馈给下一轮会话；有效的失败
报告直接封存。认证和限流不触发无进展切换，所有恢复沿用原次数及截止时间。
每次重试提供 `artifacts/recovery-context.json`，列出当前结果、报告缺项和历史证据。
环境重建通过同一冻结任务的 `previous_run_id` 追溯前次运行；仅复制有数量/大小上限的
报告、工具日志及 `recovery-notes.md`，按现有凭据脱敏规则处理，缺失或超限文件明确列出。
历史材料放在 `artifacts/recovery/<原 run_id>/`，不恢复原生 session、虚拟环境或安装状态，
也不自动作为新运行的通过证据。仅换会话时仍先核对遗留进程；重建沿用停止原容器的流程。
静默提示不等于进程挂死；只有明确退出、容器故障或达到总截止时间才触发相应恢复或结束。

### 手动恢复

需要提前恢复同一任务时，先在维护窗口停止 Worker，避免并行处理同一状态；
使用既有私有凭据环境执行，再启动服务：

```bash
systemctl --user stop triton-anchor-local-ci.service
CI_TASK_ID='<需要恢复的任务ID>'
python3 scripts/local_ci/agent_ci/worker.py \
  --config /home/anchor_ci/local_ci/config/local-ci.json --resume "$CI_TASK_ID"
systemctl --user start triton-anchor-local-ci.service
```

此操作保留次数和截止时间；耗尽任务拒绝自动重跑，已封存任务只提前安排补传。
更换认证凭据后 Worker 会重新检测，不将凭据写入命令行或公开快照。

## 健康与外部告警

| 组件 | 职责 |
| --- | --- |
| `health.py` | 约每五分钟独立采集心跳、任务进展、恢复预算、容器退出/OOM、systemd 与磁盘状态，发布 `worker-health.json` |
| Cloudflare Worker | 每五分钟读取公开快照，识别故障、维护告警 Issue 和健康缓存，不远程执行恢复 |
| Worker 页面 | 展示当前执行、恢复、资源、独立上传等待及 Cloudflare 告警 |
| Summary | CI Request 确认请求后即为 pending；派发后接收器最多每五分钟读取健康源显示进展，最终结果优先 |
| 本机 Journal / systemd 日志 | 保存具体错误和私有诊断 |

心跳、采集时间和有效任务进展分别记录。容器状态来自实际查询，查询失败标记未知；
正常完成的 oneshot 不当作停止的常驻服务。快照过期只说明上报停止，
Gitee 不可读不能单独证明服务器宕机。

公开数据仅包含允许的状态、时间、资源与错误类别；私有路径、凭据、完整 session
和原始异常留在本机。任务的恢复状态和预算只发布当前快照，不发布历史过程；
无任务或缺字段时显示未上报。

同一连续异常复用一个 Gitee Issue，当前故障集合变化时更新。
恢复需要晚于故障证据、与同一对象匹配的新鲜正常快照；
过期数据、缺失字段或另一任务成功不能证明恢复。
Cloudflare 的缓存与告警规则见 [外部告警](cloudflare/README.md)。

### 检查上报

从服务器控制目录用现有 unit 触发采集，复用服务凭据：

```bash
systemctl --user start triton-anchor-local-ci-health.service
systemctl --user show triton-anchor-local-ci-health.service \
  --property=Result,ExecMainStatus
```

核对执行结果、本地快照及 Gitee `worker-health.json` 的采集时间和 Worker ID，
再等待一次 Cloudflare 周期观察页面与 Issue。
上传失败保留本地快照和错误类别。
健康采集默认可预览，`--publish` 才发布：

```bash
python3 scripts/local_ci/maintenance/health.py \
  --config /home/anchor_ci/local_ci/config/local-ci.json
```

## 结果保留

`retention.py` 每天清理已发布满 30 天的本地大日志和证据，保留结果摘要。
准备中、运行中和待上传任务不清理；`retention_until` 可延长指定 run 的保留期。
本地清理不改 Gitee 结果，磁盘不足时暂停新任务，但封存结果优先补传。
补传按原 `task_id + run_id` 定位。

```bash
python3 scripts/local_ci/maintenance/retention.py \
  --config /home/anchor_ci/local_ci/config/local-ci.json
```

默认预览，审阅后加 `--apply` 清理。

## 部署与验证

服务器代码、服务与配置按 [环境准备与部署](../prepare/README.md) 操作。
涉及健康展示时，先更新 [Cloudflare](cloudflare/README.md)，再发布 Dashboard，
最后更新服务器控制代码，逐项核对真实部署结果。

Dashboard 发布使用 `main` 的 CI Gateway 手动 `mode=publish`，不带任务 ID；
页面发布与服务器部署分别执行。等待健康采集和 Cloudflare 周期后，
检查新鲜快照、服务/容器状态、页面及 Issues。

手动回退前停止 Worker 并保存 Journal、封存结果和 outbox，
确认目标代码能理解未完成任务。安装器的 `--rollback` 仅恢复 unit 文件；
代码回退使用控制更新器，保留任务状态与预算。

### 本地检查

```bash
python3 -m pytest scripts/local_ci/tests -q
node --test scripts/local_ci/maintenance/cloudflare/worker.test.mjs scripts/local_ci/tests/dashboard.test.cjs
```

测试使用临时目录和模拟故障，不需要生产 Token，也不创建真实 Issues。

### 正常任务验收

使用维护者派发的测试任务或适用的已有记录，记录 task_id、run_id 和 head/tested SHA。
确认前置检查、审批和投递按依赖推进，Worker 经准备、执行、封存、上传完成，
执行期间心跳持续更新，GitHub、Gitee 与 Dashboard 的结果及证据一致。

### 上传恢复演练

使用独立演练状态与结果目标，保持生产任务和健康上报可用：

1. 仅让演练结果上传失败，记录封存 `result.json` 摘要、run_id、Codex 次数和截止时间。
2. 重启演练 Worker，确认只补传原结果，不创建任务容器、不启动 Codex，也不刷新预算。
3. 恢复结果仓库访问，确认同一摘要和 run_id 发布成功。
4. 检查新鲜健康快照证明交付恢复，Cloudflare 更新并关闭对应事件。

演练保持整机 Docker、依赖目录和生产任务正常运行。
已封存未入队、响应丢失、有效 fail 报告续封存、认证等待和 session 切换可用相关回归用例验证。
反馈保留实际版本、上报时间、任务证据、演练摘要与预算保持情况，避免公开私有诊断。
