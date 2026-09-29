# Cloudflare 外部告警与健康缓存

一个定时 Worker、一份 KV、一个 Gitee Secret。即使 CI 服务器停止或无人打开 Dashboard，Cloudflare 仍会每五分钟使用 Secret 读取 Gitee 的公开健康快照，并把异常写入健康仓库 Issues。同时保存一份健康展示缓存，供 Dashboard 在 Gitee 读取失败时使用。

源码为 `worker.mjs`，不依赖 npm 包或服务器新增服务。监测对象集中在文件顶部的 `CONFIG`：

- 健康仓库：`likehupochuan/triton-anchor-worker-health`
- Worker：`race-org-localci-v2`
- 快照分支：`snapshot/race-org-localci-v2`
- 快照文件：`worker-health.json`

本套 RACE 服务每五分钟运行；旧 `local-ci-alert` 服务改为每十分钟运行，并继续使用旧 Worker ID 与 KV 状态。按当前每轮两次 KV 写入，两套合计每天 864 次写入；同账户其他应用和手动触发另计。

Cloudflare 读取服务器 health 采集发布的 `worker-health.json`，负责告警与缓存；服务器 Worker、执行器和 systemd 负责恢复。

## 部署

首次启用 Workers 前，Cloudflare 账户邮箱必须已验证，否则上传会返回 `10034`。本目录 `wrangler.jsonc` 已记录当前账户及 `ALERT_STATE` KV 的部署标识（非凭据），后续命令行部署复用这些资源，不重复创建 namespace。

新账户还需在浏览器打开一次 **Workers & Pages**，完成默认 `workers.dev` 子域初始化；否则定时器配置可能返回 `10063`。本 Worker 使用 `workers_dev: true`，开放只读健康缓存地址。

1. 在 Cloudflare 的 **Workers & Pages** 创建 Worker，名称为 `local-ci-alert-race-v2`。把 `worker.mjs` 完整内容放入代码编辑器并部署。`GET /health` 提供缓存；其他路径返回 404，告警仍只通过定时入口工作。
2. 在 Worker 的 **Bindings** 中绑定 KV，变量名必须为 **`ALERT_STATE`**。当前账户复用 `wrangler.jsonc` 中已创建的 namespace；仅在其他账户首次部署时新建。
3. 在 Worker 的 **Settings → Variables and Secrets** 添加 **Secret**，名称为 **`GITEE_TOKEN`**。令牌需要能读取健康仓库内容，以及读取、创建、更新该仓库的 Issues。由维护者直接在 Cloudflare 输入，不放入仓库、Dashboard、聊天或普通明文变量。健康快照请求通过 `Authorization` header 携带 Secret，不把令牌放入 URL、日志或公开缓存。
4. 确认 `GITEE_TOKEN` 已配置，且 `snapshot/race-org-localci-v2` 已有身份正确的新鲜快照后，在 **Settings → Triggers → Cron Triggers** 添加 `*/5 * * * *`。只配置这一条定时器，也不要另部署第二个 Worker 同时维护同一个 Worker ID 的 Issues。
5. 查看 Worker 的执行记录，确认 scheduled 调用成功。定时配置传播可能需要最多 15 分钟。此项目无需接入 GitHub Actions。

更新使用本目录配置部署，保留 KV 绑定、Secret、定时器和已有 Issues。涉及健康展示时，先部署 Cloudflare，再发布 Dashboard，最后部署服务器控制代码。快照缺字段时显示未上报。暂停监测时移除 Cron Trigger，保留 KV 和历史 Issues；缓存数据按原采集时间判断是否过期。

也可使用官方 Wrangler 命令行部署，不依赖浏览器控制扩展。`wrangler login --device --scopes account:read user:read workers_scripts:write workers_kv:write workers_tail:read` 会显示授权地址和一次性代码，由维护者在自己的浏览器确认。凭据由 Wrangler 管理，不复制进仓库。首次部署先不设置 Cron，绑定 KV、配置 Secret 并验证之后再启用定时器；以后只维护同一份部署配置。

## 告警行为

- 同一轮连续异常合并到一个 Issue，列出当前故障类别、首次发现、持续时间和源快照时间。只有故障集合或健康读取诊断发生实质变化才更新，周期刷新不重复写入 Issue。
- 恢复必须由时间晚于故障证据的有效快照明确证明；旧快照、相同时间快照、未知字段、其他任务成功都不能清除对应故障。恢复补充证据时间并关闭原 Issue，新一轮故障再创建新 Issue。关闭失败重试前重新检查本轮证据，不能在新故障出现或读取失败时继续旧关单动作。
- 任务恢复耗尽后，只有同一 task/run 明确发布终态结果才可收尾任务级告警，正文写明“恢复失败，任务已结束”；不得写成恢复成功。仍存在的服务、认证等故障继续保留。
- 心跳快照超过 20 分钟未更新：报告心跳过期、当前服务状态未知，不能仅凭这一点区分停机、断网和采集停止。
- 连续两轮读取失败：报告健康数据读取异常。不可读或过期时保留此前未确认恢复的故障，不把缺数据解释成恢复。
- 新鲜快照中明确的 Worker、Docker、Gitee 访问、Codex 连接/认证/限流/执行异常，以及磁盘、环境、任务容器意外停止/OOM、恢复等待/耗尽、结果上传异常，会列入告警。正常结束的 oneshot 服务、未知字段不会误报故障。任务 30 分钟无进展提示、60 分钟等待复查，Cloudflare 不据此终止任务。
- Codex 连接异常在尝试次数小于上限时不告警；第 10 次仍在执行也不告警，只有第 10 次结束后仍连接失败才报告。
- `result_missing`、`cli_failed`、限流、会话失效等 Worker 已在预算内自动处理的恢复过程不单独告警，包括对应的 Codex 状态告警；恢复预算用尽后再按最终状态报告。新鲜快照明确显示原任务在预算内自动恢复时，解除对应旧误报；读取失败或预算未知不构成解除证据。认证失败、执行超时仍独立告警，不能被上一轮恢复原因遮盖。
- Codex 告警以实际运行或成功状态作为恢复依据。
- Issue 写入失败会在下一轮重试。新建请求响应丢失时先查找本 Worker 的活动告警，避免直接重复创建。KV 是最终一致性存储，本实现用于单 Worker、单定时器，不提供多个实例同时写入时的严格去重。
- 找回活动告警只查询 `open` 和 `progressing`，每种状态最多 3 页、每页 100 条，总计最多 6 次列表请求；不扫描已关闭历史。分页到上限仍未找回且最后一页满额时，不能确认原 Issue 不存在，因此本轮报错并跳过创建，下一轮重试；可检查活动 Issues 与执行日志处理积压。页面缓存的历史列表仍只取一页，查询失败或达到上限不阻止健康缓存刷新。

Gitee 故障同时可能影响健康读取和 Issue 写入；此时 Cloudflare 执行记录会报告失败，待 Gitee 恢复后重试。只接入 Gitee Issues 无法保证在 Gitee 本身不可用时送达告警。

创建或更新 Issue 不保证发送邮件。若需要邮件提醒，维护者还应检查 Gitee 的仓库订阅与个人通知设置，并实际验证送达。本实现没有接入额外邮件或机器人服务。

## Dashboard

发布新的 `dashboard/worker.html` 后，页面每五分钟优先匿名读取 Gitee 健康快照和告警 Issues；读取失败才用 Cloudflare 缓存替代这些数据。这个缓存读取不访问 Gitee，也不写 KV。明确遇到 Gitee 限流时，15 分钟冷却期间直接读缓存，之后再尝试 Gitee。两边都失败时保留最后取得的数据并说明读取失败；数据过期不能表示服务器宕机。

当前缓存地址为 `https://local-ci-alert-race-v2.2272640910.workers.dev/health`，与 `dashboard/health.js` 的 `source.cacheUrl` 保持一致。这个公开接口只读取 KV，支持跨域 GET；不会接收令牌、写入 Issue 或触发 Gitee 请求。首次部署后需等待一次定时采集，缓存尚不存在或 KV 不可读时返回 503。Cloudflare 的 Gitee 写入令牌和内部告警去重状态不会放入公开缓存。

每轮定时采集合并健康快照和最近告警为一个缓存条目，保留原始 `collected_at`、`updated_at`；外层 `updated_at` 只表示缓存刷新时间。某个来源读取失败时保留该部分旧数据，不把原始异常信息发布出去。新增可选 `health_read` 记录 Cloudflare 读取健康快照的结果：固定错误类别、HTTP 状态码、耗时、连续失败次数、本轮检测时间和最近成功读取时间。失败时输出相同的脱敏诊断日志；429 或正文明确包含限流标记的 403 才归类为限流。原始响应正文、异常和 Token 不进入日志或公开缓存。同一连续故障只有错误类别或 HTTP 状态等实质变化才更新原 Issue，次数和耗时变化不反复更新；定时任务执行完成不代表读取成功。

Dashboard 分别展示网页读取 Gitee、Cloudflare 读取 Gitee、服务器轮询任务仓库的状态，并标明当前健康数据来源。Cloudflare 读取失败不会覆盖网页直读取得的新鲜服务器状态；Cloudflare 缓存不可读或过期时，其监测状态显示未知。旧缓存缺少诊断字段时显示未上报；读取成功但源快照陈旧仍显示快照过期。Issue 写入失败仍会刷新缓存，缓存写入失败也不会阻止本轮告警处理。较旧快照不能覆盖较新的缓存。结果上传等待单独展示，明确只重传、不重新测试。

正常运行时每五分钟各写一次内部告警状态和合并缓存，合计约 576 次 KV 写入/天；页面访问只读缓存。浏览器缓存有效期为 60 秒。告警变化无需触发 GitHub 工作流、Pages 重新发布或 Git 提交。

Issue 正文用 `<!-- local-ci-alert:race-org-localci-v2 -->` 标记归属，不能删除这个标记。不要把 Issue 手工关闭当成修复服务；页面也会区分 Issue 状态与实时健康状态。

## 验证

本地运行 `node --test scripts/local_ci/maintenance/cloudflare/worker.test.mjs scripts/local_ci/tests/dashboard.test.cjs`，使用内存 KV 和模拟 Gitee API 检查故障、持续去重、恢复及写失败重试，不会向真实仓库写 Issue。

上线验收确认正常 scheduled 调用、故障 Issue、持续故障去重和新鲜快照触发的恢复关闭。检查恢复后到达的过期数据不会倒退状态，以及只读 `/health` 请求不会写入 KV。浏览器核对重试次数、下次重试、截止时间、独立上传等待和历史展开。故障测试使用独立测试仓库与 Worker，不修改生产心跳。

参考：[Cron Triggers](https://developers.cloudflare.com/workers/configuration/cron-triggers/)、[KV 写入限制](https://developers.cloudflare.com/kv/api/write-key-value-pairs/)、[Secrets](https://developers.cloudflare.com/workers/configuration/secrets/)、[Gitee API](https://gitee.com/api/v5/swagger)。
