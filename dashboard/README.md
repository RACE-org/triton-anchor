# CI Dashboard

提供三个业务视图：`local-ci.html` 的任务与证据，`index.html` 的全量算子和后端与性能；另有独立的 `worker.html`「Worker 运行状态」页面。

## 数据来源与业务发布

业务视图统一读取 Gateway 生成的 `data/tasks.json`，schema 为 `triton-anchor-dashboard`；
发布时从 `_site/data/tasks.json` 复制到页面的 `data/`。[data.js](data.js) 将结果投影到三个界面，
保留选测原因、审查依据、失败详情、筛选、分页及 CSV/XLSX 下载。

全量算子明细单独保存在
`runs/ci_full_flaggems/<tested_sha>/<run_id>/flaggems-summary.json`，
schema 沿用 `triton-anchor-local-ci/flaggems-v1`，不并入任务仓库的 `result.json`。
Gateway 校验结构后按 tested SHA 与 run ID 关联到任务 feed；路径和内容的校验见
[历史结果采集](../scripts/ci/dashboard_history.py)及[封存校验](../scripts/local_ci/agent_ci/delivery.py)。

全量算子和后端与性能使用 [data.js](data.js) 中配置的业务仓库、发布分支和 Profile；
PR 结果保留在任务详情，不能覆盖这两个业务视图。

- 全量算子优先选择发布分支显式 `full=true`、实际执行 FlagGems `mode=full` 的合规结果。
  尚无这类结果时，回退到指定历史样例；兼容旧的
  `runs/ci_full_flaggems/<tested_sha>/flaggems-summary.json` 路径。
  样例身份常量见 `data.js`，不能用任意旧记录替代。
- 后端与性能接受发布分支的 `push` 和 `manual` 任务。后端汇总保留各后端最近一次已执行检查；
  三项性能指标必须来自同一次任务中通过校验的固定 runner 结果，不能跨任务拼接。
  提交、环境指纹、Profile、LLVM 版本及各工具的固定 kernel 和采样参数必须匹配；无有效测量时留空。

任务详情分别展示 `environment.variants.base` 与 `candidate` 的 Profile；
业务视图要求 candidate 明确 `backend_enabled=true` 且记录 `backend_profile`，
分别标注后端和 Profile。单环境结果读取同层字段；缺少后端身份的记录只留在任务视图，
缺少 Profile 时显示“未记录”。
“后端测试”读取 `backend_smoke`（基本功能与真实 JIT）；报告漏写时 Worker 从本次候选环境的匹配工具结果补齐，
缺少记录时显示“未记录”，不以构建、安装或其他测试的状态代替。

接收器还读取结果仓库 `runs/` 中的历史结果及旧版 `delivery-summary.txt` 对应的真实报告。
旧版摘要明确记录的 `backend_profile` 转为候选后端身份；`triton_profile`、`triton_version`
按原始记录保留，缺失时不从后端名或分支名补造。
历史记录只用于展示，不重跑、不回写旧任务的 GitHub 门禁。
业务快照保留来源提交与测量时间；新任务未选择或尚未完成相关检查时，保留最近一次有效数据。
编译时间展示四个固定 kernel，Pass 过滤汇总项后显示前 10 个热点，IR 展示五个固定指标在四个 kernel 上的中位数。

## 状态与证据

结果与所选文件随同一 Git 提交发布。页面展示所有检查状态、审查结论和文件链接；
未选择、未执行和不适用的检查保留原始状态及说明，未选中或超预算的文件保留在主机并说明省略原因。
PR 评论列出实际执行的检查，并链接回本页面查看完整记录。

“未通过”（fail）表示检查或审查明确未通过，“执行错误”（infra_error）表示执行过程出错；
两者分别展示和筛选，算子统计及后端状态保留此区别。未知错误不根据非零退出码猜测根因。

任务详情的“阻塞原因”优先逐项展示阻塞 findings 的结论、分析和代码位置，不把失败检查或审查诊断重复列为缺陷。
没有阻塞 finding 时展示 `blocking_reasons`；失败报告仍缺少原因时使用任务摘要兜底。
检查状态、诊断与证据保留在详情和完整执行报告中。
`limitations` 单列“限制说明”，说明环境、工具、验证或证据发布限制及其对结论的影响，独立于整体通过或失败结论。
未提供该字段的旧结果兼容原有诊断分类；缺少审查不等于代码有问题，普通超时也不自动归因于网络。
发现项保留风险等级（严重/高/中/低/提示），缺失时显示“未标注”。

## Worker 运行状态

`worker.html` 独立加载 `health.js`，每五分钟匿名读取 Gitee 文件 API 的
`snapshot/<worker>/worker-health.json`。健康仓库、Worker ID、缓存地址和过期阈值
集中在 [health.js](health.js) 的 `source` 中；普通 raw URL 无浏览器跨域许可，不能替代文件 API。
页面展示运行概览、服务与资源、任务执行和恢复状态，结果上传等待单独列出。
预算内的自动恢复显示进展，耗尽后显示异常；缺字段或采集失败显示未知，心跳过期不沿用旧的正常状态。

Gitee 读取失败时使用 Cloudflare 缓存，限流后冷却 15 分钟。页面标明来源和更新时间，
两边均不可用时保留旧数据并标记待确认。缓存接口须先部署，操作及告警规则见
[Cloudflare 文档](../scripts/local_ci/maintenance/cloudflare/README.md)。

告警记录读取健康仓库最近 50 条更新的 Issues，按 Worker 标记筛选后展示最近 10 条。
Issue 状态与实时健康快照分别展示；外部监测及通知由 Cloudflare 执行。

## 本地预览与验证

```bash
python3 -m http.server 8000 --directory dashboard --bind 127.0.0.1
```

页面测试：`node --test scripts/local_ci/tests/dashboard.test.cjs`。
