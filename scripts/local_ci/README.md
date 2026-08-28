# Local CI 脚本

`scripts/local_ci/` 是 Triton-anchor 确定性 Local CI、Codex AI 辅助审查和结果发布链路的可信控制面。服务器配置中的 `LOCAL_CI_SCRIPT_DIR` 必须指向这个目录，使每次任务复制或快照时都包含同一份完整模块树。

## 先理解这条链路

```text
Gitee ci/* task ref
        |
        v
poll_gitee_and_run.sh
        |
        +--> orchestration/fetch_task_metadata.sh
        +--> orchestration/run_deterministic_ci_in_container.sh
        |          |
        |          +--> deterministic_ci/
        |          +--> results and performance artifacts
        |
        +--> codex_ai/run_codex_ai_ci.sh  (non-blocking)
        |
        +--> results/publish_gitee_result.py
                   |
                   +--> Gitee local-ci-results branch
                   +--> results/bridge_gitee_to_github_status.py
                              |
                              +--> GitHub status and PR comment
```

确定性 CI 的退出码决定 Local CI 是否成功；Codex AI 是非阻塞的补充审查层，不能覆盖确定性 CI 的结果。任务只有在结果成功发布后才会被写入已处理状态，避免发布失败造成任务丢失。

## 入口和模块边界

根目录只保留稳定 poller 入口；状态回收入口统一放在 `maintenance/`：

```bash
bash scripts/local_ci/poll_gitee_and_run.sh
python3 scripts/local_ci/maintenance/manage_local_ci_state.py \
  --state-dir /home/localci/local_ci/local-ci-state
```

poller 由 systemd 和人工运维调用，负责轮询、锁、防重复处理、任务编排、每日维护和结果发布。维护入口默认只预览，显式 `--apply` 才删除受管数据。其他运行代码按职责放在以下目录：

| 模块 | 负责什么 |
| --- | --- |
| `orchestration/` | 获取 PR metadata，向 Docker 容器传递任务参数。 |
| `deterministic_ci/` | 构建、前端/后端 smoke、FlagGems 和三类独立性能测量。 |
| `deterministic_ci/performance/` | compile-time、pass profile、IR serialization benchmark 和比较器。 |
| `codex_ai/` | exact-SHA checkout、临时容器、prompt、schema、报告和测试预算。 |
| `results/` | 固定 allowlist 复制产物、发布 Gitee 结果、回写 GitHub。 |
| `shared/` | task metadata、结果路径和 shell 路径归一化等跨模块协议。 |
| `maintenance/` | 记录和发布 Worker 健康状态，执行受管状态、artifact 和 Codex 残留回收。 |

依赖方向应保持单向：poller 调用 orchestration、deterministic、Codex、results 和维护入口；Codex 与 results 只通过 `shared/` 使用共享协议。不要重新增加根目录兼容 wrapper。

## 文件结构速览

```text
scripts/local_ci/
├── poll_gitee_and_run.sh                  # 稳定入口：轮询 Gitee task ref，串起确定性 CI、Codex 和结果发布
├── config.example.env                     # 部署配置模板；生产配置放在服务器环境，不提交仓库
├── README.md                              # 面向日常维护者的使用说明
├── orchestration/                         # 任务上下文准备和容器内确定性 CI 编排
│   ├── fetch_task_metadata.sh             # 读取 PR 标题、描述、base/head 等元数据
│   └── run_deterministic_ci_in_container.sh # 把任务参数传入 Local CI 容器并启动确定性 runner
├── deterministic_ci/                      # 真正执行确定性检查的脚本集合
│   ├── run_deterministic_ci.sh            # 容器内主 runner：构建、smoke、FlagGems、benchmark
│   ├── flaggems/                          # FlagGems 用例选择、批量执行和白名单数据
│   └── performance/                       # 三类性能采集、比较器和共享读取工具
├── codex_ai/                              # Codex AI 非阻塞辅助审查链路
│   ├── run_codex_ai_ci.sh                 # 渲染 prompt、启动临时容器、收集 AI 报告
│   ├── classify_codex_review_context.py    # 按 changed-files manifest 生成审查 profile、提示和文件分组
│   ├── prepare_codex_checkout.sh           # 准备并校验 exact-SHA disposable checkout
│   ├── setup_codex_ai_container.sh         # 部署前只读 prerequisite check
│   ├── validate_codex_ai_credentials.py    # 校验独立 Codex home 和凭据边界
│   ├── prompts/                           # success/failure prompt 和 prompt 维护记录
│   ├── codex_ai_analysis.schema.json      # Codex 语义输出契约（逐文件说明、行为覆盖和审查结论）
│   ├── codex_jsonl_evidence.py            # 解析可信命令、退出码、耗时和完成事件
│   ├── build_codex_ai_report.py           # 合并 Git、JSONL 和工作区事实，生成 canonical 报告
│   ├── codex_ai_report.schema.json        # 下游 canonical v3 报告契约
│   ├── render_codex_ai_report.py          # 校验 JSON 并渲染 Markdown 报告和 PR comment
│   └── tests/                             # Codex prompt、报告和容器 harness 测试
├── results/                               # 结果发布和 GitHub 回写
│   ├── publish_gitee_result.py            # 按 allowlist 发布 run 产物到 Gitee 结果分支
│   ├── bridge_gitee_to_github_status.py   # 将 Gitee 结果转换为 GitHub status / PR comment
│   └── tests/                             # bridge 和发布协议相关测试
├── maintenance/                           # Worker 可观测性和服务器侧保留治理
│   ├── local_ci_health.py                 # 原子记录状态、采集快照并发布到专用 Gitee branch
│   └── manage_local_ci_state.py           # 预览或回收受管状态、artifact 和 Codex 残留
├── shared/                                # 跨模块共享协议，避免各模块重复实现路径和 metadata 规则
│   ├── result_paths.py                    # Python 侧结果路径协议
│   ├── finding_locations.py               # finding 文件位置和行号边界校验
│   ├── capped_tee.py                       # 限制单日志大小并生成输出超限标记
│   ├── dump_artifacts.py                  # 归档当前任务失败 IR 并清理受控 dump 目录
│   ├── output_limits.py                    # 检查和收束任务 artifact 预算
│   ├── task_tmp.py                        # 创建、校验和清理任务级临时目录
│   ├── path_utils.sh                      # Shell 侧路径归一化
│   └── validate_task_metadata.py          # PR metadata 校验
└── tests/                                 # Local CI 模块布局和入口约束测试
```

阅读或修改时可以按下面顺序定位：

1. 任务没有被处理，先看根入口 `poll_gitee_and_run.sh` 和 `orchestration/`。
2. 构建、smoke、FlagGems 或性能结果异常，先看 `deterministic_ci/`。
3. AI 报告、prompt、schema 或 PR comment 异常，先看 `codex_ai/`。
4. Gitee 结果、GitHub status 或 PR 评论发布异常，先看 `results/`。
5. 涉及路径、task metadata、结果目录命名的兼容问题，先看 `shared/`。

`__pycache__/`、`.pyc`、临时 run 目录和服务器上的生产配置都不是源码结构的一部分，不应作为稳定接口引用。

## Task Ref

轮询器只接受 `GITEE_BRANCH_INCLUDE_REGEX` 匹配的任务分支，并始终排除结果分支：

| task ref | 用途 | 处理方式 |
| --- | --- | --- |
| `ci/push/<branch>` | 普通 push | 与上一次成功处理的 push 比较。 |
| `ci/pr-<number>/<branch>` | PR head | 使用对应目标分支的 exact base SHA 做 merge-base diff。 |
| `ci/base/pr-<number>/<branch>` | PR base 指针 | 不直接执行，供性能 baseline 和 PR 元数据链路使用。 |
| `ci/full/<branch>` | 手工全量 FlagGems | 使用 full 模式，不应作为普通 push 的默认开销。 |
| `ci/meta/pr-<number>/<branch>` | PR 功能声明 | 由 metadata fetch/validator 读取，不作为 CI task 执行。 |

结果目录映射由 `shared/result_paths.py` 固定：

```text
runs/ci_push/ci_push_<branch>/<sha>/<run-id>/
runs/ci_pr/ci_pr-<number>_<branch>/h-<head12>_m-<merge12>/<run-id>/
runs/ci_pr/ci_base_pr-<number>_<branch>/<sha>/<run-id>/
runs/ci_full/ci_full_<branch>/<sha>/<run-id>/
```

PR candidate 目录同时标识 head 和 Merge-Result；旧的纯 Merge SHA 目录不再由 receiver 读取。`safe_path_part` 会压缩非法字符并可能发生碰撞，`shared/path_utils.sh` 和 `shared/result_paths.py` 必须保持相同的分支名归一化语义。

## 一次任务的生命周期

1. Poller 发现符合规则的 Gitee ref，并用 lock 文件保证同一服务器不会并发处理相同 poller 状态。
2. PR 任务读取并校验与 GitHub test merge SHA 匹配的 `task-metadata.json`，同时保留 base/head SHA 供 diff 和身份校验。标题和描述只作为声明证据，不能作为命令执行。
3. Poller 从可信提交读取 `triton/cmake/llvm-hash.txt`，按 `LOCAL_CI_PROFILE_DIR/<llvm-hash>.env` 选择服务器 profile。PR 使用可信 base 的 hash，并要求被测提交保持相同 hash；push 使用被测提交的 hash。未知 hash、缺失 profile 或升级 PR 均报告未部署匹配环境，不回退到默认 Sophgo 容器。
4. Runner 将 Local CI 控制脚本复制到所选 profile 的容器内临时目录，并执行 `deterministic_ci/run_deterministic_ci.sh`。确定性 runner 在独立 artifact 目录写入 smoke、可选后端/FlagGems/benchmark、比较结果和 `delivery-summary.txt`；每条受控命令使用独立 Triton dump 目录，失败时只归档本命令的 `.ttir`、`.linalg`、`.pplir`，然后清空 `/workspace/triton-dump-dir`、root fallback 和任务 dump。任务退出时还会执行 best-effort `uv cache prune --ci`；失败只记录警告，不改变门禁结果。
5. 如果分支匹配 `CODEX_AI_CI_BRANCH_REGEX`，Codex 从当前 profile 的 Local CI 容器快照创建一次性容器，使用只读 `/workspace` 和目标 SHA 的 writable checkout；snapshot 阶段本身不再增加清理或审计。它继续复用确定性 CI 已安装到 venv 和容器可写层的依赖，并透传 profile 的 LLVM、PPL、backend 和构建并发配置。PR 只 source poller 从可信 base 提取的 `envsetup.sh`，push 才使用被测提交中的脚本；环境准备失败时本次非阻塞 Codex 审查直接报告环境启动失败，不在残缺环境中继续测试。只有启用后端阶段的 profile 才 source backend envsetup；任务内统一把 `TRITON_DUMP_DIR` 改到自身 `/tmp`，避免定向复跑写只读 volume。
6. Codex 先使用 changed-files manifest 生成轻量上下文分组和审查 profile，再只输出 schema 约束的 JSON；renderer 校验 manifest、中文说明、测试状态、命令退出码和预算后生成 Markdown 报告与 PR comment。
7. Publisher 只复制固定 allowlist 的结果文件，写入 `publish-manifest.json`，原子更新 `latest.txt`，更新 SHA/profile 性能 cache 和 dashboard，然后带 rebase retry push `local-ci-results`。
8. Bridge 读取 `latest.txt`、`publish-manifest.json`、summary、`result.json` 和 Codex comment，校验 SHA/run/schema 后发布 GitHub statuses，并只对 PR 创建或更新带 marker 的 advisory comment。

## 配置

从模板开始：

```bash
cp scripts/local_ci/config.example.env /home/localci/local_ci/config.env
```

生产配置不得提交到仓库。至少需要确认以下几组变量：

| 配置组 | 关键变量 | 检查重点 |
| --- | --- | --- |
| Gitee relay | `GITEE_REPO_URL`、`GITEE_TOKEN`、`GITEE_BRANCH_INCLUDE_REGEX` | token 只放部署环境；结果分支不能被轮询。 |
| Local CI 状态 | `LOCAL_CI_STATE_DIR`、`LOCAL_CI_PROFILE_DIR`、`LOCAL_CI_SCRIPT_DIR` | 状态目录可写；profile 目录由服务器维护；脚本目录是当前 checkout 的完整 `scripts/local_ci`。 |
| 版本 profile | `<llvm-hash>.env` 中的 `LOCAL_CI_PROFILE_NAME`、`LOCAL_CI_CONTAINER`、`LOCAL_CI_WORKSPACE_HOST`、`LLVM_BUILD_DIR`、`PYTHON_VENV_ACTIVATE`、`RUN_BACKEND_STAGES` 和各可选阶段开关 | 文件名必须是可信 LLVM hash；一个任务只加载一个 profile，配置只在该任务子进程中生效。启用后端时还必须提供 backend profile 名、路径、发现名、JIT 命令和 wheel 匹配模式。 |
| 结果发布 | `GITEE_RESULTS_*`、`PUBLISH_GITEE_RESULTS` | 结果仓库、branch 和 Web URL 必须互相对应。 |
| Worker 监控 | `LOCAL_CI_HEALTH_*`、`LOCAL_CI_WORKER_ID`、`GITEE_WORKER_HEALTH_REPO_URL`、`GITEE_WORKER_HEALTH_BRANCH` | Poller 状态写入 `LOCAL_CI_STATE_DIR/health`；最新完整快照发布到专用公开 Health 仓库。 |
| Codex | `RUN_CODEX_AI_CI`、`CODEX_BIN`、`CODEX_AI_CI_HOME` | 使用独立 `config.toml`/`auth.json`；runner 通过 Local CI 容器 snapshot 运行，凭据只复制到临时容器的 `/root/.codex`。 |
| Codex 预算 | `CODEX_AI_CI_TIMEOUT_SECONDS`、`CODEX_AI_CI_PREPARE_TIMEOUT_SECONDS`、`CODEX_AI_CI_STARTUP_TIMEOUT_SECONDS`、`CODEX_AI_CI_MAX_TEST_COMMANDS`、`CODEX_AI_CI_RECOMMENDED_COMMAND_TIMEOUT_SECONDS`、`CODEX_AI_CI_TEST_BUDGET_SECONDS` | hard timeout 仍为 3600 秒，报告预留仍为 450 秒；最多 50 条命令、单条建议 900 秒、累计建议 2700 秒。容器准备默认限时 1500 秒，准备成功后 900 秒内没有首个有效进展会提前终止；建议预算超限只产生 warning。 |
| 资源和时限 | `LOCAL_CI_TASK_TIMEOUT_SECONDS`、`LOCAL_CI_FULL_TASK_TIMEOUT_SECONDS`、`MAX_JOBS`、`CMAKE_BUILD_PARALLEL_LEVEL`、`NINJAFLAGS`、`CODEX_AI_CI_CPUS`、`CODEX_AI_CI_MEMORY`、`CODEX_AI_CI_PIDS_LIMIT` | 普通任务 6 小时、full 任务 48 小时；生产 `config.env` 建议将构建并行度三项统一为 8，脚本缺省值仍为 1；Codex 临时容器由 runner 设置 cgroup，持久 profile 容器由服务器设置。 |
| 输出预算 | `LOCAL_CI_LOG_MAX_BYTES`、`LOCAL_CI_ARTIFACT_FILE_MAX_BYTES`、`LOCAL_CI_ARTIFACT_MAX_BYTES`、`GITEE_RESULT_MAX_BYTES` | 单日志 512 MiB、单文件 2 GiB、单任务 5 GiB、单次 Gitee 发布 256 MiB。 |
| 保留维护 | `LOCAL_CI_MAINTENANCE_*`、`LOCAL_CI_ARTIFACT_HOST_ROOTS` | 每 24 小时在任务轮次之间执行；成功 14 天、失败 28 天、无结果目录 7 天、Codex Docker 残留 72 小时。 |
| backend | `BACKEND_PROFILE`、`BACKEND_PATH`、`BACKEND_ENVSETUP` | profile、backend commit 和环境脚本必须与性能 baseline 相匹配。 |
| benchmark | `RUN_COMPILE_BENCHMARK`、`RUN_PASS_PROFILE`、`RUN_IR_SERIALIZATION_BENCHMARK` | 三类测量有独立 cache namespace，不能混用阈值或结果。 |

## Worker 运行状态

Poller 以 fail-open 方式维护以下本地文件：

```text
LOCAL_CI_STATE_DIR/health/
├── poller.json
├── active-task.json
├── last-result.json
└── snapshot.json
```

这些文件只记录运行事实。当前任务耗时、磁盘可用空间、容器 CPU/内存/PID 限制均不在本模块中做阈值判定，不会终止任务、清理数据或改变 Local CI 结果。Dashboard 将 Docker CPU 百分比按 `100% = 1 CPU` 换算成当前使用核数，并结合容器 CPU 限额计算限额利用率；顶部卡片直接显示限额利用率和 CPU 限额。Publisher 将完整快照作为根目录唯一的 `worker-health.json` force-push 到 `GITEE_WORKER_HEALTH_REPO_URL` 的配置分支；该仓库不属于 `ci/*` task ref，也不参与任务队列生命周期。

服务器可用独立的 oneshot service 每次生成并发布一份快照：

```ini
[Unit]
Description=Publish Triton Anchor Local CI worker health
After=network-online.target gitee-poll-race.service

[Service]
Type=oneshot
User=localci
EnvironmentFile=/home/localci/local_ci/config.env
WorkingDirectory=/home/localci/local_ci/control_anchor/triton-anchor
ExecStart=/usr/bin/python3 scripts/local_ci/maintenance/local_ci_health.py snapshot
ExecStart=/usr/bin/python3 scripts/local_ci/maintenance/local_ci_health.py publish
```

对应 timer 只负责定时刷新，不发送通知：

```ini
[Unit]
Description=Refresh Triton Anchor Local CI worker health

[Timer]
OnBootSec=2min
OnUnitActiveSec=5min
RandomizedDelaySec=30
Persistent=true

[Install]
WantedBy=timers.target
```

Dashboard 进入 `Worker 运行状态` 页签后，通过公开 Gitee Contents API 读取 `worker-health.json`，并在页签保持打开时每五分钟刷新。页面显示 Poller 心跳、当前任务、常驻容器、存储和最近结果；快照缺失或超过二十分钟未更新时显示未知或离线。GitHub Pages workflow 不再定时搬运健康数据，监控链路失败不会回写 GitHub status，也不会影响正在运行的任务。服务器侧原始状态可通过 `LOCAL_CI_STATE_DIR/health/*.json`、`systemctl status` 和 `journalctl` 查看。

自动处理不可信 PR 时，容器内优先使用只读 relay token 或不传 token。当前部署选择保留示例默认值 `LOCAL_CI_ALLOW_WRITE_TOKEN_IN_CONTAINER=1`，因此缺少独立只读 token 时，候选代码容器可能获得 Gitee 写 token；这是明确保留的残余风险。Codex 通过 Local CI 容器 snapshot 运行，并只复制 exact-SHA checkout 和必要输入；AI 仍以 root、联网和 `danger-full-access` 运行，不能把它描述为完整 hostile-code 隔离。

PR 的 deterministic supervisor 和后续 Codex 临时容器都不 source Merge-Result 中的 `envsetup.sh`。poller 从已验证的精确 base ref 提取该文件放入本次 runner 快照，候选版本仅执行 `bash -n`；Codex 的 writable checkout 仍是被测 exact SHA，但环境入口来自同一可信 base。任何成功选中的 profile 都继续要求现有 `frontend_build` 和 `frontend_smoke`；`RUN_BACKEND_STAGES=true` 时沿用现有后端 required 逻辑，`false` 时后端构建、JIT、FlagGems 和性能阶段统一记录为 `skipped`。计划部署的 3.3/3.6 frontend profile 使用后一种配置；服务器部署并真实运行前，不能把代码中的路由目标描述为已经验证，部署后的绿色结果也只表示当前前端范围通过。

所有 profile 都先完成 frontend build、安装、导入和 `frontend_smoke`，这部分不加载厂商 backend 环境。启用后端阶段时，runner 随后加载 profile 的 backend `envsetup.sh` 并重建 backend wheel；重建完成后再次加载 frontend 和 backend 环境，再执行 backend discovery、smoke 和 JIT。frontend-only profile 在 frontend smoke 后直接结束，不读取 backend 配置。

部署前只检查依赖、不创建长期 Docker 资源：

```bash
CODEX_AI_CI_HOME=/home/localci/local_ci/secrets/codex-ai \
  bash scripts/local_ci/codex_ai/setup_codex_ai_container.sh
```

首次启用维护前先预览，然后设置持久容器资源并核验：

```bash
# 用途：只读预览将按 14/28/7 天规则回收的 run、runner snapshot、
# artifact 和带 Local CI 标签的 Codex Docker 残留；没有 --apply，不会删除。
python3 scripts/local_ci/maintenance/manage_local_ci_state.py \
  --state-dir /home/localci/local_ci/local-ci-state \
  --artifact-root /home/localci/local_ci/workspace/local-ci-artifacts \
  --artifact-root /home/localci/local_ci/profile-workspaces/sophgo-cmodel/local-ci-artifacts \
  --artifact-root /home/localci/local_ci/profile-workspaces/triton-3.3-frontend/local-ci-artifacts \
  --artifact-root /home/localci/local_ci/profile-workspaces/triton-3.6-frontend/local-ci-artifacts \
  --success-days 14 --failure-days 28 --incomplete-days 7 \
  --docker-orphan-grace-hours 72

# 用途：只对受管 artifact 根及其现有内容授权 poller 账号清理权限；
# 默认 ACL 会被容器以后以 root 创建的新目录和文件继承。
# Ubuntu/Debian 服务器首次配置时安装提供 setfacl/getfacl 的 acl 软件包。
command -v setfacl >/dev/null || sudo apt-get install -y acl
command -v setfacl >/dev/null || {
  echo "setfacl is required; install the server acl package first." >&2
  exit 1
}
ARTIFACT_ROOTS=(
  /home/localci/local_ci/workspace/local-ci-artifacts
  /home/localci/local_ci/profile-workspaces/sophgo-cmodel/local-ci-artifacts
  /home/localci/local_ci/profile-workspaces/triton-3.3-frontend/local-ci-artifacts
  /home/localci/local_ci/profile-workspaces/triton-3.6-frontend/local-ci-artifacts
)
for root in "${ARTIFACT_ROOTS[@]}"; do
  [[ -d "${root}" ]] || continue
  sudo find "${root}" -type d \
    -exec setfacl -m u:localci:rwx,d:u:localci:rwx {} +
  sudo find "${root}" -type f \
    -exec setfacl -m u:localci:rw- {} +
done

# 用途：逐根验证 root 新建的 artifact 可以由宿主机 poller 账号删除。
# 默认 workspace 没有独立持久容器，直接使用宿主机 root 模拟写入。
probe_artifact_root() {
  local host_root="$1"
  local writer_container="${2:-}"
  local probe_host="${host_root}/.poller-acl-probe"

  if [[ -n "${writer_container}" ]]; then
    if ! docker exec "${writer_container}" bash -lc \
      'probe=/workspace/local-ci-artifacts/.poller-acl-probe
       rm -rf -- "${probe}" && mkdir -p "${probe}/child" &&
       : > "${probe}/child/probe.log"'; then
      echo "artifact ACL probe create failed: ${host_root}" >&2
      return 1
    fi
  else
    sudo rm -rf -- "${probe_host}" || return 1
    sudo mkdir -p "${probe_host}/child" || return 1
    sudo touch "${probe_host}/child/probe.log" || return 1
  fi

  if [[ ! -e "${probe_host}/child/probe.log" ]]; then
    echo "artifact ACL probe is not visible on host: ${host_root}" >&2
    return 1
  fi
  if rm -rf -- "${probe_host}" && [[ ! -e "${probe_host}" ]]; then
    echo "artifact ACL probe: PASS ${host_root}"
    return 0
  fi
  echo "artifact ACL probe delete failed: ${host_root}" >&2
  return 1
}

probe_failed=0
probe_artifact_root /home/localci/local_ci/workspace/local-ci-artifacts \
  || probe_failed=1
probe_artifact_root /home/localci/local_ci/profile-workspaces/sophgo-cmodel/local-ci-artifacts \
  anchor-sophgo-ci-prod || probe_failed=1
probe_artifact_root /home/localci/local_ci/profile-workspaces/triton-3.3-frontend/local-ci-artifacts \
  anchor-triton-3.3-ci || probe_failed=1
probe_artifact_root /home/localci/local_ci/profile-workspaces/triton-3.6-frontend/local-ci-artifacts \
  anchor-triton-3.6-ci || probe_failed=1
[[ "${probe_failed}" == "0" ]] || {
  echo "artifact ACL probes: FAIL" >&2
  exit 1
}
echo "artifact ACL probes: PASS"

# 用途：实时统一设置三个持久 CI 容器的 CPU、内存和 PID 硬上限；
# 命令不会重启容器，但容器重建时需要在创建配置中重新声明这些值。
docker update --cpus 16 --memory 64g --memory-swap 64g --pids-limit 4096 \
  anchor-sophgo-ci-prod anchor-triton-3.3-ci anchor-triton-3.6-ci

# 用途：只读核验三个持久 CI 容器最终生效的 cgroup 配置。
docker inspect --format \
  '{{.Name}} cpus={{.HostConfig.NanoCpus}} memory={{.HostConfig.Memory}} swap={{.HostConfig.MemorySwap}} pids={{.HostConfig.PidsLimit}}' \
  anchor-sophgo-ci-prod anchor-triton-3.3-ci anchor-triton-3.6-ci
```

这里的 `--memory-swap` 是内存与 swap 的总上限；与 `--memory` 设为相同值表示容器不额外使用主机 swap。

维护由长驻 poller 以其宿主机账号执行，不需要额外的 root service/timer。容器仍可保持
root 身份；服务器只为 `LOCAL_CI_ARTIFACT_HOST_ROOTS` 配置 poller 账号的访问 ACL，禁止对
整个 profile workspace 执行递归 `chown` 或 `chmod 777`。dry-run 只验证候选识别，不会尝试
删除，因此启用 `--apply` 前必须通过上述“容器创建、宿主机删除”探针。

## 结果和报告

每个 run 目录通常包含：

```text
delivery-summary.txt
result.json
local-ci.log
codex-ai-ci-summary.txt
codex-ai-report.json
codex-ai-report.md
codex-ai-comment.md
codex-changed-files-manifest.json
codex-context-summary.json
publish-manifest.json
codex-workspace-status.txt
codex-workspace.patch
codex-generated-files.tar.gz
```

容器本地 artifact 目录可能额外包含 `failure-ir/<stage>/{manifest.json,task/,sophgo/,root/}` 和 `failure-ir-collection.log`。`failure-ir/` 只在失败命令实际产生 `.ttir`、`.linalg` 或 `.pplir` 时创建；不会复制 `.so`、成功命令 dump 或旧任务 dump。它供紧随确定性 CI 的 Codex 失败诊断读取，当前 publisher allowlist 不把原始 IR 推送到 Gitee 结果分支。确定性 runner 在 `/tmp/triton-anchor-local-ci-task.<sha>.<random>/` 下统一持有 `TMPDIR`、dump、runner、临时凭据和 benchmark 隔离目录；失败 IR 提升到该目录之外的 artifact 后，阶段 dump 立即清理，任务退出时按 ownership marker 回收整个任务目录。

任务清理不会扫描 `/tmp/[0-9]+-[0-9]+` 或其他全局路径，也不会触碰 `/root/.triton/cache`、`/root/.flaggems/code_cache`、`/root/.cache/uv`、`/root/.cache/pip` 和 `/opt/venv`。这些共享缓存继续供后续 Local CI 与 Codex 复用。每日维护只处理配置列出的 state/artifact 根目录，以及带 `triton-anchor.role` 标签、已经停止的过期 Codex 容器和未被任何容器引用的过期 snapshot 镜像。

本次不把 `TRITON_CACHE_DIR` 改为任务级临时目录。它保存以源码、Triton/backend 和编译配置为 key 的可复用编译产物，命中时会跳过编译 pipeline；与只用于诊断的 dump 不同。compile benchmark 已使用并清理独立 session cache。应先上线 dump 清理并观察 snapshot 计时与 cache 体积，再决定是否需要独立的 cache 生命周期方案。

Codex runner 会额外写入 `codex-context-summary.json`，用 `docs_only`、`codex_ai_ci_maintenance`、`local_ci_protocol`、`performance`、`local_ci_failure`、`large_diff` 等轻量 profile 提示模型优先读哪些文件、日志和 artifact。该 profile 只影响阅读和验证优先级，不改变必须覆盖全部 changed-files manifest、finding 证据标准或 schema 输出契约。其中 `scripts/local_ci/codex_ai/` 下的 Codex AI-CI 自身维护变更不纳入 triton-anchor 产品代码审查，不应生成产品 finding；同步性由专用契约测试和人工维护审查负责。

Codex 报告格式为 `triton-anchor-codex-ai-report/v3`。除审查摘要、文件变更、行为覆盖、findings、测试证据和剩余风险外，`change_request_assessment` 还必须说明：

- 贡献者的修改目标；
- 声明的预期行为；
- 当前 diff 实际实现情况；
- 支持该判断的代码、测试或 Local CI 证据；多条依据必须拆成独立条目。

实现状态 `implemented`、`partially_implemented`、`not_implemented`、`not_assessable` 和 `not_applicable` 只表示声明与实现的一致程度，不替代 `PASS`/`WARNING`/`FAIL`。PR comment 会优先展示审查摘要、本地确定性 CI 检查简述、贡献者目标与实现情况、需要处理的问题、验证情况和剩余风险；判断依据和验证说明应使用提交者和审核者可理解的自然语言，并按条目展示。`AI-001`、`TEST-001`、`RUN-001` 等机器关联 ID 只保留在结构化 JSON 和完整报告中，PR comment 将问题和建议测试显示为自然序号，并将 `RUN-xxx` 替换为对应执行记录的 `purpose`，例如“缓存失效定向测试”“Python 语法检查”或“扩展模块构建”；审查主体统一称为“Codex AI 自动审查”。文件级明细折叠在评论底部；完整报告保留全部证据。

机器 ID 至少使用三位数字，但不设置三位上限；大规模差异中的 `FILE-1000`、`AI-1000`、`TEST-1000` 和 `RUN-1000` 仍属于合法关联 ID，公开评论同样会隐藏或转换这些内部编号。

每个 finding 必须定位到本次 diff 中未删除的文件，并使用单行或起止有序的连续范围；prompt 要求优先选择能够定位根因的最窄范围，但范围宽度不再作为整份报告失效条件。Renderer 会在 exact-SHA checkout 中确认文件存在且范围没有越界；`code_role` 说明该行实际负责的功能。两类 finding 都会在 PR comment 保留核心证据。Bridge 从结构化报告生成固定到审查 SHA 的 GitHub 链接：定位有效时生成精确行链接；行号无效但文件仍能通过可信 FILE-ID 映射时生成不带行锚点的文件链接，并明确标注“具体行号待核对”；无法映射到可信变更文件时不伪造链接。模型遗漏逐文件说明或提供重复引用时，Runner 会保留可信 Git 文件清单并明确标记证据缺口；finding 定位无法验证时，其原始严重度、标题、证据、影响和修复方向会完整保留为“定位待核对的问题”，不会作废整份报告或隐藏问题。

PR comment 的“验证情况”不展示 `evidence_level` 或 `test_execution.status` 的内部状态标签，而是直接分为“验证内容与结果”“限制与未覆盖”两组事实。“验证内容与结果”按验证目标展示 Codex 对现有 Local CI 证据、静态审查范围、正式验证、诊断结果和覆盖路径的最终判断，以及任务级归档的测试文件；不再逐条复述 shell 命令。可信 JSONL 命令账本仍完整保留执行事实，`test_assessment.commands.role` 将测试、构建和 lint 等正式验证标记为 `validation`，将搜索、日志检查和环境探查标记为 `diagnostic`；`purpose` 是稳定的验证目标，只有失败之后通过不同方式完成的等价检查才能关闭该目标，同一命令出现通过和失败仍保留为非确定性结果。未关闭目标在“限制与未覆盖”中按目标说明原因和影响。未分类非零记录不改变正式验证状态或 verdict，也不进入公开评论；完整命令和退出码继续保留在 JSON、完整报告和诊断摘要中。未关闭的正式验证目标直接影响执行状态和 verdict；未关闭的诊断目标只有在造成实质证据缺口、使 `evidence_level=insufficient` 时才将 verdict 提升为 WARNING。PR comment 将该值表达为“需关注（非阻塞）”，避免与附带条件的“可以合入”建议形成阻塞语义。未执行建议测试和动态验证缺口同样进入“限制与未覆盖”，由缺口产生的具体行为风险才进入“剩余风险”。内部 `evidence_level`、执行状态和完整命令记录继续保留在结构化 JSON、完整报告和诊断摘要中，并独立参与 verdict 计算。

如果 Codex 审查在形成可信语义载荷前失败，fallback 仍从可信命令账本保留已经执行的命令、退出码和耗时，并尽量保留已收集的测试文件；账本为空才表示没有执行记录，账本不可读才表示执行事实不可确认。由于失败 fallback 无法可靠区分正式验证与诊断，已有命令会保守记为未分类并保持警告，不会被改写成正式验证通过或失败。

Codex AI-CI 的审查目标是服务 `triton-anchor` 仓库及其后续分支，而不是做泛化 AI 审查平台。主要关注：Triton/AnchorIR 前端语义、TTIR pipeline、adapter/ABI、C++/MLIR binding、Public API 兼容性、Local CI 任务/结果协议、后端 smoke/FlagGems/性能证据是否支持本次 diff。纯风格、泛化重构或与这些主线无关的建议不应扩大成阻塞 finding。

## 性能测量边界

性能程序共享固定 kernel/spec、统计汇总、JSON 读取、CSV 投影和相邻 compile worker 定位；publisher、dashboard 和 runner 也只复用与指标无关的 cache、baseline、URL 和状态读取机械逻辑。三个比较器仍保留各自的判定语义：

- compile-time：按 kernel 比较 compile estimate median，默认对称阈值。
- pass profile：可选择 slowdown 或 symmetric，另有最小 baseline 和最小绝对变化量。
- IR serialization：只对 serialize/deserialize 指标判断 slowdown，并使用自己的最小噪声门槛。

缺失 baseline、非法数据和超阈值通常写入 warning 并保留诊断结果；benchmark worker、构建或 smoke 失败才会让确定性 stage 失败。性能 warning 映射到 GitHub status 时可能仍是 success，必须查看详细 comparison artifact。

## 验证

不依赖 Docker、LLVM、backend 或 Codex 凭据的最小 Python 契约测试：

```bash
PYTHONPATH=python:scripts/local_ci \
  python -m pytest \
    scripts/local_ci/codex_ai/tests \
    scripts/local_ci/tests \
    scripts/local_ci/results/tests \
    -v --tb=short
```

Linux CI 还会执行三个 Shell harness：

```bash
bash scripts/local_ci/codex_ai/tests/test_local_ci_codex_ai.sh
bash scripts/local_ci/codex_ai/tests/test_local_ci_codex_container_setup.sh
bash scripts/local_ci/codex_ai/tests/test_local_ci_codex_container.sh
```

修改 runner 或 workflow 后至少检查：

```bash
bash -n scripts/local_ci/poll_gitee_and_run.sh
bash -n scripts/local_ci/deterministic_ci/run_deterministic_ci.sh
python -m py_compile \
  scripts/local_ci/codex_ai/render_codex_ai_report.py \
  scripts/local_ci/results/bridge_gitee_to_github_status.py \
  scripts/local_ci/deterministic_ci/performance/common.py
git diff --check
```

Windows Git Bash 不能替代 Linux harness：`python3`、`/tmp`、symlink 权限、Docker 命令和默认编码语义不同。无法执行的验证必须在结果或报告中标明未执行，不能写成通过。

## 排障顺序

| 现象 | 先检查 |
| --- | --- |
| 任务一直未处理 | `LOCAL_CI_STATE_DIR/poll.lock`、branch include regex、Gitee ref 是否存在、`last_processed` 是否已推进。 |
| 确定性 CI 未启动 | `local-ci.log`、容器是否运行、`LOCAL_CI_SCRIPT_DIR` 是否包含完整 canonical 模块树。 |
| 版本环境未部署 | 查看可信/被测 `llvm-hash.txt`、`LOCAL_CI_PROFILE_DIR/<hash>.env` 和 profile 中的容器、LLVM、venv 配置；系统不会回退到其他版本环境。 |
| 结果已生成但 GitHub pending | `local-ci-results` 是否 push 成功、`latest.txt` 是否指向当前 run、bridge 是否能读取 Gitee API。 |
| Codex 失败但 Local CI 通过 | 查看 `codex-ai-ci-summary.txt` 的 `failure_code`、`failure_reason`、Codex log 和凭据校验；这是非阻塞路径。 |
| 报告生成失败 | 先按 `failure_code` 区分 `analysis_contract_failed`（Codex 语义载荷不满足公开契约）、`trusted_report_input_failed`（manifest、命令账本或生成文件归档异常）、`report_contract_failed`（Runner canonical 报告内部契约异常）和 `report_metadata_failed`（Runner 无法读取执行事实）；finding 定位无效本身会保留为“定位待核对的问题”，不再作废整份报告。 |
| 性能 warning | 先确认 baseline SHA/profile、backend commit 和 artifact 有效性，再判断是否是真回归。 |
| PR comment 没有更新 | task ref 必须是 `ci/pr-<number>/...`，comment 必须包含稳定 marker 且由 Bot 发布。 |

更完整的协议和已知风险见 [`docs/ci_guide_zh_updated.md`](../../docs/ci_guide_zh_updated.md)、[`config.example.env`](config.example.env) 和 [`codex_ai/prompts/prompt_change_log.md`](codex_ai/prompts/prompt_change_log.md)。
