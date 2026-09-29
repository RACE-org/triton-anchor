# 环境准备与部署

`prepare/` 管理共享镜像、只读依赖、Rootless Docker 任务容器及用户服务。
部署使用固定控制提交、镜像 digest 和完整 LLVM SHA，源码与控制更新经 Gitee 到达服务器。

## 配置与环境

`config.example.json` 是 `anchor_ci` 服务器完整非敏感部署配置的唯一维护来源。
在开发仓库修改并提交，经 Gitee 部署；运行副本由安装器或控制更新器生成，
不单独维护覆盖 JSON。`profiles/config.template.json` 仅供其他部署参考。

> **迁移部署前必须确认 Rootless socket。** 示例中的 `unix:///run/user/1007/docker.sock` 保留自来源部署，`1007` **不是已确认的 `anchor_ci` UID，不能直接照抄部署**。服务器同事先运行 `id -u anchor_ci`，确认该用户实际 Rootless Docker socket 后，把仓库中 `config.example.json` 的 `runtime.endpoint` 改为真实路径并提交。`runtime.context` 已改为 `anchor-ci-rootless`，须与服务器实际创建的 context 一致。镜像名、digest 和依赖哈希也保留自来源部署，应核对目标用户能使用这些既有制品，再安装或启用控制更新。

| 内容 | 配置或位置 |
| --- | --- |
| 控制来源 | `control_repo_url` 为 `https://gitee.com/race-org/triton-anchor.git`，`control_branch` 为 `CI_dev`；任务与结果仍使用 `gitee_repo_url` 指定的 `triton-anchor-local-ci-results` 仓库 |
| 控制 checkout | `/home/anchor_ci/local_ci/control_anchor` |
| 运行配置 | `/home/anchor_ci/local_ci/config/local-ci.json` |
| 持久状态 | `/home/anchor_ci/local_ci/state` |
| 私有凭据 | `/home/anchor_ci/local_ci/config/credentials.env` 与 `codex-source/` |
| 宿主 Python | `/home/anchor_ci/local_ci/local-ci-control-venv/bin/python` |

安装器读取所在 checkout 的 HEAD 配置，控制更新器读取目标 SHA 的配置，
均按 JSON 结构完整同步运行副本。`--config` 指定运行副本位置，不是额外配置来源。
同步时内容相同则跳过写入，有差异先校验再原子替换，保持 CI 用户所有和 600 权限。

base / candidate 按各自冻结源码的 **Triton major.minor + 完整 LLVM SHA** 唯一选择 profile。
无匹配或多匹配均报告环境配置问题。仓库配置覆盖 3.0–3.6 和 3.8，
只有 3.0 开启后端；3.1 复用 3.0 LLVM，但使用独立 profile。
版本表、元数据解析和挂载规则见 [DEPENDENCY_MOUNTS.md](DEPENDENCY_MOUNTS.md)。

所有 profile 使用顶层 `image` 的同一镜像 digest，差异由只读依赖和环境变量提供。
公共 Python / 系统包在 [共享镜像](profiles/slim/README.md) 中准备。
构建默认并行度由 `max_jobs` / `MAX_JOBS` 配置，工具的 `jobs` 可按任务资源调整（1–64）。

## 部署前确认

使用普通 CI 用户、user systemd、Rootless Docker 和持久用户会话。
记录实际控制 SHA、运行配置差异、服务状态与在途任务；保存配置和 unit 备份，
保留 runs、封存结果、outbox 及恢复预算。

目标必须是经审核、已提交且可从配置的 Gitee 控制分支到达的完整 SHA。
检查目标配置包含要部署的全部 profile；服务器存在未入库差异时，先查明来源。
可恢复的未封存任务先完成恢复，封存后待上传的结果可继续补传。
部署失败时保留证据，检查代码与配置实际停在哪一步。

凭据文件属于 CI 用户、权限为 600，格式为 `KEY=value`，含空格的值使用引号。
安装器通过 `--credentials-env` 加载；控制更新命令在现有私有凭据环境中执行。
不要在命令行展开或打印凭据值。

## 新服务器安装

先通过可信配置管理或制品通道投放 `bootstrap_control.py`、目标提交的
`config.example.json` 分发副本和私有凭据。引导脚本创建固定版本 checkout，
然后调用该版本的正式安装器，确保代码与配置来自同一提交。

从投放目录预览，核对后对同一命令加 `--apply`：

```bash
CONTROL_SHA='<已审核的40位控制SHA>'
python3 bootstrap_control.py \
  --config /absolute/path/config.json \
  --credentials-env /absolute/path/credentials.env \
  --expected-revision "$CONTROL_SHA"
```

引导脚本在克隆前后核对固定 SHA。目标目录已存在时，要求来源一致、checkout 干净
且处于该 SHA；安装失败可在修复问题后用同一命令重试。

已有固定版本 checkout 时，从控制目录运行安装器：

```bash
cd /home/anchor_ci/local_ci/control_anchor
CI_PYTHON=/home/anchor_ci/local_ci/local-ci-control-venv/bin/python
CI_CONFIG=/home/anchor_ci/local_ci/config/local-ci.json
"$CI_PYTHON" scripts/local_ci/prepare/install.py \
  --config "$CI_CONFIG" \
  --credentials-env /home/anchor_ci/local_ci/config/credentials.env
```

预览显示配置差异与计划安装的 units；加 `--apply` 后，安装器获取控制锁，
准备所有 profile、实测容器、备份并同步 units、写入运行配置，再启动 Worker 和定时器。
不要在外层持有同一把 `control.lock`。

`--render-dir <目录>` 可保存 unit 文件。`--rollback <备份目录> --apply` 只恢复 unit 文件，
systemd 的 enabled/active 状态需要另行恢复；它不是代码、配置和任务状态的整体回滚。

## 更新已有服务器

### 更新代码与配置

从现有控制目录执行，`CONTROL_SHA` 替换为实际目标：

```bash
cd /home/anchor_ci/local_ci/control_anchor
CONTROL_SHA='<已提交并同步至Gitee的40位SHA>'
CI_PYTHON=/home/anchor_ci/local_ci/local-ci-control-venv/bin/python
CI_CONFIG=/home/anchor_ci/local_ci/config/local-ci.json

"$CI_PYTHON" scripts/local_ci/prepare/control_update.py \
  --config "$CI_CONFIG" --expected-revision "$CONTROL_SHA"
```

核对输出的 `revision` 与 `config_fields`。更新器检查 Gitee 可达性、前进或撤回的同一历史关系及任务占用；
通过后应用：

```bash
"$CI_PYTHON" scripts/local_ci/prepare/control_update.py \
  --config "$CI_CONFIG" --expected-revision "$CONTROL_SHA" --apply
```

成功返回 `updated` 或 `current`。返回 `deferred-active-task` 表示尚未部署，
即使退出码为 0，也应等待任务和清理完成后重试。
相同 SHA 下可用此流程修复配置偏差；它在需要时重启 Worker，让代码与配置一起生效。

### 同步服务或准备环境

按改动选择对应操作，避免重复准备：

| 改动 | 操作与目的 |
| --- | --- |
| 仅控制代码或普通配置 | 使用控制更新器；相同环境可复用已验证 runtime |
| systemd units 或宿主部署锚点 | 在空闲维护窗口停止 Worker，运行目标版本安装器预览并 `--apply`，同步服务与配置 |
| profile、只读依赖或镜像 | 准备对应 runtime 后执行正式预检；安装器已完成这两项时可复用其结果 |

`control_root`、`state_dir`、`python_bin` 和 `runtime` 属于宿主部署锚点，
控制更新器不热切换这些设置；变更前按安装器要求安排持久数据和服务。

仅新增 profile 时，在更新后的控制目录准备环境：

```bash
"$CI_PYTHON" - <<'PY'
import json
from pathlib import Path
import sys

config = json.loads(Path('/home/anchor_ci/local_ci/config/local-ci.json').read_text())
sys.path.insert(0, str(Path(config['control_root']) / 'scripts/local_ci'))
from prepare.runtime import EnvironmentManager

manager = EnvironmentManager(config, config['state_dir'])
for key, profile in config['profiles'].items():
    runtime = manager.ensure_image(key, profile['llvm_hash'])
    print(key, runtime['llvm_hash'], runtime['image_id'], runtime['state'])
PY
```

每个 profile 应返回 `ready`，共享相同镜像 ID，LLVM SHA 与配置一致。
`ensure_image` 复用相同环境的验证记录，并校验新环境的依赖和基础导入。依赖更新也可用
`rotate.py --config <运行配置> --profile <名称>` 登记和探测新环境。

## 部署验收

### 配置与容器

所有 profile 准备完成后，以同一 CI 用户加载私有环境并运行预检：

```bash
"$CI_PYTHON" - <<'PY'
from pathlib import Path
import sys

sys.path.insert(0, str(Path('scripts/local_ci').resolve()))
from prepare.install import load_environment
from prepare.preflight import main

load_environment(Path('/home/anchor_ci/local_ci/config/credentials.env'))
sys.argv = ['preflight.py', '--config',
            '/home/anchor_ci/local_ci/config/local-ci.json', '--probe-runtime']
raise SystemExit(main())
PY
```

预检验证真实挂载、访问权限、Rootless Docker 资源限制、运行账户及凭据配置。
退出码为 0、`ready: true` 且 `runtime_probe.status: pass` 表示基础运行条件通过。
`--configuration-only` 用于单独检查配置；`--skip-notifications` 仅用于配置开发，
正式部署需要校验通知凭据。

### 服务与任务

```bash
git rev-parse HEAD
systemctl --user is-active triton-anchor-local-ci.service
systemctl --user is-active triton-anchor-local-ci-health.timer
systemctl --user is-active triton-anchor-local-ci-retention.timer
systemctl --user start triton-anchor-local-ci-health.service
```

Worker 与两个 timer 应处于 active。health 和 control-update 是 oneshot，
正常完成后 inactive 属于正常状态。
核对实际 HEAD、运行配置和目标提交一致，再检查健康采集的时间、Worker 身份及上传结果。
检查方法见 [运行维护](../maintenance/README.md)。

真实任务沿 GitHub → Gitee → Worker 执行，记录 task_id、run_id、head/tested SHA、
构建结果及证据路径。相同源码和环境的有效记录可复用。
任务派发见 [网关文档](../../ci/README.md)。

服务器安装、Cloudflare 部署与 Dashboard 发布各有独立入口。
外部展示和告警更新见 [维护部署](../maintenance/README.md#部署与验证)。
验收记录包含控制 SHA、非敏感配置差异、runtime/服务结论与真实任务证据。

## 任务运行边界

每任务一个容器，Codex、构建和测试使用同一个 `identities.task` / `identities.gid`
非 root 身份。candidate、base 和实验目录属于任务数据，各自 venv、构建、缓存及产物独立。

| 挂载 | 用途 |
| --- | --- |
| `control_root` 下的 `scripts`、`api_contract`、`envsetup.sh` | 只读挂载到 `/opt/local-ci/control/` 对应位置 |
| 版本化 LLVM 和后端依赖 | 只读挂载到 `/opt/local-ci/runtime/deps/` |
| `work/<head_sha>/<run_id>` | 可写挂载到 `/task` |
| 当前运行的 `artifacts` | 可写挂载到 `/task/artifacts` |

`.git`、宿主凭据、状态、私有日志和封存结果留在宿主。
任务结束后清理临时工作目录和私有会话，保留持久化证据与其他运行。
`runtime.py` 管理镜像和容器；`container_fs.py` 准备工作目录、venv 与 Codex 会话。

新任务入场且无在途执行时，Worker 检查 Gitee `control_branch`；版本不同时，
通过 `control-update/request.json` 请求同步并重启，支持前进及回退到祖先提交。
访问失败或历史分叉时等待并报告原因，后续扫描重试。

Worker 校验进程与磁盘代码版本一致，并在任务期间持有 `control.lock`。
更新器取得独占锁且没有未清理任务容器后才切换 checkout；
切换版本会覆盖本地代码修改。
恢复中的有效未封存任务保留当前控制版本，只有封存上传等待不阻止升级。
控制检查由新任务或显式命令触发，无独立定时器。

## 容器内 CI Python

`container_python` 指向镜像中的 CI 虚拟环境，通常为 `/opt/venv/bin/python`。
未显式配置时使用 profile 的 `SEED_PYTHON` 或 `PYTHON_VENV_ACTIVATE` 对应解释器。
它用于可信管理操作，并为 candidate/base 各自准备可写 venv；任务环境修复不污染预置环境。

任务 venv 复制预置包（包括 pip），安装依赖使用 `"$PYTHON_BIN" -m pip`。
构建、安装、测试及辅助脚本都使用所选任务的 CI Python，不使用系统 Python。
宿主服务的 `python_bin` 与容器解释器独立。
