# Triton / LLVM 版本兼容性矩阵

| 厂商 | Triton 版本 | 原始 Triton 工具链 commit ID | LLVM 版本 | LLVM commit ID |
|---|---|---|---|---|
| 算能（Sophgo） | 3.0.0 | `757b6a61e7df814ba806f498f8bb3160f84b120c` | 19.0.0 | `10dc3a8e916d73291269e5e2b82dd22681489aa1` |
| 清微智能（Tsingmicro） | 3.3.0 | `d654e0f2d91f07496454e0fcbec2a9b97df37d47` | 21.0.0 | `a66376b0dc3b2ea8a84fda26faca287980986f78` |
| 风华创智（Fantasy） | 3.0.0 | `757b6a61e7df814ba806f498f8bb3160f84b120c` | 19.0.0 | `10dc3a8e916d73291269e5e2b82dd22681489aa1` |
| 进迭时空（SpacemiT） | 3.6.0 | `6cc4505027d7b39fe18a44a7f89085b8babb7400` | 22.0.0 | `a992f29451b9e140424f35ac5e20177db4afbdc0` |

## Local CI 路由与部署目标

Local CI 按 `triton/cmake/llvm-hash.txt` 选择服务器 profile，一个 PR 或 push 只运行其目标版本。仓库代码支持以下路由目标，但不负责部署 LLVM、venv、长期容器或 profile：

| LLVM commit ID | profile 部署后的计划执行范围 |
| --- | --- |
| `10dc3a8e916d73291269e5e2b82dd22681489aa1` | 3.0 Sophgo profile：执行现有前端与 Sophgo 后端阶段。 |
| `a66376b0dc3b2ea8a84fda26faca287980986f78` | 3.3 frontend profile：执行现有前端 build、安装、导入与 `tests/test_smoke.py`。 |
| `a992f29451b9e140424f35ac5e20177db4afbdc0` | 3.6 frontend profile：执行现有前端 build、安装、导入与 `tests/test_smoke.py`。 |

3.3 和 3.6 当前没有部署可供测试的厂商后端；对应 frontend profile、LLVM 和容器也需先在服务器完成部署并通过真实任务验证。部署后不执行后端构建、JIT、FlagGems 和性能验证，绿色结果只表示当前前端范围通过；以后部署匹配后端时，可通过服务器 profile 启用现有后端阶段。未知 LLVM hash 或缺失 profile 不会回退到其他版本环境。
