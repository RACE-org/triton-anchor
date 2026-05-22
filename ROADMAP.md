# triton-anchor Roadmap

> 本文档列出 triton-anchor 项目的近期里程碑与中远期规划。路线图可能随技术评估和社区反馈动态调整。

---

## Phase 0 — 基础框架建设 `v0.1` （当前）

| 目标 | 状态 | 说明 |
|------|------|------|
| TTIR Pipeline 7-pass 管线 | ✅ 完成 | inliner → combine → canonicalizer → reorder_broadcast → cse → licm → symbol_dce |
| AnchorIR 双轨规范 + 两阶段验证 | ✅ 完成 | Linalg Track / TritonGPU Track 白名单与 pre/post hook 验证 |
| Linalg Adapter 接口层 | ✅ 完成 | ILinalgOptAdapter / ILinalgPybindAdapter ABI 隔离 |
| triton-linalg Adapter 参考实现 | ✅ 完成 | 基于 Pybind 的 triton-linalg pass 管线 |
| 硬件后端 entry_points 自动发现 | ✅ 完成 | pull 模式注册机制 |

---

## Phase 1 — FlagGems 算子库对接 `v0.2` （近期， 2026年6月）

| 目标 | 状态 | 说明 |
|------|------|------|
| FlagGems 算子兼容性验证 | 🔲 计划中 | 逐个验证 FlagGems 算子在 Linalg Track/ TritonGPU 下的编译通过率 |
| FlagGems 核心算子集覆盖 | 🔲 计划中 | 覆盖 BLAS (matmul, addmm)、Reduction (sum, mean, softmax)、Elementwise (relu, gelu, silu) 等高频算子 |
| 端到端 benchmark 框架 | 🔲 计划中 | 性能基准测试，对比 FlagGems + triton-anchor vs. 原生实现 |

---

## Phase 2 — 多后端支撑完善 `v0.3`（近期，2026年7月）

| 目标 | 状态 | 说明 |
|------|------|------|
| AME Matrix 后端原型 | 🔲 计划中 | RISC-V Matrix 扩展指令集后端的参考实现 |
| gpGPU (SIMT) 后端原型 | 🔲 计划中 | 自定义 LLVM 路径打通 |
| AnchorIR 扩展 Op 注册机制增强 | 🔲 计划中 | 支持后端动态声明扩展 Op 白名单 |
| CI/CD 多后端测试矩阵 | 🔲 计划中 | GitHub Actions 自动化测试覆盖多种后端 |

---

## Phase 3 — 生态完善 `v0.4+` （中期， 2026年底之前）

| 目标 | 状态 | 说明 |
|------|------|------|
| FlagGems 全量算子覆盖 | 🔲 计划中 | 覆盖 FlagGems 支持的全部 Triton 算子 |
| 性能调优工具集成 | 🔲 计划中 | Profiler / IR Visualizer / Tiling Advisor |
| 多后端 Kernel 调度策略（包括开芯院自研芯片后端） | 🔲 计划中 | 根据硬件能力自动选择最优 Track 路径 |
| 社区文档与教程体系 | 🔲 计划中 | 完善贡献指南、API 文档、最佳实践 |

---

## 未来探索方向

- **动态 Shape 支持**：探索 Triton 动态 shape 场景下的编译前端适配
- **稀疏计算原语**：在 AnchorIR 中引入结构化稀疏计算支持
- **量化算子路径**：INT8/FP8 量化算子在 Linalg Track 下的编译支持
- **跨后端 Kernel 融合**：探索跨计算范式的算子融合优化

---

> 📌 路线图更新频率：每季度更新一次。如有紧急调整，会在 [Discussions](../../discussions) 中同步。
>
> 🤝 欢迎通过 [Issue](../../issues) 或 [Pull Request](../../pulls) 参与讨论和贡献！
