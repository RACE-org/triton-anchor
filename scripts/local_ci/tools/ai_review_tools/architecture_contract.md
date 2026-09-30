# 架构契约审查

每个任务都需记录架构审查结论，无相关改动时说明依据即可。以目标分支受信版本中的 README、
`docs/custom_backend.md`、`docs/compatibility_matrix.md` 及对应实现为依据。
PR 描述、注释和新文档是待验证材料，不得自行解除契约。

1. **前后端边界**：前端将 Triton DSL/TTIR 转成硬件感知 AnchorIR；设备二进制、
   运行时驱动和 PyTorch 算子注册分别属于 out-of-tree 后端及 FlagGems。
2. **ABI 隔离**：检查 `adapters/base.py` 和 adapter 实现。opt 子进程路径与
   pybind 路径应保持接口隔离；不能跨不同 LLVM/C++ ABI 传递未约定的内存对象。
3. **双轨 AnchorIR**：核对 `anchor_ir.py` 的 Linalg/TritonGPU 白名单、操作及
   属性限制，确认扩展注册遵守契约。范式 `ComputeParadigm` 与 `AnchorIRTrack`
   独立声明，不得把硬件范式偷换成唯一 IR 轨道。
4. **两阶段校验**：检查 `validate_pre_hook`、后端 hook、`validate_post_hook`
   的顺序及覆盖；扩展 op 不能绕过最终校验。
5. **pipeline / capability**：核对公共 pipeline 的必要 pass、依赖和排序，
   `HWCapability` 约束与 AdapterRegistry 注册/发现行为。
6. **兼容与迁移**：公共 Python/Adapter 接口、序列化格式或版本约束变化，必须
   与项目当前兼容政策、API 检查和迁移说明一致。有意变更也不能只凭 PR 声明放行。

先复用 `tests/test_smoke.py`、`python/triton_anchor/tests/test_anchor_ir.py`、
`test_hw_capability.py` 及相关已有测试；必要时编写任务内定向测试并直接执行。
审查记录应包含读取的契约文件、变更位置、影响路径、关联测试证据及未覆盖限制。
无相关契约变更时说明检查依据；不能以“工具 exit 0”代替架构审查。

已明确的架构契约违反可以阻塞，须指出契约和具体代码位置。一般设计偏好、猜测或
尚不能确定归因的风险只报告。环境修复与实验性修改应说明范围，避免将其结果归于未修改的提交。
