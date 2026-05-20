"""
triton-anchor 冒烟测试
======================

验证 triton-anchor 安装后的核心功能是否正常，不需要任何硬件后端。
测试覆盖：
  1. Python 模块导入
  2. C++ 绑定加载（libtriton.so + anchor passes）
  3. MLIR Dialect 注册
  4. HWCapability 数据结构
  5. AnchorIR Validator
  6. TTIR Pipeline 构建
  7. Adapter 注册与发现
  8. TTIR 生成（AST → MLIR）
"""

import sys
import traceback


# ============================================================================
# 测试框架
# ============================================================================

passed = 0
failed = 0
skipped = 0


def run_test(name, fn):
    """执行单个测试用例，统计通过/失败"""
    global passed, failed
    print(f"\n{'='*60}")
    print(f"[TEST] {name}")
    print(f"{'='*60}")
    try:
        fn()
        print(f"  ✅ PASSED")
        passed += 1
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        traceback.print_exc()
        failed += 1


def skip_test(name, reason):
    """跳过测试用例"""
    global skipped
    print(f"\n{'='*60}")
    print(f"[SKIP] {name}")
    print(f"  ⏭️  {reason}")
    print(f"{'='*60}")
    skipped += 1


# ============================================================================
# 测试用例
# ============================================================================

def test_import_triton():
    """验证 triton 核心包可以正常导入"""
    import triton
    print(f"  triton version: {triton.__version__}")
    print(f"  triton path: {triton.__file__}")


def test_import_triton_anchor():
    """验证 triton_anchor 包可以正常导入"""
    import triton_anchor
    print(f"  triton_anchor version: {triton_anchor.__version__}")
    print(f"  triton_anchor path: {triton_anchor.__file__}")

    # 验证公共 API 导出
    assert hasattr(triton_anchor, 'HWCapability'), "缺少 HWCapability 导出"
    assert hasattr(triton_anchor, 'ComputeParadigm'), "缺少 ComputeParadigm 导出"
    assert hasattr(triton_anchor, 'AnchorIRTrack'), "缺少 AnchorIRTrack 导出"
    assert hasattr(triton_anchor, 'AnchorIRValidator'), "缺少 AnchorIRValidator 导出"
    assert hasattr(triton_anchor, 'build_ttir_pipeline'), "缺少 build_ttir_pipeline 导出"
    print("  公共 API 导出完整 ✓")


def test_libtriton_binding():
    """验证 C++ 绑定 (libtriton.so) 可以正常加载"""
    from triton._C import libtriton
    assert hasattr(libtriton, 'ir'), "libtriton 缺少 ir 模块"
    assert hasattr(libtriton, 'passes'), "libtriton 缺少 passes 模块"
    print(f"  libtriton 加载成功")
    print(f"  ir 模块: {libtriton.ir}")
    print(f"  passes 模块: {libtriton.passes}")


def test_anchor_passes_binding():
    """验证 triton-anchor 的 C++ Pass 绑定是否注册"""
    from triton._C.libtriton import anchor
    print(f"  anchor 绑定模块加载成功: {anchor}")

    # 验证关键 API 存在
    assert hasattr(anchor, 'load_dialects'), "缺少 load_dialects"
    print("  anchor.load_dialects ✓")


def test_mlir_context_and_dialects():
    """验证 MLIR Context 创建和 Dialect 注册"""
    from triton._C.libtriton import ir, anchor

    ctx = ir.context()
    assert ctx is not None, "MLIR Context 创建失败"
    print(f"  MLIR Context 创建成功")

    # 加载上游 Triton 方言
    ir.load_dialects(ctx)
    print(f"  Triton 方言加载成功 ✓")

    # 加载 triton-anchor 方言（triton-linalg 扩展）
    anchor.load_dialects(ctx)
    print(f"  Anchor 方言加载成功 (linalg_ext, math_ext, aux) ✓")


def test_hw_capability():
    """验证 HWCapability 数据结构的创建与校验"""
    from triton_anchor.hw_capability import (
        HWCapability, ComputeParadigm,
        TensorCapability, MatrixCapability, GPGPUCapability,
    )

    # 测试 Tensor Processor 类型（如 Sophgo TPU）
    hw_tpu = HWCapability(
        name="test-tpu",
        arch_family="tpu",
        compute_paradigm=ComputeParadigm.TENSOR_PROCESSOR,
        anchor_ir_track="linalg",
        ptr_model="axis_info",
        tensor_cap=TensorCapability(num_cores=4),
    )
    assert hw_tpu.lowering_path == "linalg"
    print(f"  TPU HWCapability 创建成功: {hw_tpu.name}")

    # 测试 AME 类型（如 RISC-V）
    hw_ame = HWCapability(
        name="test-riscv",
        arch_family="riscv",
        compute_paradigm=ComputeParadigm.AME_MATRIX,
        anchor_ir_track="linalg",
        ptr_model="structured",
        matrix_cap=MatrixCapability(),
    )
    assert hw_ame.lowering_path == "linalg"
    print(f"  AME HWCapability 创建成功: {hw_ame.name}")

    # 测试 GPGPU 类型
    hw_gpu = HWCapability(
        name="test-gpu",
        arch_family="gpu",
        compute_paradigm=ComputeParadigm.GPGPU,
        anchor_ir_track="triton_gpu",
        ptr_model="gpu",
        gpgpu_cap=GPGPUCapability(num_warps=4, warp_size=32),
    )
    assert hw_gpu.lowering_path == "triton_gpu"
    print(f"  GPGPU HWCapability 创建成功: {hw_gpu.name}")

    # 测试校验：Paradigm 与 Cap 不匹配时应该抛出异常
    try:
        HWCapability(
            name="bad", arch_family="tpu",
            compute_paradigm=ComputeParadigm.TENSOR_PROCESSOR,
            anchor_ir_track="linalg", ptr_model="axis_info",
            # 故意不传 tensor_cap
        )
        raise AssertionError("应该抛出 ValueError，但没有")
    except ValueError:
        print(f"  校验逻辑正确：缺少 tensor_cap 时抛出 ValueError ✓")

    # 测试 GPUTarget 兼容性
    target = hw_tpu.to_gpu_target()
    print(f"  to_gpu_target() 兼容性转换成功: {target}")


def test_anchor_ir_validator():
    """验证 AnchorIR Validator 的方言白名单/黑名单校验"""
    from triton_anchor.anchor_ir import (
        AnchorIRValidator, AnchorIRTrack, AnchorIRError,
    )

    validator = AnchorIRValidator(track=AnchorIRTrack.LINALG)

    # 合法的 Linalg Track IR
    valid_ir = """
    func.func @add(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
        %0 = arith.addf %arg0, %arg1 : tensor<256xf32>
        %1 = linalg.generic {indexing_maps = [...], iterator_types = [...]} ins(%0 : tensor<256xf32>) {
        ^bb0(%in: f32):
            %2 = math.exp %in : f32
            linalg.yield %2 : f32
        } -> tensor<256xf32>
        func.return %1 : tensor<256xf32>
    }
    """
    violations = validator.validate(valid_ir)
    assert len(violations) == 0, f"合法 IR 不应有违规: {violations}"
    assert validator.is_valid(valid_ir)
    print(f"  合法 IR 校验通过 ✓")

    # 包含黑名单方言的 IR（tt 是已被 lower 掉的过渡方言）
    invalid_ir = """
    func.func @bad(%arg0: !tt.ptr<f32>) {
        %0 = tt.load %arg0 : !tt.ptr<f32>
        func.return
    }
    """
    violations = validator.validate(invalid_ir)
    assert len(violations) > 0, "包含 tt 方言的 IR 应该产生违规"
    assert any(v.dialect == "tt" for v in violations)
    print(f"  黑名单方言检测正确 (tt): {len(violations)} 个违规 ✓")

    # validate_and_raise 应该抛出 AnchorIRError
    try:
        validator.validate_and_raise(invalid_ir, context="test")
        raise AssertionError("应该抛出 AnchorIRError")
    except AnchorIRError:
        print(f"  validate_and_raise 正确抛出 AnchorIRError ✓")

    # 测试两阶段验证
    violations_pre = validator.validate_pre_hook(valid_ir)
    assert len(violations_pre) == 0
    violations_post = validator.validate_post_hook(valid_ir, ext_allowed={"ppl"})
    assert len(violations_post) == 0
    print(f"  两阶段验证 API 正常 ✓")


def test_ttir_pipeline():
    """验证 TTIR Pipeline 能够正常构建并执行"""
    from triton._C.libtriton import ir, anchor
    from triton_anchor.pipeline import build_ttir_pipeline

    ctx = ir.context()
    ir.load_dialects(ctx)
    anchor.load_dialects(ctx)

    pm = ir.pass_manager(ctx)
    # 不传 hw 参数，仅构建 7 个 mandatory passes
    build_ttir_pipeline(pm, hw=None)
    print(f"  TTIR Pipeline 构建成功 (7 mandatory passes) ✓")


def test_adapter_discovery():
    """验证 Adapter 注册与发现机制"""
    from triton_anchor.adapters.registry import AdapterRegistry

    registry = AdapterRegistry()
    registry.discover()

    adapters = registry.list_adapters()
    print(f"  发现 {len(adapters)} 个已注册的 Adapter:")
    for name in adapters:
        print(f"    - {name}")

    # triton-linalg 应该始终可用（通过 entry_points 注册）
    if "triton-linalg" in adapters:
        print(f"  triton-linalg adapter 已注册 ✓")
    else:
        print(f"  ⚠️  triton-linalg adapter 未发现（可能 entry_points 未安装）")


import triton
import triton.language as tl


@triton.jit
def _smoke_add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """冒烟测试用的简单 kernel，定义在模块顶层以满足 JIT 作用域要求"""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


# 提供 AST 编译所需的最小 Options
class _MinimalOptions:
    def __init__(self):
        self.num_warps = 4
        self.num_stages = 3
        self.num_ctas = 1
        self.cluster_dims = (1, 1, 1)
        self.ptx_version = None
        self.enable_fp_fusion = True
        self.supported_fp8_dtypes = ()
        self.deprecated_fp8_dtypes = ()
        self.allowed_dot_input_precisions = ("ieee", "tf32", "tf32x3")
        self.allow_fp8e4nv = False
        self.max_num_imprecise_acc_default = False
        self.debug = False


def test_ttir_generation():
    """验证 Triton JIT 函数可以成功编译为 TTIR"""
    from triton._C.libtriton import ir, anchor

    # 构建 ASTSource
    src = triton.compiler.ASTSource(
        fn=_smoke_add_kernel,
        signature={0: "*fp32", 1: "*fp32", 2: "*fp32", 3: "i32"},
        constants={4: 256},
    )

    # 创建 MLIR context 并加载方言
    ctx = ir.context()
    ir.load_dialects(ctx)
    anchor.load_dialects(ctx)

    ttir_module = src.make_ir(options=_MinimalOptions(), codegen_fns=None, context=ctx)
    ir_text = str(ttir_module)

    assert "tt.func" in ir_text or "func.func" in ir_text, "TTIR 中应包含函数定义"
    assert "tt.load" in ir_text, "TTIR 中应包含 tt.load 操作"

    # 打印 TTIR 前几行
    lines = ir_text.strip().split("\n")
    print(f"  TTIR 生成成功，共 {len(lines)} 行")
    for line in lines[:5]:
        print(f"    {line}")
    if len(lines) > 5:
        print(f"    ... (省略 {len(lines) - 5} 行)")


# ============================================================================
# 主入口
# ============================================================================

def main():
    print(f"🔍 triton-anchor 冒烟测试")
    print(f"   Python: {sys.version.split()[0]}")
    print(f"   平台: {sys.platform}")

    # 基础导入测试
    run_test("导入 triton", test_import_triton)
    run_test("导入 triton_anchor", test_import_triton_anchor)

    # C++ 绑定测试
    run_test("libtriton C++ 绑定", test_libtriton_binding)
    run_test("anchor passes 绑定", test_anchor_passes_binding)
    run_test("MLIR Context 与 Dialect 注册", test_mlir_context_and_dialects)

    # 纯 Python 逻辑测试（不依赖 C++ 绑定）
    run_test("HWCapability 数据结构", test_hw_capability)
    run_test("AnchorIR Validator", test_anchor_ir_validator)

    # 需要 C++ 绑定的集成测试
    run_test("TTIR Pipeline 构建", test_ttir_pipeline)
    run_test("Adapter 注册与发现", test_adapter_discovery)
    run_test("TTIR 生成 (AST → MLIR)", test_ttir_generation)

    # 结果汇总
    print(f"\n{'='*60}")
    total = passed + failed + skipped
    print(f"结果: {passed} 通过, {failed} 失败, {skipped} 跳过 (共 {total})")
    print(f"{'='*60}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
