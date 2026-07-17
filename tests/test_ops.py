"""
Triton JIT 自动编译与执行测试

通过 @triton.jit 自动触发对应后端的编译流水线与内核启动。
无需手动 import 后端，Triton 会通过 entry_points 自动加载。

用法:
    python tests/test_ops.py [--rand] [--verbose-tensors]
"""
import os
import sys
import argparse
import traceback

# 测试每次都走完整编译流水线，不复用 Triton 磁盘编译缓存。
os.environ["TRITON_ALWAYS_COMPILE"] = "1"

import torch
import triton
import triton.backends
import triton.language as tl
from triton.runtime.driver import driver

USE_RANDOM = False
VERBOSE_TENSORS = False
DUMP_DIR = "./test_data_dump"


def init_backend():
    """通过 Triton 注册的后端动态加载对应的 PyTorch 扩展。"""
    active_backends = triton.backends.backends.keys()
    if not active_backends:
        raise RuntimeError(
            "未检测到任何已安装的 Triton 硬件后端插件 "
            "(active_backends 为空)！"
        )

    if "sophgo" in active_backends:
        import torch_tpu  # noqa: F401
        return "sophgo", "tpu:0"

    if "tsingmicro" in active_backends:
        import torch_txda  # noqa: F401
        return "tsingmicro", "txda:0"

    if "fantasy" in active_backends:
        import torch_fant  # noqa: F401
        return "fantasy", "fant:0"

    if "spacemit" in active_backends:
        return "spacemit", "cpu"

    return "cpu", "cpu"


BACKEND_NAME, DEVICE_NAME = init_backend()
passed = 0
failed = 0


def print_status_block(title, *details):
    """用醒目的区块打印失败等需要重点关注的状态。"""
    separator = "=" * 72
    print(f"\n{separator}")
    print(title)
    for label, value in details:
        print(f"   {label}: {value}")
    print(f"{separator}\n", flush=True)


def run_test(name, fn):
    """执行单个测试用例并统计通过和失败数量。"""
    global passed, failed
    print(f"\n{'=' * 60}")
    print(f"[TEST] {name}")
    print(f"{'=' * 60}")
    try:
        fn()
        print("  ✅ PASSED")
        passed += 1
    except Exception as error:
        print(f"  ❌ FAILED: {error}")
        traceback.print_exc()
        failed += 1


def assert_close(dev_result, cpu_result, test_name, rtol=1e-3, atol=1e-3):
    """将设备结果搬回 CPU 并与参考值进行比较。"""
    dev_cpu = dev_result.cpu()
    if not torch.allclose(dev_cpu, cpu_result, rtol=rtol, atol=atol):
        max_diff = (dev_cpu - cpu_result).abs().max().item()
        print_status_block(
            "❌ 数值验证失败",
            ("测试用例", test_name),
            ("最大误差", f"{max_diff:.6f}"),
            ("相对容差", rtol),
            ("绝对容差", atol),
        )
        raise AssertionError(
            f"{test_name}: 数值不匹配, max_diff={max_diff:.6f}, "
            f"rtol={rtol}, atol={atol}"
        )

    print(
        f"  ✅ 数值验证成功: {test_name} "
        f"(rtol={rtol}, atol={atol})",
        flush=True,
    )


def to_dev(tensor):
    """将 CPU tensor 拷贝到当前测试设备。"""
    if DEVICE_NAME == "cpu":
        return tensor
    return tensor.to(DEVICE_NAME)


def print_env_info():
    """打印当前测试环境。"""
    print(f"Backend: {BACKEND_NAME}")
    print(f"设备 (Device): {DEVICE_NAME}")
    print(f"TRITON_ALWAYS_COMPILE: {os.environ['TRITON_ALWAYS_COMPILE']}")
    dump_dir = os.getenv("TRITON_DUMP_DIR")
    if dump_dir:
        print(f"TRITON_DUMP_DIR: {dump_dir}")
    else:
        print("TRITON_DUMP_DIR: 未设置 (使用 Triton 默认缓存行为)")


def print_results():
    """打印测试结果汇总。"""
    print(f"\n{'=' * 60}")
    print(f"结果: {passed} 通过, {failed} 失败")
    print(f"{'=' * 60}")


def has_failures():
    """返回是否存在失败的测试。"""
    return failed > 0


def print_kernel_status(title, display_name):
    """用紧凑单行显示 JIT 编译和 Kernel 执行状态。"""
    print(f"  ✅ {title}: {display_name}", flush=True)


def initialize_jit_kernel(kernel):
    """兼容不同 Triton 版本，提前初始化 JIT 编译入口和参数 binder。"""
    if hasattr(kernel, "device_caches"):
        # Triton 3.3/3.6：defaultdict 会为当前设备调用 create_binder()，
        # 并保存 kernel cache、target、backend 和 binder 等初始化结果。
        device = driver.active.get_current_device()
        _ = kernel.device_caches[device]
        return

    # Triton 3.0：binder 直接保存在 JITFunction 实例上。
    if getattr(kernel, "binder", None) is None:
        kernel.create_binder()


def compile_and_run(kernel, grid, *args, log_name=None, **kwargs):
    """正常启动一次 kernel，并在实际编译和执行返回后分别报告状态。"""
    display_name = log_name or kernel.__name__

    # 提前初始化 kernel.compile，再只包装当前 kernel 实例并在调用后恢复。
    initialize_jit_kernel(kernel)
    original_compile = kernel.compile

    def compile_with_log(*compile_args, **compile_kwargs):
        compiled_kernel = original_compile(*compile_args, **compile_kwargs)
        print_kernel_status("JIT 编译成功", display_name)
        return compiled_kernel

    kernel.compile = compile_with_log
    try:
        result = kernel[grid](*args, **kwargs)
    finally:
        kernel.compile = original_compile

    print_kernel_status("Kernel 执行成功", display_name)
    return result


def dump_tensors(test_name, **tensors):
    print(f"\n  📊 张量数据: {test_name} ({len(tensors)} 个张量)")

    # 控制台默认只打印摘要；完整数据始终写入文本文件。
    for name, tensor in tensors.items():
        t_cpu = tensor.cpu() if hasattr(tensor, 'cpu') else tensor
        if name == "expected":
            role = "CPU 期望值"
        elif name == "output":
            role = "设备输出"
        else:
            role = "输入张量"

        print(f"  [{role}] {name}")
        print(
            f"     形状: {list(t_cpu.shape)}, "
            f"类型: {t_cpu.dtype}, "
            f"元素数: {t_cpu.numel()}"
        )

        if t_cpu.numel() > 0:
            flat = t_cpu.reshape(-1)
            sample = flat[:min(6, flat.numel())].tolist()
            print(f"     样本: {sample}")

        if VERBOSE_TENSORS:
            print(f"     完整数据:\n{t_cpu}")
        print()

    os.makedirs(DUMP_DIR, exist_ok=True)
    
    file_path = os.path.join(DUMP_DIR, f"{test_name}.txt")
    with open(file_path, "w") as f:
        # 临时取消省略号，完整输出大张量
        torch.set_printoptions(threshold=1000000, linewidth=200)
        
        f.write(f"========== Test: {test_name} ==========\n\n")
        for name, tensor in tensors.items():
            t_cpu = tensor.cpu() if hasattr(tensor, 'cpu') else tensor
            f.write(f"--- [ {name} ] ---\n")
            f.write(f"shape: {list(t_cpu.shape)}\n")
            f.write(f"dtype: {t_cpu.dtype}\n")
            f.write(f"data:\n{t_cpu}\n\n")
            
        # 恢复默认打印设置
        torch.set_printoptions(profile="default")

    print(f"  ✅ 张量数据导出成功: {file_path}", flush=True)


# ============================================================================
# Triton 内核定义
# ============================================================================

@triton.jit
def jit_add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """JIT 路径: 向量加法 (泛型)"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def jit_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    """JIT 路径: 基础单 Block 矩阵乘法 (泛型)"""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # 加载张量块
    a = tl.load(a_ptrs)
    b = tl.load(b_ptrs)
    
    # 核心矩阵乘算子 (自动根据 a, b 类型推导输出类型)
    c = tl.dot(a, b)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    
    # 存储结果
    tl.store(c_ptrs, c)


@triton.jit
def jit_fma_kernel(
    a_ptr, b_ptr, c_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """JIT 路径: 融合乘加"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = tl.load(c_ptr + offsets, mask=mask)
    output = a * b + c
    tl.store(output_ptr + offsets, output, mask=mask)


# ============================================================================
# 测试用例
# ============================================================================

def test_jit_add_float():
    """JIT: 向量加法 (Float32)"""

    n = 1024
    BLOCK_SIZE = 256

    if USE_RANDOM:
        x_cpu = torch.randn(n, dtype=torch.float32)
        y_cpu = torch.randn(n, dtype=torch.float32)
    else:
        x_cpu = torch.ones(n, dtype=torch.float32)
        y_cpu = torch.ones(n, dtype=torch.float32) * 2.0
        
    expected = x_cpu + y_cpu

    x_dev = to_dev(x_cpu)
    y_dev = to_dev(y_cpu)
    output_dev = torch.empty(n, dtype=torch.float32, device=x_dev.device)

    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    compile_and_run(
        jit_add_kernel, grid,
        x_dev, y_dev, output_dev, n,
        BLOCK_SIZE=BLOCK_SIZE,
        log_name="jit_add_float",
    )
    
    assert_close(output_dev, expected, "jit_add_float")
    dump_tensors("jit_add_float", x=x_cpu, y=y_cpu, expected=expected, output=output_dev.cpu())


def test_jit_add_int():
    """JIT: 向量加法 (Int32)"""

    n = 1024
    BLOCK_SIZE = 256

    if USE_RANDOM:
        x_cpu = torch.randint(-100, 100, (n,), dtype=torch.int32)
        y_cpu = torch.randint(-100, 100, (n,), dtype=torch.int32)
    else:
        x_cpu = torch.ones(n, dtype=torch.int32)
        y_cpu = torch.ones(n, dtype=torch.int32) * 2
        
    expected = x_cpu + y_cpu

    x_dev = to_dev(x_cpu)
    y_dev = to_dev(y_cpu)
    output_dev = torch.empty(n, dtype=torch.int32, device=x_dev.device)

    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    compile_and_run(
        jit_add_kernel, grid,
        x_dev, y_dev, output_dev, n,
        BLOCK_SIZE=BLOCK_SIZE,
        log_name="jit_add_int",
    )
    
    assert_close(output_dev, expected, "jit_add_int")
    dump_tensors("jit_add_int", x=x_cpu, y=y_cpu, expected=expected, output=output_dev.cpu())


def test_jit_matmul_float():
    """JIT: 矩阵乘法 (Float32)"""

    M, N, K = 32, 32, 32
    BM, BN, BK = 32, 32, 32

    if USE_RANDOM:
        a_cpu = torch.randn((M, K), dtype=torch.float32)
        b_cpu = torch.randn((K, N), dtype=torch.float32)
    else:
        a_cpu = torch.ones((M, K), dtype=torch.float32)
        b_cpu = torch.ones((K, N), dtype=torch.float32) * 2.0
        
    expected = torch.matmul(a_cpu, b_cpu)

    a_dev = to_dev(a_cpu)
    b_dev = to_dev(b_cpu)
    output_dev = torch.empty((M, N), dtype=torch.float32, device=a_dev.device)

    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    compile_and_run(
        jit_matmul_kernel, grid,
        a_dev, b_dev, output_dev,
        M, N, K,
        a_dev.stride(0), a_dev.stride(1),
        b_dev.stride(0), b_dev.stride(1),
        output_dev.stride(0), output_dev.stride(1),
        BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK,
        log_name="jit_matmul_float",
    )
    
    assert_close(output_dev, expected, "jit_matmul_float", atol=1e-2, rtol=1e-2)
    dump_tensors("jit_matmul_float", a=a_cpu, b=b_cpu, expected=expected, output=output_dev.cpu())


def test_jit_fma_float():
    """JIT: 融合乘加 (Float32)"""


    n = 2048
    BLOCK_SIZE = 256

    if USE_RANDOM:
        a_cpu = torch.randn(n, dtype=torch.float32)
        b_cpu = torch.randn(n, dtype=torch.float32)
        c_cpu = torch.randn(n, dtype=torch.float32)
    else:
        a_cpu = torch.ones(n, dtype=torch.float32) * 2.0
        b_cpu = torch.ones(n, dtype=torch.float32) * 3.0
        c_cpu = torch.ones(n, dtype=torch.float32) * 4.0
        
    expected = a_cpu * b_cpu + c_cpu

    a_dev = to_dev(a_cpu)
    b_dev = to_dev(b_cpu)
    c_dev = to_dev(c_cpu)
    output_dev = torch.empty(n, dtype=torch.float32, device=a_dev.device)

    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    compile_and_run(
        jit_fma_kernel, grid,
        a_dev, b_dev, c_dev, output_dev, n,
        BLOCK_SIZE=BLOCK_SIZE,
        log_name="jit_fma_float",
    )

    assert_close(output_dev, expected, "jit_fma_float")
    dump_tensors("jit_fma_float", a=a_cpu, b=b_cpu, c=c_cpu, expected=expected, output=output_dev.cpu())


# ============================================================================
# 入口
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton 统一前端测试")
    parser.add_argument("--rand", action="store_true", help="使用随机数据而不是固定常数数据")
    parser.add_argument(
        "--verbose-tensors",
        action="store_true",
        help="在控制台打印完整张量；默认仅显示摘要",
    )
    parser.add_argument("--dump-dir", type=str, default="./test_data_dump", help="指定测试数据(文本)导出目录")
    args = parser.parse_args()
    
    USE_RANDOM = args.rand
    VERBOSE_TENSORS = args.verbose_tensors
    DUMP_DIR = args.dump_dir

    # 不清空用户指定的目录；各测试生成的同名文本文件会自行覆盖。
    os.makedirs(DUMP_DIR, exist_ok=True)

    print("Triton JIT Backend 自动编译路径测试")
    print(f"数据模式: {'随机数据 (--rand)' if USE_RANDOM else '固定数据 (全1/常数，方便人工核对)'}")
    print_env_info()

    run_test("1. JIT 向量加法 (Float32)", test_jit_add_float)
    run_test("2. JIT 向量加法 (Int32)", test_jit_add_int)
    run_test("3. JIT 矩阵乘法 (Float32)", test_jit_matmul_float)
    run_test("4. JIT 融合乘加 (Float32)", test_jit_fma_float)

    print_results()
    sys.exit(1 if has_failures() else 0)
