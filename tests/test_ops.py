"""
Triton JIT 自动编译与执行测试

通过 @triton.jit 自动触发对应后端的编译流水线与内核启动。
无需手动 import 后端，Triton 会通过 entry_points 自动加载。

用法:
    python tests/test_ops.py [--rand]
"""
import os
import shutil
import sys
import argparse

import torch
import triton
import triton.language as tl

from conftest import (
    run_test, assert_close,
    to_dev, print_env_info, print_results, has_failures,
)

USE_RANDOM = False
DUMP_DIR = "./test_data_dump"

def dump_tensors(test_name, **tensors):
    # 先在控制台打印简要信息
    for name, tensor in tensors.items():
        t_cpu = tensor.cpu() if hasattr(tensor, 'cpu') else tensor
        if name == "expected":
            prefix = "CPU Expected"
        elif name == "output":
            prefix = "DeviceOutput"
        else:
            prefix = f"Input {name:<5}"
        print(f"  [{prefix:12}] shape={list(t_cpu.shape)} {t_cpu.dtype}:\n{t_cpu}")

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

    print(f"  [Info] 完整数据(文本)已导出至 {file_path}")


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
    jit_add_kernel[grid](x_dev, y_dev, output_dev, n, BLOCK_SIZE=BLOCK_SIZE)
    
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
    jit_add_kernel[grid](x_dev, y_dev, output_dev, n, BLOCK_SIZE=BLOCK_SIZE)
    
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
    jit_matmul_kernel[grid](
        a_dev, b_dev, output_dev,
        M, N, K,
        a_dev.stride(0), a_dev.stride(1),
        b_dev.stride(0), b_dev.stride(1),
        output_dev.stride(0), output_dev.stride(1),
        BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK,
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
    jit_fma_kernel[grid](a_dev, b_dev, c_dev, output_dev, n, BLOCK_SIZE=BLOCK_SIZE)

    assert_close(output_dev, expected, "jit_fma_float")
    dump_tensors("jit_fma_float", a=a_cpu, b=b_cpu, c=c_cpu, expected=expected, output=output_dev.cpu())


# ============================================================================
# 入口
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton 统一前端测试")
    parser.add_argument("--rand", action="store_true", help="使用随机数据而不是固定常数数据")
    parser.add_argument("--dump-dir", type=str, default="./test_data_dump", help="指定测试数据(文本)导出目录")
    args = parser.parse_args()
    
    USE_RANDOM = args.rand
    DUMP_DIR = args.dump_dir

    if os.path.exists(DUMP_DIR):
        shutil.rmtree(DUMP_DIR)

    print("Triton JIT Backend 自动编译路径测试")
    print(f"数据模式: {'随机数据 (--rand)' if USE_RANDOM else '固定数据 (全1/常数，方便人工核对)'}")
    print_env_info()

    run_test("1. JIT 向量加法 (Float32)", test_jit_add_float)
    run_test("2. JIT 向量加法 (Int32)", test_jit_add_int)
    run_test("3. JIT 矩阵乘法 (Float32)", test_jit_matmul_float)
    run_test("4. JIT 融合乘加 (Float32)", test_jit_fma_float)

    print_results()
    sys.exit(1 if has_failures() else 0)
