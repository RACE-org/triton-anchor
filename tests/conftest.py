"""
triton-anchor 统一前端测试公共工具

提供测试框架、数值验证、环境打印及辅助函数，
供本目录下所有的自动化测试用例（如 test_ops.py, test_smoke.py 等）共用。
"""
import os
import traceback

import torch

import triton.backends

def _init_backend():
    """通过 triton 挂载的后端来动态加载对应的 torch 扩展"""
    active_backends = triton.backends.backends.keys()
    if not active_backends:
        raise RuntimeError("未检测到任何已安装的 Triton 硬件后端插件 (active_backends 为空)！")

    if "sophgo" in active_backends:
        import torch_tpu
        return "sophgo", "tpu:0"

    if "tsingmicro" in active_backends:
        import torch_txda
        return "tsingmicro", "txda:0"

    if "fantasy" in active_backends:
        import torch_fant
        return "fantasy", "fant:0"

    if "spacemit" in active_backends:
        return "spacemit", "cpu"
        
    return "cpu", "cpu"

BACKEND_NAME, DEVICE_NAME = _init_backend()


# ============================================================================
# 测试框架
# ============================================================================

passed = 0
failed = 0


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


def assert_close(dev_result, cpu_result, test_name, rtol=1e-3, atol=1e-3):
    """数值对比：将 Device 结果搬回 CPU 后与参考值比较"""
    dev_cpu = dev_result.cpu()
    if not torch.allclose(dev_cpu, cpu_result, rtol=rtol, atol=atol):
        max_diff = (dev_cpu - cpu_result).abs().max().item()
        raise AssertionError(
            f"{test_name}: 数值不匹配, max_diff={max_diff:.6f}, "
            f"rtol={rtol}, atol={atol}"
        )
    print(f"  数值验证通过 (rtol={rtol}, atol={atol})")



def to_dev(tensor):
    """将 CPU tensor 拷贝到设备"""
    if DEVICE_NAME == "cpu":
        return tensor
    return tensor.to(DEVICE_NAME)


def print_env_info():
    """打印环境信息"""
    print(f"Backend: {BACKEND_NAME}")
    print(f"设备 (Device): {DEVICE_NAME}")
    dump_dir = os.getenv("TRITON_DUMP_DIR")
    if dump_dir:
        print(f"TRITON_DUMP_DIR: {dump_dir}")
    else:
        print(f"TRITON_DUMP_DIR: 未设置 (使用 Triton 默认缓存行为)")


def print_results():
    """打印测试结果汇总"""
    print(f"\n{'='*60}")
    print(f"结果: {passed} 通过, {failed} 失败")
    print(f"{'='*60}")


def has_failures():
    """是否有失败的测试"""
    return failed > 0
