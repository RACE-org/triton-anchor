import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_DIR = _REPO_ROOT / "python"
_SOPHGO_BACKEND_DIR = _REPO_ROOT.parent / "triton-sophgo-backend"
for _path in (_PYTHON_DIR, _REPO_ROOT, _SOPHGO_BACKEND_DIR):
    if _path.exists() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import triton
import triton.language as tl


@triton.jit
def bad_pplir_cdim_broadcast_kernel(
    x_ptr,
    out_ptr,
    BLOCK_A: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    offs_a = tl.arange(0, BLOCK_A)
    offs_b = tl.arange(0, BLOCK_B)
    offs_c = tl.arange(0, BLOCK_C)
    x = tl.load(x_ptr + offs_a[:, None] * BLOCK_C + offs_c[None, :])
    y = x[:, None, :] + tl.zeros((BLOCK_A, BLOCK_B, BLOCK_C), tl.float32)
    out_offsets = (
        offs_a[:, None, None] * BLOCK_B * BLOCK_C
        + offs_b[None, :, None] * BLOCK_C
        + offs_c[None, None, :]
    )
    tl.store(out_ptr + out_offsets, y)


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


def _is_diagnose_enabled() -> bool:
    return os.getenv("TRITON_ANCHOR_DIAGNOSE_ON_ERROR", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _load_sophgo_dialects(ctx) -> None:
    from triton_anchor.diagnostics import _load_sophgo_passes_module

    sophgo_passes = _load_sophgo_passes_module()

    load_dialects = getattr(sophgo_passes, "load_dialects", None)
    if load_dialects is not None:
        load_dialects(ctx)


def _diagnose_bad_pplir_kernel() -> int:
    from triton._C.libtriton import anchor, ir
    from triton_anchor.diagnose import _print_result
    from triton_anchor.diagnostics import PassDiagnostic

    output_root = Path(
        os.getenv(
            "TRITON_ANCHOR_DIAGNOSE_DIR",
            str(Path(__file__).resolve().parent / "bad-pplir-diagnose"),
        )
    )

    src = triton.compiler.ASTSource(
        fn=bad_pplir_cdim_broadcast_kernel,
        signature={0: "*fp32", 1: "*fp32"},
        constants={2: 4, 3: 8, 4: 16},
    )

    ctx = ir.context()
    ir.load_dialects(ctx)
    anchor.load_dialects(ctx)
    _load_sophgo_dialects(ctx)

    mod = src.make_ir(options=_MinimalOptions(), codegen_fns=None, context=ctx)
    mod.context = ctx

    print(f"[AnchorDiagnose] output root: {output_root}")
    dump_root = output_root / "triton-dump"
    (dump_root / "bad_pplir_cdim_broadcast_kernel").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TRITON_DUMP_DIR", str(dump_root))

    ttir_result = PassDiagnostic(output_dir=output_root / "ttir").diagnose_ttir(mod)
    _print_result(ttir_result)
    if not ttir_result.ok:
        return 1

    linalg_result = PassDiagnostic(
        output_dir=output_root / "triton-linalg"
    ).diagnose_triton_linalg(mod)
    _print_result(linalg_result)
    if not linalg_result.ok:
        return 1

    pplir_result = PassDiagnostic(
        output_dir=output_root / "sophgo-pplir"
    ).diagnose_sophgo_pplir(mod)
    _print_result(pplir_result)
    return 0 if pplir_result.ok else 1


if __name__ == "__main__":
    print("Compiling bad_pplir_cdim_broadcast_kernel for Sophgo PPLIR diagnostics.")
    if not _is_diagnose_enabled():
        print("Set TRITON_ANCHOR_DIAGNOSE_ON_ERROR=1 to run diagnostics.")
        raise SystemExit(0)
    raise SystemExit(_diagnose_bad_pplir_kernel())
