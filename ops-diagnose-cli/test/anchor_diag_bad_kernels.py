import os
import sys
from pathlib import Path

import triton
import triton.language as tl

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_DIR = _REPO_ROOT / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))


@triton.jit
def bad_inline_asm_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.inline_asm_elementwise(
        "mov.u32 $0, $1;",
        "=r,r",
        [x.to(tl.int32)],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    tl.store(out_ptr + offs, y.to(tl.float32), mask=mask)


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


def _diagnose_bad_inline_asm_kernel() -> int:
    from triton._C.libtriton import anchor, ir
    from triton_anchor.diagnose import _print_result
    from triton_anchor.diagnostics import PassDiagnostic

    output_root = Path(
        os.getenv(
            "TRITON_ANCHOR_DIAGNOSE_DIR",
            str(Path(__file__).resolve().parent / "auto-diagnose"),
        )
    )

    src = triton.compiler.ASTSource(
        fn=bad_inline_asm_kernel,
        signature={0: "*fp32", 1: "*fp32", 2: "i32"},
        constants={3: 256},
    )

    ctx = ir.context()
    ir.load_dialects(ctx)
    anchor.load_dialects(ctx)
    mod = src.make_ir(options=_MinimalOptions(), codegen_fns=None, context=ctx)
    mod.context = ctx

    print(f"[AnchorDiagnose] output root: {output_root}")

    ttir_result = PassDiagnostic(output_dir=output_root / "ttir").diagnose_ttir(mod)
    _print_result(ttir_result)
    if not ttir_result.ok:
        return 1

    linalg_result = PassDiagnostic(
        output_dir=output_root / "triton-linalg"
    ).diagnose_triton_linalg(mod)
    _print_result(linalg_result)
    return 0 if linalg_result.ok else 1


def _is_diagnose_enabled() -> bool:
    return os.getenv("TRITON_ANCHOR_DIAGNOSE_ON_ERROR", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


if __name__ == "__main__":
    print("Compiling bad_inline_asm_kernel for Triton Anchor diagnostics.")
    if not _is_diagnose_enabled():
        print("Set TRITON_ANCHOR_DIAGNOSE_ON_ERROR=1 to run diagnostics.")
        raise SystemExit(0)
    raise SystemExit(_diagnose_bad_inline_asm_kernel())
