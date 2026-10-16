import os
import sys
from pathlib import Path

import triton
import triton.language as tl

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_DIR = _REPO_ROOT / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@triton.jit
def bad_frontend_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.this_builtin_does_not_exist(x)
    tl.store(out_ptr + offs, y, mask=mask)


def _is_diagnose_enabled() -> bool:
    return os.getenv("TRITON_ANCHOR_DIAGNOSE_ON_ERROR", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _run_frontend_diagnostic() -> int:
    from triton_anchor.diagnose import main

    output_dir = Path(
        os.getenv(
            "TRITON_ANCHOR_DIAGNOSE_DIR",
            str(Path(__file__).resolve().parent / "bad-frontend-diagnose"),
        )
    )
    print(f"[AnchorDiagnose] output dir: {output_dir}")
    return main(
        [
            "--python",
            "anchor_diag_bad_frontend_kernel:bad_frontend_kernel",
            "--signature",
            "*fp32,*fp32,i32",
            "--constant",
            "BLOCK=256",
            "--pipeline",
            "ttir",
            "--output-dir",
            str(output_dir),
        ]
    )


if __name__ == "__main__":
    print("Compiling bad_frontend_kernel for Triton frontend diagnostics.")
    if not _is_diagnose_enabled():
        print("Set TRITON_ANCHOR_DIAGNOSE_ON_ERROR=1 to run diagnostics.")
        raise SystemExit(0)
    raise SystemExit(_run_frontend_diagnostic())
