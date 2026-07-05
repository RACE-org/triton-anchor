import triton
import triton.language as tl


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
