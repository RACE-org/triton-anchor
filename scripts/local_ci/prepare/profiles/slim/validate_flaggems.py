"""Exercise read-only source-only FlagGems with a small Sophgo operator."""

import importlib.metadata
import os
from pathlib import Path
import tempfile


def main():
    source = Path(os.environ["FLAGGEMS_CLONE_DIR"]).resolve()
    try:
        importlib.metadata.distribution("flag_gems")
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        raise RuntimeError("FlagGems must not be installed in this image")
    with tempfile.TemporaryDirectory(prefix="flaggems-mount-") as temporary:
        root = Path(temporary)
        os.environ.update(
            FLAGGEMS_CACHE_DIR=str(root / "cache"),
            TRITON_CACHE_DIR=str(root / "triton"),
            TRITON_DUMP_DIR=str(root / "dump"),
            PYTHONDONTWRITEBYTECODE="1",
            GEMS_VENDOR="sophgo",
        )
        import torch
        import torch_tpu  # noqa: F401
        import flag_gems

        assert Path(flag_gems.__file__).resolve().is_relative_to(source / "src")
        assert not flag_gems.has_c_extension
        a = torch.arange(128, dtype=torch.float32)
        b = torch.ones_like(a)
        actual = flag_gems.add(a.to("tpu:0"), b.to("tpu:0")).cpu()
        torch.testing.assert_close(actual, a + b)
        print(
            "FlagGems: read-only source import, no installed wheel, no C extension, add=pass"
        )


if __name__ == "__main__":
    main()
