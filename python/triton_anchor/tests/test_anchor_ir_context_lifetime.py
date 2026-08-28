"""Regression coverage for ModuleOp ownership across Python context GC."""

import os
from pathlib import Path
import subprocess
import sys

import triton


def test_parsed_module_retains_context_for_standalone_validation_and_normalization(
    tmp_path,
):
    source = tmp_path / "context-lifetime.mlir"
    source.write_text(
        """
module {
  func.func @compute() {
    func.return
  }
}
""",
        encoding="utf-8",
    )

    # Run the lifetime scenario in a child process: the missing ownership edge
    # in the old binding could segfault the interpreter instead of raising.
    script = """
import gc
import sys

from triton._C.libtriton import anchor, ir
from triton_anchor import (
    ANCHOR_IR_NORMALIZATION_VERSION,
    ANCHOR_IR_SPEC_VERSION,
    AnchorIRNormalizer,
    AnchorIRPhase,
    AnchorIRTrack,
    StructuredAnchorIRValidator,
)


def parse_with_ephemeral_context(path):
    context = ir.context()
    ir.load_dialects(context)
    anchor.load_dialects(context)
    return ir.parse_mlir_module(path, context)


module = parse_with_ephemeral_context(sys.argv[1])
gc.collect()

report = StructuredAnchorIRValidator().validate_module(
    module,
    spec_version=ANCHOR_IR_SPEC_VERSION,
    track=AnchorIRTrack.LINALG,
    phase=AnchorIRPhase.PRE_HOOK,
)
normalized = AnchorIRNormalizer().normalize_module(
    module,
    normalization_version=ANCHOR_IR_NORMALIZATION_VERSION,
    spec_version=ANCHOR_IR_SPEC_VERSION,
    track=AnchorIRTrack.LINALG,
    phase=AnchorIRPhase.PRE_HOOK,
)
assert report.valid, report
assert normalized.acceptable, normalized
assert normalized.normalized_text is not None
print("standalone ModuleOp validation and normalization succeeded")
"""
    repository = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    # Keep the child on the exact native package selected by the parent test
    # environment.  A source checkout may also contain an older in-place
    # libtriton.so next to ``triton/python``; putting that directory first
    # would turn this regression into a test of a stale binary.
    native_package_root = str(Path(triton.__file__).resolve().parents[1])
    pythonpath = [
        native_package_root,
        os.path.join(repository, "python"),
        os.path.join(repository, "triton", "python"),
    ]
    inherited_pythonpath = os.environ.get("PYTHONPATH")
    if inherited_pythonpath:
        pythonpath.append(inherited_pythonpath)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(pythonpath)

    completed = subprocess.run(
        [sys.executable, "-c", script, str(source)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, (
        "standalone ModuleOp use must not crash after its parse context is "
        "garbage-collected; return code=%s\nstdout:\n%s\nstderr:\n%s"
        % (completed.returncode, completed.stdout, completed.stderr)
    )
    assert "standalone ModuleOp validation and normalization succeeded" in (
        completed.stdout
    )
