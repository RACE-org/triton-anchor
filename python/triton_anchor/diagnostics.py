"""Pass-level diagnostics for Triton Anchor compilation pipelines."""

from __future__ import annotations

import json
import os
import re
import tempfile
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from .hw_capability import ComputeParadigm, HWCapability


PassAdder = Callable[[Any], None]


@dataclass(frozen=True)
class PassDescriptor:
    """A single pass in a diagnosable pipeline."""

    name: str
    add_to_pass_manager: PassAdder
    optional: bool = False


@dataclass
class PassRunRecord:
    """Execution record for one pass."""

    index: int
    name: str
    ok: bool
    before_ir: Path
    after_ir: Optional[Path] = None
    diagnostic_path: Optional[Path] = None
    error: Optional[str] = None
    diagnostic_text: Optional[str] = None
    mlir_location: Optional["MLIRDiagnosticLocation"] = None
    traceback: Optional[str] = None


@dataclass
class MLIRDiagnosticLocation:
    """Best-effort MLIR diagnostic location for a failed pass."""

    raw: str
    file: Optional[str] = None
    line: Optional[int] = None
    column: Optional[int] = None
    operation: Optional[str] = None
    ir_line: Optional[int] = None
    ir_snippet: Optional[str] = None


@dataclass
class PassDiagnosticResult:
    """Summary for a pass diagnostic run."""

    ok: bool
    pipeline: str
    total_passes: int
    output_dir: Path
    records: list[PassRunRecord] = field(default_factory=list)
    failed_pass: Optional[str] = None
    failed_index: Optional[int] = None
    error: Optional[str] = None
    mlir_location: Optional[MLIRDiagnosticLocation] = None
    summary_path: Optional[Path] = None

    @property
    def executed_passes(self) -> int:
        return len(self.records)

    @property
    def failed_record(self) -> Optional[PassRunRecord]:
        if self.failed_index is None:
            return None
        for record in self.records:
            if record.index == self.failed_index:
                return record
        return None


class PassDiagnostic:
    """Run compiler pipelines one pass at a time and save IR snapshots."""

    def __init__(
        self,
        output_dir: Optional[str | Path] = None,
        *,
        enable_debug: bool = True,
        save_success_snapshots: bool = True,
        write_summary_json: bool = True,
        capture_stderr: bool = True,
    ) -> None:
        self.output_dir = Path(
            output_dir or tempfile.mkdtemp(prefix="triton-anchor-diagnose-")
        )
        self.enable_debug = enable_debug
        self.save_success_snapshots = save_success_snapshots
        self.write_summary_json = write_summary_json
        self.capture_stderr = capture_stderr
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def diagnose_ttir(
        self,
        mod: Any,
        *,
        hw: Optional[HWCapability] = None,
    ) -> PassDiagnosticResult:
        """Diagnose the standard TTIR pipeline at pass granularity.

        Args:
            mod: MLIR module to mutate in-place.
            hw: Optional hardware capability. When provided, conditional TTIR
                passes are appended using the same policy as the normal
                ``build_ttir_pipeline`` path.

        Returns:
            ``PassDiagnosticResult`` containing pass records and failure data.
        """
        return self._diagnose_pipeline(
            mod,
            pipeline="ttir",
            descriptors=list(build_ttir_pass_descriptors(hw=hw)),
        )

    def diagnose_triton_linalg(self, mod: Any) -> PassDiagnosticResult:
        """Diagnose the triton-linalg adapter conversion pipeline."""
        return self._diagnose_pipeline(
            mod,
            pipeline="triton-linalg",
            descriptors=list(build_triton_linalg_pass_descriptors()),
        )

    def _diagnose_pipeline(
        self,
        mod: Any,
        *,
        pipeline: str,
        descriptors: list[PassDescriptor],
    ) -> PassDiagnosticResult:
        from triton._C.libtriton import ir

        records: list[PassRunRecord] = []
        result = PassDiagnosticResult(
            ok=True,
            pipeline=pipeline,
            total_passes=len(descriptors),
            output_dir=self.output_dir,
            records=records,
        )

        for index, descriptor in enumerate(descriptors, start=1):
            before_ir_text = str(mod)
            before_path = self._write_ir(index, descriptor.name, "before", mod)
            pm = ir.pass_manager(mod.context)
            if self.enable_debug:
                pm.enable_debug()
            descriptor.add_to_pass_manager(pm)

            try:
                diagnostic_text = self._run_pass(pm, mod)
            except Exception as exc:
                diagnostic_text = getattr(exc, "diagnostic_text", "")
                original_exc = getattr(exc, "original", exc)
                error = str(original_exc)
                tb = "".join(
                    traceback.format_exception(
                        type(original_exc),
                        original_exc,
                        original_exc.__traceback__,
                    )
                )
                mlir_location = _extract_mlir_location(
                    error,
                    diagnostic_text,
                    before_ir_text,
                    before_path,
                )
                diagnostic_path = self._write_failure_diagnostic(
                    index,
                    descriptor.name,
                    error,
                    diagnostic_text,
                    tb,
                    mlir_location,
                )
                record = PassRunRecord(
                    index=index,
                    name=descriptor.name,
                    ok=False,
                    before_ir=before_path,
                    diagnostic_path=diagnostic_path,
                    error=error,
                    diagnostic_text=diagnostic_text or None,
                    mlir_location=mlir_location,
                    traceback=tb,
                )
                records.append(record)
                result.ok = False
                result.failed_pass = descriptor.name
                result.failed_index = index
                result.error = error
                result.mlir_location = mlir_location
                self._write_summary(result)
                return result

            after_path = None
            if self.save_success_snapshots:
                after_path = self._write_ir(index, descriptor.name, "after", mod)
            records.append(
                PassRunRecord(
                    index=index,
                    name=descriptor.name,
                    ok=True,
                    before_ir=before_path,
                    after_ir=after_path,
                )
            )

        self._write_summary(result)
        return result

    def _run_pass(self, pm: Any, mod: Any) -> str:
        if not self.capture_stderr:
            pm.run(mod)
            return ""
        return _run_with_stderr_capture(pm, mod)

    def _write_ir(self, index: int, pass_name: str, stage: str, mod: Any) -> Path:
        path = self.output_dir / f"{index:02d}-{_safe_filename(pass_name)}.{stage}.mlir"
        path.write_text(str(mod), encoding="utf-8")
        return path

    def _write_summary(self, result: PassDiagnosticResult) -> None:
        if not self.write_summary_json:
            return
        path = self.output_dir / "summary.json"
        result.summary_path = path
        path.write_text(
            json.dumps(_result_to_jsonable(result), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _write_failure_diagnostic(
        self,
        index: int,
        pass_name: str,
        error: str,
        diagnostic_text: str,
        tb: str,
        mlir_location: Optional[MLIRDiagnosticLocation],
    ) -> Path:
        path = self.output_dir / f"{index:02d}-{_safe_filename(pass_name)}.diagnostic.txt"
        parts = [
            f"pass: {pass_name}",
            f"error: {error}",
        ]
        if mlir_location is not None:
            parts.extend(
                [
                    "location:",
                    f"  raw: {mlir_location.raw}",
                    f"  file: {mlir_location.file}",
                    f"  line: {mlir_location.line}",
                    f"  column: {mlir_location.column}",
                    f"  operation: {mlir_location.operation}",
                    f"  ir_line: {mlir_location.ir_line}",
                    f"  ir_snippet: {mlir_location.ir_snippet}",
                ]
            )
        if diagnostic_text:
            parts.extend(["captured diagnostics:", diagnostic_text.rstrip()])
        if tb:
            parts.extend(["traceback:", tb.rstrip()])
        path.write_text("\n".join(parts) + "\n", encoding="utf-8")
        return path


def extract_mlir_location(
    error: str,
    diagnostic_text: str = "",
    before_ir_text: str = "",
    before_path: str | Path | None = None,
) -> Optional[MLIRDiagnosticLocation]:
    """Extract a best-effort MLIR source/op location from diagnostic text."""
    return _extract_mlir_location(
        error,
        diagnostic_text,
        before_ir_text,
        Path(before_path) if before_path is not None else Path("<unknown>"),
    )


def build_ttir_pass_descriptors(
    *, hw: Optional[HWCapability] = None
) -> Iterable[PassDescriptor]:
    """Build diagnosable descriptors for the Triton Anchor TTIR pipeline."""
    from triton._C.libtriton import passes

    descriptors: list[PassDescriptor] = [
        PassDescriptor("common.inliner", passes.common.add_inliner),
        PassDescriptor("ttir.combine", passes.ttir.add_combine),
        PassDescriptor("common.canonicalizer", passes.common.add_canonicalizer),
        PassDescriptor("ttir.reorder_broadcast", passes.ttir.add_reorder_broadcast),
        PassDescriptor("common.cse", passes.common.add_cse),
        PassDescriptor("common.licm", passes.common.add_licm),
        PassDescriptor("common.symbol_dce", passes.common.add_symbol_dce),
    ]

    if hw is None:
        return descriptors

    if hw.compute_paradigm == ComputeParadigm.GPGPU:
        descriptors.append(
            PassDescriptor(
                "ttir.rewrite_tensor_pointer",
                _require_pass_adder(passes.ttir, "add_rewrite_tensor_pointer"),
            )
        )

    if hw.enable_loop_unroll:
        optional = _optional_pass_adder(passes.ttir, "add_loop_unroll")
        if optional is not None:
            descriptors.append(PassDescriptor("ttir.loop_unroll", optional, optional=True))

    optional = _optional_pass_adder(passes.ttir, "add_expression_restructing")
    if optional is not None:
        descriptors.append(
            PassDescriptor("ttir.expression_restructing", optional, optional=True)
        )

    return descriptors


def build_triton_linalg_pass_descriptors() -> Iterable[PassDescriptor]:
    """Build diagnosable descriptors for the triton-linalg adapter pipeline."""
    from triton._C.libtriton.anchor import anchor_passes
    from triton._C.libtriton.passes import common

    if not hasattr(anchor_passes, "triton_to_linalg"):
        raise RuntimeError("anchor_passes.triton_to_linalg not available.")

    tl = anchor_passes.triton_to_linalg
    return [
        PassDescriptor(
            "triton_linalg.wrap_func_body_with_single_block.pre",
            _require_pass_adder(tl, "add_wrap_func_body_with_single_block"),
        ),
        PassDescriptor("common.inliner", common.add_inliner),
        PassDescriptor("common.canonicalizer.pre", common.add_canonicalizer),
        PassDescriptor(
            "triton_linalg.canonicalize_triton",
            _require_pass_adder(tl, "add_canonicalize_triton"),
        ),
        PassDescriptor(
            "triton_linalg.pointer_strength_reduction",
            _require_pass_adder(tl, "add_pointer_strength_reduction"),
        ),
        PassDescriptor(
            "common.canonicalizer.after_pointer_strength_reduction",
            common.add_canonicalizer,
        ),
        PassDescriptor(
            "triton_linalg.triton_to_linalg",
            _require_pass_adder(tl, "add_triton_to_linalg"),
        ),
        PassDescriptor(
            "triton_linalg.extract_like_move_backward",
            _require_pass_adder(tl, "add_extract_like_move_backward"),
        ),
        PassDescriptor("common.canonicalizer.post_conversion", common.add_canonicalizer),
        PassDescriptor(
            "triton_linalg.arith_to_linalg",
            _require_pass_adder(tl, "add_arith_to_linalg"),
        ),
        PassDescriptor(
            "triton_linalg.math_to_linalg",
            _require_pass_adder(tl, "add_math_to_linalg"),
        ),
        PassDescriptor("common.cse", common.add_cse),
        PassDescriptor("common.licm", common.add_licm),
        PassDescriptor(
            "triton_linalg.wrap_func_body_with_single_block.final",
            _require_pass_adder(tl, "add_wrap_func_body_with_single_block"),
        ),
    ]


def _optional_pass_adder(module: Any, pass_name: str) -> Optional[PassAdder]:
    fn = getattr(module, pass_name, None)
    if fn is None:
        return None
    return fn


def _require_pass_adder(module: Any, pass_name: str) -> PassAdder:
    fn = getattr(module, pass_name, None)
    if fn is None:
        mod_name = getattr(module, "__name__", str(module))
        raise RuntimeError(
            f"Required pass '{pass_name}' not found in module '{mod_name}'. "
            "This pass is critical for the current compilation path. "
            "Check your Triton version and backend installation."
        )
    return fn


class _CapturedPassError(Exception):
    def __init__(self, original: Exception, diagnostic_text: str) -> None:
        super().__init__(str(original))
        self.original = original
        self.diagnostic_text = diagnostic_text


def _run_with_stderr_capture(pm: Any, mod: Any) -> str:
    saved_stderr_fd = os.dup(2)
    original: Optional[Exception] = None
    with tempfile.TemporaryFile(mode="w+b") as captured:
        try:
            os.dup2(captured.fileno(), 2)
            try:
                pm.run(mod)
            except Exception as exc:
                original = exc
            finally:
                os.dup2(saved_stderr_fd, 2)
        finally:
            os.close(saved_stderr_fd)

        captured.seek(0)
        diagnostic_text = captured.read().decode("utf-8", errors="replace")

    if original is not None:
        raise _CapturedPassError(original, diagnostic_text) from original
    return diagnostic_text


def _extract_mlir_location(
    error: str,
    diagnostic_text: str,
    before_ir_text: str,
    before_path: Path,
) -> Optional[MLIRDiagnosticLocation]:
    text = "\n".join(part for part in [diagnostic_text, error] if part)
    if not text:
        return None

    match = _find_location_match(text)
    if match is None:
        op_from_text = _extract_operation_from_text(text)
        if op_from_text is None:
            return None
        return MLIRDiagnosticLocation(raw=op_from_text, operation=op_from_text)

    raw = match.group(0)
    file_name = match.groupdict().get("file")
    line = _to_int(match.groupdict().get("line"))
    column = _to_int(match.groupdict().get("column"))
    ir_line, ir_snippet = _find_ir_line_for_location(
        before_ir_text,
        raw,
        file_name,
        line,
        before_path,
    )
    operation = _extract_operation_from_text(text)
    if operation is None and ir_snippet is not None:
        operation = _extract_operation_from_ir_line(ir_snippet)

    return MLIRDiagnosticLocation(
        raw=raw,
        file=file_name,
        line=line,
        column=column,
        operation=operation,
        ir_line=ir_line,
        ir_snippet=ir_snippet,
    )


def _find_location_match(text: str) -> Optional[re.Match[str]]:
    patterns = [
        r'loc\("(?P<file>[^"]+)":(?P<line>\d+):(?P<column>\d+)\)',
        r'(?P<file>[^\s:"\']+\.m?lir):(?P<line>\d+):(?P<column>\d+)',
        r'(?P<file><[^>]+>):(?P<line>\d+):(?P<column>\d+)',
        r'(?P<line>\d+):(?P<column>\d+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match is not None:
            return match
    return None


def _find_ir_line_for_location(
    before_ir_text: str,
    raw_location: str,
    file_name: Optional[str],
    line: Optional[int],
    before_path: Path,
) -> tuple[Optional[int], Optional[str]]:
    lines = before_ir_text.splitlines()

    if raw_location:
        for index, text_line in enumerate(lines, start=1):
            if raw_location in text_line:
                return index, text_line.strip()

    if line is None or line < 1 or line > len(lines):
        return None, None

    if file_name is None or _location_file_matches_ir(file_name, before_path):
        return line, lines[line - 1].strip()

    location_pattern = f"{file_name}:{line}:"
    for index, text_line in enumerate(lines, start=1):
        if location_pattern in text_line:
            return index, text_line.strip()

    return None, None


def _location_file_matches_ir(file_name: str, before_path: Path) -> bool:
    if file_name in {"<stdin>", "<unknown>", "-"}:
        return True
    try:
        return Path(file_name).resolve() == before_path.resolve()
    except OSError:
        return Path(file_name).name == before_path.name


def _extract_operation_from_text(text: str) -> Optional[str]:
    patterns = [
        r"failed to legalize operation ['\"](?P<op>[\w.]+)['\"]",
        r"operation ['\"](?P<op>[\w.]+)['\"]",
        r"op ['\"](?P<op>[\w.]+)['\"]",
        r"(?P<op>[A-Za-z_][\w]*\.[A-Za-z_][\w]*) op",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match is not None:
            return match.group("op")
    return None


def _extract_operation_from_ir_line(line: str) -> Optional[str]:
    text = line.strip()
    if not text or text.startswith("//") or text.startswith("#"):
        return None

    if " = " in text:
        text = text.split(" = ", 1)[1].lstrip()

    match = re.match(r'"(?P<quoted>[\w.]+)"', text)
    if match is not None:
        return match.group("quoted")

    match = re.match(r"(?P<op>[A-Za-z_][\w]*\.[A-Za-z_][\w]*)\b", text)
    if match is not None:
        return match.group("op")

    return None


def _to_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def _safe_filename(value: str) -> str:
    chars = []
    for char in value:
        chars.append(char if char.isalnum() or char in "._-" else "_")
    filename = "".join(chars).strip("._")
    return filename or "pass"


def _result_to_jsonable(result: PassDiagnosticResult) -> dict[str, Any]:
    data = asdict(result)
    return _paths_to_strings(data)


def _paths_to_strings(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_paths_to_strings(item) for item in value]
    if isinstance(value, dict):
        return {key: _paths_to_strings(item) for key, item in value.items()}
    return value
