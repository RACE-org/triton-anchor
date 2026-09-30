"""Pass-level diagnostics for Triton Anchor compilation pipelines."""

from __future__ import annotations

import json
import importlib.machinery
import importlib.util
import os
import re
import sys
import tempfile
import time
import traceback
import types
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
    disable_verifier: bool = False


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
    duration_ms: float = 0.0
    before_ir_bytes: int = 0
    after_ir_bytes: int = 0
    ir_delta_bytes: int = 0
    peak_rss_bytes: int = 0


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
    total_duration_ms: float = 0.0
    input_ir_bytes: int = 0
    output_ir_bytes: int = 0
    peak_rss_bytes: int = 0

    @property
    def executed_passes(self) -> int:
        return len(self.records)

    @property
    def slowest_pass(self) -> Optional[PassRunRecord]:
        """Return the executed pass with the longest wall-clock duration."""
        if not self.records:
            return None
        return max(self.records, key=lambda record: record.duration_ms)

    @property
    def failed_record(self) -> Optional[PassRunRecord]:
        if self.failed_index is None:
            return None
        for record in self.records:
            if record.index == self.failed_index:
                return record
        return None


@dataclass
class StageDiagnosticResult:
    """Summary for a non-pass compile/runtime diagnostic."""

    ok: bool
    stage: str
    output_dir: Path
    diagnostic_path: Path
    error: str
    diagnostic_text: Optional[str] = None
    command: Optional[list[str]] = None
    returncode: Optional[int] = None
    input_path: Optional[Path] = None
    mlir_location: Optional[MLIRDiagnosticLocation] = None
    traceback: Optional[str] = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    summary_path: Optional[Path] = None


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

    def diagnose_sophgo_pplir(self, mod: Any) -> PassDiagnosticResult:
        """Diagnose the Sophgo Linalg/PPL lowering pipeline."""
        return self._diagnose_pipeline(
            mod,
            pipeline="sophgo-pplir",
            descriptors=list(build_sophgo_pplir_pass_descriptors()),
        )

    @staticmethod
    def _ir_bytes(ir_text_or_module: Any) -> int:
        """Return the UTF-8 size of an IR value, or zero if unavailable."""
        try:
            text = (
                ir_text_or_module
                if isinstance(ir_text_or_module, str)
                else str(ir_text_or_module)
            )
            return len(text.encode("utf-8"))
        except Exception:
            return 0

    @staticmethod
    def _sample_peak_rss_bytes() -> int:
        """Return this process's peak RSS in bytes, or zero if unavailable."""
        try:
            import resource

            peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return peak_rss if sys.platform == "darwin" else peak_rss * 1024
        except Exception:
            return 0

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
        result.input_ir_bytes = self._ir_bytes(mod)

        for index, descriptor in enumerate(descriptors, start=1):
            before_ir_text = str(mod)
            before_path = self._write_ir(index, descriptor.name, "before", mod)
            before_ir_bytes = self._ir_bytes(before_ir_text)
            pm = ir.pass_manager(mod.context)
            if self.enable_debug:
                pm.enable_debug()
            if descriptor.disable_verifier and hasattr(pm, "enable_verifier"):
                pm.enable_verifier(False)
            descriptor.add_to_pass_manager(pm)

            started_at = time.monotonic()
            try:
                diagnostic_text = self._run_pass(pm, mod)
            except Exception as exc:
                duration_ms = (time.monotonic() - started_at) * 1000.0
                peak_rss_bytes = self._sample_peak_rss_bytes()
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
                    duration_ms=duration_ms,
                    before_ir_bytes=before_ir_bytes,
                    peak_rss_bytes=peak_rss_bytes,
                )
                records.append(record)
                result.ok = False
                result.failed_pass = descriptor.name
                result.failed_index = index
                result.error = error
                result.mlir_location = mlir_location
                result.total_duration_ms = sum(
                    record.duration_ms for record in records
                )
                result.output_ir_bytes = before_ir_bytes
                result.peak_rss_bytes = max(
                    (record.peak_rss_bytes for record in records), default=0
                )
                self._write_summary(result)
                return result

            duration_ms = (time.monotonic() - started_at) * 1000.0
            after_ir_bytes = self._ir_bytes(mod)
            peak_rss_bytes = self._sample_peak_rss_bytes()
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
                    duration_ms=duration_ms,
                    before_ir_bytes=before_ir_bytes,
                    after_ir_bytes=after_ir_bytes,
                    ir_delta_bytes=after_ir_bytes - before_ir_bytes,
                    peak_rss_bytes=peak_rss_bytes,
                )
            )

        result.total_duration_ms = sum(record.duration_ms for record in records)
        result.output_ir_bytes = self._ir_bytes(mod)
        result.peak_rss_bytes = max(
            (record.peak_rss_bytes for record in records), default=0
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


class StageDiagnostic:
    """Write diagnostics for stages that are not MLIR pass pipelines."""

    def __init__(
        self,
        output_dir: Optional[str | Path] = None,
        *,
        write_summary_json: bool = True,
    ) -> None:
        self.output_dir = Path(
            output_dir or tempfile.mkdtemp(prefix="triton-anchor-stage-diagnose-")
        )
        self.write_summary_json = write_summary_json
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def record_failure(
        self,
        stage: str,
        error: str,
        *,
        diagnostic_text: str = "",
        command: Optional[Iterable[Any]] = None,
        returncode: Optional[int] = None,
        input_path: Optional[str | Path] = None,
        input_text: str = "",
        artifacts: Optional[dict[str, Any]] = None,
        extra: Optional[dict[str, Any]] = None,
        original: Optional[BaseException] = None,
    ) -> StageDiagnosticResult:
        input_path_obj = Path(input_path) if input_path is not None else None
        if not input_text and input_path_obj is not None:
            try:
                input_text = input_path_obj.read_text(encoding="utf-8")
            except OSError:
                input_text = ""

        tb = ""
        if original is not None:
            tb = "".join(
                traceback.format_exception(
                    type(original),
                    original,
                    original.__traceback__,
                )
            )

        location = extract_mlir_location(
            error,
            diagnostic_text,
            input_text,
            input_path_obj,
        )
        diagnostic_path = self.output_dir / f"{_safe_filename(stage)}.diagnostic.txt"
        result = StageDiagnosticResult(
            ok=False,
            stage=stage,
            output_dir=self.output_dir,
            diagnostic_path=diagnostic_path,
            error=error,
            diagnostic_text=diagnostic_text or None,
            command=[str(item) for item in command] if command is not None else None,
            returncode=returncode,
            input_path=input_path_obj,
            mlir_location=location,
            traceback=tb or None,
            artifacts=artifacts or {},
            extra=extra or {},
        )
        self._write_failure_diagnostic(result)
        self._write_summary(result)
        return result

    def _write_summary(self, result: StageDiagnosticResult) -> None:
        if not self.write_summary_json:
            return
        path = self.output_dir / "summary.json"
        result.summary_path = path
        path.write_text(
            json.dumps(_paths_to_strings(asdict(result)), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _write_failure_diagnostic(self, result: StageDiagnosticResult) -> None:
        parts = [
            f"stage: {result.stage}",
            f"error: {result.error}",
        ]
        if result.command is not None:
            parts.append(f"command: {' '.join(result.command)}")
        if result.returncode is not None:
            parts.append(f"returncode: {result.returncode}")
        if result.input_path is not None:
            parts.append(f"input: {result.input_path}")
        if result.mlir_location is not None:
            location = result.mlir_location
            parts.extend(
                [
                    "location:",
                    f"  raw: {location.raw}",
                    f"  file: {location.file}",
                    f"  line: {location.line}",
                    f"  column: {location.column}",
                    f"  operation: {location.operation}",
                    f"  ir_line: {location.ir_line}",
                    f"  ir_snippet: {location.ir_snippet}",
                ]
            )
        if result.artifacts:
            parts.extend(["artifacts:", json.dumps(_paths_to_strings(result.artifacts), indent=2, sort_keys=True)])
        if result.extra:
            parts.extend(["extra:", json.dumps(_paths_to_strings(result.extra), indent=2, sort_keys=True)])
        if result.diagnostic_text:
            parts.extend(["captured diagnostics:", result.diagnostic_text.rstrip()])
        if result.traceback:
            parts.extend(["traceback:", result.traceback.rstrip()])
        result.diagnostic_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


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


def build_sophgo_pplir_pass_descriptors() -> Iterable[PassDescriptor]:
    """Build diagnosable descriptors for Sophgo Linalg/PPL lowering."""
    from triton._C.libtriton.passes import common

    sophgo_passes = _load_sophgo_passes_module()

    return [
        PassDescriptor(
            "sophgo.triton_to_ppl",
            _require_pass_adder(sophgo_passes, "add_triton_to_ppl"),
        ),
        PassDescriptor(
            "sophgo.linalg_to_ppl",
            _require_pass_adder(sophgo_passes, "add_linalg_to_ppl"),
            disable_verifier=True,
        ),
        PassDescriptor("common.cse.pplir", common.add_cse, disable_verifier=True),
        PassDescriptor(
            "common.canonicalizer.pplir",
            common.add_canonicalizer,
            disable_verifier=True,
        ),
    ]


def _load_sophgo_passes_module():
    try:
        from triton_sophgo._C import passes as sophgo_passes

        return sophgo_passes
    except Exception as exc:
        normal_import_error = exc

    extension_path = _find_sophgo_extension()
    if extension_path is None:
        raise RuntimeError(
            "triton_sophgo._C.passes is not available. Install "
            "triton-sophgo-backend before running sophgo-pplir diagnostics."
        ) from normal_import_error

    package = types.ModuleType("triton_sophgo")
    package.__path__ = [str(extension_path.parent)]
    package.__spec__ = importlib.machinery.ModuleSpec(
        "triton_sophgo",
        loader=None,
        is_package=True,
    )
    sys.modules["triton_sophgo"] = package

    spec = importlib.util.spec_from_file_location("triton_sophgo._C", extension_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"failed to load Sophgo extension from {extension_path}"
        ) from normal_import_error

    module = importlib.util.module_from_spec(spec)
    sys.modules["triton_sophgo._C"] = module
    spec.loader.exec_module(module)
    sophgo_passes = getattr(module, "passes", None)
    if sophgo_passes is None:
        raise RuntimeError(
            f"Sophgo extension {extension_path} does not export passes"
        ) from normal_import_error
    sys.modules["triton_sophgo._C.passes"] = sophgo_passes
    return sophgo_passes


def _find_sophgo_extension() -> Optional[Path]:
    for entry in sys.path:
        if not entry:
            entry = os.getcwd()
        package_dir = Path(entry) / "triton_sophgo"
        for candidate in package_dir.glob("_C*.so"):
            return candidate
    return None


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

    operation = _extract_operation_from_text(text)
    match = _find_location_match(text)
    if match is None:
        if operation is None:
            return None
        return MLIRDiagnosticLocation(raw=operation, operation=operation)

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
        operation,
    )
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
    best_match: Optional[re.Match[str]] = None
    best_score: Optional[int] = None
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            line_start = text.rfind("\n", 0, match.start()) + 1
            line_end = text.find("\n", match.end())
            if line_end == -1:
                line_end = len(text)
            diagnostic_line = text[line_start:line_end]
            score = 0
            if "error:" in diagnostic_line:
                score += 100
            if "failed" in diagnostic_line.lower():
                score += 25
            if "cannot open output file" in diagnostic_line:
                score -= 100
            if best_score is None or score > best_score:
                best_score = score
                best_match = match
    return best_match


def _find_ir_line_for_location(
    before_ir_text: str,
    raw_location: str,
    file_name: Optional[str],
    line: Optional[int],
    before_path: Path,
    operation: Optional[str] = None,
) -> tuple[Optional[int], Optional[str]]:
    lines = before_ir_text.splitlines()

    if raw_location:
        for index, text_line in enumerate(lines, start=1):
            if raw_location in text_line:
                alias = _extract_location_alias(text_line)
                if alias is not None:
                    alias_match = _find_ir_line_for_alias(lines, alias, operation)
                    if alias_match is not None:
                        return alias_match
                return index, text_line.strip()

    if line is None or line < 1 or line > len(lines):
        return None, None

    if file_name is None or _location_file_matches_ir(file_name, before_path):
        return line, lines[line - 1].strip()

    location_pattern = f"{file_name}:{line}:"
    for index, text_line in enumerate(lines, start=1):
        if location_pattern in text_line:
            alias = _extract_location_alias(text_line)
            if alias is not None:
                alias_match = _find_ir_line_for_alias(lines, alias, operation)
                if alias_match is not None:
                    return alias_match
            return index, text_line.strip()

    return None, None


def _extract_location_alias(text_line: str) -> Optional[str]:
    match = re.match(r"\s*(#loc\d+)\s*=", text_line)
    if match is None:
        return None
    return match.group(1)


def _find_ir_line_for_alias(
    lines: list[str],
    alias: str,
    operation: Optional[str],
) -> Optional[tuple[int, str]]:
    alias_ref = f"loc({alias})"
    if operation is not None:
        for index, text_line in enumerate(lines, start=1):
            if alias_ref in text_line and operation in text_line:
                return index, text_line.strip()
    for index, text_line in enumerate(lines, start=1):
        if alias_ref in text_line and _extract_operation_from_ir_line(text_line):
            return index, text_line.strip()
    return None


def _location_file_matches_ir(file_name: str, before_path: Path) -> bool:
    if file_name in {"<stdin>", "<unknown>", "-"}:
        return True
    try:
        return Path(file_name).resolve() == before_path.resolve()
    except OSError:
        return Path(file_name).name == before_path.name


def _extract_operation_from_text(text: str) -> Optional[str]:
    if "C dimension affected" in text:
        return "linalg.broadcast"
    if "unsupported transpose" in text or "cw transpose" in text:
        return "linalg.transpose"

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
