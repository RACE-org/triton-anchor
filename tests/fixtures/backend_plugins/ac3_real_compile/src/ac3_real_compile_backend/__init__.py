"""AC3 fixture backend: minimal real compiler for the Registry -> triton.compile() proof.

This is an acceptance fixture only, not a product backend.  It implements
exactly the ``triton.compile()`` contract needed to prove that the compiler
class selected by BackendPluginRegistry is instantiated and driven through the
real Triton compilation pipeline (ASTSource -> TTIR stage -> final stage).
"""

from __future__ import annotations

import hashlib
import json
import re

from triton._C.libtriton import anchor
from triton.backends.compiler import BaseBackend, GPUTarget
from triton.backends.driver import DriverBase


TARGET = "ac3_mock"


class AC3RealCompileOptions:
    """Minimal options object required by triton.compile() (must expose hash())."""

    def __init__(self, values):
        self.num_warps = values.get("num_warps", 1)
        self.num_stages = values.get("num_stages", 1)
        self.num_ctas = values.get("num_ctas", 1)
        self.cluster_dims = values.get("cluster_dims", (1, 1, 1))
        self.ptx_version = values.get("ptx_version", None)
        self.enable_fp_fusion = values.get("enable_fp_fusion", True)
        self.supported_fp8_dtypes = values.get("supported_fp8_dtypes", ())
        self.deprecated_fp8_dtypes = values.get("deprecated_fp8_dtypes", ())
        self.allowed_dot_input_precisions = values.get(
            "allowed_dot_input_precisions", ("ieee", "tf32", "tf32x3")
        )
        self.allow_fp8e4nv = values.get("allow_fp8e4nv", False)
        self.max_num_imprecise_acc_default = values.get(
            "max_num_imprecise_acc_default", False
        )
        self.debug = values.get("debug", False)

    def hash(self) -> str:
        payload = json.dumps(self.__dict__, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class AC3RealCompileCompiler(BaseBackend):
    """Real BaseBackend implementation with the minimum compile pipeline."""

    # Class-level stage log shared by every instance so the probe can prove
    # triton.compile() drove the pipeline through this plugin's compiler.
    stage_calls = []
    # CompiledKernel reads the final stage artifact using this extension.
    binary_ext = "asm"

    @classmethod
    def supports_target(cls, target):
        return getattr(target, "backend", None) == TARGET

    def hash(self) -> str:
        return "ac3-real-compile"

    def parse_options(self, options):
        if options.get("break_parse_options"):
            raise RuntimeError("AC3 fixture injected compile failure")
        return AC3RealCompileOptions(options)

    def add_stages(self, stages, options):
        def ttir_stage(module, metadata):
            type(self).stage_calls.append("ttir")
            # CompiledKernel requires metadata["name"]; stage functions are the
            # documented hook for backends to populate the metadata dict.
            match = re.search(r"@([A-Za-z_][A-Za-z0-9_.]*)", str(module))
            assert match is not None, "TTIR module has no function name"
            metadata["name"] = match.group(1)
            return module

        def asm_stage(module, metadata):
            type(self).stage_calls.append("asm")
            return str(module).encode("utf-8")

        stages["ttir"] = ttir_stage
        stages["asm"] = asm_stage

    def load_dialects(self, context):
        # tt.load/tt.store dialect registration required by the AST frontend.
        anchor.load_dialects(context)

    def get_codegen_implementation(self):
        return None

    def pack_metadata(self, metadata):
        # CompiledKernel stores this for the launcher; the acceptance probe
        # never launches, so the namedtuple metadata is the packed form.
        return metadata


class AC3RealCompileDriver(DriverBase):

    @classmethod
    def is_active(cls):
        return True

    def get_current_target(self):
        return GPUTarget(TARGET, 1, 1)


compiler_cls = AC3RealCompileCompiler
driver_cls = AC3RealCompileDriver
