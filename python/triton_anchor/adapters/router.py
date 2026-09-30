"""
Adapter Router
==============

T6.1 deterministic adapter selection.

The registry owns registration and discovery.  The router owns policy:
matching ``HWCapability`` to an adapter, rejecting incompatible candidates, and
recording selection metadata for downstream compilation diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence, Tuple

from .base import AdapterNotFoundError, ITritonToLinalgAdapter


ADAPTER_ROUTING_POLICY_VERSION = "t6.1.v1"


@dataclass(frozen=True)
class AdapterRoutingPolicy:
    """Configurable adapter routing policy.

    ``strict=True`` forbids fallback.  To allow fallback, use ``strict=False``
    and declare an ordered fallback allowlist for the requested adapter.
    """

    version: str = ADAPTER_ROUTING_POLICY_VERSION
    strict: bool = True
    fallback_allowlist: Mapping[str, Sequence[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class AdapterDecision:
    """A complete, metadata-friendly routing result."""

    selected_adapter: Optional[str]
    adapter: Optional[ITritonToLinalgAdapter]
    policy_version: str
    requested_adapter: str
    reason: str
    reject_reasons: Dict[str, str]
    fallback_reason: str = ""
    fallback_chain: Tuple[str, ...] = ()
    candidates: Tuple[str, ...] = ()
    policy_strict: bool = True
    decision_key: Mapping[str, str] = field(default_factory=dict)

    @property
    def reject_reason(self) -> str:
        if not self.reject_reasons:
            return ""
        return "; ".join(
            f"{name}: {reason}" for name, reason in sorted(self.reject_reasons.items())
        )

    def write_metadata(self, metadata: dict) -> None:
        metadata["selected_adapter"] = self.selected_adapter
        metadata["adapter_policy_version"] = self.policy_version
        metadata["adapter_reject_reason"] = self.reject_reason
        metadata["adapter_reject_reasons"] = dict(sorted(self.reject_reasons.items()))
        metadata["adapter_fallback_reason"] = self.fallback_reason
        metadata["adapter_fallback_chain"] = list(self.fallback_chain)
        metadata["adapter_selection_reason"] = self.reason
        metadata["adapter_requested_adapter"] = self.requested_adapter
        metadata["adapter_candidates"] = list(self.candidates)
        metadata["adapter_policy_strict"] = self.policy_strict
        metadata["adapter_decision_key"] = dict(sorted(self.decision_key.items()))


class AdapterRouter:
    """Deterministic router for TTIR-to-AnchorIR adapters."""

    _PTR_MODEL_TO_ADAPTER = {
        "structured": "triton-shared",
        "axis_info": "triton-linalg",
        "hybrid": "hybrid",
        "gpu": "triton-gpu",
    }

    def __init__(
        self,
        *,
        adapters: Optional[Mapping[str, ITritonToLinalgAdapter]] = None,
        registry=None,
        policy: Optional[AdapterRoutingPolicy] = None,
    ):
        if adapters is not None and registry is not None:
            raise ValueError("Pass either adapters or registry, not both.")
        self._adapters = dict(adapters) if adapters is not None else None
        self._registry = registry
        self.policy = policy or AdapterRoutingPolicy()

    def select(self, hw, metadata: Optional[dict] = None) -> AdapterDecision:
        """Select an adapter or raise ``AdapterNotFoundError``.

        Metadata is updated for both successful and failed selections.
        """
        if metadata is None:
            metadata = {}

        adapters = self._available_adapters()
        candidates = tuple(sorted(adapters))
        reject_reasons: Dict[str, str] = {}

        requested_adapter, capability_error = self._requested_adapter_for(hw)
        if capability_error:
            reject_reasons["hw_capability"] = capability_error
            return self._fail(
                metadata, hw, requested_adapter, reject_reasons, candidates
            )

        preferred_adapter = getattr(hw, "preferred_adapter", None)
        if preferred_adapter:
            requested_adapter = preferred_adapter
            adapter = adapters.get(preferred_adapter)
            if adapter is None:
                reject_reasons[preferred_adapter] = (
                    "preferred adapter is not registered"
                )
                return self._fail(
                    metadata, hw, requested_adapter, reject_reasons, candidates
                )

            rejection = self._adapter_rejection(adapter, hw)
            if rejection:
                reject_reasons[preferred_adapter] = rejection
                return self._fail(
                    metadata, hw, requested_adapter, reject_reasons, candidates
                )

            reject_reasons.update(
                self._nonselected_rejections(
                    adapters, preferred_adapter, hw, requested_adapter
                )
            )
            return self._success(
                metadata,
                hw,
                adapter,
                requested_adapter,
                "preferred adapter matched declared capabilities",
                reject_reasons,
                candidates,
            )

        adapter = adapters.get(requested_adapter)
        if adapter is not None:
            rejection = self._adapter_rejection(adapter, hw)
            if not rejection:
                reject_reasons.update(
                    self._nonselected_rejections(
                        adapters, requested_adapter, hw, requested_adapter
                    )
                )
                return self._success(
                    metadata,
                    hw,
                    adapter,
                    requested_adapter,
                    "automatic route matched track and ptr_model",
                    reject_reasons,
                    candidates,
                )
            reject_reasons[requested_adapter] = rejection
        else:
            reject_reasons[requested_adapter] = "required adapter is not registered"

        fallback = self._select_fallback(
            requested_adapter, adapters, hw, reject_reasons
        )
        if fallback is not None:
            fallback_name, fallback_adapter, fallback_reason = fallback
            reject_reasons.update(
                self._nonselected_rejections(
                    adapters, fallback_name, hw, requested_adapter
                )
            )
            return self._success(
                metadata,
                hw,
                fallback_adapter,
                requested_adapter,
                "fallback route matched policy allowlist",
                reject_reasons,
                candidates,
                fallback_reason=fallback_reason,
                fallback_chain=(requested_adapter, fallback_name),
            )

        return self._fail(metadata, hw, requested_adapter, reject_reasons, candidates)

    def get_adapter(self, hw, metadata: Optional[dict] = None) -> ITritonToLinalgAdapter:
        """Return only the selected adapter, preserving the historical API."""
        decision = self.select(hw, metadata=metadata)
        assert decision.adapter is not None
        return decision.adapter

    def _available_adapters(self) -> Dict[str, ITritonToLinalgAdapter]:
        if self._adapters is not None:
            return dict(self._adapters)

        registry = self._registry
        if registry is None:
            from .registry import AdapterRegistry

            registry = AdapterRegistry
        return registry.snapshot()

    def _requested_adapter_for(self, hw) -> Tuple[str, str]:
        track = self._track_value(getattr(hw, "anchor_ir_track", ""))
        ptr_model = str(getattr(hw, "ptr_model", ""))

        if ptr_model == "gpu" and track != "triton_gpu":
            return (
                self._PTR_MODEL_TO_ADAPTER[ptr_model],
                "ptr_model 'gpu' requires AnchorIR track 'triton_gpu'",
            )
        if track == "triton_gpu" and ptr_model != "gpu":
            return (
                self._PTR_MODEL_TO_ADAPTER.get(ptr_model, ""),
                "AnchorIR track 'triton_gpu' requires ptr_model 'gpu'",
            )
        if track == "linalg" and ptr_model == "gpu":
            return (
                self._PTR_MODEL_TO_ADAPTER[ptr_model],
                "AnchorIR track 'linalg' cannot use ptr_model 'gpu'",
            )

        adapter_name = self._PTR_MODEL_TO_ADAPTER.get(ptr_model)
        if adapter_name is None:
            return "", f"unsupported ptr_model '{ptr_model}'"
        return adapter_name, ""

    def _adapter_rejection(self, adapter: ITritonToLinalgAdapter, hw) -> str:
        track_rejection = self._adapter_track_rejection(adapter, hw)
        if track_rejection:
            return track_rejection

        ptr_models = self._supported_ptr_models(adapter)
        ptr_model = str(getattr(hw, "ptr_model", ""))
        if not ptr_models:
            return "adapter has no declared ptr_model capability"
        if ptr_model not in ptr_models:
            return (
                f"ptr_model '{ptr_model}' is not in declared ptr_models "
                f"{sorted(ptr_models)}"
            )
        return ""

    def _adapter_track_rejection(self, adapter: ITritonToLinalgAdapter, hw) -> str:
        tracks = self._supported_tracks(adapter)
        track = self._track_value(getattr(hw, "anchor_ir_track", ""))
        if track not in tracks:
            return f"track '{track}' is not in declared tracks {sorted(tracks)}"
        return ""

    def _select_fallback(
        self,
        requested_adapter: str,
        adapters: Mapping[str, ITritonToLinalgAdapter],
        hw,
        reject_reasons: Dict[str, str],
    ) -> Optional[Tuple[str, ITritonToLinalgAdapter, str]]:
        if self.policy.strict:
            reject_reasons["fallback"] = "strict policy forbids fallback"
            return None

        for fallback_name in self.policy.fallback_allowlist.get(
            requested_adapter, ()
        ):
            fallback_adapter = adapters.get(fallback_name)
            if fallback_adapter is None:
                reject_reasons[fallback_name] = "fallback adapter is not registered"
                continue

            rejection = self._adapter_track_rejection(fallback_adapter, hw)
            if rejection:
                reject_reasons[fallback_name] = rejection
                continue

            reason = (
                f"requested adapter '{requested_adapter}' rejected; "
                f"policy allowlisted fallback '{fallback_name}'"
            )
            return fallback_name, fallback_adapter, reason

        if requested_adapter:
            reject_reasons["fallback"] = (
                f"no allowlisted fallback available for '{requested_adapter}'"
            )
        return None

    def _nonselected_rejections(
        self,
        adapters: Mapping[str, ITritonToLinalgAdapter],
        selected_name: str,
        hw,
        requested_adapter: str,
    ) -> Dict[str, str]:
        reasons: Dict[str, str] = {}
        for name in sorted(adapters):
            if name == selected_name:
                continue
            rejection = self._adapter_rejection(adapters[name], hw)
            if rejection:
                reasons[name] = rejection
            elif name != requested_adapter:
                reasons[name] = (
                    f"not selected because route requested '{requested_adapter}'"
                )
        return reasons

    def _success(
        self,
        metadata: dict,
        hw,
        adapter: ITritonToLinalgAdapter,
        requested_adapter: str,
        reason: str,
        reject_reasons: Dict[str, str],
        candidates: Tuple[str, ...],
        *,
        fallback_reason: str = "",
        fallback_chain: Tuple[str, ...] = (),
    ) -> AdapterDecision:
        decision = AdapterDecision(
            selected_adapter=adapter.name(),
            adapter=adapter,
            policy_version=self.policy.version,
            requested_adapter=requested_adapter,
            reason=reason,
            reject_reasons=reject_reasons,
            fallback_reason=fallback_reason,
            fallback_chain=fallback_chain,
            candidates=candidates,
            policy_strict=self.policy.strict,
            decision_key=self._decision_key(hw, requested_adapter),
        )
        decision.write_metadata(metadata)
        return decision

    def _fail(
        self,
        metadata: dict,
        hw,
        requested_adapter: str,
        reject_reasons: Dict[str, str],
        candidates: Tuple[str, ...],
    ) -> AdapterDecision:
        if not candidates:
            reject_reasons["registry"] = "no adapters registered"
        decision = AdapterDecision(
            selected_adapter=None,
            adapter=None,
            policy_version=self.policy.version,
            requested_adapter=requested_adapter,
            reason="no adapter matched routing policy",
            reject_reasons=reject_reasons,
            candidates=candidates,
            policy_strict=self.policy.strict,
            decision_key=self._decision_key(hw, requested_adapter),
        )
        decision.write_metadata(metadata)
        raise AdapterNotFoundError(
            "No adapter selected by policy "
            f"{self.policy.version}: {decision.reject_reason}"
        )

    @staticmethod
    def _track_value(track) -> str:
        return str(getattr(track, "value", track))

    def _decision_key(self, hw, requested_adapter: str) -> Dict[str, str]:
        """Return the stable routing inputs recorded for diagnostics."""
        return {
            "anchor_ir_track": self._track_value(getattr(hw, "anchor_ir_track", "")),
            "ptr_model": str(getattr(hw, "ptr_model", "")),
            "preferred_adapter": str(getattr(hw, "preferred_adapter", "") or ""),
            "requested_adapter": requested_adapter,
            "policy_version": self.policy.version,
            "policy_strict": str(self.policy.strict),
        }

    @classmethod
    def _supported_tracks(cls, adapter: ITritonToLinalgAdapter) -> set:
        return {
            cls._track_value(track)
            for track in adapter.get_supported_tracks()
        }

    @staticmethod
    def _supported_ptr_models(adapter: ITritonToLinalgAdapter) -> set:
        return {str(model) for model in adapter.get_supported_ptr_models()}


def get_adapter(hw, metadata: Optional[dict] = None) -> ITritonToLinalgAdapter:
    """Shortcut for deterministic adapter routing."""
    return AdapterRouter().get_adapter(hw, metadata=metadata)
