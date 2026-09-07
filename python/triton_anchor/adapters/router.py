"""AdapterRouter for TTIR -> AnchorIR adapter selection.

T6.1 keeps adapter policy in this module.  ``AdapterRegistry`` owns only
registration and discovery; it does not select a weaker adapter by itself.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Tuple

from ..anchor_ir import AnchorIRTrack
from ..hw_capability import HWCapability
from .base import AdapterConversionContext, AdapterInfo, IAnchorAdapter


DEFAULT_ROUTING_TABLE = {
    (AnchorIRTrack.LINALG, "structured"): ("triton-shared",),
    (AnchorIRTrack.LINALG, "axis_info"): ("triton-linalg",),
    (AnchorIRTrack.LINALG, "hybrid"): ("hybrid",),
    (AnchorIRTrack.TRITON_GPU, "gpu"): ("triton-gpu",),
}

DEFAULT_FALLBACK_TABLE = {
    (AnchorIRTrack.LINALG, "structured"): ("triton-linalg",),
    (AnchorIRTrack.LINALG, "hybrid"): ("triton-linalg",),
}

ADAPTER_DECISION_POLICY_VERSION = "t6.1.3.3.v1"


def _string_values(values: Optional[Iterable[str] | str]) -> Tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        return (values,)
    return tuple(str(value) for value in values)


def _unique_sorted(values: Optional[Iterable[str] | str]) -> Tuple[str, ...]:
    return tuple(sorted(set(_string_values(values))))


def _unique_in_order(values: Optional[Iterable[str] | str]) -> Tuple[str, ...]:
    seen = set()
    ordered = []
    for item in _string_values(values):
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return tuple(ordered)


def _coerce_track(value: AnchorIRTrack | str) -> AnchorIRTrack:
    if isinstance(value, AnchorIRTrack):
        return value
    return AnchorIRTrack(value)


@dataclass(frozen=True)
class AdapterRouterConfig:
    """User-controlled adapter routing policy."""

    allow_fallback: bool = False
    fallback_order: Tuple[str, ...] = ("triton-linalg",)
    strict_backend_capabilities: bool = True
    metadata_key: str = "adapter_decision"

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> "AdapterRouterConfig":
        if data is None:
            return cls()
        fallback_order = data.get("fallback_order", cls.fallback_order)
        return cls(
            allow_fallback=bool(data.get("allow_fallback", cls.allow_fallback)),
            fallback_order=_unique_in_order(fallback_order),
            strict_backend_capabilities=bool(
                data.get(
                    "strict_backend_capabilities",
                    cls.strict_backend_capabilities,
                )
            ),
            metadata_key=str(data.get("metadata_key", cls.metadata_key)),
        )


@dataclass(frozen=True)
class OpCoverage:
    """Kernel op requirements and optional per-adapter coverage data."""

    required_ops: Tuple[str, ...] = ()
    adapter_supported_ops: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: Any) -> "OpCoverage":
        if value is None:
            return cls()
        if isinstance(value, OpCoverage):
            return cls(
                required_ops=_unique_sorted(value.required_ops),
                adapter_supported_ops={
                    str(name): _unique_sorted(ops)
                    for name, ops in value.adapter_supported_ops.items()
                },
            )
        if isinstance(value, Mapping):
            required_ops = value.get("required_ops", ())
            supported = value.get("adapter_supported_ops", None)
            if supported is None:
                supported = {
                    name: ops
                    for name, ops in value.items()
                    if name not in {"required_ops", "adapter_supported_ops"}
                }
            return cls(
                required_ops=_unique_sorted(required_ops),
                adapter_supported_ops={
                    str(name): _unique_sorted(ops)
                    for name, ops in dict(supported).items()
                },
            )
        return cls(required_ops=_unique_sorted(value))


@dataclass(frozen=True)
class AdapterRequest:
    """Complete router input for one kernel after TTIR optimization."""

    hw: HWCapability
    anchor_ir_track: AnchorIRTrack
    ptr_model: str
    preferred_adapter: Optional[str] = None
    backend_capabilities: Tuple[str, ...] = ()
    config: AdapterRouterConfig = field(default_factory=AdapterRouterConfig)
    op_coverage: OpCoverage = field(default_factory=OpCoverage)
    user_config: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AdapterRejection:
    """Why one adapter candidate was not selected."""

    adapter_name: str
    reason: str
    details: Tuple[str, ...] = ()

    def to_metadata(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter_name,
            "reason": self.reason,
            "details": list(self.details),
        }


@dataclass(frozen=True)
class AdapterDecision:
    """Serializable result of adapter routing."""

    selected_adapter: Optional[str]
    selected_reason: str
    rejected_adapters: Tuple[AdapterRejection, ...]
    fallback: bool
    fallback_from: Optional[str]
    fallback_authorized: bool
    fallback_candidates: Tuple[str, ...]
    anchor_ir_track: AnchorIRTrack
    ptr_model: str
    preferred_adapter: Optional[str]
    candidate_order: Tuple[str, ...]
    backend_capabilities: Tuple[str, ...]
    required_ops: Tuple[str, ...]
    hw_name: str
    adapter_decision_policy_version: str = ADAPTER_DECISION_POLICY_VERSION
    fallback_chain: Tuple[str, ...] = ()
    fallback_reason: Optional[str] = None
    decision_key: str = ""

    def to_metadata(self) -> dict[str, Any]:
        return {
            "selected_adapter": self.selected_adapter,
            "adapter_decision_policy_version": (
                self.adapter_decision_policy_version
            ),
            "selected_reason": self.selected_reason,
            "rejected_adapters": [
                rejection.to_metadata() for rejection in self.rejected_adapters
            ],
            "reject_reasons": {
                rejection.adapter_name: rejection.reason
                for rejection in self.rejected_adapters
            },
            "fallback": self.fallback,
            "fallback_from": self.fallback_from,
            "fallback_chain": list(self.fallback_chain),
            "fallback_reason": self.fallback_reason,
            "fallback_authorized": self.fallback_authorized,
            "fallback_candidates": list(self.fallback_candidates),
            "anchor_ir_track": self.anchor_ir_track.value,
            "ptr_model": self.ptr_model,
            "preferred_adapter": self.preferred_adapter,
            "candidate_order": list(self.candidate_order),
            "backend_capabilities": list(self.backend_capabilities),
            "required_ops": list(self.required_ops),
            "hw_name": self.hw_name,
            "decision_key": self.decision_key,
        }

    def write_metadata(self, metadata: MutableMapping[str, Any], key: str) -> None:
        metadata[key] = self.to_metadata()

    def conversion_context(
        self,
        hw: HWCapability,
        metadata_key: str = "adapter_decision",
        extras: Optional[Mapping[str, Any]] = None,
    ) -> AdapterConversionContext:
        return AdapterConversionContext(
            hw=hw,
            decision=self,
            fallback_authorized=self.fallback_authorized,
            fallback_candidates=self.fallback_candidates,
            metadata_key=metadata_key,
            extras=extras or {},
        )


@dataclass(frozen=True)
class AdapterRoute:
    """Selected adapter instance paired with the router decision."""

    adapter: IAnchorAdapter
    decision: AdapterDecision
    metadata_key: str = "adapter_decision"

    def conversion_context(
        self,
        hw: HWCapability,
        extras: Optional[Mapping[str, Any]] = None,
    ) -> AdapterConversionContext:
        return self.decision.conversion_context(
            hw,
            metadata_key=self.metadata_key,
            extras=extras,
        )


class AdapterRoutingError(Exception):
    """Raised when no adapter can satisfy an ``AdapterRequest``."""

    def __init__(self, message: str, decision: Optional[AdapterDecision] = None):
        self.decision = decision
        super().__init__(message)


class AdapterRouter:
    """Deterministic adapter selection for TTIR -> AnchorIR lowering."""

    def __init__(
        self,
        registry: Any = None,
        adapters: Optional[Mapping[str, IAnchorAdapter]] = None,
    ):
        self._registry = registry
        self._adapters = dict(adapters) if adapters is not None else None

    def make_request(
        self,
        hw: HWCapability,
        *,
        anchor_ir_track: Optional[AnchorIRTrack | str] = None,
        ptr_model: Optional[str] = None,
        preferred_adapter: Optional[str] = None,
        backend_capabilities: Iterable[str] = (),
        config: Optional[AdapterRouterConfig | Mapping[str, Any]] = None,
        user_config: Optional[Mapping[str, Any]] = None,
        op_coverage: Any = None,
    ) -> AdapterRequest:
        if config is None:
            router_config = AdapterRouterConfig.from_mapping(user_config)
        elif isinstance(config, AdapterRouterConfig):
            router_config = config
        else:
            router_config = AdapterRouterConfig.from_mapping(config)

        return AdapterRequest(
            hw=hw,
            anchor_ir_track=_coerce_track(anchor_ir_track or hw.anchor_ir_track),
            ptr_model=str(ptr_model or hw.ptr_model),
            preferred_adapter=preferred_adapter
            if preferred_adapter is not None
            else hw.preferred_adapter,
            backend_capabilities=_unique_sorted(backend_capabilities),
            config=router_config,
            op_coverage=OpCoverage.from_value(op_coverage),
            user_config=dict(user_config or {}),
        )

    def select(
        self,
        hw: HWCapability,
        *,
        anchor_ir_track: Optional[AnchorIRTrack | str] = None,
        ptr_model: Optional[str] = None,
        preferred_adapter: Optional[str] = None,
        backend_capabilities: Iterable[str] = (),
        config: Optional[AdapterRouterConfig | Mapping[str, Any]] = None,
        user_config: Optional[Mapping[str, Any]] = None,
        op_coverage: Any = None,
        metadata: Optional[MutableMapping[str, Any]] = None,
    ) -> AdapterDecision:
        request = self.make_request(
            hw,
            anchor_ir_track=anchor_ir_track,
            ptr_model=ptr_model,
            preferred_adapter=preferred_adapter,
            backend_capabilities=backend_capabilities,
            config=config,
            user_config=user_config,
            op_coverage=op_coverage,
        )
        decision = self._select_from_request(request)
        if metadata is not None:
            decision.write_metadata(metadata, request.config.metadata_key)
        return decision

    def route(
        self,
        hw: HWCapability,
        *,
        anchor_ir_track: Optional[AnchorIRTrack | str] = None,
        ptr_model: Optional[str] = None,
        preferred_adapter: Optional[str] = None,
        backend_capabilities: Iterable[str] = (),
        config: Optional[AdapterRouterConfig | Mapping[str, Any]] = None,
        user_config: Optional[Mapping[str, Any]] = None,
        op_coverage: Any = None,
        metadata: Optional[MutableMapping[str, Any]] = None,
    ) -> AdapterRoute:
        request = self.make_request(
            hw,
            anchor_ir_track=anchor_ir_track,
            ptr_model=ptr_model,
            preferred_adapter=preferred_adapter,
            backend_capabilities=backend_capabilities,
            config=config,
            user_config=user_config,
            op_coverage=op_coverage,
        )
        decision = self._select_from_request(request)
        adapter = self._adapter_map()[decision.selected_adapter]
        if metadata is not None:
            decision.write_metadata(metadata, request.config.metadata_key)
        return AdapterRoute(
            adapter=adapter,
            decision=decision,
            metadata_key=request.config.metadata_key,
        )

    def _adapter_map(self) -> dict[str, IAnchorAdapter]:
        if self._adapters is not None:
            return dict(sorted(self._adapters.items()))

        if self._registry is None:
            from .registry import AdapterRegistry

            self._registry = AdapterRegistry

        self._registry.discover()
        return dict(sorted(self._registry.items().items()))

    def _select_from_request(self, request: AdapterRequest) -> AdapterDecision:
        adapters = self._adapter_map()
        decision_key = self._decision_key(request, adapters)
        primary_names = self._primary_candidates(request)
        fallback_names = self._fallback_candidates(request, primary_names)
        candidate_order = _unique_in_order(primary_names + fallback_names)

        if not candidate_order:
            decision = self._failure_decision(
                request,
                (),
                (),
                "no routing rule for requested AnchorIR track and pointer model",
                decision_key=decision_key,
            )
            raise AdapterRoutingError(decision.selected_reason, decision)

        rejections = []
        primary_set = set(primary_names)

        for name in candidate_order:
            is_fallback_candidate = name not in primary_set
            if is_fallback_candidate and not request.config.allow_fallback:
                rejections.append(
                    AdapterRejection(
                        name,
                        "fallback disabled by AdapterRouterConfig",
                    )
                )
                continue

            adapter = adapters.get(name)
            allow_ptr_override = (
                request.preferred_adapter == name or is_fallback_candidate
            )
            rejection = self._reject_reason(
                name,
                adapter,
                request,
                allow_ptr_override=allow_ptr_override,
            )
            if rejection is not None:
                rejections.append(rejection)
                continue

            fallback = is_fallback_candidate
            fallback_from = primary_names[0] if fallback and primary_names else None
            fallback_reason = self._fallback_reason(
                fallback_from,
                tuple(rejections),
            )
            fallback_chain = (
                tuple(name for name in (fallback_from, name) if name)
                if fallback
                else ()
            )
            decision = AdapterDecision(
                selected_adapter=name,
                selected_reason=self._selected_reason(
                    request,
                    name,
                    fallback=fallback,
                    fallback_from=fallback_from,
                ),
                rejected_adapters=tuple(rejections),
                fallback=fallback,
                fallback_from=fallback_from,
                fallback_authorized=self._fallback_authorized(
                    request,
                    name,
                    fallback_names,
                ),
                fallback_candidates=fallback_names,
                anchor_ir_track=request.anchor_ir_track,
                ptr_model=request.ptr_model,
                preferred_adapter=request.preferred_adapter,
                candidate_order=candidate_order,
                backend_capabilities=request.backend_capabilities,
                required_ops=request.op_coverage.required_ops,
                hw_name=request.hw.name,
                fallback_chain=fallback_chain,
                fallback_reason=fallback_reason,
                decision_key=decision_key,
            )
            return decision

        decision = self._failure_decision(
            request,
            candidate_order,
            tuple(rejections),
            "no adapter satisfied requested AnchorIR track, capabilities, and coverage",
            decision_key=decision_key,
        )
        raise AdapterRoutingError(decision.selected_reason, decision)

    def _primary_candidates(self, request: AdapterRequest) -> Tuple[str, ...]:
        if request.preferred_adapter:
            return (request.preferred_adapter,)
        return DEFAULT_ROUTING_TABLE.get(
            (request.anchor_ir_track, request.ptr_model),
            (),
        )

    def _fallback_candidates(
        self,
        request: AdapterRequest,
        primary_names: Tuple[str, ...],
    ) -> Tuple[str, ...]:
        if request.preferred_adapter:
            return ()

        allowed_by_route = DEFAULT_FALLBACK_TABLE.get(
            (request.anchor_ir_track, request.ptr_model),
            (),
        )
        ordered = [
            name
            for name in request.config.fallback_order
            if name in allowed_by_route and name not in primary_names
        ]
        if request.config.allow_fallback:
            return _unique_in_order(ordered)
        return _unique_in_order(
            name for name in allowed_by_route if name not in primary_names
        )

    def _reject_reason(
        self,
        name: str,
        adapter: Optional[IAnchorAdapter],
        request: AdapterRequest,
        *,
        allow_ptr_override: bool,
    ) -> Optional[AdapterRejection]:
        if adapter is None:
            return AdapterRejection(name, "adapter is not registered")

        info = adapter.describe()

        if request.anchor_ir_track not in info.supported_tracks:
            tracks = tuple(track.value for track in info.supported_tracks)
            return AdapterRejection(
                name,
                "AnchorIR track mismatch",
                (f"supports={tracks}", f"requested={request.anchor_ir_track.value}"),
            )

        if (
            request.ptr_model not in info.supported_ptr_models
            and not allow_ptr_override
        ):
            return AdapterRejection(
                name,
                "pointer model mismatch",
                (
                    f"supports={info.supported_ptr_models}",
                    f"requested={request.ptr_model}",
                ),
            )

        if request.config.strict_backend_capabilities:
            missing_caps = tuple(
                cap
                for cap in info.required_backend_capabilities
                if cap not in request.backend_capabilities
            )
            if missing_caps:
                return AdapterRejection(
                    name,
                    "missing backend capabilities",
                    missing_caps,
                )

        missing_ops = self._missing_ops(info, request.op_coverage)
        if missing_ops is not None:
            reason = (
                "op coverage unavailable"
                if missing_ops == ("<unknown>",)
                else "missing op coverage"
            )
            return AdapterRejection(name, reason, missing_ops)

        return None

    def _missing_ops(
        self,
        info: AdapterInfo,
        op_coverage: OpCoverage,
    ) -> Optional[Tuple[str, ...]]:
        required = set(op_coverage.required_ops)
        if not required:
            return None

        supported = op_coverage.adapter_supported_ops.get(info.name, info.supported_ops)
        if supported is None:
            return ("<unknown>",)
        if "*" in supported:
            return None

        missing = tuple(sorted(required.difference(supported)))
        return missing or None

    def _fallback_authorized(
        self,
        request: AdapterRequest,
        selected_name: str,
        fallback_names: Tuple[str, ...],
    ) -> bool:
        if not request.config.allow_fallback:
            return False
        if selected_name in fallback_names:
            return True
        adapter = self._adapter_map().get(selected_name)
        return bool(adapter and adapter.describe().supports_internal_fallback)

    def _selected_reason(
        self,
        request: AdapterRequest,
        name: str,
        *,
        fallback: bool,
        fallback_from: Optional[str],
    ) -> str:
        if request.preferred_adapter == name:
            return (
                f"preferred adapter '{name}' selected for "
                f"AnchorIR track '{request.anchor_ir_track.value}'"
            )
        if fallback:
            return (
                f"fallback adapter '{name}' selected after '{fallback_from}' "
                "was rejected"
            )
        return (
            f"adapter '{name}' selected for AnchorIR track "
            f"'{request.anchor_ir_track.value}' and ptr_model "
            f"'{request.ptr_model}'"
        )

    def _fallback_reason(
        self,
        fallback_from: Optional[str],
        rejections: Tuple[AdapterRejection, ...],
    ) -> Optional[str]:
        if fallback_from is None:
            return None

        for rejection in rejections:
            if rejection.adapter_name != fallback_from:
                continue
            detail = ""
            if rejection.details:
                detail = f" ({', '.join(rejection.details)})"
            return (
                f"primary adapter '{fallback_from}' rejected: "
                f"{rejection.reason}{detail}"
            )

        return f"primary adapter '{fallback_from}' rejected before fallback"

    def _decision_key(
        self,
        request: AdapterRequest,
        adapters: Mapping[str, IAnchorAdapter],
    ) -> str:
        payload = {
            "policy_version": ADAPTER_DECISION_POLICY_VERSION,
            "hw": {
                "name": request.hw.name,
                "arch_family": request.hw.arch_family,
                "compute_paradigm": request.hw.compute_paradigm.value,
            },
            "anchor_ir_track": request.anchor_ir_track.value,
            "ptr_model": request.ptr_model,
            "preferred_adapter": request.preferred_adapter,
            "backend_capabilities": list(request.backend_capabilities),
            "config": {
                "allow_fallback": request.config.allow_fallback,
                "fallback_order": list(request.config.fallback_order),
                "strict_backend_capabilities": (
                    request.config.strict_backend_capabilities
                ),
                "metadata_key": request.config.metadata_key,
            },
            "required_ops": list(request.op_coverage.required_ops),
            "adapter_supported_ops": {
                name: list(ops)
                for name, ops in sorted(
                    request.op_coverage.adapter_supported_ops.items()
                )
            },
            "registered_adapters": self._adapter_inventory(adapters),
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()

    def _adapter_inventory(
        self,
        adapters: Mapping[str, IAnchorAdapter],
    ) -> dict[str, Any]:
        inventory = {}
        for name in sorted(adapters):
            info = adapters[name].describe()
            inventory[name] = {
                "supported_tracks": sorted(
                    track.value for track in info.supported_tracks
                ),
                "supported_ptr_models": sorted(info.supported_ptr_models),
                "required_backend_capabilities": sorted(
                    info.required_backend_capabilities
                ),
                "supported_ops": (
                    None
                    if info.supported_ops is None
                    else sorted(info.supported_ops)
                ),
                "supports_internal_fallback": info.supports_internal_fallback,
            }
        return inventory

    def _failure_decision(
        self,
        request: AdapterRequest,
        candidate_order: Tuple[str, ...],
        rejections: Tuple[AdapterRejection, ...],
        reason: str,
        *,
        decision_key: str = "",
    ) -> AdapterDecision:
        return AdapterDecision(
            selected_adapter=None,
            selected_reason=reason,
            rejected_adapters=rejections,
            fallback=False,
            fallback_from=None,
            fallback_authorized=False,
            fallback_candidates=self._fallback_candidates(
                request,
                self._primary_candidates(request),
            ),
            anchor_ir_track=request.anchor_ir_track,
            ptr_model=request.ptr_model,
            preferred_adapter=request.preferred_adapter,
            candidate_order=candidate_order,
            backend_capabilities=request.backend_capabilities,
            required_ops=request.op_coverage.required_ops,
            hw_name=request.hw.name,
            decision_key=decision_key,
        )
