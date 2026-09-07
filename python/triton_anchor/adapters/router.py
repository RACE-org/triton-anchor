"""Deterministic Adapter Router for Triton 3.6.

The registry owns adapter discovery.  This module owns the final routing
decision so T6.1 selection, compatibility checks, fallback policy, and cache
metadata are produced by one deterministic policy.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple, TYPE_CHECKING

from .base import ITritonToLinalgAdapter

if TYPE_CHECKING:
    from ..hw_capability import HWCapability


ADAPTER_ROUTE_SCHEMA = "triton-anchor-adapter-route-v1"
ADAPTER_DECISION_POLICY_VERSION = "triton-anchor-adapter-router-v1"

_TRACK_GPU = "triton_gpu"
_TRACK_LINALG = "linalg"

_ROUTE_TO_ADAPTER = {
    (_TRACK_GPU, "*"): "triton-gpu",
    (_TRACK_LINALG, "structured"): "triton-shared",
    (_TRACK_LINALG, "axis_info"): "triton-linalg",
    (_TRACK_LINALG, "hybrid"): "hybrid",
}

_FALLBACK_CHAINS = {
    "triton-shared": ("triton-shared",),
    "triton-linalg": ("triton-linalg",),
    "hybrid": ("triton-shared", "triton-linalg"),
    "triton-gpu": ("triton-gpu",),
}

_BUILTIN_SUPPORTED_ROUTES = {
    "triton-shared": ((_TRACK_LINALG, "structured"),),
    "triton-linalg": ((_TRACK_LINALG, "axis_info"),),
    "hybrid": ((_TRACK_LINALG, "hybrid"),),
    "triton-gpu": ((_TRACK_GPU, "*"),),
}


class AdapterNotFoundError(Exception):
    """Raised when no compatible adapter route can be selected."""

    pass


@dataclass(frozen=True)
class AdapterRoute:
    """Deterministic Router decision for one ``HWCapability``."""

    ptr_model: str
    anchor_ir_track: str
    selected_adapter: str
    adapter_components: Dict[str, Any]
    fallback_chain: Tuple[str, ...]
    fallback_policy: str = "managed"
    preferred_adapter: Optional[str] = None
    explicit_preference: bool = False
    strict_selection: bool = True
    decision_policy_version: str = ADAPTER_DECISION_POLICY_VERSION
    rejected_candidates: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    reject_reasons: Dict[str, str] = field(default_factory=dict)
    fallback_reason: Optional[str] = None
    input_decision_key: str = ""
    schema: str = ADAPTER_ROUTE_SCHEMA

    def to_cache_payload(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "decision_policy_version": self.decision_policy_version,
            "ptr_model": self.ptr_model,
            "anchor_ir_track": self.anchor_ir_track,
            "selected_adapter": self.selected_adapter,
            "adapter_components": self.adapter_components,
            "fallback_chain": list(self.fallback_chain),
            "fallback_policy": self.fallback_policy,
            "fallback_reason": self.fallback_reason,
            "preferred_adapter": self.preferred_adapter,
            "explicit_preference": self.explicit_preference,
            "strict_selection": self.strict_selection,
            "rejected_candidates": list(self.rejected_candidates),
            "reject_reasons": self.reject_reasons,
            "input_decision_key": self.input_decision_key,
        }

    def cache_key(self) -> str:
        payload = json.dumps(
            self.to_cache_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"{self.schema}:sha256:{digest}"

    def to_metadata(self) -> Dict[str, Any]:
        payload = self.to_cache_payload()
        payload["cache_key"] = self.cache_key()
        return {
            "selected_adapter": self.selected_adapter,
            "anchor_adapter": self.selected_adapter,
            "anchor_adapter_route": payload,
            "anchor_adapter_cache_key": payload["cache_key"],
            "anchor_adapter_decision_policy_version": self.decision_policy_version,
            "anchor_adapter_fallback_chain": list(self.fallback_chain),
            "anchor_adapter_fallback_policy": self.fallback_policy,
            "anchor_adapter_fallback_reason": self.fallback_reason,
            "anchor_adapter_rejected_candidates": list(self.rejected_candidates),
            "anchor_adapter_reject_reasons": dict(self.reject_reasons),
            "anchor_adapter_input_decision_key": self.input_decision_key,
            "anchor_adapter_strict_selection": self.strict_selection,
        }


class AdapterRouter:
    """Select a compatible adapter from a discovered adapter registry."""

    def __init__(self, adapters: Mapping[str, ITritonToLinalgAdapter]):
        self._adapters = adapters

    def resolve(
        self, hw: HWCapability, metadata: Optional[dict] = None
    ) -> Tuple[ITritonToLinalgAdapter, AdapterRoute]:
        track = _track_value(hw)
        ptr_model = str(getattr(hw, "ptr_model", ""))
        preferred_adapter = getattr(hw, "preferred_adapter", None)
        explicit_preference = bool(preferred_adapter)

        if explicit_preference:
            selected_name = self._resolve_preferred(hw, preferred_adapter)
            fallback_chain = (selected_name,)
            fallback_reason = None
        else:
            selected_name = self._resolve_policy_route(track, ptr_model)
            fallback_chain = _FALLBACK_CHAINS[selected_name]
            fallback_reason = (
                "hybrid structured-first fallback to axis_info is allowed"
                if selected_name == "hybrid"
                else None
            )

        adapter = self._adapters.get(selected_name)
        if adapter is None:
            raise AdapterNotFoundError(
                f"Adapter '{selected_name}' required for "
                f"anchor_ir_track='{track}', ptr_model='{ptr_model}' was not found. "
                f"Available: {sorted(self._adapters)}"
            )

        input_decision_key = self._input_decision_key(hw)
        rejected_candidates, reject_reasons = self._rejections(
            hw, selected_adapter=selected_name
        )
        route = AdapterRoute(
            ptr_model=ptr_model,
            anchor_ir_track=track,
            selected_adapter=selected_name,
            adapter_components=adapter.cache_key_components(),
            fallback_chain=fallback_chain,
            fallback_policy=getattr(hw, "adapter_fallback_policy", "managed"),
            preferred_adapter=preferred_adapter,
            explicit_preference=explicit_preference,
            rejected_candidates=tuple(rejected_candidates),
            reject_reasons=reject_reasons,
            fallback_reason=fallback_reason,
            input_decision_key=input_decision_key,
        )
        if metadata is not None:
            metadata.update(route.to_metadata())
        return adapter, route

    def _resolve_policy_route(self, track: str, ptr_model: str) -> str:
        if track == _TRACK_GPU:
            return _ROUTE_TO_ADAPTER[(_TRACK_GPU, "*")]
        try:
            return _ROUTE_TO_ADAPTER[(track, ptr_model)]
        except KeyError as exc:
            raise AdapterNotFoundError(
                f"Unsupported adapter route: anchor_ir_track='{track}', "
                f"ptr_model='{ptr_model}'. Supported routes: "
                f"{sorted(_ROUTE_TO_ADAPTER)}"
            ) from exc

    def _resolve_preferred(self, hw: HWCapability, preferred_adapter: str) -> str:
        adapter = self._adapters.get(preferred_adapter)
        if adapter is None:
            raise AdapterNotFoundError(
                f"Preferred adapter '{preferred_adapter}' not found. "
                f"Available: {sorted(self._adapters)}"
            )
        if not self._adapter_supports(adapter, hw):
            raise AdapterNotFoundError(
                f"Preferred adapter '{preferred_adapter}' is incompatible with "
                f"anchor_ir_track='{_track_value(hw)}', "
                f"ptr_model='{getattr(hw, 'ptr_model', '')}'."
            )
        return preferred_adapter

    def _adapter_supports(self, adapter: ITritonToLinalgAdapter, hw: HWCapability) -> bool:
        name = adapter.name()
        builtin_routes = _BUILTIN_SUPPORTED_ROUTES.get(name)
        if builtin_routes is not None:
            return _route_matches_any(_track_value(hw), getattr(hw, "ptr_model", ""), builtin_routes)
        supports = getattr(adapter, "supports", None)
        if supports is None:
            return True
        return bool(supports(hw))

    def _rejections(
        self, hw: HWCapability, *, selected_adapter: str
    ) -> Tuple[list[Dict[str, Any]], Dict[str, str]]:
        rejected = []
        reasons = {}
        for name in sorted(self._adapters):
            if name == selected_adapter:
                continue
            adapter = self._adapters[name]
            if self._adapter_supports(adapter, hw):
                reason = (
                    "compatible but not selected by "
                    f"{ADAPTER_DECISION_POLICY_VERSION}"
                )
            else:
                reason = (
                    f"incompatible with anchor_ir_track='{_track_value(hw)}', "
                    f"ptr_model='{getattr(hw, 'ptr_model', '')}'"
                )
            item = {"adapter": name, "reason": reason}
            rejected.append(item)
            reasons[name] = reason
        return rejected, reasons

    def _input_decision_key(self, hw: HWCapability) -> str:
        material = {
            "decision_policy_version": ADAPTER_DECISION_POLICY_VERSION,
            "anchor_ir_track": _track_value(hw),
            "ptr_model": getattr(hw, "ptr_model", ""),
            "preferred_adapter": getattr(hw, "preferred_adapter", None),
            "fallback_policy": getattr(hw, "adapter_fallback_policy", "managed"),
            "registered_adapters": {
                name: adapter.cache_key_components()
                for name, adapter in sorted(self._adapters.items())
            },
        }
        payload = json.dumps(
            material,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"{ADAPTER_DECISION_POLICY_VERSION}:sha256:{digest}"


def _track_value(hw: HWCapability) -> str:
    track = getattr(hw, "anchor_ir_track", "")
    return str(getattr(track, "value", track))


def _route_matches_any(
    track: str, ptr_model: str, supported_routes: Tuple[Tuple[str, str], ...]
) -> bool:
    for supported_track, supported_ptr_model in supported_routes:
        if supported_track != track:
            continue
        if supported_ptr_model == "*" or supported_ptr_model == ptr_model:
            return True
    return False
