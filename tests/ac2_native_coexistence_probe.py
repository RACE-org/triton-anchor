"""Real-wheel proof that two independent native plugins coexist."""

import json
import os
from pathlib import Path
import sys
import tempfile

import w10_native_wheel_probe as w10_support


FIXTURES = {
    "alpha": {
        "distribution": "triton-anchor-ac2-native-alpha-backend",
        "entry_point": "ac2_native_alpha",
        "module": "ac2_native_alpha",
        "marker": "alpha.imported",
        "plugin_id": "ac2.native.alpha",
        "target": "ac2_native_alpha",
        "soname": "libtriton_anchor_ac2_native_alpha.so",
        "symbol": "triton_anchor_ac2_native_alpha_symbol",
    },
    "beta": {
        "distribution": "triton-anchor-ac2-native-beta-backend",
        "entry_point": "ac2_native_beta",
        "module": "ac2_native_beta",
        "marker": "beta.imported",
        "plugin_id": "ac2.native.beta",
        "target": "ac2_native_beta",
        "soname": "libtriton_anchor_ac2_native_beta.so",
        "symbol": "triton_anchor_ac2_native_beta_symbol",
    },
}


def _assert_installed(fixture):
    distribution, entry_point = w10_support._assert_installed_native_wheel(
        fixture
    )
    Path(distribution.locate_file("")).resolve().relative_to(
        Path(sys.prefix).resolve()
    )
    assert entry_point.value == fixture["module"]
    return distribution, entry_point


def _marker_count(marker_dir, fixture):
    marker = marker_dir / fixture["marker"]
    if not marker.exists():
        return 0
    return marker.read_text(encoding="utf-8").splitlines().count("imported")


def run_probe(marker_dir):
    installed = {
        name: _assert_installed(fixture)
        for name, fixture in FIXTURES.items()
    }
    original_load, load_counts = w10_support._install_load_counter(
        tuple(entry_point for _, entry_point in installed.values())
    )
    try:
        environment = w10_support._assert_core_origin_and_environment()
        assert all(
            w10_support._module_or_submodule_imported(fixture["module"])
            is False
            for fixture in FIXTURES.values()
        )
        assert all(
            _marker_count(marker_dir, fixture) == 0
            for fixture in FIXTURES.values()
        )
        assert set(load_counts.values()) == {0}

        from triton.backends.compiler import GPUTarget
        from triton_anchor.backends import (
            BackendPluginRegistry,
            PluginLifecycleState,
        )

        registry = BackendPluginRegistry()
        records = registry.discover()
        coexist_records = {
            name: w10_support._record_by_distribution(
                records, fixture["distribution"]
            )
            for name, fixture in FIXTURES.items()
        }
        assert len({record.record_id for record in coexist_records.values()}) == 2
        assert {
            record.plugin_id for record in coexist_records.values()
        } == {fixture["plugin_id"] for fixture in FIXTURES.values()}
        assert set(load_counts.values()) == {0}

        validated = {
            name: registry.validate(record.record_id)
            for name, record in coexist_records.items()
        }
        for name, record in validated.items():
            fixture = FIXTURES[name]
            assert record.state is PluginLifecycleState.VALIDATED
            assert record.compatibility_report.compatible
            assert len(record.compatibility_report.native_artifacts) == 1
            artifact = record.compatibility_report.native_artifacts[0]
            assert artifact.identity == fixture["soname"]
            assert artifact.exported_symbols == (fixture["symbol"],)
            assert load_counts[fixture["entry_point"]] == 0
            assert _marker_count(marker_dir, fixture) == 0

        conflict_report = registry.conflicts()
        assert conflict_report.ok
        assert not conflict_report.has_fatal
        assert not conflict_report.requires_selection
        assert conflict_report.conflicts == ()
        pre_load = {
            name: {
                "state": registry.inspect(record.record_id).state.value,
                "load_calls": load_counts[FIXTURES[name]["entry_point"]],
                "import_calls": _marker_count(marker_dir, FIXTURES[name]),
            }
            for name, record in validated.items()
        }

        decisions = {}
        for name, fixture in FIXTURES.items():
            target = GPUTarget(fixture["target"], "fixture", 1)
            decision = registry.select(target, environment={})
            assert decision.record_id == validated[name].record_id
            assert decision.plugin_id == fixture["plugin_id"]
            assert decision.method.value == "sole_candidate"
            assert decision.candidate_record_ids == (validated[name].record_id,)
            assert decision.record.compiler_cls.__module__ == fixture["module"]
            assert decision.record.driver_cls.__module__ == fixture["module"]
            assert decision.record.state is PluginLifecycleState.SELECTED

            repeated = registry.select(target, environment={})
            assert repeated.record_id == decision.record_id
            assert registry.load(decision.record_id).record_id == decision.record_id
            assert (
                registry.register(decision.record_id).record_id
                == decision.record_id
            )
            assert load_counts[fixture["entry_point"]] == 1
            assert _marker_count(marker_dir, fixture) == 1
            decisions[name] = decision.to_dict()

        final_records = {
            name: registry.inspect(record.record_id)
            for name, record in validated.items()
        }
        assert {
            record.state for record in final_records.values()
        } == {PluginLifecycleState.SELECTED}
        assert set(load_counts.values()) == {1}
        assert all(
            _marker_count(marker_dir, fixture) == 1
            for fixture in FIXTURES.values()
        )

        return {
            "mode": "ac2-two-independent-native-wheels",
            "python_prefix": sys.prefix,
            "core_abi_fingerprint": environment.core_abi_fingerprint,
            "distributions": {
                name: {
                    "name": distribution.metadata["Name"],
                    "version": distribution.version,
                    "root": str(
                        Path(distribution.locate_file("")).resolve()
                    ),
                    "entry_point": entry_point.name,
                    "entry_point_value": entry_point.value,
                    "plugin_id": FIXTURES[name]["plugin_id"],
                    "target": FIXTURES[name]["target"],
                    "soname": FIXTURES[name]["soname"],
                    "symbol": FIXTURES[name]["symbol"],
                }
                for name, (distribution, entry_point) in installed.items()
            },
            "pre_load": pre_load,
            "conflict_report": conflict_report.to_dict(),
            "selections": decisions,
            "final_records": {
                name: {
                    "record_id": record.record_id,
                    "plugin_id": record.plugin_id,
                    "state": record.state.value,
                    "selected_targets": list(record.selected_targets),
                    "compiler_module": record.compiler_cls.__module__,
                    "driver_module": record.driver_cls.__module__,
                }
                for name, record in final_records.items()
            },
            "entry_point_load_calls": load_counts,
            "module_import_calls": {
                name: _marker_count(marker_dir, fixture)
                for name, fixture in FIXTURES.items()
            },
        }
    finally:
        w10_support.metadata.EntryPoint.load = original_load


def main():
    assert not os.environ.get("PYTHONPATH")
    with tempfile.TemporaryDirectory(
        prefix="triton-anchor-ac2-coexist-markers."
    ) as marker_dir:
        marker_path = Path(marker_dir)
        os.environ["TRITON_ANCHOR_AC2_MARKER_DIR"] = str(marker_path)
        result = run_probe(marker_path)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
