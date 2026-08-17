"""Installed-wheel rejection proof for native metadata/artifact failures."""

import argparse
import base64
import csv
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import sys

import w10_native_wheel_probe as w10_support


DISTRIBUTION = "triton-anchor-ac2-native-alpha-backend"
ENTRY_POINT = "ac2_native_alpha"
MODULE = "ac2_native_alpha"
CASES = {
    "malformed_json": ("BackendPluginManifestError", "manifest_file"),
    "coverage_error": ("BackendPluginManifestError", "plugins[].entry_point"),
    "missing_abi": ("BackendPluginManifestError", "abi_fingerprint"),
    "path_escape": ("BackendPluginManifestError", "native_libraries"),
    "missing_native": ("BackendPluginManifestError", "native_libraries"),
    "hidden_native": ("BackendPluginCompatibilityError", "python_only wheel contents"),
    "record_tamper": ("BackendPluginManifestError", "RECORD"),
    "bad_elf": ("BackendPluginCompatibilityError", "native binary format"),
    "wrong_arch": ("BackendPluginCompatibilityError", "native architecture"),
    "wrong_wheel_tag": ("BackendPluginCompatibilityError", "wheel platform tag"),
    "subprocess_contract": ("BackendPluginCompatibilityError", "subprocess IR contract"),
}


def _file(distribution, name):
    matches = tuple(
        item
        for item in distribution.files or ()
        if item.name == name or (name == ".so" and item.name.endswith(".so"))
    )
    assert len(matches) == 1
    return Path(distribution.locate_file(matches[0])).resolve(), str(matches[0])


def _rehash(distribution, changed_path, record_name):
    record_path, _ = _file(distribution, "RECORD")
    rows = list(csv.reader(record_path.read_text(encoding="utf-8").splitlines()))
    digest = base64.urlsafe_b64encode(
        hashlib.sha256(changed_path.read_bytes()).digest()
    ).decode("ascii").rstrip("=")
    for row in rows:
        if row[0] == record_name:
            row[1:] = ["sha256=" + digest, str(changed_path.stat().st_size)]
            break
    else:
        raise AssertionError(record_name)
    with record_path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(rows)


def _mutate(case, distribution):
    manifest_path, manifest_name = _file(distribution, "triton_anchor_backend.json")
    native_path, native_name = _file(distribution, ".so")
    wheel_path, wheel_name = _file(distribution, "WHEEL")
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    plugin = document["plugins"][0]
    if case == "malformed_json":
        manifest_path.write_text("{not-json\n", encoding="utf-8")
        _rehash(distribution, manifest_path, manifest_name)
        return
    if case == "coverage_error":
        plugin["entry_point"] = "undeclared_metadata_entry"
    elif case == "missing_abi":
        plugin.pop("abi_fingerprint")
    elif case == "path_escape":
        plugin["native_libraries"] = ["../escaped.so"]
    elif case == "missing_native":
        native_path.unlink()
        return
    elif case == "hidden_native":
        plugin["isolation_mode"] = "python_only"
        plugin.pop("native_libraries")
        plugin.pop("abi_fingerprint")
    elif case == "record_tamper":
        native_path.write_bytes(native_path.read_bytes() + b"tampered")
        return
    elif case == "bad_elf":
        native_path.write_bytes(b"not an ELF shared object\n")
        _rehash(distribution, native_path, native_name)
        return
    elif case == "wrong_arch":
        content = bytearray(native_path.read_bytes())
        assert content[:4] == b"\x7fELF"
        content[18:20] = (183).to_bytes(2, "little")
        native_path.write_bytes(content)
        _rehash(distribution, native_path, native_name)
        return
    elif case == "wrong_wheel_tag":
        lines = wheel_path.read_text(encoding="utf-8").splitlines()
        lines = ["Tag: cp312-cp312-win_amd64" if line.startswith("Tag:") else line for line in lines]
        wheel_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        _rehash(distribution, wheel_path, wheel_name)
        return
    elif case == "subprocess_contract":
        plugin["isolation_mode"] = "subprocess"
        plugin.pop("abi_fingerprint")
    else:
        raise AssertionError(case)
    manifest_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    _rehash(distribution, manifest_path, manifest_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=tuple(CASES), required=True)
    parser.add_argument("--mode", choices=("load", "register", "select"), required=True)
    args = parser.parse_args()
    assert not os.environ.get("PYTHONPATH")
    distribution = metadata.distribution(DISTRIBUTION)
    entry_point = tuple(ep for ep in distribution.entry_points if ep.group == "triton.backends")[0]
    assert entry_point.name == ENTRY_POINT and entry_point.value == MODULE
    Path(distribution.locate_file("")).resolve().relative_to(Path(sys.prefix).resolve())
    _mutate(args.case, distribution)
    original, counts = w10_support._install_load_counter((entry_point,))
    try:
        from triton_anchor.backends import BackendPluginRegistry
        registry = BackendPluginRegistry(distribution_provider=lambda: (distribution,))
        error = None
        try:
            record = registry.discover()[0]
            if args.mode == "load":
                registry.load(record.record_id)
            elif args.mode == "register":
                registry.register(record.record_id)
            else:
                registry.select(
                    "ac2_native_alpha",
                    explicit_selector=record.registry_key,
                    environment={},
                )
        except Exception as caught:
            error = caught
        record = registry.list()[0]
        expected_type, expected_field = CASES[args.case]
        diagnostic = error.to_dict()
        assert type(error).__name__ == expected_type
        assert diagnostic["field"] == expected_field
        assert diagnostic["expected"] and diagnostic["actual"] and diagnostic["remediation"]
        assert record.state.value == "rejected"
        assert counts[ENTRY_POINT] == 0
        assert MODULE not in sys.modules
        result = {"case": args.case, "mode": args.mode, "state": record.state.value,
                  "error_type": type(error).__name__, "error": diagnostic,
                  "entry_point_load_calls": counts, "module_import_calls": 0}
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        metadata.EntryPoint.load = original


if __name__ == "__main__":
    main()
