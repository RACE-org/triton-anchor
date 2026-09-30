#!/usr/bin/env python3
"""Select impact-based operators, with full coverage available to manual tasks."""

from __future__ import annotations

import argparse
import random
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path


IGNORED_MARKERS = {
    "filterwarnings",
    "parametrize",
    "skip",
    "skipif",
    "usefixtures",
    "xfail",
}
MARK_RE = re.compile(r"@pytest\.mark\.([A-Za-z_][A-Za-z0-9_]*)")
MAX_IMPACT_OPS = 6
# One known-supported operator from each category in the pass whitelist.
DEFAULT_IMPACT_OPS = ("abs", "maximum", "mm", "arange", "exponential_", "embedding")


@dataclass(frozen=True)
class Entry:
    category: str
    op: str
    marker: str
    test_file: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("impact", "sample", "full", "single"), default="impact"
    )
    parser.add_argument(
        "--ops", default="", help="Comma-separated affected operator identifiers"
    )
    parser.add_argument(
        "--categories", default="", help="Comma-separated affected categories"
    )
    parser.add_argument("--sample-size", type=int, default=6)
    parser.add_argument("--seed", default="")
    parser.add_argument("--op", default="")
    parser.add_argument("--whitelist", required=True)
    parser.add_argument("--full-list", default="")
    parser.add_argument("--flaggems-dir", required=True)
    parser.add_argument("--python-bin", default="python3")
    parser.add_argument("--selected-output", default="")
    return parser.parse_args()


def read_entries(path: Path) -> list[Entry]:
    entries: list[Entry] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) not in (3, 4):
            raise ValueError(
                f"{path}:{line_number}: expected 3 or 4 columns, got {len(parts)}"
            )
        test_file = parts[3] if len(parts) == 4 else ""
        entries.append(Entry(parts[0], parts[1], parts[2], test_file))
    if not entries:
        raise ValueError(f"{path}: no FlagGems operators found")
    return entries


def discover_marker_files(flaggems_dir: Path) -> dict[str, list[str]]:
    tests_dir = flaggems_dir / "tests"
    if not tests_dir.is_dir():
        raise ValueError(f"FlagGems tests directory does not exist: {tests_dir}")

    marker_files: dict[str, list[str]] = {}
    for test_file in sorted(tests_dir.glob("test_*.py")):
        rel_path = test_file.relative_to(flaggems_dir).as_posix()
        text = test_file.read_text(encoding="utf-8", errors="replace")
        for match in MARK_RE.finditer(text):
            marker = match.group(1)
            if marker in IGNORED_MARKERS:
                continue
            marker_files.setdefault(marker, [])
            if rel_path not in marker_files[marker]:
                marker_files[marker].append(rel_path)
    return marker_files


def marker_aliases(entry: Entry) -> list[str]:
    aliases = [entry.marker, entry.op]
    aliases.append(entry.marker.removesuffix("_tensor"))
    aliases.append(entry.marker.removesuffix("_dim"))
    aliases.append(entry.marker.removeprefix("native_"))
    aliases.append(entry.op.removesuffix("_tensor"))
    aliases.append(entry.op.removesuffix("_dim"))
    aliases.append(entry.op.removeprefix("native_"))
    return unique_in_order([alias for alias in aliases if alias])


def attach_discovered_files(
    entries: list[Entry], marker_files: dict[str, list[str]]
) -> list[Entry]:
    resolved: list[Entry] = []
    for entry in entries:
        chosen_marker = ""
        files: list[str] = []
        for alias in marker_aliases(entry):
            if alias in marker_files:
                chosen_marker = alias
                files = marker_files[alias]
                break
        if not files:
            print(
                f"warning: no pytest marker found for {entry.op}({entry.marker}); "
                "the batch runner will record this operator separately",
                file=sys.stderr,
            )
            resolved.append(entry)
            continue
        for test_file in files:
            resolved.append(Entry(entry.category, entry.op, chosen_marker, test_file))
    return resolved


def group_entries_by_category(entries: list[Entry]) -> dict[str, list[Entry]]:
    grouped: dict[str, list[Entry]] = {}
    for entry in entries:
        grouped.setdefault(entry.category, []).append(entry)
    return grouped


def select_sample_entries(
    entries: list[Entry], requested_size: int, seed: str
) -> list[Entry]:
    if not 1 <= requested_size <= MAX_IMPACT_OPS:
        raise ValueError(
            "FlagGems sample size must be 1..6; use --mode full for broader coverage"
        )
    rng = random.Random(seed) if seed else random.SystemRandom()
    grouped = group_entries_by_category(entries)
    categories = sorted(grouped)
    rng.shuffle(categories)
    selected_ops: set[str] = set()
    for category in categories:
        available = sorted({entry.op for entry in grouped[category]} - selected_ops)
        if available:
            selected_ops.add(rng.choice(available))
        if len(selected_ops) == requested_size:
            break
    pool = sorted({entry.op for entry in entries} - selected_ops)
    selected_ops.update(
        rng.sample(pool, min(requested_size - len(selected_ops), len(pool)))
    )
    return [entry for entry in entries if entry.op in selected_ops]


def select_entries(args: argparse.Namespace) -> list[Entry]:
    marker_files = discover_marker_files(Path(args.flaggems_dir))
    if args.mode == "full":
        if not args.full_list:
            raise ValueError("--full-list is required in full mode")
        selected = read_entries(Path(args.full_list))
        return attach_discovered_files(selected, marker_files)

    entries = read_entries(Path(args.whitelist))
    if args.mode == "impact":
        ops = {
            value.strip() for value in getattr(args, "ops", "").split(",") if value.strip()
        }
        categories = {
            value.strip()
            for value in getattr(args, "categories", "").split(",")
            if value.strip()
        }
        # Explicitly affected operators may be outside the historical pass list.
        # They must still be tested, rather than being silently discarded.
        catalogue = read_entries(Path(args.full_list)) if args.full_list else entries
        known_ops = {value for entry in catalogue for value in (entry.op, entry.marker)}
        known_categories = {entry.category for entry in catalogue}
        if ops - known_ops or categories - known_categories:
            raise ValueError(
                f"Unknown FlagGems impact selection: ops={sorted(ops - known_ops)}, categories={sorted(categories - known_categories)}"
            )
        selected = [
            entry
            for entry in catalogue
            if entry.op in ops or entry.marker in ops or entry.category in categories
        ]
        if not ops and not categories:
            missing = set(DEFAULT_IMPACT_OPS) - {entry.op for entry in entries}
            if missing:
                raise ValueError(
                    f"Default FlagGems operators missing from pass whitelist: {sorted(missing)}"
                )
            selected = [entry for entry in entries if entry.op in DEFAULT_IMPACT_OPS]
        if not selected:
            raise ValueError("Impact selection contains no operators")
    elif args.mode == "single":
        selected = [entry for entry in entries if args.op in (entry.op, entry.marker)]
        if not selected:
            raise ValueError(
                f"FlagGems op {args.op!r} was not found in the pass whitelist"
            )
    else:
        selected = select_sample_entries(entries, args.sample_size, args.seed)

    count = len({entry.op for entry in selected})
    if count > MAX_IMPACT_OPS:
        raise ValueError(
            f"FlagGems non-full selection contains {count} operators (maximum 6); "
            "use --mode full for broader coverage"
        )
    return attach_discovered_files(selected, marker_files)


def unique_in_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def write_selected(
    path_text: str, selected: list[Entry], args: argparse.Namespace, command: str
) -> None:
    if not path_text:
        return
    path = Path(path_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    unique_ops = unique_in_order([entry.op for entry in selected])
    unique_markers = unique_in_order([entry.marker for entry in selected])
    unique_categories = unique_in_order([entry.category for entry in selected])
    lines = [
        f"mode: {args.mode}",
        f"sample_size: {args.sample_size}",
        f"seed: {args.seed}",
        f"selected_op_count: {len(unique_ops)}",
        f"selected_marker_count: {len(unique_markers)}",
        f"selected_category_count: {len(unique_categories)}",
        "selected_ops:",
    ]
    lines.extend(f"- {op}" for op in unique_ops)
    lines.append("selected_categories:")
    lines.extend(f"- {category}" for category in unique_categories)
    lines.append("selected_markers:")
    lines.extend(f"- {marker}" for marker in unique_markers)
    lines.append("selected_entries:")
    lines.extend(
        f"- op: {entry.op}, marker: {entry.marker}, category: {entry.category}, file: {entry.test_file}"
        for entry in selected
    )
    lines.extend(["command:", command])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    selected = select_entries(args)
    test_files = unique_in_order(
        [entry.test_file for entry in selected if entry.test_file]
    )
    markers = unique_in_order([entry.marker for entry in selected])
    marker_expr = " or ".join(markers)

    pytest_parts = [
        args.python_bin,
        "-m",
        "pytest",
        "-s",
        *test_files,
        "-m",
        marker_expr,
        "--ref",
        "cpu",
        "-vs",
    ]
    command = f"cd {shlex.quote(args.flaggems_dir)} && " + " ".join(
        shlex.quote(part) for part in pytest_parts
    )
    write_selected(args.selected_output, selected, args, command)
    print(command)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"failed to select FlagGems tests: {exc}", file=sys.stderr)
        raise SystemExit(1)
