import ast
from pathlib import Path

import mkdocs_gen_files


TARGET_PATH = "index.md"
PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "python" / "triton_anchor"
SECTION_NAMES = ("Core API", "Adapters", "Extensions")


def _section_name(relative_path: Path) -> str:
    if relative_path.parts[0] == "adapters":
        return "Adapters"
    if relative_path.parts[0] in {"extensions", "language"}:
        return "Extensions"
    return "Core API"


def _public_source_files(package_root: Path):
    for source_path in sorted(package_root.rglob("*.py")):
        relative_path = source_path.relative_to(package_root)
        if "tests" in relative_path.parts:
            continue
        if relative_path.name == "__init__.py" or relative_path.name.startswith("_"):
            continue
        yield source_path, relative_path


def _public_definitions(source_path: Path):
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                yield node


def _module_name(relative_path: Path) -> str:
    return "triton_anchor." + ".".join(relative_path.with_suffix("").parts)


def _discover_public_api(package_root: Path):
    sections = {section_name: [] for section_name in SECTION_NAMES}
    for source_path, relative_path in _public_source_files(package_root):
        module_name = _module_name(relative_path)
        section_name = _section_name(relative_path)
        for node in _public_definitions(source_path):
            sections[section_name].append(f"{module_name}.{node.name}")
    return tuple((name, tuple(sections[name])) for name in SECTION_NAMES)


def main() -> None:
    api_sections = _discover_public_api(PACKAGE_ROOT)
    api_count = sum(len(api_objects) for _, api_objects in api_sections)
    rendered_api_count = 0

    with mkdocs_gen_files.open(TARGET_PATH, "w") as page:
        page.write("# API Reference\n\n")
        page.write(
            "1. API Reference 的用途：本页汇总 `triton-anchor` 对外提供的 Python API。\n\n"
        )
        page.write("2. 内容范围：涵盖核心编译能力、适配器和扩展机制。\n\n")
        page.write(
            "3. 文档阅读方式：\n\n"
            "    ```text\n"
            "    APIName(parameter: Type) -> ReturnType\n"
            "    Brief description.\n\n"
            "    Parameters: parameter (Type) — Description.\n"
            "    Returns: ReturnType — Description.\n"
            "    Raises: ErrorType — Condition.\n"
            "    ```\n\n"
            "    第一行用于确认 API 名称、参数类型和返回类型，下一行说明 API 的主要功能。\n\n"
            "    `Parameters`、`Returns` 和 `Raises` 分别说明输入参数、返回结果和可能出现的异常。\n\n"
        )

        for section_name, api_objects in api_sections:
            page.write(f"## {section_name}\n\n")
            for api_object in api_objects:
                page.write(f"::: {api_object}\n\n")
                rendered_api_count += 1
                if rendered_api_count < api_count:
                    page.write("---\n\n")


main()
