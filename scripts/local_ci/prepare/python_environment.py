"""Select the prepared CI interpreter without falling back to system Python."""

from pathlib import PurePosixPath


def ci_python(config=None, environment=None):
    config, environment = config or {}, environment or {}
    executable = config.get("container_python") or environment.get("SEED_PYTHON")
    if not executable:
        executable = str(
            PurePosixPath(environment.get("PYTHON_VENV_ACTIVATE", "/opt/venv/bin/activate")).parent
            / "python"
        )
    path = PurePosixPath(executable)
    if (
        not path.is_absolute()
        or ".." in path.parts
        or PurePosixPath("/task") in path.parents
        or path.parent in {PurePosixPath("/usr/bin"), PurePosixPath("/usr/local/bin"), PurePosixPath("/bin")}
        or any(char in executable for char in "\n\r\x00")
    ):
        raise ValueError("container_python must point to the prepared CI virtual environment, e.g. /opt/venv/bin/python")
    return str(path)
