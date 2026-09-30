"""Read the two dedicated Codex files; Codex validates its own configuration."""

import json
from pathlib import Path


class CredentialValidationError(ValueError):
    pass


def validate_credentials(codex_home, personal_codex_home, *, warnings=None):
    home = Path(codex_home)
    if (
        not home.is_absolute()
        or home.resolve() == Path(personal_codex_home).expanduser().resolve()
    ):
        raise CredentialValidationError(
            "Use an absolute, dedicated Codex CI configuration directory"
        )
    for name in ("config.toml", "auth.json"):
        path = home / name
        if not path.is_file() or path.is_symlink():
            raise CredentialValidationError(f"Missing regular Codex file: {path}")
        if not path.read_text(encoding="utf-8").strip():
            raise CredentialValidationError(f"Empty Codex file: {path}")
    if not isinstance(json.loads((home / "auth.json").read_text()), dict):
        raise CredentialValidationError("Codex auth.json must be an object")
    return home
