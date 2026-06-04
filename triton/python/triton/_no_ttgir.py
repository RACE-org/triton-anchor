import os


def _read_no_ttgir():
    value = os.getenv("NO_TTGIR", "0").strip().upper()
    if value in {"", "0", "OFF", "FALSE", "NO"}:
        return False
    if value in {"1", "ON", "TRUE", "YES"}:
        return True
    raise RuntimeError(f"Invalid NO_TTGIR value: {value}")


NO_TTGIR = _read_no_ttgir()
