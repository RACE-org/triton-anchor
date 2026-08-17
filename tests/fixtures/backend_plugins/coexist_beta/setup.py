import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from native_fixture_build import build_native_fixture  # noqa: E402

build_native_fixture(
    fixture_root=Path(__file__).parent,
    distribution="triton-anchor-ac2-native-beta-backend",
    description="Second independent AC2 native coexistence backend",
    package="ac2_native_beta",
    entry_point="ac2_native_beta",
    library="libtriton_anchor_ac2_native_beta.so",
    soname="libtriton_anchor_ac2_native_beta.so",
)
