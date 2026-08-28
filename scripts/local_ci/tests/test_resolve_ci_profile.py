import sys
from pathlib import Path

import pytest


SHARED = Path(__file__).resolve().parents[1] / "shared"
sys.path.insert(0, str(SHARED))

import resolve_ci_profile as profiles


LEGACY_HASH = "10dc3a8e916d73291269e5e2b82dd22681489aa1"
TRITON_33_HASH = "a66376b0dc3b2ea8a84fda26faca287980986f78"
TRITON_36_HASH = "a992f29451b9e140424f35ac5e20177db4afbdc0"
KNOWN_HASHES = (LEGACY_HASH, TRITON_33_HASH, TRITON_36_HASH)


def test_empty_profile_dir_never_falls_back() -> None:
    with pytest.raises(profiles.ProfileResolutionError, match="not configured"):
        profiles.resolve_profile_file(
            profile_dir="",
            llvm_hash=TRITON_33_HASH,
        )


def test_profile_is_selected_only_by_its_exact_hash_filename(tmp_path: Path) -> None:
    selected_profiles = {}
    for llvm_hash in KNOWN_HASHES:
        profile = tmp_path / f"{llvm_hash}.env"
        profile.write_text(f'LOCAL_CI_PROFILE_NAME="profile-{llvm_hash[:8]}"\n')
        selected_profiles[llvm_hash] = profile.resolve()

    for llvm_hash, expected_profile in selected_profiles.items():
        assert profiles.resolve_profile_file(
            profile_dir=str(tmp_path),
            llvm_hash=llvm_hash,
        ) == expected_profile

    with pytest.raises(profiles.ProfileResolutionError, match="No server-owned"):
        profiles.resolve_profile_file(
            profile_dir=str(tmp_path),
            llvm_hash="b" * 40,
        )


def test_profile_directory_must_be_absolute(tmp_path: Path) -> None:
    with pytest.raises(profiles.ProfileResolutionError, match="absolute path"):
        profiles.resolve_profile_file(
            profile_dir="relative/profiles",
            llvm_hash=TRITON_33_HASH,
        )


@pytest.mark.parametrize(
    "value",
    ["", "a" * 39, "a" * 41, "A" * 40, "../" + "a" * 40],
)
def test_invalid_hash_cannot_construct_a_profile_path(value: str) -> None:
    with pytest.raises(profiles.ProfileResolutionError, match="lowercase 40-character"):
        profiles.resolve_profile_file(
            profile_dir="",
            llvm_hash=value,
        )
