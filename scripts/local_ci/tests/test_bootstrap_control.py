"""An empty server bootstraps only an explicitly approved control commit."""

import json
import subprocess

import pytest

from prepare.bootstrap_control import bootstrap_control


def git(root, *arguments):
    return subprocess.check_output(["git", *arguments], cwd=root, text=True).strip()


def fixture_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(
        ["git", "init", "-q", "-b", "CI_dev"],
        cwd=source,
        check=True,
    )
    install = source / "scripts/local_ci/prepare/install.py"
    install.parent.mkdir(parents=True)
    install.write_text("print('pinned installer')\n")
    (source / "control.txt").write_text("approved\n")
    subprocess.run(["git", "add", "."], cwd=source, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-q",
            "-m",
            "approved",
        ],
        cwd=source,
        check=True,
    )
    return source, git(source, "rev-parse", "HEAD")


def fixture_config(tmp_path, source):
    config_path = tmp_path / "config.json"
    credentials_path = tmp_path / "credentials.env"
    credentials_path.write_text("FIXTURE=private\n")
    credentials_path.chmod(0o600)
    config = {
        "control_root": str(tmp_path / "control"),
        "state_dir": str(tmp_path / "state"),
        "control_repo_url": str(source),
        "control_branch": "CI_dev",
        "python_bin": "/usr/bin/python3",
    }
    config_path.write_text(json.dumps(config))
    return config, config_path, credentials_path


@pytest.mark.parametrize("existing", [False, True])
def test_bootstrap_uses_exact_revision_and_invokes_pinned_installer(tmp_path, existing):
    source, revision = fixture_source(tmp_path)
    config, config_path, credentials_path = fixture_config(tmp_path, source)
    if existing:
        subprocess.run(["git", "clone", "-q", str(source), config["control_root"]], check=True)
    installed = []

    result = bootstrap_control(
        config,
        config_path,
        credentials_path,
        revision,
        apply=True,
        allow_local=True,
        installer=lambda *arguments: installed.append(arguments),
    )

    control = tmp_path / "control"
    assert result["applied"] and result["cloned"] is not existing
    assert git(control, "rev-parse", "HEAD") == revision
    assert git(control, "config", "--get", "remote.origin.url") == str(source)
    assert len(installed) == 1
    assert installed[0][0] == control
    marker = json.loads((tmp_path / "state/control-bootstrap.json").read_text())
    assert marker["revision"] == revision


def test_preview_and_revision_mismatch_do_not_create_control_root(tmp_path):
    source, revision = fixture_source(tmp_path)
    config, config_path, credentials_path = fixture_config(tmp_path, source)

    preview = bootstrap_control(
        config,
        config_path,
        credentials_path,
        revision,
        allow_local=True,
    )
    assert preview["action"] == "clone"
    assert not (tmp_path / "control").exists()

    with pytest.raises(ValueError, match="does not match expected_revision"):
        bootstrap_control(
            config,
            config_path,
            credentials_path,
            "0" * 40,
            apply=True,
            allow_local=True,
            installer=lambda *arguments: None,
        )
    assert not (tmp_path / "control").exists()


def test_existing_unrelated_directory_is_preserved(tmp_path):
    source, revision = fixture_source(tmp_path)
    config, config_path, credentials_path = fixture_config(tmp_path, source)
    control = tmp_path / "control"
    control.mkdir()
    evidence = control / "preserve.txt"
    evidence.write_text("keep\n")

    with pytest.raises((ValueError, RuntimeError), match="Git checkout|git rev-parse"):
        bootstrap_control(
            config,
            config_path,
            credentials_path,
            revision,
            apply=True,
            allow_local=True,
            installer=lambda *arguments: None,
        )

    assert evidence.read_text() == "keep\n"
