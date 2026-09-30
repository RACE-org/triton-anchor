"""Prepare task workspaces and launch Codex inside the owned task container."""

from __future__ import annotations

import hashlib
import copy
import shlex
import subprocess
import tarfile
from pathlib import Path

from .protocol import ContractError, atomic_json
from prepare.runtime import docker_command
from prepare.python_environment import ci_python

LAUNCH_PROGRAM = r"""
import json,os,pathlib,signal,sys,time
child=os.fork()
if child==0:
    try:
        os.setsid()
        root=pathlib.Path('/task/.processes'); root.mkdir(exist_ok=True)
        pid=os.getpid()
        start=pathlib.Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]
        path=root/(os.environ['LOCAL_CI_EXECUTION_ID']+'.json')
        tmp=root/(path.name+'.tmp.'+str(pid))
        tmp.write_text(json.dumps({'pid':pid,'start':start,'started_at':time.time()}))
        os.replace(tmp,path)
        os.execvpe(sys.argv[1],sys.argv[1:],os.environ)
    except BaseException as exc:
        print('Local CI launcher failed: '+str(exc),file=sys.stderr)
        os._exit(125)
def forward(signum,frame):
    try: os.killpg(child,signum)
    except ProcessLookupError: pass
for signum in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):
    signal.signal(signum,forward)
while True:
    try:
        _,status=os.waitpid(child,0)
        break
    except InterruptedError:
        pass
if os.WIFEXITED(status): sys.exit(os.WEXITSTATUS(status))
if os.WIFSIGNALED(status): sys.exit(128+os.WTERMSIG(status))
sys.exit(125)
"""

STOP_PROGRAM = r"""
import json,os,pathlib,signal,sys,time
p=pathlib.Path('/task/.processes')/(sys.argv[1]+'.json')
if not p.exists(): print(json.dumps({'verified':True,'remaining':[]}));sys.exit()
r=json.loads(p.read_text()); pid=r['pid']
try:
    start=pathlib.Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()[19]
    if start!=r['start']: raise RuntimeError('Process identity changed')
except FileNotFoundError:
    p.unlink(missing_ok=True);print(json.dumps({'verified':True,'remaining':[]}));sys.exit()
for sig in (signal.SIGTERM,signal.SIGKILL):
    try: os.killpg(pid,sig)
    except ProcessLookupError: break
    time.sleep(.1)
p.unlink(missing_ok=True)
print(json.dumps({'verified':True,'remaining':[]}))
"""

CODEX_LAUNCH_PROGRAM = r"""
import json,os,sys
with open('/task/session/environment.json') as f: env=json.load(f)
os.chdir(env['LOCAL_CI_CODEX_WORKSPACE'])
os.environ.clear();os.environ.update(env)
os.environ['LOCAL_CI_EXECUTION_ID']='codex'
program=sys.argv[1];sys.argv=sys.argv[1:];exec(program)
"""


class DockerExecutor:
    def __init__(self, config, state_dir, generation, task, relay, *, manager):
        self.config, self.generation, self.task = config, generation, task
        self.manager, self.relay = manager, relay
        self.artifacts = Path(generation["artifacts_host"])
        self.run_dir = self.artifacts.parent
        self.uid, self.gid = generation["execution_uid"], generation["execution_gid"]
        if self.uid <= 0 or self.gid <= 0:
            raise ContractError("Codex must run as the task's non-root user")

    def prepare(self, variant="candidate"):
        if variant not in {"candidate", "base"}:
            raise ContractError("Unknown source version")
        root = self.run_dir / "inputs" / variant
        checkout = root / "checkout"
        root.mkdir(parents=True, exist_ok=True)
        sha = self.task["base_sha" if variant == "base" else "tested_sha"]
        self.relay.checkout(sha, checkout)
        self.relay.checkout_submodules(self.task, sha, checkout)
        archive = root / "checkout.tar"
        with tarfile.open(archive, "w") as stream:
            for child in sorted(checkout.iterdir()):
                stream.add(child, arcname=child.name)
        with archive.open("rb") as stream:
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        self.manager.import_checkout(
            self.generation, variant, archive, digest.hexdigest()
        )
        archive.unlink()
        self.manager.prepare_workspace(self.generation, variant)
        return checkout

    def environment(self, variant="candidate"):
        root = Path("/task") / variant
        runtime = self.generation["variants"][variant]
        env = {str(k): str(v) for k, v in runtime["env"].items()}
        jobs = str(self.config.get("max_jobs", 12))
        env.update(
            ANCHOR_DIR=str(root / "checkout"),
            LOCAL_CI_TASK_ROOT=str(root),
            WORKSPACE="/task",
            LOCAL_CI_ARTIFACT_DIR=f"/task/artifacts/{variant}",
            LOCAL_CI_TASK_ID=self.task["task_id"],
            LOCAL_CI_TESTED_SHA=runtime["source_sha"],
            LOCAL_CI_VARIANT=variant,
            LOCAL_CI_LLVM_HASH=runtime["llvm_hash"],
            LOCAL_CI_TRITON_VERSION=runtime["triton_version"],
            PYTHON_BIN=str(root / "venv/bin/python"),
            PYTHON_VENV_ACTIVATE=str(root / "venv/bin/activate"),
            VIRTUAL_ENV=str(root / "venv"),
            PYTHONNOUSERSITE="1",
            PIP_REQUIRE_VIRTUALENV="true",
            LOCAL_CI_SEED_PYTHON=ci_python(self.config, runtime["env"]),
            BASH_ENV=self.config.get("container_control_root", "/opt/local-ci/control")
            + "/scripts/local_ci/tools/basic_tools/ci_python_env.sh",
            PATH=str(root / "venv/bin")
            + ":"
            + env.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            HOME=str(root / "home"),
            TMPDIR=str(root / "tmp"),
            XDG_CACHE_HOME=str(root / "cache"),
            TRITON_CACHE_DIR=str(root / "cache/triton"),
            MAX_JOBS=jobs,
            CMAKE_BUILD_PARALLEL_LEVEL=jobs,
            LANG="C.UTF-8",
        )
        if runtime["backend_enabled"]:
            env["BACKEND_PATH"] = str(root / "backend")
            env["TRITON_SOURCE_DIR"] = str(root / "checkout/triton")
            env["TRITON_ANCHOR_SOURCE_DIR"] = str(root / "checkout")
        env.pop("PYTHONHOME", None)
        return env

    def tool_context(self, variant="candidate"):
        root = Path("/task") / variant
        env = self.environment(variant)
        runtime = self.generation["variants"][variant]
        tools = copy.deepcopy(runtime["tools"])
        original = runtime["env"]
        tools["llvm_dir"] = original["LLVM_BUILD_DIR"]
        fallbacks = {
            "flaggems_dir": original.get("FLAGGEMS_CLONE_DIR"),
            "expected_backend": original.get("EXPECTED_TRITON_BACKEND"),
        }
        for key, value in fallbacks.items():
            if value:
                tools.setdefault(key, value)
        tools.update(
            python_bin=env["PYTHON_BIN"],
            backend_dir=str(root / "backend"),
            env={**tools.get("env", {}), **env},
        )
        if original.get("TRUSTED_ANCHOR_ENVSETUP"):
            tools.setdefault(
                "env_scripts",
                [{"path": original["TRUSTED_ANCHOR_ENVSETUP"], "args": []}],
            )
        if original.get("BACKEND_ENVSETUP"):
            path = Path(original["BACKEND_ENVSETUP"])
            tools.setdefault(
                "backend_env_scripts",
                [
                    {
                        "path": str(
                            path if path.is_absolute() else root / "backend" / path
                        ),
                        "args": shlex.split(original.get("BACKEND_ENVSETUP_ARGS", "")),
                    }
                ],
            )
        if original.get("BACKEND_TEST_COMMAND"):
            tools.setdefault(
                "backend_smoke_argv", ["bash", "-c", original["BACKEND_TEST_COMMAND"]]
            )
        tools.setdefault(
            "backend_test_paths",
            shlex.split(original.get("BACKEND_TEST_PATHS", "tests")),
        )
        tools.setdefault(
            "backend_wheel_pattern", original.get("BACKEND_WHEEL_PATTERN") or "*.whl"
        )
        return {
            "variant": variant,
            "runtime_env": env,
            "source_dir": str(root / "checkout"),
            "artifact_dir": f"/task/artifacts/{variant}",
            "task_root": str(root),
            "task_id": self.task["task_id"],
            "target_sha": env["LOCAL_CI_TESTED_SHA"],
            "base_sha": self.task["base_sha"],
            "triton_version": runtime["triton_version"],
            "llvm_hash": runtime["llvm_hash"],
            "backend_enabled": runtime["backend_enabled"],
            "python_bin": env["PYTHON_BIN"],
            "task_venv": str(root / "venv"),
            "trusted_python_bin": ci_python(self.config, runtime["env"]),
            "tools_dir": self.config.get(
                "container_control_root", "/opt/local-ci/control"
            )
            + "/scripts/local_ci/tools",
            "environment_fingerprint": runtime["environment_fingerprint"],
            "profile": {
                "id": runtime["profile"],
                "llvm_revision": runtime["llvm_hash"],
                "backend_enabled": runtime["backend_enabled"],
                "tools": tools,
            },
        }

    def write_context(self, policy, changes):
        self.artifacts.mkdir(parents=True, exist_ok=True)
        atomic_json(
            self.artifacts / "task-context.json",
            {
                "task": self.task,
                "policy": policy,
                "changes": changes,
                "candidate": "/task/candidate/checkout",
                "base": "/task/base/checkout",
                "artifacts": "/task/artifacts",
            },
        )
        for variant in ("candidate", "base"):
            path = self.artifacts / f"{variant}-context.json"
            atomic_json(path, self.tool_context(variant))
            path.chmod(0o644)
        (self.artifacts / "task-context.json").chmod(0o644)

    def deploy_session(self, files, environment):
        return self.manager.deploy_session(self.generation, files, environment)

    def codex_command(self, arguments):
        binary = self.config.get("codex_bin", "")
        if not Path(binary).is_absolute():
            raise ContractError("codex_bin must be an absolute container path")
        return docker_command(
            self.config,
            "exec",
            "--interactive",
            "--user",
            f"{self.uid}:{self.gid}",
            self.generation["container_id"],
            ci_python(self.config, self.generation.get("env")),
            "-I",
            "-S",
            "-c",
            CODEX_LAUNCH_PROGRAM,
            LAUNCH_PROGRAM,
            binary,
            *arguments,
        )

    def stop_codex(self):
        subprocess.run(
            docker_command(
                self.config,
                "exec",
                "--user",
                f"{self.uid}:{self.gid}",
                self.generation["container_id"],
                ci_python(self.config, self.generation.get("env")),
                "-I",
                "-S",
                "-c",
                STOP_PROGRAM,
                "codex",
            ),
            check=True,
            capture_output=True,
            timeout=self.config.get("cleanup_timeout_seconds", 60),
        )
