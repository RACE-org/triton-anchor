# Read-only Host Toolchains

LLVM, PPL, backend sources and FlagGems live in versioned directories owned by
the CI account. Profiles share one immutable runtime image and mount their
dependencies read-only under `/opt/local-ci/runtime/deps`. Frozen sources,
venvs, build output and caches remain in task storage.

## Source Identity and Profiles

The gateway and server read LLVM metadata from the frozen source's
`triton/cmake/` directory. Recognized names are `llvm-hash` and `llvm-info`,
with no extension, `.txt` or `.json`. Plain text contains a full 40-character SHA;
JSON provides `llvm_hash`. All recognized files present must agree.
`amd-llvm-info.json` and `llvm-build-info.json` describe other build inputs and
do not select the task LLVM. Triton 3.8 uses `llvm-info.json`.

Each variant resolves a unique profile by **Triton major.minor and full LLVM SHA**.
The deployment configuration contains:

| Triton profile | LLVM SHA | Backend |
| --- | --- | --- |
| 3.0 | `10dc3a8e916d73291269e5e2b82dd22681489aa1` | Enabled |
| 3.1 | `10dc3a8e916d73291269e5e2b82dd22681489aa1` | Disabled |
| 3.2 | `86b69c31642e98f8357df62c09d118ad1da4e16a` | Disabled |
| 3.3 | `a66376b0dc3b2ea8a84fda26faca287980986f78` | Disabled |
| 3.4 | `8957e64a20fc7f4277565c6cfe3e555c119783ce` | Disabled |
| 3.5 | `7d5de3033187c8a3bb4d2e322f5462cdaf49808f` | Disabled |
| 3.6 | `a992f29451b9e140424f35ac5e20177db4afbdc0` | Disabled |
| 3.8 | `941a04e69ee8fe4c7a162b2f1e215aa8df867534` | Disabled |

The runtime selects profiles from the frozen source declarations.
Triton 3.5.1 selects profile 3.5. Triton 3.0 and 3.1 share a read-only LLVM
installation but keep distinct capabilities. Only 3.0 uses backend, PPL,
FlagGems, backend-src and deploy-tools mounts. Frontend profiles have
`backend_enabled=false`, `devices=[]`, common frontend environment variables
and one LLVM mount. Paths and directory digests are maintained in
[config.example.json](config.example.json).

## Directory Requirements

Set an absolute `dependency_root` outside control, state and credential directories.
Each source must be a canonical directory strictly beneath that root.
Directories and files must belong to the CI account, permit mapped container
users to read files and traverse directories, and have no group/other write permission.
Links must be relative, resolve inside the same version directory and have
existing targets. Sockets and special files are rejected.

Record package provenance or build records proving the full LLVM commit.
A directory name or `llvm-config --version` alone does not prove that identity.
Verify `llvm-config --version`, `--host-target`, `--shared-mode`, headers and
CMake packages, and check ABI compatibility with the shared image.

## Configure a Dependency

Prepare the version directory and finalize its permissions and links, then
calculate `prepare.artifacts.tree_digest(Path(source))` using trusted control code.
This is a directory-content digest, not an archive checksum.

Use a matching LLVM recipe and read-only mount. Replace the placeholders below
with literal values; `source` must be an absolute path under the configured
`dependency_root`. JSON paths do not expand shell variables.

```json
{
  "llvm_hash": "<full LLVM commit>",
  "llvm": {"mode": "mount", "commit": "<full LLVM commit>"},
  "mounts": [
    {
      "source": "<dependency_root>/llvm-<commit>",
      "target": "/opt/local-ci/runtime/deps/llvm-<commit>",
      "read_only": true,
      "sha256": "<tree_digest of source>"
    }
  ]
}
```

Edit and commit the repository configuration, then follow the
[deployment procedure](README.md). Environment preparation and deployment preflight call
`dependency_mounts(..., verify_content=True)` and `validate_mounted_llvm(...)`
to check content, ownership, permissions and the LLVM recipe.
Task startup checks the pinned image and read-only mount identity.

Read-only container mounts do not prevent host-side modification.
To change an in-use dependency, prepare a new version directory, update its
configuration and repeat runtime preflight. Retain directories needed by active
tasks. Control-code updates reuse unchanged dependencies.

Install common Python/system packages in the shared image; their ABI must match
all configured toolchains. Probes check imports and mounted LLVM paths.
Formal tasks build and install frontend/backend wheels in their own directories.

Only fixed read-only directory binds are accepted; recursive nested mounts are
disabled. Host credentials, Docker sockets, task roots and home directories are
not dependency mounts. Management recovery uses task/session volumes so missing
toolchains do not prevent evidence export or cleanup.
