# W11 Triton-version backend wheels

## AC2 independent native-wheel coexistence

`coexist_alpha` and `coexist_beta` are independent native distributions with
unique plugin IDs, entry points, targets, SONAMEs, and exported symbols. The
repository acceptance entry builds Core and both fixtures, installs all three
wheels into one clean venv, clears `PYTHONPATH`, and runs the shared Registry
lifecycle proof:

```bash
python tools/run_ac2_wheel_coexistence.py
```

The fixture `setup.py` files delegate ABI-manifest injection and native build
mechanics to `native_fixture_build.py`; the probe reuses the W10 installed
Wheel, ELF, Core-origin, and entry-point counting checks.

These fixtures are the W11 Triton-only real-wheel matrix:

- `compatible`: `requires_triton.version` is `>=3.0,<3.1`;
- `incompatible`: `requires_triton.version` is `>=9.0,<10.0`;
- `malformed`: the required `requires_triton` field is absent.

All three wheels are `python_only`.  They intentionally do not declare Core,
LLVM/MLIR, platform, native-library, or ABI constraints.

Build one fixture with:

```bash
uv build --wheel --out-dir <wheel-dir> \
  tests/fixtures/backend_plugins/<case>
```

Install the newly built triton-anchor wheel and all three fixture wheels into
a clean environment, then run:

```bash
python tests/w11_triton_version_probe.py --mode mixed
```

Install only triton-anchor plus the incompatible fixture in a second clean
environment, then run:

```bash
python tests/w11_triton_version_probe.py --mode incompatible-only
```

The negative fixture modules write an import marker and immediately fail if
their real entry point is loaded.  A successful probe therefore proves that
Triton-version rejection happened before `EntryPoint.load()`.

## W10 native ABI and ELF-conflict wheels

The four `native_*` fixtures extend the matrix with Linux ELF
`native_in_process` plugins:

- `native_compatible` declares the exact Core ABI fingerprint and loads only
  after every Protocol/Core/Triton/LLVM/MLIR/platform/ABI check succeeds;
- `native_bad_abi` derives a fingerprint that is guaranteed to differ from the
  supplied Core fingerprint and must fail before Python import;
- `native_collision_a` and `native_collision_b` are independent wheels whose
  libraries intentionally share one SONAME and one exported symbol.  Both pass
  individual validation, but conflict analysis and selection must reject the
  pair before either entry point is imported.

Every fixture wheel is platform-specific (`Root-Is-Purelib: false`), contains
one RECORD-hashed ELF shared object, and exports only its single fixture
symbol.  The Manifests target Core 0.2, Protocol 1, Triton 3.0 plus vendored
commit `757b6a61e7df814ba806f498f8bb3160f84b120c`, and LLVM/MLIR 19.0 plus
llvm-project commit `10dc3a8e916d73291269e5e2b82dd22681489aa1`.

Build these wheels only after installing the freshly built triton-anchor wheel
in the environment that will run the probe.  Obtain the fingerprint from that
installed wheel and pass the same value to every fixture build:

```bash
python -c \
  'from triton_anchor.backends import collect_core_environment as c; print(c().core_abi_fingerprint)'

export TRITON_ANCHOR_TEST_CORE_ABI_FINGERPRINT=<printed-sha256-value>

uv build --wheel --out-dir <wheel-dir> \
  tests/fixtures/backend_plugins/native_compatible
uv build --wheel --out-dir <wheel-dir> \
  tests/fixtures/backend_plugins/native_bad_abi
uv build --wheel --out-dir <wheel-dir> \
  tests/fixtures/backend_plugins/native_collision_a
uv build --wheel --out-dir <wheel-dir> \
  tests/fixtures/backend_plugins/native_collision_b
```

The build fails rather than writing a guessed ABI value when the environment
variable is missing or malformed.  Install all four wheels into the same clean
environment as the Core wheel, then run the probe without source-tree
injection:

```bash
env -u PYTHONPATH python tests/w10_native_wheel_probe.py
```

The probe verifies wheel layout and RECORD coverage, the full compatibility
report, exact ABI rejection, SONAME and dynamic-symbol collisions, and the
`EntryPoint.load()`/module-import counters.

## W11 full-profile backend wheels

The second matrix extends the real-wheel check to all currently implemented
Python-only compatibility parameters.  It is deliberately bound to the
generated triton-anchor `0.2.0` wheel built from this frozen environment:

- Backend Plugin Protocol `1.0`;
- Triton `3.0.0`, commit
  `757b6a61e7df814ba806f498f8bb3160f84b120c`;
- LLVM and MLIR `19.0.0`, actual toolchain commit
  `10dc3a8e916d73291269e5e2b82dd22681489aa1`.

`full_compatible` declares matching Protocol, Core, Triton, LLVM, and MLIR
constraints.  The seven independent negative wheels each change exactly one
parameter:

- `bad_protocol`;
- `bad_core`;
- `bad_triton_commit`;
- `bad_llvm_version`;
- `bad_llvm_commit`;
- `bad_mlir_version`;
- `bad_mlir_commit`.

Every negative module writes a case-specific marker and raises immediately if
it reaches `EntryPoint.load()`.  The full-profile probe requires all seven
records to be `REJECTED`, checks each structured error `field`, and proves that
their load count and import-marker count remain zero.  The compatible record
must validate against generated build metadata and load exactly once.

Build the eight fixture wheels without rebuilding Core:

```bash
fixture_wheel_dir="$(mktemp -d)"
for fixture_case in \
  full_compatible \
  bad_protocol \
  bad_core \
  bad_triton_commit \
  bad_llvm_version \
  bad_llvm_commit \
  bad_mlir_version \
  bad_mlir_commit
do
  uv build --wheel --out-dir "${fixture_wheel_dir}" \
    "tests/fixtures/backend_plugins/${fixture_case}"
done
```

Create a clean environment and install an existing, freshly built
triton-anchor wheel (whose `_build_info.json` has `"generated": true`) plus all
eight fixture wheels:

```bash
python3 -m venv <full-profile-venv>
<full-profile-venv>/bin/python -m pip install \
  <fresh-triton-anchor-wheel> \
  "${fixture_wheel_dir}"/*.whl
```

Run the probe with no checkout injected through `PYTHONPATH`:

```bash
env -u PYTHONPATH \
  <full-profile-venv>/bin/python \
  tests/w11_full_profile_probe.py
```

The probe also verifies that imported `triton` and `triton_anchor` modules come
from the clean environment, that build metadata is generated, and that the
Registry is using the `full` preflight profile.

## W12 installed Legacy backend wheels

The `legacy/generate.py` template creates two independent pure-Python
distributions without a Backend Manifest:

- `legacy_good` exposes a complete legacy compiler/driver pair;
- `legacy_bad_interface` imports successfully but exposes a non-class
  `driver_cls`.

Generate temporary build projects, then use the existing wheel build flow:

```bash
python tests/fixtures/backend_plugins/legacy/generate.py \
  --case legacy_good --output-root <project-root>
python tests/fixtures/backend_plugins/legacy/generate.py \
  --case legacy_bad_interface --output-root <project-root>
uv build --wheel --out-dir <wheel-dir> <project-root>/legacy_good
uv build --wheel --out-dir <wheel-dir> \
  <project-root>/legacy_bad_interface
```

Install the freshly built Core wheel plus one Legacy fixture wheel into a
clean venv. Run each probe mode in a separate process without `PYTHONPATH`:

```bash
python tests/w12_legacy_wheel_probe.py \
  --mode discovery --wheel-path <legacy-good-wheel> \
  --evidence-dir <new-evidence-dir>
python tests/w12_legacy_wheel_probe.py \
  --mode good-runtime --wheel-path <legacy-good-wheel> \
  --evidence-dir <new-evidence-dir>
```

The bad-interface modes are `bad-register`, `bad-compiler`, and
`bad-runtime`. The probe verifies installed distribution metadata, a single
real `triton.backends` entry point, pure-Python wheel contents, import
counters, lifecycle state, structured errors, and module origin. Because
these wheels have no Manifest, they remain `LEGACY_UNVERIFIED`; pure-Python
wheel contents do not imply protocol-level `python_only` compatibility.
