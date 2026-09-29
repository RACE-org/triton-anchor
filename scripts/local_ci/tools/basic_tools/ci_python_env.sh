#!/usr/bin/env bash
# BASH_ENV runs after login profiles too, which may otherwise reset PATH.
export PYTHON_BIN="${PYTHON_BIN:-/task/candidate/venv/bin/python}"
export VIRTUAL_ENV="${PYTHON_BIN%/bin/*}"
export PYTHON_VENV_ACTIVATE="$VIRTUAL_ENV/bin/activate"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export PIP_REQUIRE_VIRTUALENV=true
unset PYTHONHOME
