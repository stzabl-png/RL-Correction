#!/bin/bash
# Create .venv-pico (py3.10) for scripts/teleop_pico_producer.py.
#
# Why a separate venv: xrobotoolkit_sdk ships a cp310 .so; the repo's .venv
# and .venv-isaac are py3.11 and cannot import it.
#
# Zero-setup alternative (GR00T's teleop venv already has everything):
#   ~/magicsim/GR00T-WholeBodyControl/.venv_teleop/bin/python scripts/teleop_pico_producer.py
set -e
cd "$(dirname "$0")/.."

# SDK 查找顺序:环境变量 > 仓库自带 external/(发布仓库有)> 原始检出位置
if [ -z "${SDK_DIR:-}" ]; then
    if [ -d "$PWD/external/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64" ]; then
        SDK_DIR="$PWD/external/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64"
    else
        SDK_DIR="$HOME/magicsim/GR00T-WholeBodyControl/external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64"
    fi
fi

if [ ! -d .venv-pico ]; then
    uv venv .venv-pico --python 3.10
fi
# pytest 与 scipy 是 tests/test_pico_math.py 要用的(check_all 用这个环境跑它)
# setuptools + pybind11: build deps for the --no-build-isolation editable install
# below (uv venvs ship without setuptools; CMakeLists does find_package(pybind11)).
uv pip install --python .venv-pico/bin/python numpy pyzmq msgpack pytest scipy setuptools pybind11

# Prefer a proper editable install (builds via CMake, same as GR00T's
# install_pico.sh); fall back to dropping the prebuilt cp310 .so in place.
if ! uv pip install --python .venv-pico/bin/python --no-build-isolation -e "$SDK_DIR"; then
    echo "[setup] editable build failed - falling back to the prebuilt .so"
    SITE=$(.venv-pico/bin/python -c 'import site; print(site.getsitepackages()[0])')
    cp "$SDK_DIR"/xrobotoolkit_sdk.cpython-310-x86_64-linux-gnu.so "$SITE/"
fi

.venv-pico/bin/python -c "import xrobotoolkit_sdk, zmq, msgpack, numpy; print('[setup] .venv-pico OK')"
echo "[setup] run: .venv-pico/bin/python scripts/teleop_pico_producer.py"
