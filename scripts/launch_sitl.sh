#!/usr/bin/env bash
# scripts/launch_sitl.sh
# ─────────────────────────────────────────────────────────────────────────────
# Launches PX4 SITL for local development.
#
# As of late 2025, PX4 removed jMAVSim and Gazebo Classic targets. The current
# Mac-friendly path is:
#   • Backend: SIH (Simulation-In-Hardware) — physics inside the PX4 process,
#     no external sim engine.
#   • Visualizer: Hawkeye (separate terminal) — `brew install PX4/px4/hawkeye`
#                 then run `hawkeye` while SIH is up.
#
# Prerequisites:
#   1. PX4-Autopilot cloned: https://github.com/PX4/PX4-Autopilot
#   2. macOS dev env installed:  bash ./Tools/setup/macos.sh
#   3. Set PX4_ROOT below, or  export PX4_ROOT=/path/to/PX4-Autopilot
# ─────────────────────────────────────────────────────────────────────────────

set -e

PX4_ROOT="${PX4_ROOT:-$HOME/PX4-Autopilot}"
TARGET="${1:-sihsim_quadx}"      # sihsim_quadx (default) | gz_x500 | …
BUILD="${2:-px4_sitl_sih}"       # px4_sitl_sih (default, lighter) | px4_sitl

if [ ! -d "$PX4_ROOT" ]; then
    echo "❌ PX4-Autopilot not found at $PX4_ROOT"
    echo ""
    echo "To install PX4 (macOS):"
    echo "  git clone https://github.com/PX4/PX4-Autopilot.git --recursive ~/PX4-Autopilot"
    echo "  cd ~/PX4-Autopilot && bash ./Tools/setup/macos.sh"
    echo ""
    echo "Then optionally:  export PX4_ROOT=/custom/path/to/PX4-Autopilot"
    exit 1
fi

# PX4's CMake configure step needs Python build deps (kconfiglib, jinja2, etc.).
# We keep them in a dedicated venv at $PX4_ROOT/.venv so they don't pollute the
# Swarm agent's venv. Activate it fully here so $VIRTUAL_ENV (which CMake's
# FindPythonInterp honors) and $PATH both point at the right Python — modifying
# only $PATH isn't enough when the calling shell already has another venv active.
PX4_VENV="${PX4_VENV:-$PX4_ROOT/.venv}"
if [ -d "$PX4_VENV" ]; then
    # Unset the parent shell's venv markers so they don't override us
    unset VIRTUAL_ENV
    unset PYTHONHOME
    # shellcheck disable=SC1091
    source "$PX4_VENV/bin/activate"
fi

if ! python3 -c "import kconfiglib" >/dev/null 2>&1; then
    echo "❌ Python build dep 'kconfiglib' is missing on PATH for python3."
    echo "   PX4 needs ~30 Python packages (kconfiglib, jinja2, pymavlink, …)"
    echo "   listed in $PX4_ROOT/Tools/setup/requirements.txt."
    echo ""
    echo "One-time setup (creates an isolated PX4 venv):"
    echo "  python3 -m venv \"$PX4_VENV\""
    echo "  source \"$PX4_VENV/bin/activate\""
    echo "  pip install -r \"$PX4_ROOT/Tools/setup/requirements.txt\""
    echo "  deactivate"
    echo ""
    echo "Then re-run this script."
    exit 1
fi

echo "🚀 Launching PX4 SITL: ${BUILD} ${TARGET}"
echo "   Using Python: $(command -v python3)"
echo "   MAVSDK will connect on udp://:14540"
echo "   Hawkeye visualizer (run in another terminal): \`hawkeye\`"
echo ""

cd "$PX4_ROOT"
make "${BUILD}" "${TARGET}"
