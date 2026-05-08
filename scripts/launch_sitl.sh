#!/usr/bin/env bash
# scripts/launch_sitl.sh
# ─────────────────────────────────────────────────────────────────────────────
# Launches PX4 SITL (Software In The Loop) for local development.
# Prerequisites:
#   1. PX4-Autopilot cloned: https://github.com/PX4/PX4-Autopilot
#   2. PX4 build dependencies installed (Ubuntu 20.04/22.04 recommended)
#   3. Set PX4_ROOT below or export PX4_ROOT=/path/to/PX4-Autopilot
# ─────────────────────────────────────────────────────────────────────────────

set -e

PX4_ROOT="${PX4_ROOT:-$HOME/PX4-Autopilot}"
VEHICLE="${1:-iris}"          # iris (quad) | standard_vtol | plane
SIM="${2:-gazebo-classic}"    # gazebo-classic | gz | jmavsim

if [ ! -d "$PX4_ROOT" ]; then
    echo "❌ PX4-Autopilot not found at $PX4_ROOT"
    echo ""
    echo "To install PX4:"
    echo "  git clone https://github.com/PX4/PX4-Autopilot.git --recursive"
    echo "  cd PX4-Autopilot && bash ./Tools/setup/ubuntu.sh"
    echo ""
    echo "Then set: export PX4_ROOT=/path/to/PX4-Autopilot"
    exit 1
fi

echo "🚀 Launching PX4 SITL: ${SIM}_${VEHICLE}"
echo "   MAVSDK will connect on udp://:14540"
echo ""

cd "$PX4_ROOT"
make "px4_sitl" "${SIM}_${VEHICLE}"
