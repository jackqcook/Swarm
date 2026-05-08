"""
drone/agent.py
──────────────
The top-level drone agent.  Wires together:
  • CommandInterpreter  (Tier 1 — VLM + KnowNo)
  • PerceptionLoop      (Tier 2 — YOLOv8n + depth)
  • Planner             (waypoint generation + local replanning)
  • MavlinkBridge       (PX4 / SITL interface via MAVSDK)

Entry point:
  python -m drone.agent --config configs/sim_default.yaml [--sim]
"""

import asyncio
import logging
import argparse
from pathlib import Path
from typing import Optional

import yaml

from drone.mavlink_bridge import MavlinkBridge
from drone.command_interpreter import CommandInterpreter
from perception.perception_loop import PerceptionLoop
from planner.waypoint_planner import WaypointPlanner

logger = logging.getLogger(__name__)


class DroneAgent:
    """
    Orchestrates all subsystems for a single drone.

    Tier 1 (seconds-latency):
        User text → CommandInterpreter → structured Goal

    Tier 2 (milliseconds-latency):
        PerceptionLoop → obstacles → Planner → PX4 velocity setpoints
    """

    def __init__(self, config: dict, sim: bool = False):
        self.config = config
        self.sim = sim
        self._running = False

        # Subsystems (initialized in start())
        self.bridge: Optional[MavlinkBridge] = None
        self.interpreter: Optional[CommandInterpreter] = None
        self.perception: Optional[PerceptionLoop] = None
        self.planner: Optional[WaypointPlanner] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self):
        """Initialize all subsystems and connect to the flight controller."""
        logger.info("Starting DroneAgent (sim=%s)", self.sim)

        # 1. MAVLink bridge
        conn_str = (
            self.config["mavlink"]["sitl_address"]
            if self.sim
            else self.config["mavlink"]["serial_port"]
        )
        self.bridge = MavlinkBridge(conn_str, config=self.config["mavlink"])
        await self.bridge.connect()

        # 2. Command interpreter (loads VLM lazily on first command)
        self.interpreter = CommandInterpreter(config=self.config.get("interpreter", {}))

        # 3. Perception loop (starts background camera + inference thread)
        self.perception = PerceptionLoop(config=self.config.get("perception", {}))
        await self.perception.start()

        # 4. Waypoint planner
        self.planner = WaypointPlanner(
            bridge=self.bridge,
            perception=self.perception,
            config=self.config.get("planner", {}),
        )

        self._running = True
        logger.info("DroneAgent ready. Waiting for commands.")

    async def stop(self):
        """Graceful shutdown: RTL, disarm, disconnect."""
        logger.info("Shutting down DroneAgent...")
        self._running = False
        if self.bridge and self.bridge.is_armed:
            logger.info("Returning to launch before shutdown...")
            await self.bridge.return_to_launch()
        if self.perception:
            await self.perception.stop()
        if self.bridge:
            await self.bridge.disconnect()
        logger.info("DroneAgent stopped.")

    # ── Command interface ──────────────────────────────────────────────────────

    async def handle_command(self, raw_text: str) -> str:
        """
        Public interface: accept a natural-language command, interpret it,
        execute it, and return a human-readable status string.

        Returns a clarification question if ambiguous.
        """
        if not self._running:
            return "Agent not running. Call start() first."

        logger.info("Received command: %r", raw_text)

        # Tier 1: interpret
        result = await self.interpreter.interpret(
            text=raw_text,
            image=self.perception.latest_frame,
        )

        if result.needs_clarification:
            question = result.clarification_question
            logger.info("Ambiguous command — asking for clarification: %s", question)
            return f"CLARIFY: {question}"

        goal = result.goal
        logger.info("Interpreted goal: %s", goal)

        # Tier 2: execute
        await self.planner.execute_goal(goal)
        return f"Executing: {goal.description}"

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def run_interactive(self):
        """Simple interactive CLI loop for testing. Replace with your UI/API."""
        print("\n🚁 Swarm Drone Agent — Interactive Mode")
        print("Type a command, or 'quit' to exit.\n")

        while self._running:
            try:
                raw = await asyncio.get_event_loop().run_in_executor(
                    None, input, "Command > "
                )
            except (EOFError, KeyboardInterrupt):
                break

            if raw.strip().lower() in ("quit", "exit", "q"):
                break

            response = await self.handle_command(raw.strip())
            print(f"  → {response}\n")

        await self.stop()


# ── Entry point ───────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


async def main():
    parser = argparse.ArgumentParser(description="Swarm Drone Agent")
    parser.add_argument(
        "--config", default="configs/sim_default.yaml", help="Path to config YAML"
    )
    parser.add_argument(
        "--sim", action="store_true", help="Use SITL simulation instead of real hardware"
    )
    parser.add_argument(
        "--command", "-c", type=str, default=None,
        help="Single command to execute (non-interactive)"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(args.config)
    agent = DroneAgent(config=config, sim=args.sim)

    try:
        await agent.start()
        if args.command:
            response = await agent.handle_command(args.command)
            print(response)
            await agent.stop()
        else:
            await agent.run_interactive()
    except Exception as e:
        logger.exception("Fatal error in DroneAgent: %s", e)
        await agent.stop()
        raise


if __name__ == "__main__":
    asyncio.run(main())
