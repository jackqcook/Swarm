#!/usr/bin/env python3
"""
scripts/send_command.py
────────────────────────
Send a text command to a running DroneAgent via ZeroMQ.

Usage:
  python scripts/send_command.py "fly to the northeast clearing at 30 meters"
  python scripts/send_command.py --interactive

The DroneAgent must be running with comms enabled (Stage 1: it reads from stdin
or ZMQ socket depending on launch mode).

For quick testing without ZMQ, use --direct to spawn an agent inline:
  python scripts/send_command.py --direct --sim "hover at 20 meters"
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))


async def send_direct(command: str, config_path: str, sim: bool):
    """Spawn a DroneAgent inline and send a single command (good for testing)."""
    import yaml
    from drone.agent import DroneAgent

    with open(config_path) as f:
        config = yaml.safe_load(f)

    agent = DroneAgent(config=config, sim=sim)
    await agent.start()
    response = await agent.handle_command(command)
    print(f"\n→ {response}")

    # If clarification is needed, prompt user
    if response.startswith("CLARIFY:"):
        print(response[8:])
        follow_up = input("\nYour clarification > ").strip()
        if follow_up:
            response2 = await agent.handle_command(follow_up)
            print(f"\n→ {response2}")

    await agent.stop()


async def send_zmq(command: str, address: str = "tcp://localhost:5555"):
    """Send a command to a running agent over ZeroMQ (Stage 2+)."""
    try:
        import zmq
        import zmq.asyncio
    except ImportError:
        print("pyzmq not installed. Use --direct for in-process testing.")
        sys.exit(1)

    ctx = zmq.asyncio.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(address)
    await sock.send_string(command)
    response = await sock.recv_string()
    print(f"→ {response}")
    sock.close()


def main():
    parser = argparse.ArgumentParser(description="Send a command to the Swarm drone agent")
    parser.add_argument("command", nargs="?", help="Natural language command")
    parser.add_argument("--direct", action="store_true", help="Spawn agent in-process (no ZMQ)")
    parser.add_argument("--sim", action="store_true", help="Use SITL simulation")
    parser.add_argument("--config", default="configs/sim_default.yaml")
    parser.add_argument("--zmq", default="tcp://localhost:5555", help="ZMQ address of running agent")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    if args.interactive or not args.command:
        print("🚁 Swarm Command Interface (type 'quit' to exit)\n")
        while True:
            try:
                cmd = input("Command > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if cmd.lower() in ("quit", "exit", "q"):
                break
            if not cmd:
                continue
            if args.direct:
                asyncio.run(send_direct(cmd, args.config, args.sim))
            else:
                asyncio.run(send_zmq(cmd, args.zmq))
    else:
        if args.direct:
            asyncio.run(send_direct(args.command, args.config, args.sim))
        else:
            asyncio.run(send_zmq(args.command, args.zmq))


if __name__ == "__main__":
    main()
