"""
drone/mavlink_bridge.py
───────────────────────
Thin async wrapper around MAVSDK-Python.

Exposes clean methods the rest of the system uses:
  - arm / disarm
  - takeoff / land / return_to_launch
  - fly_to_position (global GPS)
  - fly_to_local    (NED local frame)
  - send_velocity   (body-frame or NED velocity — used by the reactive planner)
  - telemetry properties: position, velocity, is_armed, flight_mode

On real hardware: serial port (e.g. /dev/ttyUSB0)
In SITL:         UDP (e.g. udp://:14540)
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

try:
    from mavsdk import System
    from mavsdk.offboard import (
        OffboardError,
        VelocityBodyYawspeed,
        VelocityNedYaw,
        PositionNedYaw,
    )
    from mavsdk.action import ActionError
    MAVSDK_AVAILABLE = True
except ImportError:
    MAVSDK_AVAILABLE = False
    logging.warning(
        "mavsdk not installed. MavlinkBridge will run in MOCK mode. "
        "Install with: pip install mavsdk"
    )

logger = logging.getLogger(__name__)


@dataclass
class DronePosition:
    lat_deg: float
    lon_deg: float
    abs_alt_m: float
    rel_alt_m: float


@dataclass
class DroneVelocity:
    north_m_s: float
    east_m_s: float
    down_m_s: float


class MavlinkBridge:
    """
    Async interface to a PX4 flight controller via MAVSDK.

    Usage:
        bridge = MavlinkBridge("udp://:14540")
        await bridge.connect()
        await bridge.arm()
        await bridge.takeoff(altitude_m=10.0)
        await bridge.fly_to_position(lat, lon, alt)
    """

    def __init__(self, connection_string: str, config: dict = None):
        self.connection_string = connection_string
        self.config = config or {}
        self._drone: Optional[object] = None  # mavsdk.System
        self._mock_mode = not MAVSDK_AVAILABLE
        self._position: Optional[DronePosition] = None
        self._velocity: Optional[DroneVelocity] = None
        self._is_armed = False
        self._flight_mode = "UNKNOWN"
        self._telemetry_task: Optional[asyncio.Task] = None

    # ── Connection ────────────────────────────────────────────────────────────

    async def connect(self, timeout_s: float = 30.0):
        if self._mock_mode:
            logger.warning("[MOCK] MavlinkBridge.connect() — no real hardware")
            self._is_armed = False
            self._position = DronePosition(37.7749, -122.4194, 100.0, 0.0)
            return

        self._drone = System()
        logger.info("Connecting to drone at %s ...", self.connection_string)
        await self._drone.connect(system_address=self.connection_string)

        # Wait for connection
        async for state in self._drone.core.connection_state():
            if state.is_connected:
                logger.info("Connected to drone.")
                break

        # Start telemetry background tasks
        self._telemetry_task = asyncio.create_task(self._telemetry_loop())

    async def disconnect(self):
        if self._telemetry_task:
            self._telemetry_task.cancel()
        logger.info("MavlinkBridge disconnected.")

    # ── Arming & basic actions ─────────────────────────────────────────────────

    async def arm(self):
        if self._mock_mode:
            logger.info("[MOCK] arm()")
            self._is_armed = True
            return
        logger.info("Arming...")
        await self._drone.action.arm()
        self._is_armed = True

    async def disarm(self):
        if self._mock_mode:
            logger.info("[MOCK] disarm()")
            self._is_armed = False
            return
        await self._drone.action.disarm()
        self._is_armed = False

    async def takeoff(self, altitude_m: float = 10.0):
        if self._mock_mode:
            logger.info("[MOCK] takeoff(altitude=%.1f)", altitude_m)
            return
        await self._drone.action.set_takeoff_altitude(altitude_m)
        await self._drone.action.takeoff()
        logger.info("Taking off to %.1f m", altitude_m)

    async def land(self):
        if self._mock_mode:
            logger.info("[MOCK] land()")
            return
        await self._drone.action.land()

    async def return_to_launch(self):
        if self._mock_mode:
            logger.info("[MOCK] return_to_launch()")
            return
        await self._drone.action.return_to_launch()

    # ── Navigation ────────────────────────────────────────────────────────────

    async def fly_to_position(
        self,
        lat_deg: float,
        lon_deg: float,
        alt_m: float,
        yaw_deg: float = float("nan"),
    ):
        """
        Fly to a global GPS position.
        Uses PX4 mission-item goto for accuracy over long distances.
        """
        if self._mock_mode:
            logger.info("[MOCK] fly_to_position(%.6f, %.6f, %.1f)", lat_deg, lon_deg, alt_m)
            return

        logger.info("Flying to (%.6f, %.6f) alt=%.1f m", lat_deg, lon_deg, alt_m)
        await self._drone.action.goto_location(lat_deg, lon_deg, alt_m, yaw_deg)

    async def send_velocity_ned(
        self,
        north_m_s: float,
        east_m_s: float,
        down_m_s: float,
        yaw_deg: float = 0.0,
    ):
        """
        Send a NED velocity setpoint (used by the reactive obstacle-avoidance planner).
        Drone must be in OFFBOARD mode.
        """
        if self._mock_mode:
            return

        setpoint = VelocityNedYaw(north_m_s, east_m_s, down_m_s, yaw_deg)
        await self._drone.offboard.set_velocity_ned(setpoint)

    async def start_offboard(self):
        """Switch to OFFBOARD mode (required for velocity setpoints)."""
        if self._mock_mode:
            logger.info("[MOCK] start_offboard()")
            return
        # Must send at least one setpoint before switching
        await self._drone.offboard.set_velocity_ned(VelocityNedYaw(0, 0, 0, 0))
        await self._drone.offboard.start()
        logger.info("Offboard mode started.")

    async def stop_offboard(self):
        if self._mock_mode:
            return
        await self._drone.offboard.stop()

    # ── Telemetry ─────────────────────────────────────────────────────────────

    @property
    def position(self) -> Optional[DronePosition]:
        return self._position

    @property
    def velocity(self) -> Optional[DroneVelocity]:
        return self._velocity

    @property
    def is_armed(self) -> bool:
        return self._is_armed

    @property
    def flight_mode(self) -> str:
        return self._flight_mode

    async def _telemetry_loop(self):
        """Background task: continuously update position, velocity, flight mode."""
        async def update_position():
            async for pos in self._drone.telemetry.position():
                self._position = DronePosition(
                    pos.latitude_deg,
                    pos.longitude_deg,
                    pos.absolute_altitude_m,
                    pos.relative_altitude_m,
                )

        async def update_velocity():
            async for vel in self._drone.telemetry.velocity_ned():
                self._velocity = DroneVelocity(
                    vel.north_m_s, vel.east_m_s, vel.down_m_s
                )

        async def update_armed():
            async for armed in self._drone.telemetry.armed():
                self._is_armed = armed

        async def update_mode():
            async for mode in self._drone.telemetry.flight_mode():
                self._flight_mode = str(mode)

        await asyncio.gather(
            update_position(),
            update_velocity(),
            update_armed(),
            update_mode(),
        )
