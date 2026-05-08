"""
planner/waypoint_planner.py
────────────────────────────
Converts a DroneGoal (from Tier 1) into concrete flight behavior,
while using real-time perception for obstacle avoidance (Tier 2).

Two layers:
  1. GLOBAL PLANNER  (runs once per new goal, ~seconds)
     - Converts goal.region_description → GPS waypoints
     - Uses GPS + basic geometry for Stage 1
     - Stage 2 will add ORB-SLAM3 map-based global planning

  2. LOCAL PLANNER   (reactive, ~10–30 Hz)
     - Monitors perception queue
     - If obstacle within safety_radius_m → generate avoidance velocity
     - Uses a simplified potential-field approach (EGO-Planner style)
     - Sends velocity setpoints to PX4 via OFFBOARD mode

Stage 1 simplification:
  - region_description is treated as a relative bearing or cardinal direction
    ("north", "northeast", "the clearing ahead") rather than a VLM-grounded
    pixel coordinate. Full visual grounding (NanoOWL) comes in Stage 2.
"""

import asyncio
import logging
import math
import time
from dataclasses import dataclass
from typing import Optional, List, Tuple

from drone.command_interpreter import DroneGoal
from perception.perception_loop import PerceptionLoop, ObstacleMap

logger = logging.getLogger(__name__)


# ── Coordinate helpers ────────────────────────────────────────────────────────

def haversine_m(lat1, lon1, lat2, lon2) -> float:
    """Distance in meters between two GPS coordinates."""
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def offset_gps(lat, lon, north_m, east_m) -> Tuple[float, float]:
    """Apply a local NED offset (metres) to a GPS coordinate."""
    R = 6_371_000.0
    new_lat = lat + math.degrees(north_m / R)
    new_lon = lon + math.degrees(east_m / (R * math.cos(math.radians(lat))))
    return new_lat, new_lon


DIRECTION_MAP = {
    "north": (1, 0),
    "northeast": (0.707, 0.707),
    "east": (0, 1),
    "southeast": (-0.707, 0.707),
    "south": (-1, 0),
    "southwest": (-0.707, -0.707),
    "west": (0, -1),
    "northwest": (0.707, -0.707),
    "forward": (1, 0),
    "back": (-1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


# ── Planner data structures ────────────────────────────────────────────────────

@dataclass
class Waypoint:
    lat: float
    lon: float
    alt_m: float
    loiter_s: float = 0.0
    label: str = ""


# ── Global planner ─────────────────────────────────────────────────────────────

class GlobalPlanner:
    """
    Converts a DroneGoal into a list of Waypoints.

    Stage 1: keyword-based direction resolution + simple geometry.
    Stage 2: will add NanoOWL visual grounding to find GPS of named targets.
    """

    DEFAULT_TRAVEL_DISTANCE_M = 100.0  # Default "fly over there" distance
    DEFAULT_SURVEY_RADIUS_M = 50.0

    def generate_waypoints(
        self,
        goal: DroneGoal,
        current_lat: float,
        current_lon: float,
        current_alt_m: float,
    ) -> List[Waypoint]:

        alt = goal.target_alt_m or 30.0
        desc = (goal.region_description or "").lower()

        if goal.action == "return":
            # Caller must supply home lat/lon — placeholder here
            return [Waypoint(current_lat, current_lon, alt, loiter_s=0, label="RTL")]

        if goal.action in ("hover", "land"):
            return [Waypoint(current_lat, current_lon, alt if goal.action == "hover" else 0.0,
                             loiter_s=999.0, label=goal.action)]

        if goal.action == "orbit":
            return self._orbit_waypoints(current_lat, current_lon, alt, radius_m=30.0)

        if goal.action == "survey_area":
            return self._survey_waypoints(
                current_lat, current_lon, alt,
                radius_m=self.DEFAULT_SURVEY_RADIUS_M,
                direction_hint=desc,
            )

        # Default: fly_to
        # Resolve direction from description
        north_m, east_m = self._resolve_direction(desc, self.DEFAULT_TRAVEL_DISTANCE_M)
        target_lat, target_lon = offset_gps(current_lat, current_lon, north_m, east_m)

        return [
            Waypoint(target_lat, target_lon, alt, loiter_s=5.0, label=f"fly_to:{desc[:20]}"),
        ]

    def _resolve_direction(self, desc: str, distance_m: float) -> Tuple[float, float]:
        """Map a region description to (north_m, east_m) offset."""
        for keyword, (n_unit, e_unit) in DIRECTION_MAP.items():
            if keyword in desc:
                return n_unit * distance_m, e_unit * distance_m
        # No keyword match — go north as a safe default and log
        logger.info("No direction keyword found in %r — defaulting to north", desc)
        return distance_m, 0.0

    def _orbit_waypoints(
        self, lat, lon, alt_m, radius_m=30.0, n_points=8
    ) -> List[Waypoint]:
        wps = []
        for i in range(n_points + 1):
            angle = 2 * math.pi * i / n_points
            north = radius_m * math.cos(angle)
            east = radius_m * math.sin(angle)
            wp_lat, wp_lon = offset_gps(lat, lon, north, east)
            wps.append(Waypoint(wp_lat, wp_lon, alt_m, loiter_s=0.0, label=f"orbit_{i}"))
        return wps

    def _survey_waypoints(
        self, lat, lon, alt_m, radius_m=50.0, direction_hint="", n_rows=4
    ) -> List[Waypoint]:
        """Simple lawnmower survey pattern."""
        north_offset, east_offset = self._resolve_direction(direction_hint, radius_m)
        center_lat, center_lon = offset_gps(lat, lon, north_offset, east_offset)

        wps = []
        for i in range(n_rows):
            row_east = -radius_m + i * (2 * radius_m / (n_rows - 1)) if n_rows > 1 else 0
            row_north = radius_m if i % 2 == 0 else -radius_m
            wp_lat1, wp_lon1 = offset_gps(center_lat, center_lon, row_north, row_east)
            wp_lat2, wp_lon2 = offset_gps(center_lat, center_lon, -row_north, row_east)
            wps.append(Waypoint(wp_lat1, wp_lon1, alt_m, label=f"survey_{i}a"))
            wps.append(Waypoint(wp_lat2, wp_lon2, alt_m, label=f"survey_{i}b"))
        return wps


# ── Local planner (reactive obstacle avoidance) ────────────────────────────────

class LocalPlanner:
    """
    Simplified potential-field reactive avoidance.

    When an obstacle is within safety_radius_m:
      - Compute repulsion velocity away from obstacle bearing
      - Add attraction velocity toward current waypoint
      - Send resultant NED velocity setpoint to PX4

    This runs in the async execution loop at ~10–30 Hz.
    It's the inner part of what EGO-Planner does (gradient descent on ESDF).
    Stage 3 will replace this with a full EGO-Planner port.
    """

    def __init__(
        self,
        safety_radius_m: float = 5.0,
        max_speed_m_s: float = 3.0,
        attraction_gain: float = 0.5,
        repulsion_gain: float = 2.0,
    ):
        self.safety_radius_m = safety_radius_m
        self.max_speed_m_s = max_speed_m_s
        self.attraction_gain = attraction_gain
        self.repulsion_gain = repulsion_gain

    def compute_velocity(
        self,
        obstacle_map: ObstacleMap,
        target_lat: float,
        target_lon: float,
        current_lat: float,
        current_lon: float,
        current_heading_deg: float = 0.0,
    ) -> Tuple[float, float, float]:
        """
        Returns (vN, vE, vD) in m/s (NED frame).
        vD=0 (altitude hold handled by PX4).
        """
        # Attraction toward waypoint
        d_lat = target_lat - current_lat
        d_lon = target_lon - current_lon
        dist_m = haversine_m(current_lat, current_lon, target_lat, target_lon)

        if dist_m < 1.0:
            return 0.0, 0.0, 0.0

        # Normalize and scale
        vN_attract = self.attraction_gain * d_lat / abs(d_lat + 1e-9) * min(dist_m, self.max_speed_m_s)
        vE_attract = self.attraction_gain * d_lon / abs(d_lon + 1e-9) * min(dist_m, self.max_speed_m_s)

        # Repulsion from nearest obstacle
        vN_repulse, vE_repulse = 0.0, 0.0
        nearest = obstacle_map.nearest_obstacle_m
        bearing = math.radians(obstacle_map.nearest_obstacle_bearing_deg)

        if nearest < self.safety_radius_m:
            repulsion = self.repulsion_gain * (1.0 / nearest - 1.0 / self.safety_radius_m)
            # Obstacle bearing is in body frame; convert to NED using heading
            heading_rad = math.radians(current_heading_deg)
            obs_ned_bearing = heading_rad + bearing
            vN_repulse = -repulsion * math.cos(obs_ned_bearing)
            vE_repulse = -repulsion * math.sin(obs_ned_bearing)
            logger.info(
                "OBSTACLE at %.1f m bearing %.1f° → repulsion (%.2f, %.2f)",
                nearest, math.degrees(bearing), vN_repulse, vE_repulse,
            )

        vN = vN_attract + vN_repulse
        vE = vE_attract + vE_repulse

        # Clamp to max speed
        speed = math.sqrt(vN ** 2 + vE ** 2)
        if speed > self.max_speed_m_s:
            vN = vN / speed * self.max_speed_m_s
            vE = vE / speed * self.max_speed_m_s

        return vN, vE, 0.0


# ── Top-level planner ──────────────────────────────────────────────────────────

class WaypointPlanner:
    """
    Orchestrates global + local planning.
    Called by DroneAgent.handle_command() after interpretation.
    """

    WAYPOINT_ARRIVAL_RADIUS_M = 5.0   # Accept waypoint as reached when within this distance
    CONTROL_HZ = 10                    # Reactive avoidance loop rate

    def __init__(self, bridge, perception: PerceptionLoop, config: dict = None):
        self.bridge = bridge
        self.perception = perception
        self.config = config or {}
        self.global_planner = GlobalPlanner()
        self.local_planner = LocalPlanner(
            safety_radius_m=self.config.get("safety_radius_m", 5.0),
            max_speed_m_s=self.config.get("max_speed_m_s", 3.0),
        )
        self._active_task: Optional[asyncio.Task] = None

    async def execute_goal(self, goal: DroneGoal):
        """
        Cancel any active mission and start executing the new goal.
        """
        if self._active_task and not self._active_task.done():
            logger.info("Cancelling active mission for new goal.")
            self._active_task.cancel()
            try:
                await self._active_task
            except asyncio.CancelledError:
                pass

        self._active_task = asyncio.create_task(self._execute(goal))

    async def _execute(self, goal: DroneGoal):
        """
        Main execution coroutine for a single goal.
        """
        pos = self.bridge.position
        if not pos:
            logger.error("No GPS position available.")
            return

        # Special cases
        if goal.action == "return":
            await self.bridge.return_to_launch()
            return
        if goal.action == "land":
            await self.bridge.land()
            return

        # Generate waypoints
        waypoints = self.global_planner.generate_waypoints(
            goal,
            current_lat=pos.lat_deg,
            current_lon=pos.lon_deg,
            current_alt_m=pos.rel_alt_m,
        )

        if not waypoints:
            logger.warning("No waypoints generated for goal: %s", goal)
            return

        logger.info("Executing %d waypoint(s) for goal: %s", len(waypoints), goal.description)

        # Takeoff if on ground
        if not self.bridge.is_armed:
            await self.bridge.arm()
            await asyncio.sleep(1.0)

        if pos.rel_alt_m < 2.0:
            alt = goal.target_alt_m or 20.0
            await self.bridge.takeoff(altitude_m=alt)
            await asyncio.sleep(5.0)  # Wait for takeoff

        # Hover action: just set altitude and hold
        if goal.action == "hover":
            target_alt = goal.target_alt_m or 20.0
            current_pos = self.bridge.position
            if current_pos:
                await self.bridge.fly_to_position(
                    current_pos.lat_deg, current_pos.lon_deg, target_alt
                )
            return

        # Fly through waypoints with reactive avoidance
        await self.bridge.start_offboard()

        for wp_idx, wp in enumerate(waypoints):
            logger.info("Navigating to waypoint %d/%d: %s", wp_idx + 1, len(waypoints), wp.label)
            await self._navigate_to_waypoint(wp)

            if wp.loiter_s > 0:
                logger.info("Loitering at waypoint for %.1f s", wp.loiter_s)
                await asyncio.sleep(min(wp.loiter_s, 30.0))  # Cap loiter for safety

        await self.bridge.stop_offboard()
        logger.info("Goal complete: %s", goal.description)

    async def _navigate_to_waypoint(self, wp: Waypoint):
        """
        Reactive navigation loop to a single waypoint.
        Runs at CONTROL_HZ, blending attraction + repulsion velocities.
        """
        interval = 1.0 / self.CONTROL_HZ
        timeout = 120.0  # Seconds before giving up on a waypoint
        t_start = time.monotonic()

        while True:
            pos = self.bridge.position
            if not pos:
                await asyncio.sleep(interval)
                continue

            dist = haversine_m(pos.lat_deg, pos.lon_deg, wp.lat, wp.lon)
            if dist < self.WAYPOINT_ARRIVAL_RADIUS_M:
                logger.info("Waypoint reached (dist=%.1f m)", dist)
                break

            if time.monotonic() - t_start > timeout:
                logger.warning("Waypoint timeout — moving on.")
                break

            # Get latest perception
            obs_map = self.perception.latest_obstacle_map
            if obs_map is None:
                # No perception yet — use zero velocity (hold)
                await self.bridge.send_velocity_ned(0.0, 0.0, 0.0)
                await asyncio.sleep(interval)
                continue

            vN, vE, vD = self.local_planner.compute_velocity(
                obstacle_map=obs_map,
                target_lat=wp.lat,
                target_lon=wp.lon,
                current_lat=pos.lat_deg,
                current_lon=pos.lon_deg,
            )

            await self.bridge.send_velocity_ned(vN, vE, vD)
            await asyncio.sleep(interval)
