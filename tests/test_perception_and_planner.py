"""
tests/test_perception_and_planner.py
──────────────────────────────────────
Unit tests for PerceptionLoop and WaypointPlanner.
All tests use mock backends — no camera or drone hardware required.
"""

import asyncio
import pytest
import math

from perception.perception_loop import PerceptionLoop, ObstacleMap, Detection
from planner.waypoint_planner import (
    GlobalPlanner,
    LocalPlanner,
    WaypointPlanner,
    haversine_m,
    offset_gps,
)
from drone.command_interpreter import DroneGoal


# ── Coordinate helpers ────────────────────────────────────────────────────────

class TestCoordinateHelpers:

    def test_haversine_same_point(self):
        assert haversine_m(37.0, -122.0, 37.0, -122.0) == pytest.approx(0.0, abs=1e-3)

    def test_haversine_100m_north(self):
        lat, lon = 37.0, -122.0
        lat2, lon2 = offset_gps(lat, lon, 100.0, 0.0)
        dist = haversine_m(lat, lon, lat2, lon2)
        assert dist == pytest.approx(100.0, abs=0.5)

    def test_offset_east(self):
        lat, lon = 37.0, -122.0
        lat2, lon2 = offset_gps(lat, lon, 0.0, 50.0)
        assert lat2 == pytest.approx(lat, abs=1e-4)
        assert lon2 > lon  # East = higher longitude


# ── GlobalPlanner tests ────────────────────────────────────────────────────────

class TestGlobalPlanner:

    def setup_method(self):
        self.planner = GlobalPlanner()
        self.lat, self.lon, self.alt = 37.7749, -122.4194, 0.0

    def test_hover_returns_single_waypoint(self):
        goal = DroneGoal(action="hover", description="hover", target_alt_m=20.0)
        wps = self.planner.generate_waypoints(goal, self.lat, self.lon, self.alt)
        assert len(wps) == 1
        assert wps[0].alt_m == pytest.approx(20.0)
        assert wps[0].loiter_s > 100

    def test_fly_to_north_generates_north_waypoint(self):
        goal = DroneGoal(action="fly_to", description="fly north", region_description="north", target_alt_m=30.0)
        wps = self.planner.generate_waypoints(goal, self.lat, self.lon, self.alt)
        assert len(wps) == 1
        assert wps[0].lat > self.lat  # North = higher latitude

    def test_survey_generates_multiple_waypoints(self):
        goal = DroneGoal(action="survey_area", description="survey north", region_description="north", target_alt_m=40.0)
        wps = self.planner.generate_waypoints(goal, self.lat, self.lon, self.alt)
        assert len(wps) > 2

    def test_orbit_generates_closed_loop(self):
        goal = DroneGoal(action="orbit", description="orbit", target_alt_m=25.0)
        wps = self.planner.generate_waypoints(goal, self.lat, self.lon, self.alt)
        # First and last waypoints should be close (closed orbit)
        assert len(wps) >= 8
        dist = haversine_m(wps[0].lat, wps[0].lon, wps[-1].lat, wps[-1].lon)
        assert dist < 5.0

    def test_unknown_direction_defaults_to_north(self):
        goal = DroneGoal(action="fly_to", description="go somewhere vague", region_description="somewhere vague", target_alt_m=30.0)
        wps = self.planner.generate_waypoints(goal, self.lat, self.lon, self.alt)
        assert wps[0].lat > self.lat  # Should default to north


# ── LocalPlanner tests ─────────────────────────────────────────────────────────

class TestLocalPlanner:

    def setup_method(self):
        self.planner = LocalPlanner(
            safety_radius_m=5.0,
            max_speed_m_s=3.0,
            attraction_gain=0.5,
            repulsion_gain=2.0,
        )
        self.lat, self.lon = 37.7749, -122.4194
        self.target_lat, self.target_lon = offset_gps(self.lat, self.lon, 100.0, 0.0)

    def _make_obs_map(self, nearest_m=100.0, bearing_deg=0.0):
        return ObstacleMap(
            timestamp=0.0,
            frame=None,
            depth_map=None,
            detections=[],
            nearest_obstacle_m=nearest_m,
            nearest_obstacle_bearing_deg=bearing_deg,
        )

    def test_no_obstacle_moves_toward_target(self):
        obs = self._make_obs_map(nearest_m=100.0)
        vN, vE, vD = self.planner.compute_velocity(
            obs, self.target_lat, self.target_lon, self.lat, self.lon
        )
        assert vN > 0  # Target is north → positive vN
        assert vD == 0.0

    def test_obstacle_ahead_creates_repulsion(self):
        obs = self._make_obs_map(nearest_m=2.0, bearing_deg=0.0)  # Dead ahead
        vN_clear, _, _ = self.planner.compute_velocity(
            self._make_obs_map(100.0), self.target_lat, self.target_lon, self.lat, self.lon
        )
        vN_obs, _, _ = self.planner.compute_velocity(
            obs, self.target_lat, self.target_lon, self.lat, self.lon
        )
        # Repulsion should reduce or reverse forward velocity
        assert vN_obs < vN_clear

    def test_velocity_clamped_to_max_speed(self):
        obs = self._make_obs_map(nearest_m=100.0)
        # Very close target to force high attraction
        close_target = offset_gps(self.lat, self.lon, 1000.0, 0.0)
        vN, vE, _ = self.planner.compute_velocity(
            obs, close_target[0], close_target[1], self.lat, self.lon
        )
        speed = math.sqrt(vN ** 2 + vE ** 2)
        assert speed <= self.planner.max_speed_m_s + 0.01


# ── PerceptionLoop tests ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_perception_loop_starts_and_stops():
    loop = PerceptionLoop(config={"camera_backend": "mock", "fps": 10, "mock_detections": True})
    await loop.start()
    await asyncio.sleep(0.3)  # Let it run a few frames
    assert loop.latest_obstacle_map is not None
    await loop.stop()


@pytest.mark.asyncio
async def test_perception_loop_publishes_obstacle_map():
    loop = PerceptionLoop(config={"camera_backend": "mock", "fps": 20, "mock_detections": True})
    await loop.start()
    await asyncio.sleep(0.2)
    obs = await loop.get_obstacle_map(timeout=0.2)
    assert obs is not None
    assert obs.nearest_obstacle_m >= 0
    await loop.stop()


@pytest.mark.asyncio
async def test_perception_loop_frame_is_numpy():
    import numpy as np
    loop = PerceptionLoop(config={"camera_backend": "mock", "fps": 10, "mock_detections": True})
    await loop.start()
    await asyncio.sleep(0.2)
    frame = loop.latest_frame
    assert isinstance(frame, np.ndarray)
    assert frame.ndim == 3
    await loop.stop()
