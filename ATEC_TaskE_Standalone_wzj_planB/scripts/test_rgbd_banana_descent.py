"""Small state-machine checks for a contact-limited banana descent."""

import unittest

import numpy as np

from demo.rgbd_control.constants import (
    BANANA_FORWARD_TILT_DEG, DEFAULT_Q, GRIPPER_CLOSE, GRIPPER_OPEN, TABLE_TOP_Z,
)
from demo.rgbd_control.perception import ObjectEstimate
from demo.solution_rgbd import _SingleEnvironmentPolicy


class _FixedKinematics:
    def __init__(self, xyz, rotation_error=0.1):
        self.xyz = np.asarray(xyz, dtype=float)
        self.rotation_error = rotation_error
        self.last_target = None
        self.last_max_delta = None

    def forward(self, q):
        return self.xyz.copy(), np.eye(3)

    def step(self, q, xyz, rotation, max_delta=0.04):
        self.last_target = np.asarray(xyz).copy()
        self.last_max_delta = max_delta
        return np.asarray(q).copy(), float(np.linalg.norm(xyz - self.xyz)), self.rotation_error


class BananaDescentTest(unittest.TestCase):
    def _stalled_step(self, xyz, rotation_error=0.1, no_progress=None,
                      state_steps=100):
        policy = _SingleEnvironmentPolicy()
        policy.state = "DESCEND"
        policy.state_steps = state_steps
        policy.object_ptr = 2
        policy.grasp_xy = np.array([1.0, 0.06])
        policy.grasp_rotation = np.eye(3)
        policy.descend_lowest_z = xyz[2]
        policy.descend_no_progress = (
            policy._BANANA_STALL_STEPS - 1 if no_progress is None else no_progress)
        policy.kinematics = _FixedKinematics(xyz, rotation_error)
        q = DEFAULT_Q.copy()
        q[0] += 0.1  # Keep the episode-reset detector inactive.
        q[6:8] = GRIPPER_OPEN
        proprio = np.zeros(24)
        proprio[:8] = q - DEFAULT_Q
        action, stopped = policy.step(proprio, None, None)
        return policy, action, stopped

    def test_aligned_contact_stall_closes(self):
        grasp_z = TABLE_TOP_Z + 0.09
        # The observed plateau now sits about 6.8 cm above the lower target.
        policy, action, stopped = self._stalled_step([1.0, 0.06, grasp_z + 0.0678])
        self.assertEqual(policy.state, "CLOSE")
        self.assertFalse(stopped)
        np.testing.assert_allclose(action[6:8], GRIPPER_CLOSE)

    def test_stall_is_not_blocked_by_target_height_error(self):
        grasp_z = TABLE_TOP_Z + 0.09
        policy, action, stopped = self._stalled_step([1.0, 0.06, grasp_z + 0.090])
        self.assertEqual(policy.state, "CLOSE")
        self.assertFalse(stopped)
        np.testing.assert_allclose(action[6:8], GRIPPER_CLOSE)

    def test_stall_closes_without_xy_gate(self):
        grasp_z = TABLE_TOP_Z + 0.09
        policy, action, stopped = self._stalled_step([1.02, 0.06, grasp_z + 0.0678])
        self.assertEqual(policy.state, "CLOSE")
        self.assertFalse(stopped)
        np.testing.assert_allclose(action[6:8], GRIPPER_CLOSE)

    def test_timeout_without_stall_recovers_open(self):
        grasp_z = TABLE_TOP_Z + 0.09
        policy, action, stopped = self._stalled_step(
            [1.0, 0.06, grasp_z + 0.0678], no_progress=0,
            state_steps=_SingleEnvironmentPolicy._BANANA_DESCEND_TIMEOUT)
        self.assertEqual(policy.state, "RECOVER")
        self.assertFalse(stopped)
        np.testing.assert_allclose(action[6:8], GRIPPER_OPEN)

    def test_faster_descent_stays_above_slow_zone(self):
        policy, _, _ = self._stalled_step(
            [1.0, 0.06, TABLE_TOP_Z + 0.21], no_progress=0)
        self.assertEqual(policy.state, "DESCEND")
        self.assertAlmostEqual(
            policy.kinematics.last_target[2], TABLE_TOP_Z + 0.203)
        self.assertAlmostEqual(policy.kinematics.last_max_delta, 0.018)

    def test_descent_slows_near_table(self):
        policy, _, _ = self._stalled_step(
            [1.0, 0.06, TABLE_TOP_Z + 0.17], no_progress=0)
        self.assertEqual(policy.state, "DESCEND")
        self.assertAlmostEqual(
            policy.kinematics.last_target[2], TABLE_TOP_Z + 0.166)
        self.assertAlmostEqual(policy.kinematics.last_max_delta, 0.012)

    def test_banana_fingertips_tilt_toward_negative_x(self):
        policy = _SingleEnvironmentPolicy()
        policy.object_ptr = 2
        policy.estimate = ObjectEstimate(
            object_id=3, xy=np.array([0.97, 0.06]),
            long_axis_xy=np.array([1.0, 0.0]), aspect_ratio=3.0,
            point_count=100, mean_rgb=np.zeros(3))
        policy._grasp_plan()
        rotation = policy.grasp_rotation
        angle = np.deg2rad(BANANA_FORWARD_TILT_DEG)
        np.testing.assert_allclose(
            rotation[:, 2], [-np.sin(angle), 0.0, -np.cos(angle)], atol=1e-8)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)


if __name__ == "__main__":
    unittest.main()
