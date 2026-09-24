"""Standalone Piper forward kinematics and damped least-squares IK."""

import numpy as np

from .camera import rotation_from_quaternion
from .constants import (JOINT_AXIS, JOINT_MAX, JOINT_MIN, JOINT_RPY, JOINT_XYZ,
                        ROBOT_BASE_WXYZ, ROBOT_BASE_XYZ)


def _rpy_matrix(rpy):
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return np.array([
        [cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
        [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
        [-sp, cp*sr, cp*cr],
    ])


def _axis_rotation(axis, theta):
    x, y, z = axis
    c, s = np.cos(theta), np.sin(theta)
    return c*np.eye(3) + (1-c)*np.outer(axis, axis) + s*np.array([
        [0, -z, y], [z, 0, -x], [-y, x, 0],
    ])


class PiperKinematics:
    def __init__(self):
        self.base_rotation = rotation_from_quaternion(ROBOT_BASE_WXYZ)
        self.origin_rotations = [_rpy_matrix(rpy) for rpy in JOINT_RPY]

    def forward(self, q, with_jacobian=False):
        q = np.asarray(q, dtype=np.float64)[:6]
        position = ROBOT_BASE_XYZ.copy()
        rotation = self.base_rotation.copy()
        origins, axes = [], []
        for i in range(6):
            position = position + rotation @ JOINT_XYZ[i]
            rotation = rotation @ self.origin_rotations[i]
            origins.append(position.copy())
            axes.append(rotation @ JOINT_AXIS[i])
            rotation = rotation @ _axis_rotation(JOINT_AXIS[i], q[i])
        if not with_jacobian:
            return position, rotation
        jacobian = np.zeros((6, 6))
        for i, (origin, axis) in enumerate(zip(origins, axes)):
            jacobian[:3, i] = np.cross(axis, position - origin)
            jacobian[3:, i] = axis
        return position, rotation, jacobian

    def step(self, q, target_xyz, target_rotation, max_delta=0.04):
        q = np.asarray(q, dtype=np.float64)[:6]
        position, rotation, jacobian = self.forward(q, with_jacobian=True)
        linear_error = np.asarray(target_xyz) - position
        # Column cross-products give a bounded orientation error for small
        # corrections. The HOME joint pose seeds the top-down branch first.
        angular_error = 0.5 * sum(
            np.cross(rotation[:, i], target_rotation[:, i]) for i in range(3)
        )
        error = np.concatenate((np.clip(linear_error, -0.06, 0.06),
                                np.clip(angular_error, -0.20, 0.20)))
        damping = 0.08
        system = jacobian @ jacobian.T + damping*damping*np.eye(6)
        dq = jacobian.T @ np.linalg.solve(system, error)
        dq = np.clip(dq, -max_delta, max_delta)
        target_q = np.clip(q + dq, JOINT_MIN + 0.01, JOINT_MAX - 0.01)
        pose_error = float(np.linalg.norm(linear_error))
        orientation_error = float(np.linalg.norm(angular_error))
        return target_q, pose_error, orientation_error


def top_down_rotation(long_axis_xy=None, tilt_toward_neg_x_deg=0.0):
    """Gripper local X follows the object; local Z points mostly down."""
    if long_axis_xy is None:
        x = np.array([1.0, 0.0, 0.0])
    else:
        xy = np.asarray(long_axis_xy, dtype=np.float64)
        xy = xy / max(np.linalg.norm(xy), 1e-9)
        x = np.array([xy[0], xy[1], 0.0])
        if x[0] < 0:
            x = -x
    z = np.array([0.0, 0.0, -1.0])
    y = np.cross(z, x)
    rotation = np.column_stack((x, y, z))
    if tilt_toward_neg_x_deg:
        angle = np.deg2rad(tilt_toward_neg_x_deg)
        c, s = np.cos(angle), np.sin(angle)
        world_y_tilt = np.array([[c, 0.0, s],
                                 [0.0, 1.0, 0.0],
                                 [-s, 0.0, c]])
        rotation = world_y_tilt @ rotation
    return rotation


def bottle_side_rotation(down_angle_deg):
    """Enter the bottle from +X, with the jaw opening across its narrow Y side.

    Local Z is the finger/approach direction. A modest downward pitch keeps
    the pose reachable at both ends of the bottle's spawn range.
    """
    angle = np.deg2rad(down_angle_deg)
    z = np.array([-np.cos(angle), 0.0, -np.sin(angle)])
    y = np.array([0.0, 1.0, 0.0])
    x = np.cross(y, z)
    return np.column_stack((x, y, z))


def bottle_carry_rotation(target_xy, down_angle_deg):
    """Keep a carried bottle nearly upright while following the arm's reach."""
    radial_xy = np.asarray(target_xy, dtype=np.float64) - ROBOT_BASE_XYZ[:2]
    radial_xy /= max(np.linalg.norm(radial_xy), 1e-9)
    angle = np.deg2rad(down_angle_deg)
    z = np.array([radial_xy[0] * np.cos(angle),
                  radial_xy[1] * np.cos(angle), -np.sin(angle)])
    y = np.array([radial_xy[1], -radial_xy[0], 0.0])
    x = np.cross(y, z)
    return np.column_stack((x, y, z))
