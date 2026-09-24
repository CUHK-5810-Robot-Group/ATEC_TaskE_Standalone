"""Pinhole back-projection for Task E's fixed external camera."""

import numpy as np

from .constants import CAMERA_WXYZ, CAMERA_XYZ, FOCAL_LENGTH_MM, HORIZONTAL_APERTURE_MM


def rotation_from_quaternion(wxyz):
    q = np.asarray(wxyz, dtype=np.float64)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)],
        [2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)],
        [2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)],
    ])


class FixedCamera:
    def __init__(self, width, height):
        self.width = int(width)
        self.height = int(height)
        self.fx = FOCAL_LENGTH_MM / HORIZONTAL_APERTURE_MM * self.width
        self.fy = self.fx
        self.cx = (self.width - 1) / 2.0
        self.cy = (self.height - 1) / 2.0
        self.rotation = rotation_from_quaternion(CAMERA_WXYZ)

    def point_cloud(self, depth, stride=3):
        """Return sampled world points, matching pixel row/column arrays.

        Isaac Lab `depth` is optical-axis depth in metres. Its `world` camera
        convention uses +X forward, +Z up, and -Y image-right.
        """
        d = np.squeeze(np.asarray(depth))
        if d.shape != (self.height, self.width):
            raise ValueError(f"video_depth has shape {d.shape}; expected {(self.height, self.width)}")
        rows, cols = np.mgrid[0:self.height:stride, 0:self.width:stride]
        z = d[::stride, ::stride].astype(np.float64, copy=False)
        valid = np.isfinite(z) & (z > 0.1) & (z < 3.0)
        zz = z[valid]
        uu = cols[valid]
        vv = rows[valid]
        camera_points = np.stack((
            zz,
            -(uu - self.cx) * zz / self.fx,
            -(vv - self.cy) * zz / self.fy,
        ), axis=1)
        world_points = camera_points @ self.rotation.T + CAMERA_XYZ
        return world_points, vv, uu
