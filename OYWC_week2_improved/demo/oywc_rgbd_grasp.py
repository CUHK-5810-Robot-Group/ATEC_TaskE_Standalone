"""RGB-D geometry and guarded grasp planning for the OYWC Task-E branch.

This module deliberately contains no Isaac/PhysX calls.  It consumes only an
instance mask, a depth image and calibrated camera parameters, so it can be
unit-tested away from the simulator and then connected to the normal
``AlgSolution.predicts`` adapter.  The important geometric choice is that the
gripper closes along the *minor* image principal axis.  For an upright
mustard bottle this is the horizontal axis through the middle of the body,
which avoids the brittle fixed-pose grasp used by the baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics; depth is the optical-axis distance in metres."""

    fx: float
    fy: float
    cx: float
    cy: float
    depth_scale: float = 1.0

    def backproject(self, pixels_uv: np.ndarray, depth: np.ndarray) -> np.ndarray:
        uv = np.asarray(pixels_uv, dtype=np.float64)
        z = np.asarray(depth, dtype=np.float64) * self.depth_scale
        x = (uv[:, 0] - self.cx) * z / self.fx
        y = (uv[:, 1] - self.cy) * z / self.fy
        return np.column_stack((x, y, z))


@dataclass(frozen=True)
class RigidTransform:
    """A camera-to-base rigid transform represented as a 4x4 matrix."""

    matrix: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        m = np.asarray(self.matrix, dtype=np.float64)
        if m.shape != (4, 4) or not np.all(np.isfinite(m)):
            raise ValueError("transform must be a finite 4x4 matrix")
        object.__setattr__(self, "matrix", m)

    @property
    def rotation(self) -> np.ndarray:
        return self.matrix[:3, :3]

    @property
    def translation(self) -> np.ndarray:
        return self.matrix[:3, 3]

    def points(self, xyz_camera: np.ndarray) -> np.ndarray:
        xyz = np.asarray(xyz_camera, dtype=np.float64)
        return xyz @ self.rotation.T + self.translation

    def direction(self, vector_camera: Sequence[float]) -> np.ndarray:
        return self.rotation @ np.asarray(vector_camera, dtype=np.float64)


@dataclass
class ObjectObservation:
    """A mask/depth observation reduced to robust 3-D point statistics."""

    label: str
    mask: np.ndarray
    depth: np.ndarray
    confidence: float = 1.0
    bbox: Optional[tuple[int, int, int, int]] = None

    def pixels(self, min_depth: float = 0.05, max_depth: float = 3.0) -> np.ndarray:
        mask = np.asarray(self.mask, dtype=bool)
        depth = np.asarray(self.depth, dtype=np.float64)
        if mask.shape != depth.shape or mask.ndim != 2:
            raise ValueError("mask and depth must be two-dimensional arrays with equal shape")
        valid = mask & np.isfinite(depth) & (depth >= min_depth) & (depth <= max_depth)
        y, x = np.nonzero(valid)
        if len(x) < 8:
            raise ValueError("object mask has fewer than eight valid depth pixels")
        return np.column_stack((x, y))

    def robust_depth(self, pixels: np.ndarray) -> np.ndarray:
        d = np.asarray(self.depth, dtype=np.float64)[pixels[:, 1], pixels[:, 0]]
        lo, hi = np.percentile(d, [10.0, 90.0])
        keep = (d >= lo) & (d <= hi)
        if np.count_nonzero(keep) < 8:
            keep = np.ones_like(d, dtype=bool)
        return np.column_stack((pixels[keep], d[keep]))


@dataclass(frozen=True)
class GraspPose:
    """A pose expressed in the base frame and the image point used for it."""

    label: str
    position_base: np.ndarray
    closing_axis_base: np.ndarray
    approach_axis_base: np.ndarray
    yaw: float
    body_center_uv: tuple[float, float]
    width_m: float
    confidence: float
    reason: str

    def __post_init__(self) -> None:
        p = np.asarray(self.position_base, dtype=np.float64)
        c = np.asarray(self.closing_axis_base, dtype=np.float64)
        a = np.asarray(self.approach_axis_base, dtype=np.float64)
        if p.shape != (3,) or c.shape != (3,) or a.shape != (3,):
            raise ValueError("pose vectors must have shape (3,)")
        if not np.all(np.isfinite(np.r_[p, c, a, self.yaw, self.width_m])):
            raise ValueError("pose contains non-finite values")
        object.__setattr__(self, "position_base", p)
        object.__setattr__(self, "closing_axis_base", c / max(np.linalg.norm(c), 1e-12))
        object.__setattr__(self, "approach_axis_base", a / max(np.linalg.norm(a), 1e-12))


class CenterlineNarrowEdgeGraspPlanner:
    """Estimate a body-middle, narrow-edge grasp from RGB-D.

    ``mask`` may contain a cap or a few background pixels.  We trim depth
    outliers and use the central 50% along the principal axis for bottles and
    bananas.  The principal minor axis is the gripper closing direction.  A
    bottle therefore gets a horizontal, body-middle grasp when it is upright,
    while a banana/box follows its observed planar orientation.
    """

    def __init__(self, intrinsics: CameraIntrinsics, camera_to_base: RigidTransform):
        self.intrinsics = intrinsics
        self.camera_to_base = camera_to_base

    @staticmethod
    def _pca_axes(pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xy = pixels.astype(np.float64)
        center = np.mean(xy, axis=0)
        covariance = np.cov(xy - center, rowvar=False, bias=True)
        values, vectors = np.linalg.eigh(covariance)
        order = np.argsort(values)[::-1]
        major = vectors[:, order[0]]
        minor = vectors[:, order[-1]]
        # Make the direction deterministic for temporal filtering and tests.
        if major[1] < 0 or (abs(major[1]) < 1e-9 and major[0] < 0):
            major = -major
        if minor[0] < 0 or (abs(minor[0]) < 1e-9 and minor[1] < 0):
            minor = -minor
        return center, major, minor

    def plan(self, observation: ObjectObservation) -> GraspPose:
        pixels = observation.pixels()
        robust = observation.robust_depth(pixels)
        uv = robust[:, :2]
        depth = robust[:, 2]
        _, major, minor = self._pca_axes(uv)
        projections = (uv - np.mean(uv, axis=0)) @ major
        # Exclude cap/tip and edge pixels.  The middle of a bottle body is
        # safer than its shoulder, and the same rule is stable for bananas.
        q25, q75 = np.percentile(projections, [25.0, 75.0])
        middle = (projections >= q25) & (projections <= q75)
        if np.count_nonzero(middle) < 8:
            middle = np.ones_like(projections, dtype=bool)
        body_uv = uv[middle]
        body_depth = depth[middle]
        body_camera = self.intrinsics.backproject(body_uv, body_depth)
        center_camera = np.median(body_camera, axis=0)
        center_base = self.camera_to_base.points(center_camera[None, :])[0]

        # The closing direction lies in the camera image plane.  The approach
        # is normal to the table (camera z transformed into the base frame),
        # with a deterministic sign chosen toward the table-facing direction.
        closing_camera = np.array([minor[0], minor[1], 0.0], dtype=np.float64)
        closing_base = self.camera_to_base.direction(closing_camera)
        approach_base = self.camera_to_base.direction(np.array([0.0, 0.0, 1.0]))
        if approach_base[2] > 0:
            approach_base = -approach_base
        yaw = float(np.arctan2(closing_base[1], closing_base[0]))
        # Estimate the opening in metres from the reconstructed 3-D points.
        # Using the image span directly would make the same bottle appear
        # wider when it is farther from the camera and would saturate the
        # opening clamp.  This is only a conservative target; it never
        # replaces collision and force limits in the robot adapter.
        closing_camera /= max(np.linalg.norm(closing_camera), 1e-12)
        side_metric = body_camera @ closing_camera
        width_m = float(np.percentile(side_metric, 90) - np.percentile(side_metric, 10))
        width_m = float(np.clip(width_m * 1.25, 0.018, 0.080))
        confidence = float(np.clip(observation.confidence * min(1.0, len(body_uv) / 150.0), 0.0, 1.0))
        if observation.label == "mustard_bottle":
            reason = "body-middle; jaws close along image minor axis (horizontal for upright bottle)"
        else:
            reason = "central robust-depth point; jaws close along mask minor axis"
        return GraspPose(
            label=observation.label,
            position_base=center_base,
            closing_axis_base=closing_base,
            approach_axis_base=approach_base,
            yaw=yaw,
            body_center_uv=(float(np.median(body_uv[:, 0])), float(np.median(body_uv[:, 1]))),
            width_m=width_m,
            confidence=confidence,
            reason=reason,
        )


class TemporalPoseFilter:
    """Reject stale or jumping detections before they reach the controller."""

    def __init__(self, alpha: float = 0.35, max_jump_m: float = 0.06, min_confidence: float = 0.35):
        self.alpha = float(alpha)
        self.max_jump_m = float(max_jump_m)
        self.min_confidence = float(min_confidence)
        self._last: Optional[GraspPose] = None

    def reset(self) -> None:
        self._last = None

    def update(self, pose: GraspPose) -> Optional[GraspPose]:
        if pose.confidence < self.min_confidence:
            return None
        if self._last is None:
            self._last = pose
            return pose
        distance = float(np.linalg.norm(pose.position_base - self._last.position_base))
        if distance > self.max_jump_m:
            return None
        p = (1.0 - self.alpha) * self._last.position_base + self.alpha * pose.position_base
        c = (1.0 - self.alpha) * self._last.closing_axis_base + self.alpha * pose.closing_axis_base
        c /= max(np.linalg.norm(c), 1e-12)
        filtered = GraspPose(
            label=pose.label,
            position_base=p,
            closing_axis_base=c,
            approach_axis_base=pose.approach_axis_base,
            yaw=float(np.arctan2(c[1], c[0])),
            body_center_uv=pose.body_center_uv,
            width_m=pose.width_m,
            confidence=pose.confidence,
            reason=pose.reason,
        )
        self._last = filtered
        return filtered


class GraspPhase(str, Enum):
    APPROACH = "approach"
    CLOSE = "close"
    LIFT = "lift"
    TRANSPORT = "transport"
    RELEASE = "release"
    RETREAT = "retreat"
    RETRY = "retry"
    DONE = "done"


@dataclass
class GraspStateMachine:
    """Finite phase machine used by the adapter, with bounded retries."""

    max_retries: int = 2
    phase: GraspPhase = GraspPhase.APPROACH
    retries: int = 0

    def reset(self) -> None:
        self.phase = GraspPhase.APPROACH
        self.retries = 0

    def advance(self, *, reached: bool, object_followed: bool = True, released: bool = False) -> GraspPhase:
        if self.phase == GraspPhase.APPROACH and reached:
            self.phase = GraspPhase.CLOSE
        elif self.phase == GraspPhase.CLOSE and reached:
            self.phase = GraspPhase.LIFT
        elif self.phase == GraspPhase.LIFT:
            self.phase = GraspPhase.TRANSPORT if object_followed else self._retry()
        elif self.phase == GraspPhase.TRANSPORT and reached:
            self.phase = GraspPhase.RELEASE
        elif self.phase == GraspPhase.RELEASE and released:
            self.phase = GraspPhase.RETREAT
        elif self.phase == GraspPhase.RETREAT and reached:
            self.phase = GraspPhase.DONE
        elif self.phase == GraspPhase.RETRY and reached:
            # Refresh RGB-D and re-enter the guarded approach sequence.  A
            # retry must never resume at CLOSE or LIFT with a stale target.
            self.phase = GraspPhase.APPROACH
        return self.phase

    def _retry(self) -> GraspPhase:
        if self.retries < self.max_retries:
            self.retries += 1
            return GraspPhase.RETRY
        return GraspPhase.DONE


__all__ = [
    "CameraIntrinsics",
    "RigidTransform",
    "ObjectObservation",
    "GraspPose",
    "CenterlineNarrowEdgeGraspPlanner",
    "TemporalPoseFilter",
    "GraspPhase",
    "GraspStateMachine",
]
