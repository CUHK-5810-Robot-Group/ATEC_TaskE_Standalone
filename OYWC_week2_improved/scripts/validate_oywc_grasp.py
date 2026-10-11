#!/usr/bin/env python3
"""Deterministic offline checks for the OYWC RGB-D grasp planner.

The checks use synthetic RGB-D arrays so they run without Isaac Lab, a GPU or
the private base repository.  They specifically guard the failure mode that
motivated this branch: an upright bottle must be grasped across its narrow
horizontal body axis at the body's vertical middle.
"""

from __future__ import annotations

import math
import sys

import numpy as np

from demo.oywc_rgbd_grasp import (
    CameraIntrinsics,
    CenterlineNarrowEdgeGraspPlanner,
    GraspPhase,
    GraspStateMachine,
    ObjectObservation,
    RigidTransform,
    TemporalPoseFilter,
)


def synthetic_bottle() -> ObjectObservation:
    mask = np.zeros((180, 240), dtype=bool)
    mask[45:145, 105:135] = True
    depth = np.full(mask.shape, np.nan, dtype=np.float64)
    rng = np.random.default_rng(5810)
    depth[mask] = 0.72 + rng.normal(0.0, 0.002, size=np.count_nonzero(mask))
    # A few invalid and grossly wrong pixels must not move the median target.
    ys, xs = np.where(mask)
    depth[ys[:4], xs[:4]] = 2.0
    depth[ys[4:8], xs[4:8]] = np.nan
    return ObjectObservation("mustard_bottle", mask, depth, confidence=0.98)


def main() -> int:
    camera = CameraIntrinsics(fx=400.0, fy=400.0, cx=120.0, cy=90.0)
    planner = CenterlineNarrowEdgeGraspPlanner(camera, RigidTransform(np.eye(4)))
    pose = planner.plan(synthetic_bottle())

    # The bottle body is centered at y=95, and central trimming should leave
    # the estimate near the body center rather than at the shoulder or cap.
    assert abs(pose.body_center_uv[0] - 119.5) < 2.0, pose
    assert abs(pose.body_center_uv[1] - 94.5) < 4.0, pose
    assert abs(pose.position_base[2] - 0.72) < 0.01, pose
    # Jaws close along the narrow (horizontal) edge: yaw is approximately 0.
    yaw_mod = abs(((pose.yaw + math.pi / 2) % math.pi) - math.pi / 2)
    assert yaw_mod < 0.15, (pose.yaw, pose.closing_axis_base)
    assert "body-middle" in pose.reason

    filt = TemporalPoseFilter(max_jump_m=0.04)
    assert filt.update(pose) is not None
    jumped = pose.__class__(
        label=pose.label,
        position_base=pose.position_base + np.array([0.1, 0.0, 0.0]),
        closing_axis_base=pose.closing_axis_base,
        approach_axis_base=pose.approach_axis_base,
        yaw=pose.yaw,
        body_center_uv=pose.body_center_uv,
        width_m=pose.width_m,
        confidence=pose.confidence,
        reason=pose.reason,
    )
    assert filt.update(jumped) is None

    sm = GraspStateMachine(max_retries=1)
    assert sm.phase == GraspPhase.APPROACH
    sm.advance(reached=True)
    sm.advance(reached=True)
    sm.advance(reached=False, object_followed=True)
    assert sm.phase == GraspPhase.TRANSPORT
    sm.advance(reached=True)
    sm.advance(reached=True, released=True)
    sm.advance(reached=True)
    assert sm.phase == GraspPhase.DONE

    retry = GraspStateMachine(max_retries=1)
    retry.advance(reached=True)
    retry.advance(reached=True)
    retry.advance(reached=False, object_followed=False)
    assert retry.phase == GraspPhase.RETRY
    retry.advance(reached=True)
    assert retry.phase == GraspPhase.APPROACH
    print("OYWC RGB-D checks passed: body-middle horizontal bottle grasp, outlier rejection, stale-jump gate, finite phases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
