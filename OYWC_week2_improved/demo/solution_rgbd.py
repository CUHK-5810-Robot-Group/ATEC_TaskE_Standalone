"""Observation-only RGB-D controller for the OYWC week-two experiment.

The original ACT entry predicts a fixed action sequence from RGB.  This
adapter keeps the same eight-dimensional action contract, but recomputes a
grasp target from every fresh RGB-D observation.  A repository-specific Piper
IK function is injected at construction time, so this file can be tested
without importing Isaac Lab.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from .oywc_rgbd_grasp import (
    CameraIntrinsics,
    CenterlineNarrowEdgeGraspPlanner,
    GraspPhase,
    GraspPose,
    GraspStateMachine,
    ObjectObservation,
    RigidTransform,
    TemporalPoseFilter,
)


Segmenter = Callable[[Mapping[str, Any]], Sequence[ObjectObservation]]
IKAdapter = Callable[[np.ndarray, float, float, Mapping[str, Any]], np.ndarray]


class EpisodeStatus(str, Enum):
    RUNNING = "running"
    COMPLETE = "complete"
    GIVE_UP = "give_up"


@dataclass(frozen=True)
class RGBDControllerConfig:
    """Limits shared by all runs, expressed in the environment's action units."""

    intrinsics: CameraIntrinsics
    camera_to_base: RigidTransform
    basket_position_base: np.ndarray
    approach_height_m: float = 0.10
    lift_height_m: float = 0.12
    release_height_m: float = 0.08
    reach_tolerance_m: float = 0.018
    max_action_delta: float = 0.10
    max_missing_frames: int = 5
    max_retries_per_object: int = 2

    def __post_init__(self) -> None:
        basket = np.asarray(self.basket_position_base, dtype=np.float64)
        if basket.shape != (3,) or not np.all(np.isfinite(basket)):
            raise ValueError("basket_position_base must be a finite 3-vector")
        object.__setattr__(self, "basket_position_base", basket)


class ObservationOnlyRGBDController:
    """Three-object finite-state policy with body-middle bottle grasping.

    The controller deliberately uses only standard observation data.  The
    segmenter produces masks and depth; the injected IK adapter is responsible
    for converting a base-frame target and gripper opening to eight actions.
    """

    ORDER = ("sugar_box", "mustard_bottle", "banana")

    def __init__(
        self,
        config: RGBDControllerConfig,
        segmenter: Segmenter,
        ik_adapter: IKAdapter,
    ) -> None:
        self.config = config
        self.segmenter = segmenter
        self.ik_adapter = ik_adapter
        self.planner = CenterlineNarrowEdgeGraspPlanner(
            config.intrinsics, config.camera_to_base
        )
        self.filter = TemporalPoseFilter()
        self._state = GraspStateMachine(max_retries=config.max_retries_per_object)
        self._last_action = np.zeros(8, dtype=np.float32)
        self._pose: Optional[GraspPose] = None
        self._object_index = 0
        self._status = EpisodeStatus.RUNNING
        self._missing_frames = 0

    @property
    def phase(self) -> GraspPhase:
        return self._state.phase

    @property
    def status(self) -> EpisodeStatus:
        return self._status

    @property
    def current_object(self) -> Optional[str]:
        if self._object_index >= len(self.ORDER):
            return None
        return self.ORDER[self._object_index]

    def reset(self) -> None:
        self.filter.reset()
        self._state.reset()
        self._last_action.fill(0.0)
        self._pose = None
        self._object_index = 0
        self._status = EpisodeStatus.RUNNING
        self._missing_frames = 0

    @staticmethod
    def _image_payload(obs: Mapping[str, Any]) -> Mapping[str, Any]:
        image = obs.get("image", obs)
        if not isinstance(image, Mapping):
            return {"raw": obs}
        rgb = image.get("video_rgb", image.get("rgb"))
        depth = image.get("video_depth", image.get("depth"))
        return {"rgb": rgb, "depth": depth, "raw": obs}

    def _hold(self, obs: Mapping[str, Any]) -> np.ndarray:
        """Hold the measured action instead of driving on stale perception."""
        proprio = obs.get("proprio")
        if proprio is not None:
            values = np.asarray(proprio, dtype=np.float32).reshape(-1)
            if values.size >= 8 and np.all(np.isfinite(values[:8])):
                self._last_action = values[:8].copy()
        return self._last_action.copy()

    def _select_pose(self, obs: Mapping[str, Any]) -> Optional[GraspPose]:
        candidates = list(self.segmenter(self._image_payload(obs)))
        target = self.current_object
        selected = next((item for item in candidates if item.label == target), None)
        if selected is None:
            self._missing_frames += 1
            return None
        self._missing_frames = 0
        return self.filter.update(self.planner.plan(selected))

    def _target_for_phase(self, pose: GraspPose) -> tuple[np.ndarray, float]:
        target = pose.position_base.copy()
        if self.phase == GraspPhase.APPROACH:
            target[2] += self.config.approach_height_m
            return target, pose.width_m + 0.015
        if self.phase == GraspPhase.CLOSE:
            return target, pose.width_m
        if self.phase == GraspPhase.LIFT:
            target[2] += self.config.lift_height_m
            return target, pose.width_m
        if self.phase == GraspPhase.TRANSPORT:
            target = self.config.basket_position_base.copy()
            target[2] += self.config.release_height_m
            return target, pose.width_m
        if self.phase == GraspPhase.RELEASE:
            target = self.config.basket_position_base.copy()
            target[2] += self.config.release_height_m
            return target, pose.width_m + 0.025
        target = self.config.basket_position_base.copy()
        target[2] += self.config.release_height_m + 0.08
        return target, pose.width_m + 0.025

    def _advance(self, obs: Mapping[str, Any], pose: GraspPose) -> None:
        """Advance only on explicit feedback supplied by the host adapter."""
        feedback = obs.get("oywc_feedback", {})
        reached = bool(feedback.get("target_reached", False))
        followed = bool(feedback.get("object_followed", True))
        released = bool(feedback.get("released", False))
        previous = self.phase
        if self.phase == GraspPhase.CLOSE:
            # Closing is a command state.  It must be acknowledged by the
            # adapter instead of being advanced solely by elapsed time.
            reached = reached and bool(feedback.get("gripper_closed", False))
        self._state.advance(
            reached=reached, object_followed=followed, released=released
        )
        if previous == GraspPhase.RELEASE and self.phase == GraspPhase.RETREAT:
            self._object_index += 1
            self._state.reset()
            self.filter.reset()
            if self._object_index >= len(self.ORDER):
                self._status = EpisodeStatus.COMPLETE
        if self.phase == GraspPhase.DONE:
            self._status = EpisodeStatus.GIVE_UP

    def predict(self, obs: Mapping[str, Any], current_score: float = 0.0) -> dict[str, Any]:
        del current_score
        if self._status != EpisodeStatus.RUNNING:
            return {
                "action": self._hold(obs),
                "giveup": self._status == EpisodeStatus.GIVE_UP,
                "oywc_phase": self.phase.value,
                "oywc_object": self.current_object,
            }

        try:
            pose = self._select_pose(obs)
        except (ValueError, TypeError, np.linalg.LinAlgError):
            pose = None
        if pose is None:
            action = self._hold(obs)
            if self._missing_frames > self.config.max_missing_frames:
                self._state.phase = GraspPhase.RETRY
            return {
                "action": action,
                "giveup": False,
                "oywc_phase": self.phase.value,
                "oywc_object": self.current_object,
            }

        self._pose = pose
        target, opening = self._target_for_phase(pose)
        try:
            action = np.asarray(
                self.ik_adapter(target, pose.yaw, opening, obs), dtype=np.float32
            ).reshape(-1)
        except Exception:
            action = self._hold(obs)
        if action.size != 8 or not np.all(np.isfinite(action)):
            action = self._hold(obs)
        else:
            delta = np.clip(
                action - self._last_action,
                -self.config.max_action_delta,
                self.config.max_action_delta,
            )
            self._last_action = np.clip(
                self._last_action + delta, -1.0, 1.0
            ).astype(np.float32)

        self._advance(obs, pose)
        return {
            "action": self._last_action.copy(),
            "giveup": self._status == EpisodeStatus.GIVE_UP,
            "oywc_phase": self.phase.value,
            "oywc_object": self.current_object,
            "grasp_reason": pose.reason,
        }


__all__ = [
    "EpisodeStatus",
    "RGBDControllerConfig",
    "ObservationOnlyRGBDController",
]
