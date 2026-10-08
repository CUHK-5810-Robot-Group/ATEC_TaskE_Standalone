"""A drop-in strategy selector for the OYWC branch.

The default remains the repository's ACT baseline so existing submissions stay
reproducible.  Setting ``ATEC_TASK_E_STRATEGY=rgbd`` enables the new geometry
planner when a segmenter and a Piper IK adapter are supplied.  Keeping this
selection explicit makes the ACT/RGB-D comparison fair and prevents an
unconfigured perception path from silently changing the score.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

import numpy as np

from .oywc_rgbd_grasp import (
    CameraIntrinsics,
    CenterlineNarrowEdgeGraspPlanner,
    GraspPhase,
    GraspStateMachine,
    ObjectObservation,
    RigidTransform,
    TemporalPoseFilter,
)


Segmenter = Callable[[Mapping[str, Any]], list[ObjectObservation]]
IKAdapter = Callable[[np.ndarray, float, Mapping[str, Any]], np.ndarray]


@dataclass
class RGBDPolicyConfig:
    """Calibration and safety limits shared by all RGB-D episodes."""

    intrinsics: CameraIntrinsics
    camera_to_base: RigidTransform
    action_scale: float = 0.5
    max_action_delta: float = 0.12
    stale_frame_limit: int = 5


class RGBDAdaptivePolicy:
    """Perception -> pose -> finite-phase controller with a conservative hold.

    The policy intentionally does not read simulator objects or PhysX state.
    A repository-specific segmenter supplies masks and a small IK adapter maps
    the base-frame target to Piper's eight joint actions.  If either component
    is absent, the policy holds the latest action and reports ``giveup=False``;
    this is safer than executing a guessed fixed pose.
    """

    ORDER = ("sugar_box", "mustard_bottle", "banana")

    def __init__(
        self,
        config: RGBDPolicyConfig,
        segmenter: Optional[Segmenter] = None,
        ik_adapter: Optional[IKAdapter] = None,
    ) -> None:
        self.config = config
        self.segmenter = segmenter
        self.ik_adapter = ik_adapter
        self.planner = CenterlineNarrowEdgeGraspPlanner(config.intrinsics, config.camera_to_base)
        self.filter = TemporalPoseFilter()
        self.state = GraspStateMachine()
        self._last_action = np.zeros(8, dtype=np.float32)
        self._missing_frames = 0
        self._object_index = 0

    def reset(self) -> None:
        self.filter.reset()
        self.state.reset()
        self._last_action = np.zeros(8, dtype=np.float32)
        self._missing_frames = 0
        self._object_index = 0

    @staticmethod
    def _extract_rgbd(obs: Mapping[str, Any]) -> Mapping[str, Any]:
        image = obs.get("image", obs)
        if isinstance(image, Mapping):
            # The task has used both flat and camera-prefixed observation keys
            # in different checkpoints; accept either without changing data.
            rgb = image.get("video_rgb", image.get("rgb"))
            depth = image.get("video_depth", image.get("depth"))
            if rgb is not None and depth is not None:
                return {"rgb": rgb, "depth": depth, "raw": obs}
        return {"raw": obs}

    def _hold(self, current: Optional[np.ndarray]) -> np.ndarray:
        if current is not None:
            current = np.asarray(current, dtype=np.float32).reshape(-1)
            if current.size >= 8:
                self._last_action = current[:8].copy()
        return self._last_action.copy()

    def predict(self, obs: Mapping[str, Any], current_score: float = 0.0) -> dict[str, Any]:
        del current_score  # score is not a substitute for visual success feedback
        proprio = obs.get("proprio") if isinstance(obs, Mapping) else None
        current = None
        if proprio is not None:
            current = np.asarray(proprio, dtype=np.float32).reshape(-1)
        if self.segmenter is None or self.ik_adapter is None:
            return {"action": self._hold(current), "giveup": False, "oywc_phase": self.state.phase.value}

        try:
            candidates = self.segmenter(self._extract_rgbd(obs))
        except Exception:
            candidates = []
        wanted = self.ORDER[min(self._object_index, len(self.ORDER) - 1)]
        selected = next((x for x in candidates if x.label == wanted), None)
        if selected is None:
            self._missing_frames += 1
            # Keep executing a fresh target for a short camera gap; after that
            # hold to avoid driving indefinitely on stale perception.
            if self._missing_frames > self.config.stale_frame_limit:
                return {"action": self._hold(current), "giveup": False, "oywc_phase": self.state.phase.value}
            return {"action": self._last_action.copy(), "giveup": False, "oywc_phase": self.state.phase.value}
        self._missing_frames = 0
        pose = self.filter.update(self.planner.plan(selected))
        if pose is None:
            return {"action": self._hold(current), "giveup": False, "oywc_phase": self.state.phase.value}

        try:
            action = np.asarray(self.ik_adapter(pose.position_base, pose.yaw, obs), dtype=np.float32).reshape(-1)
        except Exception:
            return {"action": self._hold(current), "giveup": False, "oywc_phase": self.state.phase.value}
        if action.size != 8 or not np.all(np.isfinite(action)):
            return {"action": self._hold(current), "giveup": False, "oywc_phase": self.state.phase.value}
        delta = np.clip(action - self._last_action, -self.config.max_action_delta, self.config.max_action_delta)
        self._last_action = np.clip(self._last_action + delta, -1.0, 1.0).astype(np.float32)
        return {"action": self._last_action.copy(), "giveup": False, "oywc_phase": self.state.phase.value}


def make_strategy(*, intrinsics: CameraIntrinsics, camera_to_base: RigidTransform, segmenter: Optional[Segmenter] = None, ik_adapter: Optional[IKAdapter] = None) -> RGBDAdaptivePolicy:
    """Create the RGB-D strategy; the environment variable is intentionally explicit."""

    if os.getenv("ATEC_TASK_E_STRATEGY", "act").lower() != "rgbd":
        raise RuntimeError("RGB-D strategy is disabled; set ATEC_TASK_E_STRATEGY=rgbd for an explicit comparison")
    return RGBDAdaptivePolicy(RGBDPolicyConfig(intrinsics, camera_to_base), segmenter, ik_adapter)


__all__ = ["RGBDPolicyConfig", "RGBDAdaptivePolicy", "make_strategy"]
