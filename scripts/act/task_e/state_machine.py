"""Pick-place state machine and grasp-quaternion solver for Task E."""

import torch
from isaaclab.utils.math import matrix_from_quat, quat_from_matrix

from .config import (
    STEPS, STATE_ORDER, PRE_GRASP_HOME_DWELL,
    CARRY_Z, PLACE_HEIGHT,
    RETRACT_POS_X, RETRACT_POS_Y,
    GRASP_Z_OFFSET, GRASP_Z_OFFSET_BY_OBJECT, GRASP_XY_OFFSET_BY_OBJECT,
    SIDE_GRASP_OBJECT_IDS, SIDE_GRASP_Z_OFFSET,
    SIDE_GRASP_APPROACH_DIST, SIDE_GRASP_STANDOFF,
    SIDE_GRASP_XY_OFFSET_BY_OBJECT,
    OBJ_HALF_EXTENTS,
    BASKET_CENTER_X, BASKET_CENTER_Y,
    DEFAULT_PLACE_QUAT_W,
)


def _build_grasp_matrix(long_axis: torch.Tensor, grip_z: torch.Tensor) -> torch.Tensor:
    """Build a right-handed gripper rotation matrix given the object's long axis.

    The Piper gripper jaw opens along its LOCAL Y axis, so local Y must be
    perpendicular to the object's long axis.

    Frame layout (columns of R_grip):
      col 0 (local X) = align_dir  ∥ long_axis
      col 1 (local Y) = jaw_dir    ⊥ long_axis  ← jaw opening direction
      col 2 (local Z) = grip_z     pointing down

    Right-hand check: col0 × col1 = align_dir × jaw_dir = grip_z ✓
    """
    jaw_dir   = torch.linalg.cross(long_axis, grip_z)   # ⊥ long_axis, in XY plane
    jaw_dir   = jaw_dir / jaw_dir.norm().clamp(min=1e-6)
    align_dir = torch.linalg.cross(jaw_dir, grip_z)     # ∥ long_axis
    align_dir = align_dir / align_dir.norm().clamp(min=1e-6)
    return torch.stack([align_dir, jaw_dir, grip_z], dim=1)   # (3, 3)


def compute_grasp_quat(obj_quat_w: torch.Tensor, device: str) -> torch.Tensor:
    """Compute a top-down grasp quaternion for the object.

    Finds the object axis most aligned with the world XY-plane (the long axis),
    builds a gripper frame where the jaw (local Y) is perpendicular to that axis,
    then picks the candidate orientation closest to the default top-down pose.

    Parameters
    ----------
    obj_quat_w : (4,) tensor, (w, x, y, z)
    device     : torch device string

    Returns
    -------
    grasp_quat : (4,) tensor, (w, x, y, z)
    """
    R_obj        = matrix_from_quat(obj_quat_w.unsqueeze(0)).squeeze(0)   # (3, 3)
    grip_z       = torch.tensor([0.0, 0.0, -1.0], device=device)
    default_quat = torch.tensor(DEFAULT_PLACE_QUAT_W, dtype=torch.float32, device=device)

    # Project each object column-axis onto XY plane; keep the most horizontal one(s)
    norms, axes_xy = [], []
    for col in range(3):
        ax = torch.tensor([R_obj[0, col].item(), R_obj[1, col].item(), 0.0], device=device)
        norms.append(ax.norm().item())
        axes_xy.append(ax)

    best_norm  = max(norms)
    candidates = [
        axes_xy[c] / max(norms[c], 1e-6)
        for c in range(3)
        if norms[c] >= best_norm - 1e-3
    ]

    # Among candidates, pick the one whose grasp frame is closest to the default orientation
    best_cos  = -2.0
    long_axis = candidates[0]
    for cand in candidates:
        q_cand  = quat_from_matrix(_build_grasp_matrix(cand, grip_z).unsqueeze(0)).squeeze(0)
        cos_sim = torch.abs((q_cand * default_quat).sum()).item()
        if cos_sim > best_cos:
            best_cos  = cos_sim
            long_axis = cand

    R_grip = _build_grasp_matrix(long_axis, grip_z)                  # (3, 3)
    return quat_from_matrix(R_grip.unsqueeze(0)).squeeze(0)           # (4,) w,x,y,z


def compute_side_grasp(
    half_extents_xy: tuple[float, float],
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed side pinch for an upright object in its canonical spawn yaw.

    The frame does not follow the live object quaternion. A bottle that yaws
    while settling otherwise swings the approach onto ±Y (toward the table
    sides) and puts the jaw across the wrong face.

    Gripper local Z points from the robot (+X) toward the bottle. Local Y, the
    jaw opening, is the shorter horizontal footprint of the canonical pose so
    the ~7 cm jaw closes on that face. The collector restores that spawn yaw
    before this frame is used.

    Returns
    -------
    grasp_quat : (4,) w,x,y,z
    approach_dir : (3,) unit vector, gripper local Z in the world frame
    """
    hx, hy = half_extents_xy
    # Jaw spans the shorter canonical footprint. Approach is the other axis,
    # sign chosen so the gripper sits on the robot side of the bottle.
    if hy <= hx:
        jaw = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)
        approach = torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float32, device=device)
    else:
        jaw = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=device)
        approach = torch.tensor([0.0, -1.0, 0.0], dtype=torch.float32, device=device)
    align = torch.linalg.cross(jaw, approach)
    align = align / align.norm().clamp(min=1e-6)
    R_grip = torch.stack([align, jaw, approach], dim=1)
    grasp_quat = quat_from_matrix(R_grip.unsqueeze(0)).squeeze(0)
    return grasp_quat, approach


def _slerp(q0: torch.Tensor, q1: torch.Tensor, alpha: float) -> torch.Tensor:
    """Shortest-path quaternion slerp. Quaternions are (w, x, y, z)."""
    q0 = q0 / q0.norm().clamp(min=1e-6)
    q1 = q1 / q1.norm().clamp(min=1e-6)
    dot = torch.dot(q0, q1)
    if float(dot) < 0.0:
        q1 = -q1
        dot = -dot
    if float(dot) > 0.9995:
        out = q0 + alpha * (q1 - q0)
        return out / out.norm().clamp(min=1e-6)
    theta = torch.acos(dot.clamp(-1.0, 1.0))
    s = torch.sin(theta)
    return (
        torch.sin((1.0 - alpha) * theta) * q0 + torch.sin(alpha * theta) * q1
    ) / s


class PickPlaceStateMachine:
    """Finite state machine that sequences pick-and-place for multiple objects.

    States (in order): INIT → PRE_GRASP → REACH → CLOSE → LIFT →
                       TRANSPORT → PLACE → OPEN → RETRACT → (next object or done)
    """

    def __init__(self, object_indices: list[int], device: str):
        self._obj_indices = object_indices
        self._device      = device
        self._grasp_quat_cache: dict[int, torch.Tensor] = {}
        self._side_approach_cache: dict[int, torch.Tensor] = {}
        self.reset()

    def reset(self) -> None:
        self._ptr             = 0
        self._state_idx       = 0
        self._count           = 0
        self.done             = False
        self._cached_obj_pos: torch.Tensor | None = None
        self._transport_start: torch.Tensor | None = None
        self._retract_start: torch.Tensor | None = None
        self.needs_ik_reset      = False
        self._grasp_quat_cache.clear()
        self._side_approach_cache.clear()

    def set_grasp_quat(self, obj_idx: int, obj_quat_w: torch.Tensor) -> None:
        """Pre-compute and cache the grasp quaternion for one object."""
        if obj_idx in SIDE_GRASP_OBJECT_IDS:
            quat, approach = compute_side_grasp(
                OBJ_HALF_EXTENTS[obj_idx], self._device
            )
            self._grasp_quat_cache[obj_idx] = quat
            self._side_approach_cache[obj_idx] = approach
        else:
            self._grasp_quat_cache[obj_idx] = compute_grasp_quat(obj_quat_w, self._device)

    def tick(self, obj_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, str]:
        """Advance the state machine by one step.

        Parameters
        ----------
        obj_pos : (3,) tensor — current object position in world frame

        Returns
        -------
        ee_pos_des  : (3,) target EE position
        ee_quat_des : (4,) target EE orientation (w,x,y,z)
        gripper_cmd : "open" | "close"
        """
        s = self.state
        d = self._device

        # Freeze object position at start of PRE_GRASP. LIFT must use this
        # snapshot too: the object is in the hand, so a live XY target chases
        # any IK drift and the base joint winds away from the basket.
        if s == "PRE_GRASP" and self._count == 0:
            self._cached_obj_pos = obj_pos.clone()
        if s in ("PRE_GRASP", "REACH", "CLOSE", "LIFT", "TRANSPORT") and self._cached_obj_pos is not None:
            obj_pos = self._cached_obj_pos

        ee_pos, gripper = self._get_target_pos_gripper(s, obj_pos, d)
        ee_quat         = self._get_target_quat(s, d)

        self._count += 1
        if self._count >= STEPS[s]:
            self._count = 0
            if s == "RETRACT":
                self._ptr          += 1
                self._cached_obj_pos = None
                self._transport_start = None
                self._retract_start = None
                if self._ptr >= len(self._obj_indices):
                    self.done = True
                    return ee_pos, ee_quat, gripper
                self._state_idx = STATE_ORDER.index("PRE_GRASP")
                self.needs_ik_reset = True
            else:
                self._state_idx += 1

        return ee_pos, ee_quat, gripper

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def state(self) -> str:
        return STATE_ORDER[self._state_idx]

    @property
    def current_object_key(self) -> str:
        return f"object_{self._obj_indices[self._ptr]}"

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _current_obj_idx(self) -> int:
        return self._obj_indices[min(self._ptr, len(self._obj_indices) - 1)]

    def _grasp_z_offset(self) -> float:
        return GRASP_Z_OFFSET_BY_OBJECT.get(self._current_obj_idx(), GRASP_Z_OFFSET)

    def _apply_grasp_xy_offset(self, p: torch.Tensor) -> torch.Tensor:
        dx, dy = GRASP_XY_OFFSET_BY_OBJECT.get(self._current_obj_idx(), (0.0, 0.0))
        p = p.clone()
        p[0] += dx
        p[1] += dy
        return p

    def _uses_side_grasp(self) -> bool:
        return self._current_obj_idx() in SIDE_GRASP_OBJECT_IDS

    def _side_distances(self) -> tuple[float, float]:
        idx = self._current_obj_idx()
        close = SIDE_GRASP_APPROACH_DIST[idx]
        return close, close + SIDE_GRASP_STANDOFF[idx]

    def _side_grasp_anchor(self, obj_pos: torch.Tensor) -> torch.Tensor:
        """Object root shifted by side-grasp XY offset (world frame)."""
        p = obj_pos.clone()
        dx, dy = SIDE_GRASP_XY_OFFSET_BY_OBJECT.get(self._current_obj_idx(), (0.0, 0.0))
        p[0] += dx
        p[1] += dy
        return p

    def _side_point(self, obj_pos: torch.Tensor, distance: float, z: float | None = None) -> torch.Tensor:
        """gripper_base pose for a side grasp. Local Z points at the object."""
        anchor = self._side_grasp_anchor(obj_pos)
        approach = self._side_approach_cache[self._current_obj_idx()]
        p = anchor - approach * float(distance)
        if z is None:
            p[2] = anchor[2] + SIDE_GRASP_Z_OFFSET[self._current_obj_idx()]
        else:
            p[2] = z
        return p

    @staticmethod
    def _lerp(a: torch.Tensor, b: torch.Tensor, alpha: float) -> torch.Tensor:
        return (1.0 - alpha) * a + alpha * b

    def _retract_pose(self, d: str) -> torch.Tensor:
        return torch.tensor([RETRACT_POS_X, RETRACT_POS_Y, CARRY_Z], dtype=torch.float32, device=d)

    def _top_down_pre_grasp(self, obj_pos: torch.Tensor, d: str) -> torch.Tensor:
        """Move from the home/retract pose to hover above the object."""
        home = self._retract_pose(d)
        goal = self._apply_grasp_xy_offset(obj_pos)
        goal[2] = CARRY_Z
        total = STEPS["PRE_GRASP"]
        dwell = PRE_GRASP_HOME_DWELL if self._ptr > 0 else 0
        dwell = min(dwell, max(total - 1, 0))
        if self._count < dwell:
            return home.clone()
        alpha = (self._count - dwell) / max(total - dwell - 1, 1)
        return self._lerp(home, goal, alpha)

    def _top_down_reach(self, obj_pos: torch.Tensor) -> torch.Tensor:
        """Descend from carry height to the grasp height at fixed XY."""
        high = self._apply_grasp_xy_offset(obj_pos)
        high[2] = CARRY_Z
        low = self._apply_grasp_xy_offset(obj_pos)
        low[2] = low[2] + self._grasp_z_offset()
        alpha = self._count / max(STEPS["REACH"] - 1, 1)
        return self._lerp(high, low, alpha)

    def _basket_ee(self, z: float, d: str) -> torch.Tensor:
        p = torch.tensor([BASKET_CENTER_X, BASKET_CENTER_Y, z], dtype=torch.float32, device=d)
        if self._uses_side_grasp():
            idx = self._current_obj_idx()
            dx, dy = SIDE_GRASP_XY_OFFSET_BY_OBJECT.get(idx, (0.0, 0.0))
            p[0] += dx
            p[1] += dy
            p = p - self._side_approach_cache[idx] * SIDE_GRASP_APPROACH_DIST[idx]
            p[2] = z
        return p

    def _get_target_pos_gripper(
        self, s: str, obj_pos: torch.Tensor, d: str
    ) -> tuple[torch.Tensor, str]:
        if s == "INIT":
            return torch.tensor([RETRACT_POS_X, RETRACT_POS_Y, CARRY_Z], device=d), "open"
        elif s == "PRE_GRASP":
            if self._uses_side_grasp():
                _, dist_far = self._side_distances()
                p_low = self._side_point(obj_pos, dist_far)
                p_high = p_low.clone()
                p_high[2] = CARRY_Z
                alpha = self._count / max(STEPS["PRE_GRASP"] - 1, 1)
                return self._lerp(p_high, p_low, alpha), "open"
            return self._top_down_pre_grasp(obj_pos, d), "open"
        elif s == "REACH":
            if self._uses_side_grasp():
                dist_close, dist_far = self._side_distances()
                alpha = self._count / max(STEPS["REACH"] - 1, 1)
                return self._lerp(
                    self._side_point(obj_pos, dist_far),
                    self._side_point(obj_pos, dist_close),
                    alpha,
                ), "open"
            return self._top_down_reach(obj_pos), "open"
        elif s == "CLOSE":
            if self._uses_side_grasp():
                dist_close, _ = self._side_distances()
                return self._side_point(obj_pos, dist_close), "close"
            p = self._apply_grasp_xy_offset(obj_pos)
            p[2] += self._grasp_z_offset()
            return p, "close"
        elif s == "LIFT":
            # Raise Z in small steps at the frozen grasp XY. A single jump to
            # CARRY_Z saturates the wrist, and joint1 then yaws the arm outward.
            if self._uses_side_grasp():
                dist_close, _ = self._side_distances()
                p = self._side_point(obj_pos, dist_close)
            else:
                p = self._apply_grasp_xy_offset(obj_pos)
                p[2] = p[2] + self._grasp_z_offset()
            z_grasp = float(p[2])
            alpha = self._count / max(STEPS["LIFT"] - 1, 1)
            p[2] = (1.0 - alpha) * z_grasp + alpha * CARRY_Z
            return p, "close"
        elif s == "TRANSPORT":
            # Walk the IK target from the lift pose to the basket so each
            # differential-IK step sees a few millimetres of error.
            if self._count == 0 or self._transport_start is None:
                if self._uses_side_grasp():
                    dist_close, _ = self._side_distances()
                    p = self._side_point(obj_pos, dist_close, z=CARRY_Z)
                else:
                    p = self._apply_grasp_xy_offset(obj_pos)
                    p[2] = CARRY_Z
                self._transport_start = p.clone()
            goal = self._basket_ee(CARRY_Z, d)
            alpha = self._count / max(STEPS["TRANSPORT"] - 1, 1)
            ee_pos = (1.0 - alpha) * self._transport_start + alpha * goal
            return ee_pos, "close"
        elif s == "PLACE":
            return self._basket_ee(PLACE_HEIGHT, d), "close"
        elif s == "OPEN":
            return self._basket_ee(PLACE_HEIGHT, d), "open"
        elif s == "LIFT_RETRACT":
            return self._basket_ee(CARRY_Z, d), "open"
        elif s == "RETRACT":
            if self._count == 0 or self._retract_start is None:
                self._retract_start = self._basket_ee(CARRY_Z, d).clone()
            alpha = self._count / max(STEPS["RETRACT"] - 1, 1)
            ee_pos = self._lerp(self._retract_start, self._retract_pose(d), alpha)
            return ee_pos, "open"
        else:
            raise ValueError(f"Unknown state: {s}")

    def _get_target_quat(self, s: str, d: str) -> torch.Tensor:
        default_quat = torch.tensor(DEFAULT_PLACE_QUAT_W, dtype=torch.float32, device=d)
        # Keep the top-down grasp yaw through the carry and release. Snapping
        # back to DEFAULT_PLACE_QUAT_W at TRANSPORT makes joint1 sweep the
        # long way around before the position error pulls the arm to the basket.
        if s in ("PRE_GRASP", "REACH", "CLOSE", "LIFT",
                 "TRANSPORT", "PLACE", "OPEN", "LIFT_RETRACT"):
            cur_idx = self._obj_indices[min(self._ptr, len(self._obj_indices) - 1)]
            grasp_quat = self._grasp_quat_cache.get(cur_idx, default_quat)
            # Blend from top-down home to the object grasp over PRE_GRASP so a
            # side-grasp release does not snap wrist+yaw on the next object.
            if s == "PRE_GRASP":
                total = STEPS["PRE_GRASP"]
                dwell = PRE_GRASP_HOME_DWELL if self._ptr > 0 else 0
                dwell = min(dwell, max(total - 1, 0))
                if self._count < dwell:
                    return default_quat
                alpha = (self._count - dwell) / max(total - dwell - 1, 1)
                return _slerp(default_quat, grasp_quat, alpha)
            return grasp_quat
        if s == "RETRACT":
            cur_idx = self._obj_indices[self._ptr]
            grasp_quat = self._grasp_quat_cache.get(cur_idx, default_quat)
            alpha = self._count / max(STEPS["RETRACT"] - 1, 1)
            return _slerp(grasp_quat, default_quat, alpha)
        return default_quat
