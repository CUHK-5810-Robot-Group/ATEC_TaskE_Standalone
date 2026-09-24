"""Task E RGB-D pick/place policy using only standard observations.

The fixed robot/camera geometry comes from the supplied task and Piper model.
Per-episode object positions are estimated exclusively from video RGB-D.
"""

import os

import numpy as np

from .rgbd_control.constants import (
    ACTION_SCALE, BANANA_FORWARD_TILT_DEG, BASKET_SLOTS,
    BOTTLE_CARRY_DOWN_ANGLE_DEG,
    BOTTLE_CARRY_WAYPOINTS, BOTTLE_CARRY_Z,
    BOTTLE_PLACE_FINGER_REACH, BOTTLE_PLACE_Z,
    BOTTLE_SIDE_DOWN_ANGLE_DEG,
    BOTTLE_SIDE_APPROACH_POS_TOLERANCE,
    BOTTLE_SIDE_GRASP_X, BOTTLE_SIDE_GRASP_Z,
    BOTTLE_SIDE_NEAR_STAGE_X, BOTTLE_SIDE_PRE_Z,
    BOTTLE_SIDE_STAGE_X, BOTTLE_PRE_GUARD_X, BOTTLE_PRE_GUARD_Z,
    BOTTLE_PRE_MAX_WORLD_X, BOTTLE_PRE_MAX_X, BOTTLE_PRE_MIN_X,
    BOTTLE_FAR_PRE_Z,
    BOTTLE_STAGE_RAMP_END_X, BOTTLE_STAGE_RAMP_START_X,
    BOTTLE_SUGAR_SIDE_RAMP_END_Y, BOTTLE_SUGAR_SIDE_RAMP_START_Y,
    CARRY_Z,
    DEFAULT_Q, GRASP_XY_OFFSETS, GRASP_Z_OFFSET, GRIPPER_CLOSE, GRIPPER_OPEN,
    HOME_Q, LIFT_CLEARANCE,
    OBJECT_CENTER_Z, OBJECT_ORDER, OBJECT_Y_BANDS, PLACE_Z,
    PRE_GRASP_CLEARANCE, RETRACT_XYZ, ROBOT_BASE_XYZ,
    TABLE_TOP_Z,
)
from .rgbd_control.kinematics import (
    PiperKinematics, bottle_carry_rotation, bottle_side_rotation,
    top_down_rotation,
)
from .rgbd_control.perception import TaskPerception


def _numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class _SingleEnvironmentPolicy:
    """One episode controller; create one instance per parallel environment."""

    _MOTION_TIMEOUT = 300
    _MAX_RETRIES = 3
    _SUGAR_DESCEND_TIMEOUT = 450
    _SUGAR_STALL_STEPS = 60
    _BANANA_STALL_STEPS = 60
    _BANANA_DESCEND_TIMEOUT = 450
    _BANANA_DESCEND_Z_STEP = 0.004
    _BANANA_DESCEND_MAX_DELTA = 0.012
    _BANANA_FAST_DESCEND_Z_STEP = 0.007
    _BANANA_FAST_DESCEND_MAX_DELTA = 0.018
    _BANANA_SLOW_ZONE_Z = TABLE_TOP_Z + 0.18
    _BANANA_LIFT_CLEARANCE = 0.05
    _BANANA_LIFT_CLEAR_TIMEOUT = 300
    _BANANA_LIFT_TIMEOUT = 450
    _BOTTLE_PRE_GRASP_TIMEOUT = 650
    _BOTTLE_STAGE_TIMEOUT = 450
    _BOTTLE_STAGE_STALL_STEPS = 120
    _BOTTLE_STAGE_MIN_Z_PROGRESS = 0.003
    _BOTTLE_APPROACH_TIMEOUT = 550
    _BOTTLE_APPROACH_X_STEP = 0.004
    _BOTTLE_APPROACH_Y_CORRIDOR = 0.006
    _BOTTLE_APPROACH_Z_CORRIDOR = 0.012
    _BOTTLE_APPROACH_NEAR_X = 0.14
    _BOTTLE_APPROACH_MAX_Y_ERROR = 0.015
    _BOTTLE_APPROACH_MAX_Z_ERROR = 0.022
    _BOTTLE_MIN_BASKET_GAIN = 30
    _BOTTLE_LIFT_TIMEOUT = 450
    _BOTTLE_LIFT_MAX_DELTA = 0.01
    _BOTTLE_CARRY_TIMEOUT = 500
    _BOTTLE_CARRY_MAX_DELTA = 0.015
    _BOTTLE_RETRACT_TIMEOUT = 500
    _BOTTLE_RETRACT_CLEAR_STEPS = 6
    _BOTTLE_RETRACT_CLEAR_X = 0.15
    _BOTTLE_RETRACT_CLEAR_Y = 0.14
    _BOTTLE_RETRACT_CLEAR_Z = 0.30

    def __init__(self):
        self.kinematics = PiperKinematics()
        self.perception = TaskPerception()
        self.debug = os.environ.get("ATEC_RGBD_DEBUG", "0") == "1"
        self.reset()

    def reset(self):
        self.state = "HOME"
        self.state_steps = 0
        self.stable_steps = 0
        self.object_ptr = 0
        self.retry = 0
        self.estimate = None
        self.grasp_xy = None
        self.grasp_rotation = None
        self.initial_table_count = 0
        self.initial_basket_count = 0
        self.vision_age = 999
        self.terminal = False
        self.descend_lowest_z = None
        self.descend_no_progress = 0
        self.bottle_stage_lowest_z = None
        self.bottle_stage_no_progress = 0
        self.relocated_xy = None
        self.relocated_hits = 0
        self.bottle_retract_clear_steps = 0
        self.banana_lift_xy = None
        self.banana_lift_rotation = None
        self.banana_lift_clear_z = None

    @property
    def object_id(self):
        return OBJECT_ORDER[self.object_ptr]

    def _enter(self, state):
        if self.debug and self.state != state:
            object_id = OBJECT_ORDER[self.object_ptr] if self.object_ptr < len(OBJECT_ORDER) else "all"
            print(f"[RGBD] object={object_id} {self.state} -> {state} retry={self.retry}", flush=True)
        self.state = state
        self.state_steps = 0
        self.stable_steps = 0
        if state == "DESCEND":
            self.descend_lowest_z = None
            self.descend_no_progress = 0
        if state == "BOTTLE_SIDE_STAGE":
            self.bottle_stage_lowest_z = None
            self.bottle_stage_no_progress = 0
        if state == "SCAN":
            self.relocated_xy = None
            self.relocated_hits = 0
        if state == "RETRACT":
            self.bottle_retract_clear_steps = 0

    def _refresh_vision(self, rgb, depth, force=False):
        self.vision_age += 1
        if force or self.vision_age >= 8:
            self.perception.update(_numpy(rgb), _numpy(depth))
            self.vision_age = 0
            return True
        return False

    def _grasp_plan(self):
        oid = self.object_id
        # Keep retries object-specific. Sugar's long side follows world X, so
        # a stalled descent tries slightly closer to the arm rather than
        # shifting the jaws off its short Y side. Bottle retries stay near
        # the visible bottle centre. A dropped banana must be re-centred from
        # the new scan; the former +/-2.5 cm Y search missed its moved body.
        retry_offsets = {
            1: ((0.0, 0.0), (0.015, 0.0), (0.030, 0.0)),
            2: ((0.0, 0.0), (0.0, 0.008), (0.0, -0.008)),
            3: ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
        }
        offset = np.array(retry_offsets[oid][self.retry])
        self.grasp_xy = self.estimate.xy + GRASP_XY_OFFSETS[oid] + offset
        if self.debug:
            bottle_stage = (f" bottle_pre_x_m={self._bottle_pre_offset():.3f}"
                            f" bottle_pre_z_m={self._bottle_pre_z() - TABLE_TOP_Z:.3f}"
                            f" bottle_stage_x_m={self._bottle_stage_offset():.3f}"
                            if oid == 2 else "")
            print(f"[RGBD] grasp plan object={oid} retry={self.retry} "
                  f"target_xy={np.round(self.grasp_xy, 4).tolist()} "
                  f"base_offset_xy={GRASP_XY_OFFSETS[oid].tolist()} "
                  f"retry_offset_xy={offset.tolist()}{bottle_stage}", flush=True)
        # The sugar-box image shows the open fingers spanning its long top
        # face with the former default grasp frame. Rotate the wrist by 90 deg
        # about world Z so the opening spans the short side before descent.
        # -Y selects the wrist branch away from joint6's negative limit.
        if oid == 1:
            axis = np.array([0.0, -1.0])
        elif oid == 3:
            axis = self.estimate.long_axis_xy
        else:
            axis = None
        if oid == 2:
            self.grasp_rotation = bottle_side_rotation(BOTTLE_SIDE_DOWN_ANGLE_DEG)
        elif oid == 3:
            self.grasp_rotation = top_down_rotation(
                axis, tilt_toward_neg_x_deg=BANANA_FORWARD_TILT_DEG)
        else:
            self.grasp_rotation = top_down_rotation(axis)
        self.initial_table_count = self.estimate.point_count
        self.initial_basket_count = self.perception.basket_count(BASKET_SLOTS[oid])

    def _bottle_sugar_side_fraction(self):
        span = BOTTLE_SUGAR_SIDE_RAMP_END_Y - BOTTLE_SUGAR_SIDE_RAMP_START_Y
        return float(np.clip((self.grasp_xy[1] - BOTTLE_SUGAR_SIDE_RAMP_START_Y)
                             / span, 0.0, 1.0))

    def _bottle_pre_offset(self):
        # Keep the rotation away from the cap without asking a near-arm
        # bottle to rotate at an unreachable absolute X.
        return float(np.clip(BOTTLE_PRE_MAX_WORLD_X - self.grasp_xy[0],
                             BOTTLE_PRE_MIN_X, BOTTLE_PRE_MAX_X))

    def _bottle_pre_z(self):
        # The far edge needs a slightly lower high waypoint to avoid a wrist
        # branch that can stall before the outside-stage descent.
        return (BOTTLE_FAR_PRE_Z if self.grasp_xy[0] < BOTTLE_STAGE_RAMP_START_X
                else BOTTLE_SIDE_PRE_Z)

    def _bottle_pre_motion(self, q):
        """Rotate outside the bottle; reject an IK arc that dips near its cap."""
        target = np.r_[self.grasp_xy + [self._bottle_pre_offset(), 0.0],
                       self._bottle_pre_z()]
        candidate, _, _ = self.kinematics.step(
            q, target, self.grasp_rotation, max_delta=0.04)
        # The gripper fingers extend beyond the wrist FK point. Keep ample
        # wrist clearance during this phase, including intermediate joint
        # commands; the later outside-stage descent is handled separately.
        for fraction in (0.0, 0.5, 1.0):
            xyz, _ = self.kinematics.forward(q + fraction * (candidate - q))
            if (xyz[0] - self.grasp_xy[0] < BOTTLE_PRE_GUARD_X
                    and xyz[2] < BOTTLE_PRE_GUARD_Z):
                if self.debug:
                    print(f"[RGBD] bottle pre-grasp clearance violated "
                          f"step={self.state_steps} "
                          f"fk_xyz={np.round(xyz, 4).tolist()}; recovering",
                          flush=True)
                self._retry_or_stop()
                return np.concatenate((q, GRIPPER_OPEN))
        return self._motion(q, target, self.grasp_rotation, GRIPPER_OPEN,
                            "BOTTLE_SIDE_STAGE", orientation_tolerance=0.35,
                            approach_axis_tolerance=0.30,
                            timeout=self._BOTTLE_PRE_GRASP_TIMEOUT)

    def _bottle_stage_offset(self):
        span = BOTTLE_STAGE_RAMP_END_X - BOTTLE_STAGE_RAMP_START_X
        near_arm_fraction = np.clip(
            (self.grasp_xy[0] - BOTTLE_STAGE_RAMP_START_X) / span, 0.0, 1.0)
        fraction = max(near_arm_fraction, self._bottle_sugar_side_fraction())
        return BOTTLE_SIDE_STAGE_X + fraction * (
            BOTTLE_SIDE_NEAR_STAGE_X - BOTTLE_SIDE_STAGE_X)

    def _valid_scan_candidate(self, candidate):
        if candidate is None:
            return False
        if self.object_id != 2 or self.retry == 0 or self.estimate is None \
                or self.estimate.object_id != 2:
            return True
        # After recovery, reject sparse fragments, but allow a bottle that
        # was bumped to a new stable position to be reacquired.
        min_points = max(60, int(self.initial_table_count * 0.50))
        shift = float(np.linalg.norm(candidate.xy - self.estimate.xy))
        if candidate.point_count >= min_points and shift < 0.035:
            return True
        # A dropped bottle may roll far from its original position. Accept a
        # dense, same-colour cluster in the bottle band only after three
        # consecutive, spatially consistent scans; reject sparse fragments.
        low_y, high_y = OBJECT_Y_BANDS[2]
        color_delta = float(np.linalg.norm(candidate.mean_rgb - self.estimate.mean_rgb))
        plausible = (candidate.point_count >= min_points
                     and low_y - 0.01 <= candidate.xy[1] <= high_y + 0.01
                     and color_delta < 80.0)
        if plausible:
            if self.relocated_xy is not None and np.linalg.norm(candidate.xy - self.relocated_xy) < 0.015:
                self.relocated_hits += 1
            else:
                self.relocated_xy = candidate.xy.copy()
                self.relocated_hits = 1
            if self.relocated_hits >= 3:
                return True
        else:
            self.relocated_xy = None
            self.relocated_hits = 0
        if self.debug:
            print(f"[RGBD] pending object=2 scan xy={candidate.xy.tolist()} "
                  f"points={candidate.point_count} shift_m={shift:.4f} "
                  f"color_delta={color_delta:.1f} "
                  f"relocation_hits={self.relocated_hits}", flush=True)
        return False

    def _matches_initial_color(self, candidate):
        # A grasped object should disappear from its table band. Ignore a new
        # depth cluster if its RGB differs strongly from the pre-grasp object
        # (usually a passing arm link or a specular depth artefact).
        return candidate is not None and np.linalg.norm(
            candidate.mean_rgb - self.estimate.mean_rgb
        ) < 80.0

    @staticmethod
    def _bottle_place_base_xy():
        """Aim the held bottle, not the gripper base, at its basket slot."""
        slot_xy = BASKET_SLOTS[2]
        radial_xy = slot_xy - ROBOT_BASE_XYZ[:2]
        radial_xy /= max(np.linalg.norm(radial_xy), 1e-9)
        return slot_xy - BOTTLE_PLACE_FINGER_REACH * radial_xy

    def _bottle_carry_motion(self, q, xy, z, next_state):
        if self.state_steps > self._BOTTLE_CARRY_TIMEOUT:
            # Do not enter RECOVER: it opens the gripper and drops a bottle
            # that may still be held. Stop this run with the fingers closed.
            if self.debug:
                fk_xyz, _ = self.kinematics.forward(q)
                print(f"[RGBD] bottle carry timeout state={self.state} "
                      f"target_xy={np.round(xy, 4).tolist()} "
                      f"fk_xyz={np.round(fk_xyz, 4).tolist()}; "
                      "holding gripper closed", flush=True)
            self._enter("HOLD_BOTTLE")
            self.terminal = True
            return np.concatenate((q, GRIPPER_CLOSE))
        rotation = bottle_carry_rotation(xy, BOTTLE_CARRY_DOWN_ANGLE_DEG)
        return self._motion(
            q, np.r_[xy, z], rotation, GRIPPER_CLOSE, next_state,
            orientation_tolerance=0.45, approach_axis_tolerance=0.40,
            max_delta=self._BOTTLE_CARRY_MAX_DELTA,
            timeout=self._BOTTLE_CARRY_TIMEOUT,
        )

    def _bottle_side_approach_motion(self, q, grasp_z):
        """Advance along X only while the measured wrist stays in its Y/Z lane."""
        fk_xyz, fk_rotation = self.kinematics.forward(q)
        final_xyz = np.r_[self.grasp_xy + [BOTTLE_SIDE_GRASP_X, 0.0], grasp_z]
        y_error = float(fk_xyz[1] - final_xyz[1])
        z_error = float(fk_xyz[2] - final_xyz[2])
        near_bottle = fk_xyz[0] - self.grasp_xy[0] <= self._BOTTLE_APPROACH_NEAR_X
        if (near_bottle and (abs(y_error) > self._BOTTLE_APPROACH_MAX_Y_ERROR
                             or abs(z_error) > self._BOTTLE_APPROACH_MAX_Z_ERROR)):
            if self.debug:
                print(f"[RGBD] bottle approach left corridor step={self.state_steps} "
                      f"fk_xyz={np.round(fk_xyz, 4).tolist()} "
                      f"y_error_m={y_error:+.4f} z_error_m={z_error:+.4f}; "
                      "recovering before contact", flush=True)
            self._retry_or_stop()
            return np.concatenate((q, GRIPPER_OPEN))
        if self.state_steps > self._BOTTLE_APPROACH_TIMEOUT:
            if self.debug:
                print(f"[RGBD] bottle approach timeout step={self.state_steps} "
                      f"fk_xyz={np.round(fk_xyz, 4).tolist()}; recovering",
                      flush=True)
            self._retry_or_stop()
            return np.concatenate((q, GRIPPER_OPEN))

        aligned = (abs(y_error) <= self._BOTTLE_APPROACH_Y_CORRIDOR
                   and abs(z_error) <= self._BOTTLE_APPROACH_Z_CORRIDOR)
        waypoint_x = (max(final_xyz[0], fk_xyz[0] - self._BOTTLE_APPROACH_X_STEP)
                      if aligned else fk_xyz[0])
        waypoint_xyz = np.array([waypoint_x, final_xyz[1], final_xyz[2]])
        target_q, _, rot_error = self.kinematics.step(
            q, waypoint_xyz, self.grasp_rotation, max_delta=0.008)
        final_error = float(np.linalg.norm(final_xyz - fk_xyz))
        xy_error = float(np.linalg.norm(final_xyz[:2] - fk_xyz[:2]))
        axis_error = float(np.arccos(np.clip(
            fk_rotation[:, 2] @ self.grasp_rotation[:, 2], -1.0, 1.0)))
        if self.debug and self.state_steps in (1, 50, 150, 300, 500):
            print(f"[RGBD] bottle approach step={self.state_steps} "
                  f"target_xyz={np.round(final_xyz, 4).tolist()} "
                  f"waypoint_xyz={np.round(waypoint_xyz, 4).tolist()} "
                  f"fk_xyz={np.round(fk_xyz, 4).tolist()} "
                  f"pos_error_m={final_error:.4f} "
                  f"y_error_m={y_error:+.4f} z_error_m={z_error:+.4f} "
                  f"rot_error={rot_error:.4f} axis_error={axis_error:.4f}",
                  flush=True)
        # Contact can leave the wrist 7-9 mm high even when the fingers have
        # reached both sides of the bottle. Keep the horizontal constraint
        # tight; allow only this small vertical residual before closing.
        arrived = ((final_error < BOTTLE_SIDE_APPROACH_POS_TOLERANCE
                    or (xy_error < 0.006 and abs(z_error) < 0.010))
                   and rot_error < 0.25 and axis_error < 0.20)
        self.stable_steps = self.stable_steps + 1 if arrived else 0
        if self.stable_steps >= 6:
            if self.debug:
                print(f"[RGBD] reached bottle side approach step={self.state_steps} "
                      f"fk_xyz={np.round(fk_xyz, 4).tolist()} "
                      f"pos_error_m={final_error:.4f}", flush=True)
            self._enter("CLOSE")
        return np.concatenate((target_q, GRIPPER_OPEN))

    def _motion(self, q, xyz, rotation, gripper, next_state,
                orientation_tolerance=0.8, position_tolerance=0.025,
                heading_tolerance=None, approach_axis_tolerance=None,
                max_delta=0.04, timeout=None):
        timeout = self._MOTION_TIMEOUT if timeout is None else timeout
        target, pos_error, rot_error = self.kinematics.step(
            q, xyz, rotation, max_delta=max_delta)
        heading_error = None
        approach_axis_error = None
        if approach_axis_tolerance is not None:
            _, current_rotation = self.kinematics.forward(q)
            approach_axis_error = float(np.arccos(np.clip(
                current_rotation[:, 2] @ rotation[:, 2], -1.0, 1.0)))
        if heading_tolerance is not None:
            _, current_rotation = self.kinematics.forward(q)
            current_x = current_rotation[:2, 0].copy()
            target_x = rotation[:2, 0].copy()
            current_x /= max(np.linalg.norm(current_x), 1e-9)
            target_x /= max(np.linalg.norm(target_x), 1e-9)
            heading_error = float(np.arccos(np.clip(current_x @ target_x, -1.0, 1.0)))
        if self.debug and (self.state_steps in (1, 50, 150, 300, timeout)
                           or self.state_steps > timeout):
            fk_xyz, _ = self.kinematics.forward(q)
            print(
                f"[RGBD] motion object={self.object_id} state={self.state} "
                f"step={self.state_steps} target_xyz={np.round(xyz, 4).tolist()} "
                f"fk_xyz={np.round(fk_xyz, 4).tolist()} "
                f"pos_error_m={pos_error:.4f} rot_error={rot_error:.4f} "
                f"pos_tolerance_m={position_tolerance:.3f} "
                f"rot_tolerance={orientation_tolerance:.2f} "
                f"max_delta_rad={max_delta:.3f} "
                f"heading_error={None if heading_error is None else round(heading_error, 4)} "
                f"approach_axis_error={None if approach_axis_error is None else round(approach_axis_error, 4)} "
                f"q={np.round(q, 4).tolist()} "
                f"command_q={np.round(target, 4).tolist()}",
                flush=True,
            )
        arrived = (pos_error < position_tolerance
                   and rot_error < orientation_tolerance
                   and (heading_error is None or heading_error < heading_tolerance)
                   and (approach_axis_error is None
                        or approach_axis_error < approach_axis_tolerance))
        self.stable_steps = self.stable_steps + 1 if arrived else 0
        if self.stable_steps >= 6:
            if self.debug:
                fk_xyz, _ = self.kinematics.forward(q)
                print(
                    f"[RGBD] reached object={self.object_id} state={self.state} "
                    f"step={self.state_steps} fk_xyz={np.round(fk_xyz, 4).tolist()} "
                    f"pos_error_m={pos_error:.4f} rot_error={rot_error:.4f} "
                    f"heading_error={None if heading_error is None else round(heading_error, 4)} "
                    f"q={np.round(q, 4).tolist()}",
                    flush=True,
                )
            self._enter(next_state)
        elif self.state_steps > timeout:
            # A stalled IK target must not silently advance to a grasp.
            self._retry_or_stop()
        return np.concatenate((target, gripper))

    def _retry_or_stop(self):
        self.retry += 1
        if self.retry >= self._MAX_RETRIES:
            self.terminal = True
            self._enter("STOP")
        else:
            self._enter("RECOVER")

    def step(self, proprio, rgb, depth):
        q = np.asarray(proprio[:8], dtype=np.float64) + DEFAULT_Q
        # Evaluators may reuse one AlgSolution across episodes. A reset returns
        # the arm and previous action to their defaults.
        if self.state != "HOME" and np.max(np.abs(q[:6] - DEFAULT_Q[:6])) < 0.025 \
                and np.max(np.abs(proprio[16:24])) < 0.02:
            self.reset()
        # The final place check advances object_ptr past OBJECT_ORDER. Handle
        # DONE before any later branch can index self.object_id again.
        if self.state == "DONE":
            self.terminal = True
        if self.terminal:
            gripper = GRIPPER_CLOSE if self.state == "HOLD_BOTTLE" else GRIPPER_OPEN
            return np.concatenate((q[:6], gripper)), True

        self.state_steps += 1
        if self.state == "HOME":
            err = HOME_Q - q[:6]
            arm = q[:6] + np.clip(err, -0.04, 0.04)
            self.stable_steps = self.stable_steps + 1 if np.max(np.abs(err)) < 0.07 else 0
            if self.stable_steps >= 8:
                self._enter("SCAN")
            elif self.state_steps > 300:
                self.terminal = True
            return np.concatenate((arm, GRIPPER_OPEN)), self.terminal

        if self.state == "SCAN":
            updated = self._refresh_vision(rgb, depth, force=self.state_steps == 1)
            estimate = self.perception.detect(self.object_id) if updated else None
            if self._valid_scan_candidate(estimate):
                self.estimate = estimate
                if self.debug:
                    body_y_info = (
                        f" body_y_points={estimate.body_y_point_count} "
                        f"body_y_shift_m={estimate.body_y_shift_m:+.4f}"
                        if self.object_id == 2 else "")
                    banana_mid_info = (
                        f" banana_raw_points={estimate.banana_raw_point_count} "
                        f"banana_mid_points={estimate.banana_mid_point_count} "
                        f"banana_span_m={estimate.banana_span_m:.4f} "
                        f"banana_shift_xy_m=[{estimate.banana_shift_x_m:+.4f}, "
                        f"{estimate.banana_shift_y_m:+.4f}]"
                        if self.object_id == 3 else "")
                    print(f"[RGBD] detected object={self.object_id} xy={estimate.xy.tolist()} "
                          f"points={estimate.point_count}{body_y_info}{banana_mid_info}",
                          flush=True)
                self._grasp_plan()
                self._enter("PRE_GRASP")
            elif self.state_steps > 120:
                self._retry_or_stop()
            return np.concatenate((q[:6], GRIPPER_OPEN)), self.terminal

        if self.state == "RECOVER":
            # Clear the camera view before obtaining another grasp estimate.
            # This is a clearance move, not a grasp; its top-down orientation
            # need not meet the tighter default tolerance near the arm base.
            out = self._motion(q[:6], RETRACT_XYZ, top_down_rotation(),
                               GRIPPER_OPEN, "SCAN", orientation_tolerance=1.1)
            return out, self.terminal

        oid = self.object_id
        place_xy = BASKET_SLOTS[oid]
        grasp_z = (BOTTLE_SIDE_GRASP_Z if oid == 2
                   else OBJECT_CENTER_Z[oid] + GRASP_Z_OFFSET[oid])
        if self.state == "PRE_GRASP":
            if oid == 2:
                out = self._bottle_pre_motion(q[:6])
            else:
                # Sugar must finish wrist yaw before descending. At the far
                # edge, a small tool-tilt residual is allowed, but not a
                # heading error that would miss the box sides.
                out = self._motion(q[:6],
                                   np.r_[self.grasp_xy, grasp_z + PRE_GRASP_CLEARANCE],
                                   self.grasp_rotation, GRIPPER_OPEN, "DESCEND",
                                   orientation_tolerance=0.75 if oid == 1 else 0.8,
                                   heading_tolerance=0.2 if oid == 1 else None)
        elif self.state == "BOTTLE_SIDE_STAGE":
            # Descend outside the shoulder. Either near-arm X or proximity to
            # the sugar-side Y edge can increase clearance; the final grasp X
            # stays unchanged. Never advance merely because of a timeout.
            target = np.r_[self.grasp_xy + [self._bottle_stage_offset(), 0.0],
                           grasp_z]
            fk_xyz, _ = self.kinematics.forward(q[:6])
            if (self.bottle_stage_lowest_z is None
                    or fk_xyz[2] < self.bottle_stage_lowest_z
                    - self._BOTTLE_STAGE_MIN_Z_PROGRESS):
                self.bottle_stage_lowest_z = fk_xyz[2]
                self.bottle_stage_no_progress = 0
            else:
                self.bottle_stage_no_progress += 1
            if (self.bottle_stage_no_progress >= self._BOTTLE_STAGE_STALL_STEPS
                    and fk_xyz[2] - grasp_z > 0.025):
                if self.debug:
                    print(f"[RGBD] bottle stage stalled step={self.state_steps} "
                          f"fk_z={fk_xyz[2]:.4f} "
                          f"remaining_z_m={fk_xyz[2] - grasp_z:.4f}; "
                          "recovering", flush=True)
                self._retry_or_stop()
                out = np.concatenate((q[:6], GRIPPER_OPEN))
            else:
                out = self._motion(q[:6], target, self.grasp_rotation,
                                   GRIPPER_OPEN, "BOTTLE_SIDE_APPROACH",
                                   orientation_tolerance=0.30,
                                   approach_axis_tolerance=0.25,
                                   max_delta=0.02,
                                   timeout=self._BOTTLE_STAGE_TIMEOUT)
        elif self.state == "BOTTLE_SIDE_APPROACH":
            # A direct IK jump to the final pose can arc 3 cm sideways near
            # the bottle even when its final FK error is only a few mm.
            out = self._bottle_side_approach_motion(q[:6], grasp_z)
        elif self.state == "DESCEND":
            # Joint limits can leave a modest wrist-angle residual even when
            # the grasp position is within centimetres of its target.
            if oid == 1:
                # Keep the sugar grasp pose unchanged. Reduce joint increments
                # for the last 6 cm, and stop pushing if FK shows no further
                # descent for 60 control steps while still above the target.
                fk_xyz, _ = self.kinematics.forward(q[:6])
                remaining_z = fk_xyz[2] - grasp_z
                if self.descend_lowest_z is None or fk_xyz[2] < self.descend_lowest_z - 0.001:
                    self.descend_lowest_z = fk_xyz[2]
                    self.descend_no_progress = 0
                else:
                    self.descend_no_progress += 1
                if self.descend_no_progress >= self._SUGAR_STALL_STEPS and remaining_z > 0.025:
                    if self.debug:
                        print(f"[RGBD] sugar descent stalled object={oid} "
                              f"step={self.state_steps} fk_z={fk_xyz[2]:.4f} "
                              f"remaining_z_m={remaining_z:.4f}; recovering", flush=True)
                    self._retry_or_stop()
                    out = np.concatenate((q[:6], GRIPPER_OPEN))
                else:
                    max_delta = 0.02 if remaining_z > 0.06 else 0.008
                    out = self._motion(q[:6], np.r_[self.grasp_xy, grasp_z],
                                       self.grasp_rotation, GRIPPER_OPEN, "CLOSE",
                                       orientation_tolerance=0.45, position_tolerance=0.015,
                                       heading_tolerance=0.2, max_delta=max_delta,
                                       timeout=self._SUGAR_DESCEND_TIMEOUT)
            elif oid == 3:
                # Approach vertically through small Cartesian waypoints.
                # Close either at the low pose or after descent stops making
                # progress, even if the fixed low target was unreachable.
                fk_xyz, _ = self.kinematics.forward(q[:6])
                remaining_z = fk_xyz[2] - grasp_z
                if (self.descend_lowest_z is None
                        or fk_xyz[2] < self.descend_lowest_z - 0.001):
                    self.descend_lowest_z = fk_xyz[2]
                    self.descend_no_progress = 0
                else:
                    self.descend_no_progress += 1
                stalled = self.descend_no_progress >= self._BANANA_STALL_STEPS
                timed_out = self.state_steps > self._BANANA_DESCEND_TIMEOUT
                if stalled:
                    # Contact or a kinematic limit can stop the open fingers
                    # above the requested height. Close at the measured pose;
                    # the lift and vision check decide whether a grasp formed.
                    if self.debug:
                        xy_error = float(np.linalg.norm(fk_xyz[:2] - self.grasp_xy))
                        print(f"[RGBD] banana descent stalled; closing "
                              f"step={self.state_steps} "
                              f"fk_xyz={np.round(fk_xyz, 4).tolist()} "
                              f"remaining_z_m={remaining_z:.4f} "
                              f"xy_error_m={xy_error:.4f}", flush=True)
                    self._enter("CLOSE")
                    out = np.concatenate((q[:6], GRIPPER_CLOSE))
                elif timed_out:
                    if self.debug:
                        print(f"[RGBD] banana descent timed out "
                              f"step={self.state_steps} fk_z={fk_xyz[2]:.4f} "
                              f"remaining_z_m={remaining_z:.4f}; recovering",
                              flush=True)
                    self._retry_or_stop()
                    out = np.concatenate((q[:6], GRIPPER_OPEN))
                else:
                    above_slow_zone = fk_xyz[2] > self._BANANA_SLOW_ZONE_Z
                    z_step = (self._BANANA_FAST_DESCEND_Z_STEP if above_slow_zone
                              else self._BANANA_DESCEND_Z_STEP)
                    max_delta = (self._BANANA_FAST_DESCEND_MAX_DELTA if above_slow_zone
                                 else self._BANANA_DESCEND_MAX_DELTA)
                    waypoint_z = max(grasp_z,
                                     fk_xyz[2] - z_step)
                    target_q, _, rot_error = self.kinematics.step(
                        q[:6], np.r_[self.grasp_xy, waypoint_z],
                        self.grasp_rotation,
                        max_delta=max_delta)
                    xy_error = float(np.linalg.norm(fk_xyz[:2] - self.grasp_xy))
                    arrived = (-0.008 <= remaining_z < 0.012
                               and xy_error < 0.010 and rot_error < 0.35)
                    self.stable_steps = self.stable_steps + 1 if arrived else 0
                    if self.debug and self.state_steps in (1, 50, 150, 300):
                        print(f"[RGBD] banana vertical descent "
                              f"step={self.state_steps} "
                              f"target_xyz={np.round(np.r_[self.grasp_xy, grasp_z], 4).tolist()} "
                              f"waypoint_z={waypoint_z:.4f} "
                              f"z_step_m={z_step:.3f} "
                              f"max_delta_rad={max_delta:.3f} "
                              f"fk_xyz={np.round(fk_xyz, 4).tolist()} "
                              f"xy_error_m={xy_error:.4f} "
                              f"z_error_m={remaining_z:.4f} "
                              f"rot_error={rot_error:.4f}", flush=True)
                    if self.stable_steps >= 6:
                        if self.debug:
                            print(f"[RGBD] banana low grasp pose reached "
                                  f"step={self.state_steps} "
                                  f"fk_xyz={np.round(fk_xyz, 4).tolist()} "
                                  f"z_error_m={remaining_z:.4f}", flush=True)
                        self._enter("CLOSE")
                        out = np.concatenate((q[:6], GRIPPER_CLOSE))
                    else:
                        out = np.concatenate((target_q, GRIPPER_OPEN))
            else:
                out = self._motion(q[:6], np.r_[self.grasp_xy, grasp_z],
                                   self.grasp_rotation, GRIPPER_OPEN, "CLOSE",
                                   orientation_tolerance=0.45, position_tolerance=0.015)
        elif self.state == "CLOSE":
            out = np.concatenate((q[:6], GRIPPER_CLOSE))
            if self.debug and self.state_steps in (1, 40):
                print(f"[RGBD] gripper object={oid} state=CLOSE step={self.state_steps} "
                      f"joint_pos={np.round(q[6:8], 4).tolist()}", flush=True)
            if self.state_steps >= 40:
                if oid == 3:
                    # The banana can slide out if the wrist sweeps sideways
                    # or rotates as soon as the jaws make contact. Capture
                    # the measured grasp pose and lift vertically first.
                    fk_xyz, fk_rotation = self.kinematics.forward(q[:6])
                    self.banana_lift_xy = fk_xyz[:2].copy()
                    self.banana_lift_rotation = fk_rotation.copy()
                    self.banana_lift_clear_z = fk_xyz[2] + self._BANANA_LIFT_CLEARANCE
                    self._enter("BANANA_LIFT_CLEAR")
                else:
                    self._enter("LIFT")
        elif self.state == "BANANA_LIFT_CLEAR":
            out = self._motion(q[:6],
                               np.r_[self.banana_lift_xy, self.banana_lift_clear_z],
                               self.banana_lift_rotation, GRIPPER_CLOSE, "LIFT",
                               orientation_tolerance=0.35,
                               position_tolerance=0.012, max_delta=0.01,
                               timeout=self._BANANA_LIFT_CLEAR_TIMEOUT)
        elif self.state == "LIFT":
            lift_z = min(CARRY_Z, grasp_z + LIFT_CLEARANCE)
            if oid == 2:
                lift_xy = self.grasp_xy + [BOTTLE_SIDE_GRASP_X, 0.0]
                lift_rotation = self.grasp_rotation
                lift_max_delta = self._BOTTLE_LIFT_MAX_DELTA
                lift_timeout = self._BOTTLE_LIFT_TIMEOUT
            elif oid == 3:
                lift_xy = (self.banana_lift_xy if self.banana_lift_xy is not None
                           else self.grasp_xy)
                lift_rotation = (self.banana_lift_rotation
                                 if self.banana_lift_rotation is not None
                                 else self.grasp_rotation)
                lift_max_delta = 0.015
                lift_timeout = self._BANANA_LIFT_TIMEOUT
            else:
                lift_xy = self.grasp_xy
                lift_rotation = self.grasp_rotation
                lift_max_delta = 0.04
                lift_timeout = None
            if oid == 2 and self.state_steps > self._BOTTLE_LIFT_TIMEOUT:
                # A bottle already well clear of the table must not be
                # released merely because the exact lift target stalled.
                fk_xyz, _ = self.kinematics.forward(q[:6])
                safely_lifted = (fk_xyz[2] >= TABLE_TOP_Z + 0.29
                                 and np.linalg.norm(fk_xyz[:2] - lift_xy) < 0.05)
                if safely_lifted:
                    if self.debug:
                        print(f"[RGBD] bottle lift clearance reached "
                              f"fk_xyz={np.round(fk_xyz, 4).tolist()}; "
                              "continuing with gripper closed", flush=True)
                    self._enter("BOTTLE_CARRY_CLEAR")
                    out = np.concatenate((q[:6], GRIPPER_CLOSE))
                else:
                    if self.debug:
                        print(f"[RGBD] bottle lift stalled "
                              f"fk_xyz={np.round(fk_xyz, 4).tolist()}; "
                              "holding gripper closed", flush=True)
                    self._enter("HOLD_BOTTLE")
                    self.terminal = True
                    out = np.concatenate((q[:6], GRIPPER_CLOSE))
            else:
                out = self._motion(q[:6], np.r_[lift_xy, lift_z],
                                   lift_rotation, GRIPPER_CLOSE,
                                   "BOTTLE_CARRY_CLEAR" if oid == 2 else "VERIFY_GRASP",
                                   max_delta=lift_max_delta, timeout=lift_timeout)
        elif self.state == "VERIFY_GRASP":
            out = np.concatenate((q[:6], GRIPPER_CLOSE))
            if self.state_steps == 12:
                self._refresh_vision(rgb, depth, force=True)
                remaining = self.perception.detect(oid)
                on_table = (self._matches_initial_color(remaining)
                            and remaining.point_count >= max(12, self.initial_table_count * 0.40))
                if on_table:
                    self._retry_or_stop()
                else:
                    self._enter("TRANSPORT")
        elif self.state == "BOTTLE_CARRY_CLEAR":
            out = self._bottle_carry_motion(
                q[:6], BOTTLE_CARRY_WAYPOINTS[0], BOTTLE_CARRY_Z,
                "BOTTLE_CARRY_CROSS")
        elif self.state == "BOTTLE_CARRY_CROSS":
            out = self._bottle_carry_motion(
                q[:6], BOTTLE_CARRY_WAYPOINTS[1], BOTTLE_CARRY_Z,
                "BOTTLE_CARRY_OVER_BIN")
        elif self.state == "BOTTLE_CARRY_OVER_BIN":
            out = self._bottle_carry_motion(
                q[:6], self._bottle_place_base_xy(), BOTTLE_CARRY_Z,
                "BOTTLE_PLACE")
        elif self.state == "BOTTLE_PLACE":
            out = self._bottle_carry_motion(
                q[:6], self._bottle_place_base_xy(), BOTTLE_PLACE_Z,
                "OPEN")
        elif self.state == "TRANSPORT":
            if oid == 2:
                # Older saved controller states must not send a held bottle
                # through the top-down transport path.
                self._enter("BOTTLE_CARRY_CLEAR")
                out = np.concatenate((q[:6], GRIPPER_CLOSE))
            else:
                out = self._motion(q[:6], np.r_[place_xy, CARRY_Z],
                                   top_down_rotation(), GRIPPER_CLOSE, "PLACE", 1.0)
        elif self.state == "PLACE":
            # Release above the bin instead of requiring the gripper base to
            # reach its floor. If contact still stalls the arm, a release is
            # allowed only after FK confirms it is safely over the bin; the
            # usual retract/vision check then decides whether to advance.
            if self.state_steps > self._MOTION_TIMEOUT:
                fk_xyz, _ = self.kinematics.forward(q[:6])
                over_bin = (np.linalg.norm(fk_xyz[:2] - place_xy) < 0.05
                            and TABLE_TOP_Z + 0.20 <= fk_xyz[2] <= TABLE_TOP_Z + 0.34)
                if over_bin:
                    if self.debug:
                        print(f"[RGBD] place timeout object={oid} "
                              f"fk_xyz={np.round(fk_xyz, 4).tolist()}; "
                              "releasing over basket", flush=True)
                    self._enter("OPEN")
                    out = np.concatenate((q[:6], GRIPPER_OPEN))
                else:
                    self._retry_or_stop()
                    out = np.concatenate((q[:6], GRIPPER_CLOSE))
            else:
                out = self._motion(q[:6], np.r_[place_xy, PLACE_Z],
                                   top_down_rotation(), GRIPPER_CLOSE, "OPEN",
                                   orientation_tolerance=0.8)
        elif self.state == "OPEN":
            out = np.concatenate((q[:6], GRIPPER_OPEN))
            if self.state_steps >= 50:
                self._enter("LIFT_RETRACT")
        elif self.state == "LIFT_RETRACT":
            out = self._motion(q[:6], np.r_[place_xy, CARRY_Z],
                               top_down_rotation(), GRIPPER_OPEN, "RETRACT", 1.0)
        elif self.state == "RETRACT":
            if oid == 2:
                # After release, the purpose of retracting is to clear the
                # basket/camera view, not to hit one exact IK pose. The last
                # successful placement exceeded the generic 300-step limit
                # while already clear, which incorrectly counted as a retry.
                fk_xyz, _ = self.kinematics.forward(q[:6])
                clear = (fk_xyz[0] >= place_xy[0] + self._BOTTLE_RETRACT_CLEAR_X
                         and fk_xyz[1] >= place_xy[1] + self._BOTTLE_RETRACT_CLEAR_Y
                         and fk_xyz[2] >= TABLE_TOP_Z + self._BOTTLE_RETRACT_CLEAR_Z)
                self.bottle_retract_clear_steps = (
                    self.bottle_retract_clear_steps + 1 if clear else 0)
                if self.bottle_retract_clear_steps >= self._BOTTLE_RETRACT_CLEAR_STEPS:
                    if self.debug:
                        print(f"[RGBD] bottle post-place clearance reached "
                              f"fk_xyz={np.round(fk_xyz, 4).tolist()}; "
                              "verifying placement", flush=True)
                    self._enter("VERIFY_PLACE")
                    out = np.concatenate((q[:6], GRIPPER_OPEN))
                elif self.state_steps > self._BOTTLE_RETRACT_TIMEOUT:
                    # A release has already happened. Do not restart a grasp
                    # just because the arm could not leave the bin safely.
                    if self.debug:
                        print(f"[RGBD] bottle post-place retract stalled "
                              f"fk_xyz={np.round(fk_xyz, 4).tolist()}; "
                              "stopping without a grasp retry", flush=True)
                    self.terminal = True
                    self._enter("STOP")
                    out = np.concatenate((q[:6], GRIPPER_OPEN))
                else:
                    out = self._motion(q[:6], RETRACT_XYZ,
                                       top_down_rotation(), GRIPPER_OPEN,
                                       "VERIFY_PLACE", 1.1,
                                       timeout=self._BOTTLE_RETRACT_TIMEOUT)
            else:
                out = self._motion(q[:6], RETRACT_XYZ,
                                   top_down_rotation(), GRIPPER_OPEN, "VERIFY_PLACE", 1.1)
        elif self.state == "VERIFY_PLACE":
            out = np.concatenate((q[:6], GRIPPER_OPEN))
            if self.state_steps == 10:
                self._refresh_vision(rgb, depth, force=True)
                on_table = self.perception.detect(oid)
                basket_count = self.perception.basket_count(place_xy)
                back_on_table = self._matches_initial_color(on_table) and on_table.point_count >= max(12, self.initial_table_count * 0.40)
                basket_gain = basket_count - self.initial_basket_count
                if self.debug:
                    print(f"[RGBD] object={oid} back_on_table={back_on_table} "
                          f"basket_point_gain={basket_gain}", flush=True)
                if back_on_table:
                    self._retry_or_stop()
                elif oid == 2 and basket_gain < self._BOTTLE_MIN_BASKET_GAIN:
                    # A knocked-over bottle can leave its original detection
                    # band without entering the basket. Do not advance to the
                    # next object on absence alone, or blindly re-grasp an
                    # unlocalized bottle.
                    if self.debug:
                        print(f"[RGBD] bottle placement unverified "
                              f"basket_point_gain={basket_gain}; stopping",
                              flush=True)
                    self.terminal = True
                    self._enter("STOP")
                else:
                    self.object_ptr += 1
                    self.retry = 0
                    if self.object_ptr >= len(OBJECT_ORDER):
                        self._enter("DONE")
                        self.terminal = True
                        if self.debug:
                            print("[RGBD] all requested objects placed; "
                                  "stopping controller", flush=True)
                    else:
                        self._enter("SCAN")
        else:
            raise RuntimeError(f"Unknown state: {self.state}")
        return out, self.terminal


class AlgSolution:
    def __init__(self):
        self.controllers = []

    def reset(self):
        for controller in self.controllers:
            controller.reset()

    def get_action_spec(self):
        return {}

    def predicts(self, obs, current_score):
        if not isinstance(obs, dict) or "proprio" not in obs or "image" not in obs:
            raise ValueError("RGB-D policy requires obs['proprio'] and obs['image']; enable cameras.")
        image = obs["image"]
        if "video_rgb" not in image or "video_depth" not in image:
            raise ValueError("RGB-D policy requires video_rgb and video_depth observations.")
        proprio = _numpy(obs["proprio"])
        rgb = image["video_rgb"]
        depth = image["video_depth"]
        if proprio.ndim != 2 or proprio.shape[1] < 24:
            raise ValueError(f"Expected proprio shape (N, 24); got {proprio.shape}")
        if len(proprio) != 1:
            raise ValueError("The fixed-camera RGB-D policy currently supports --num_envs 1 only.")
        if len(self.controllers) != len(proprio):
            self.controllers = [_SingleEnvironmentPolicy() for _ in range(len(proprio))]
        actions = []
        giveup = False
        for index, controller in enumerate(self.controllers):
            target, stopped = controller.step(proprio[index], rgb[index], depth[index])
            actions.append(((target - DEFAULT_Q) / ACTION_SCALE).tolist())
            giveup = giveup or stopped
        return {"action": actions, "giveup": giveup}
