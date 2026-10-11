"""Single-episode demo collection and success checking for Task E."""

import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv
from atec_rl_lab.utils import CartesianController
from atec_rl_lab.tasks.task_e.env_cfg import (
    BASKET_SUCCESS_CENTER,
    BASKET_SUCCESS_HALF_X,
    BASKET_SUCCESS_HALF_Y,
)

from .config import (
    ACTION_SCALE,
    GRIPPER_OPEN_POS, GRIPPER_CLOSE_POS,
    RETRACT_POS_X, RETRACT_POS_Y, CARRY_Z,
    DEFAULT_PLACE_QUAT_W,
    BASKET_SUCCESS_MIN_Z, BASKET_SUCCESS_MAX_Z,
    OBJ_SPAWN_X_MIN, OBJ_SPAWN_X_MAX, OBJ_SPAWN_Z, OBJ_SPAWN_Y_BANDS,
    OBJ_HALF_EXTENTS, OBJ_BBOX_MARGIN,
    SIDE_GRASP_OBJECT_IDS,
    WARMUP_STEPS, SETTLE_STEPS,
)
from .state_machine import PickPlaceStateMachine


def _object_in_basket_region(
    env: ManagerBasedRLEnv,
    object_name: str,
    *,
    center_xy: tuple[float, float],
    half_x: float,
    half_y: float,
    min_z: float,
    max_z: float,
) -> bool:
    """Match ObjectsInBasketDone geometry for one rigid object."""
    obj = env.unwrapped.scene.rigid_objects[object_name]
    pos = obj.data.root_pos_w[0] - env.unwrapped.scene.env_origins[0]
    px, py, pz = pos[0].item(), pos[1].item(), pos[2].item()
    cx, cy = center_xy
    if abs(px - cx) > half_x or abs(py - cy) > half_y:
        return False
    return min_z <= pz <= max_z


def check_all_objects_in_basket(env: ManagerBasedRLEnv) -> bool:
    """True when all three task objects satisfy the evaluation success region."""
    center_xy = (BASKET_SUCCESS_CENTER[0], BASKET_SUCCESS_CENTER[1])
    for obj_idx in (1, 2, 3):
        if not _object_in_basket_region(
            env,
            f"object_{obj_idx}",
            center_xy=center_xy,
            half_x=BASKET_SUCCESS_HALF_X,
            half_y=BASKET_SUCCESS_HALF_Y,
            min_z=BASKET_SUCCESS_MIN_Z,
            max_z=BASKET_SUCCESS_MAX_Z,
        ):
            return False
    return True


def check_objects_in_basket(env: ManagerBasedRLEnv, pick_objects: list[int]) -> bool:
    """True when every object in *pick_objects* is inside the basket success region."""
    center_xy = (BASKET_SUCCESS_CENTER[0], BASKET_SUCCESS_CENTER[1])
    for obj_idx in pick_objects:
        if not _object_in_basket_region(
            env,
            f"object_{obj_idx}",
            center_xy=center_xy,
            half_x=BASKET_SUCCESS_HALF_X,
            half_y=BASKET_SUCCESS_HALF_Y,
            min_z=BASKET_SUCCESS_MIN_Z,
            max_z=BASKET_SUCCESS_MAX_Z,
        ):
            return False
    return True


def describe_basket_check(
    env: ManagerBasedRLEnv, pick_objects: list[int]
) -> str:
    """Human-readable per-object position vs evaluation success bounds."""
    cx, cy = BASKET_SUCCESS_CENTER[0], BASKET_SUCCESS_CENTER[1]
    lines = [
        f"  success XY center=({cx:.3f}, {cy:.3f}), "
        f"half=({BASKET_SUCCESS_HALF_X}, {BASKET_SUCCESS_HALF_Y}), "
        f"Z in [{BASKET_SUCCESS_MIN_Z:.4f}, {BASKET_SUCCESS_MAX_Z:.4f}]",
    ]
    for obj_idx in pick_objects:
        name = f"object_{obj_idx}"
        obj = env.unwrapped.scene.rigid_objects[name]
        pos = obj.data.root_pos_w[0] - env.unwrapped.scene.env_origins[0]
        px, py, pz = pos[0].item(), pos[1].item(), pos[2].item()
        ok_x = abs(px - cx) <= BASKET_SUCCESS_HALF_X
        ok_y = abs(py - cy) <= BASKET_SUCCESS_HALF_Y
        ok_z = BASKET_SUCCESS_MIN_Z <= pz <= BASKET_SUCCESS_MAX_Z
        ok = ok_x and ok_y and ok_z
        lines.append(
            f"  {name}: pos=({px:.3f}, {py:.3f}, {pz:.3f}) -> "
            f"{'OK' if ok else 'FAIL'} "
            f"(x{' OK' if ok_x else ' OUT'}, y{' OK' if ok_y else ' OUT'}, z{' OK' if ok_z else ' OUT'})"
        )
    return "\n".join(lines)


def settle_physics_after_demo(
    env: ManagerBasedRLEnv,
    default_jpos: torch.Tensor,
    steps: int,
) -> None:
    """Run extra env steps so released objects can fall/settle before success checks."""
    if steps <= 0:
        return
    zero = torch.zeros_like(default_jpos)
    for _ in range(steps):
        env.step(zero)


def _rerandomize_objects(env: ManagerBasedRLEnv, rng: np.random.Generator) -> None:
    """Place each object randomly with AABB-based overlap rejection."""
    placed: dict[int, tuple[float, float]] = {}  # obj_idx -> (x, y)

    for obj_idx in [1, 2, 3]:
        obj = env.unwrapped.scene.rigid_objects[f"object_{obj_idx}"]
        y_min, y_max = OBJ_SPAWN_Y_BANDS[obj_idx]
        hx, hy = OBJ_HALF_EXTENTS[obj_idx]

        x = y = None
        for _ in range(200):
            cx = float(rng.uniform(OBJ_SPAWN_X_MIN, OBJ_SPAWN_X_MAX))
            cy = float(rng.uniform(y_min, y_max))
            # AABB overlap check against all already-placed objects
            ok = all(
                abs(cx - px) >= hx + OBJ_HALF_EXTENTS[pi][0] + OBJ_BBOX_MARGIN or
                abs(cy - py) >= hy + OBJ_HALF_EXTENTS[pi][1] + OBJ_BBOX_MARGIN
                for pi, (px, py) in placed.items()
            )
            if ok:
                x, y = cx, cy
                break

        if x is None:  # fallback: band centre
            x = (OBJ_SPAWN_X_MIN + OBJ_SPAWN_X_MAX) / 2.0
            y = (y_min + y_max) / 2.0

        placed[obj_idx] = (x, y)
        state = obj.data.default_root_state[0:1].clone()
        state[0, 0] = x
        state[0, 1] = y
        state[0, 2] = OBJ_SPAWN_Z
        state[0, 7:] = 0.0   # zero velocities
        obj.write_root_state_to_sim(state)

    env.unwrapped.scene.write_data_to_sim()
    env.unwrapped.sim.forward()


def _restore_side_grasp_yaw(env: ManagerBasedRLEnv) -> None:
    """Put side-grasp objects back on their canonical upright yaw after warm-up/settle."""
    for obj_idx in SIDE_GRASP_OBJECT_IDS:
        obj = env.unwrapped.scene.rigid_objects[f"object_{obj_idx}"]
        state = obj.data.root_state_w[0:1].clone()
        state[0, 3:7] = obj.data.default_root_state[0, 3:7]
        state[0, 7:] = 0.0
        obj.write_root_state_to_sim(state)
    env.unwrapped.scene.write_data_to_sim()
    env.unwrapped.sim.forward()


def collect_one_demo(
    env:         ManagerBasedRLEnv,
    robot,
    ik_ctrl:     CartesianController,
    arm_ids:     list[int],
    gripper_ids: list[int],
    pick_objects: list[int],
    device:      str,
    default_jpos: torch.Tensor,
    rng:         np.random.Generator,
    camera=None,
) -> dict | None:
    """Run one full episode and return recorded data, or None on failed early stop.

    Returns a dict with keys:
      qpos    (T, 8)        absolute joint positions
      qvel    (T, 8)        joint velocities
      ee_pos  (T, 3)        end-effector position (world frame)
      ee_quat (T, 4)        end-effector quaternion (w,x,y,z)
      action  (T, 8)        env action = (joint_target - default_jpos) / ACTION_SCALE
      frames  (T, H, W, 3)  RGB uint8 — only present when camera is given
    """
    env.reset()
    robot.write_joint_state_to_sim(
        robot.data.default_joint_pos,
        torch.zeros_like(robot.data.default_joint_vel),
    )

    _rerandomize_objects(env, rng)      # write new object positions to sim + sim.forward()
    default_jpos = robot.data.default_joint_pos.clone()

    ee_home = torch.tensor([[RETRACT_POS_X, RETRACT_POS_Y, CARRY_Z]],
                            dtype=torch.float32, device=device)
    eq_home = torch.tensor([DEFAULT_PLACE_QUAT_W], dtype=torch.float32, device=device)
    g_open  = torch.tensor([GRIPPER_OPEN_POS], dtype=torch.float32, device=device)

    robot.update(dt=env.unwrapped.physics_dt)
    ik_ctrl.reset()

    # Warm-up: drive arm to HOME position (not recorded)
    for _ in range(WARMUP_STEPS):
        _step_to(env, robot, ik_ctrl, arm_ids, gripper_ids,
                 ee_home, eq_home, g_open, default_jpos)

    # Settle, then restore the canonical yaw of side-grasp objects before
    # the grasp frame is computed.
    for _ in range(SETTLE_STEPS):
        _step_to(env, robot, ik_ctrl, arm_ids, gripper_ids,
                 ee_home, eq_home, g_open, default_jpos)
    _restore_side_grasp_yaw(env)

    sm = PickPlaceStateMachine(pick_objects, device)
    for obj_idx in pick_objects:
        obj_quat = env.unwrapped.scene.rigid_objects[f"object_{obj_idx}"] \
                       .data.root_state_w[0, 3:7]
        sm.set_grasp_quat(obj_idx, obj_quat)

    ik_ctrl.reset()

    # ---- Recording loop ---- #
    qpos_buf, qvel_buf, ee_pos_buf, ee_quat_buf, action_buf = [], [], [], [], []
    frames_buf = [] if camera is not None else None

    while not sm.done:
        if sm.needs_ik_reset:
            robot.update(dt=env.unwrapped.physics_dt)
            ik_ctrl.reset()
            sm.needs_ik_reset = False

        obj_pos_w = env.unwrapped.scene.rigid_objects[sm.current_object_key] \
                        .data.root_pos_w[0].clone()
        ee_pos_des, ee_quat_des, gripper_cmd = sm.tick(obj_pos_w)

        arm_jpos_des   = ik_ctrl.compute(ee_pos_des.unsqueeze(0), ee_quat_des.unsqueeze(0))
        gripper_vals   = GRIPPER_OPEN_POS if gripper_cmd == "open" else GRIPPER_CLOSE_POS
        gripper_target = torch.tensor([gripper_vals], dtype=torch.float32, device=device)

        full_target = robot.data.joint_pos.clone()
        full_target[:, arm_ids]     = arm_jpos_des
        full_target[:, gripper_ids] = gripper_target
        env_action = (full_target - default_jpos) / ACTION_SCALE

        # Record BEFORE stepping (obs at time t, action at time t)
        qpos_buf.append(robot.data.joint_pos[0].cpu().numpy())
        qvel_buf.append(robot.data.joint_vel[0].cpu().numpy())
        ee_pos_buf.append(ik_ctrl.ee_pos_w[0].cpu().numpy())
        ee_quat_buf.append(ik_ctrl.ee_quat_w[0].cpu().numpy())
        action_buf.append(env_action[0].cpu().numpy())
        if frames_buf is not None:
            rgba = camera.data.output["rgb"][0].cpu().numpy()
            frames_buf.append(rgba[:, :, :3])

        _, _, terminated, truncated, _ = env.step(env_action)

        if bool(terminated.any()):
            if check_all_objects_in_basket(env):
                print("[INFO] Evaluation success termination — keeping recorded trajectory.")
                break
            print("[WARN] Episode terminated (non-success) — skipping demo.")
            return None
        if bool(truncated.any()):
            if sm.done:
                print("[INFO] State machine finished at timeout boundary — keeping trajectory.")
                break
            print("[WARN] Episode timed out before state machine finished — skipping demo.")
            return None

    if not qpos_buf:
        return None

    result = {
        "qpos":    np.stack(qpos_buf),
        "qvel":    np.stack(qvel_buf),
        "ee_pos":  np.stack(ee_pos_buf),
        "ee_quat": np.stack(ee_quat_buf),
        "action":  np.stack(action_buf),
    }
    if frames_buf is not None:
        result["frames"] = np.stack(frames_buf)
    return result


# ------------------------------------------------------------------ #
# Internal helper
# ------------------------------------------------------------------ #

def _step_to(env, robot, ik_ctrl, arm_ids, gripper_ids,
             ee_pos, ee_quat, gripper_target, default_jpos):
    """Single IK step toward a target pose (utility used during warm-up/settle)."""
    arm_des = ik_ctrl.compute(ee_pos, ee_quat)
    tgt = robot.data.joint_pos.clone()
    tgt[:, arm_ids]     = arm_des
    tgt[:, gripper_ids] = gripper_target
    env.step((tgt - default_jpos) / ACTION_SCALE)
    robot.update(dt=env.unwrapped.physics_dt)
