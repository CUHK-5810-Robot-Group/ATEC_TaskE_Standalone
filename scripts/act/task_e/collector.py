"""Single-episode demo collection and success checking for Task E."""

import numpy as np
import torch
import torchvision.transforms.functional as TF
from isaaclab.envs import ManagerBasedRLEnv
from atec_rl_lab.utils import CartesianController
from atec_rl_lab.tasks.task_e.env_cfg import (
    TABLE_CENTER_X, TABLE_CENTER_Y, TABLE_TOP_Z,
    BASKET_CENTER_X, BASKET_CENTER_Y,
)

from .config import (
    ACTION_SCALE,
    GRIPPER_OPEN_POS, GRIPPER_CLOSE_POS,
    RETRACT_POS_X, RETRACT_POS_Y, CARRY_Z,
    DEFAULT_PLACE_QUAT_W,
    BASKET_IN_X, BASKET_IN_Y,
    OBJ_SPAWN_X_MIN, OBJ_SPAWN_X_MAX, OBJ_SPAWN_Z, OBJ_SPAWN_Y_BANDS,
    OBJ_HALF_EXTENTS, OBJ_BBOX_MARGIN,
    WARMUP_STEPS, SETTLE_STEPS,
    STEPS,
    DIVERSITY_STEPS_JITTER, DIVERSITY_PERTURB_SIGMA,
    DIVERSITY_PERTURB_STATES, DIVERSITY_PERTURB_PROB,
)
from .state_machine import PickPlaceStateMachine


def make_steps_profile(rng: np.random.Generator,
                       jitter: tuple[float, float] = DIVERSITY_STEPS_JITTER
                       ) -> dict[str, int]:
    """给一条示范抽一套各状态步数（时序抖动）。

    为什么要：原状态机是纯步数推进的，53 条示范的夹爪切换时刻完全相同
    （见 config.DIVERSITY_* 的说明），示范里没有"时序不确定"的信息。
    给每个状态乘一个独立随机因子后，同一段路径上"何时切换"不再固定 ——
    例如 CLOSE 会在不同的下降高度触发。

    下限钳到 8 步，避免抖出 0 步导致状态机卡死（`_count >= 0` 恒真）。
    """
    lo, hi = jitter
    return {k: max(8, int(round(v * rng.uniform(lo, hi)))) for k, v in STEPS.items()}


def perturb_arm(env: ManagerBasedRLEnv, robot, ik_ctrl, rng: np.random.Generator,
                sigma: float, arm_ids: list[int]) -> None:
    """给 6 个臂关节注入一次高斯扰动（DAgger 式状态增广）。

    直接 `write_joint_state_to_sim` 把关节挪开一点，然后让状态机自己纠回来
    —— 状态机是笛卡尔目标驱动的，所以纠回来的**过程**就是一条"恢复示范"。
    这与评测器复位机器人用的是同一个 API，属于合法的状态注入。

    注意：扰动后要 `ik_ctrl.reset()`，否则差速 IK 的上一帧误差会带进来。
    """
    jp = robot.data.joint_pos.clone()
    noise = torch.from_numpy(
        rng.normal(0.0, sigma, size=(1, len(arm_ids))).astype("float32")
    ).to(jp.device)
    jp[:, arm_ids] += noise
    robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
    env.unwrapped.scene.write_data_to_sim()
    env.unwrapped.sim.forward()
    robot.update(dt=env.unwrapped.physics_dt)
    ik_ctrl.reset()


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
            # （已恢复为交接包原始实现：任一轴分开即视为不重叠。
            #   曾短暂改为加严的强分离校验，但它改变了实际采样分布，
            #   与"不得改动 OBJ_SPAWN_* 分布"的要求冲突，故撤回。）
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


def rerandomize_objects_multi(env: ManagerBasedRLEnv, seeds: list[int]) -> None:
    """给**每个环境**独立随机摆放（用于 num_envs>1 的并行多种子测试）。

    为什么需要单独一个函数
    ----------------------
    `_rerandomize_objects` 用单个 `np.random.Generator` 算出一组位置后
    **对所有环境写同一组值**（它只写 state[0]，即 env_0）。实测后果：
    num_envs=16 时 16 个环境的物体位置**完全相同**，等于把同一条 episode
    重复 16 次，起不到"并行测多个种子"的作用。

    本函数对每个环境各用一个独立 RNG（由外部传入互不相同的 seed），
    其余采样规则（X/Y 带、AABB 排斥、沉降高度）与 `_rerandomize_objects`
    **逐字一致**，因此不改变实际采样分布。

    参数
    ----
    seeds : 种子列表；长度不足时循环复用。
    """
    n = env.unwrapped.scene.num_envs
    rngs = [np.random.default_rng(int(seeds[i % len(seeds)])) for i in range(n)]

    for obj_idx in [1, 2, 3]:
        obj = env.unwrapped.scene.rigid_objects[f"object_{obj_idx}"]
        y_min, y_max = OBJ_SPAWN_Y_BANDS[obj_idx]
        hx, hy = OBJ_HALF_EXTENTS[obj_idx]

        placed: list[dict[int, tuple[float, float]]] = [dict() for _ in range(n)]
        state = obj.data.default_root_state[0:n].clone()
        for e in range(n):
            rng = rngs[e]
            x = y = None
            for _ in range(200):
                cx = float(rng.uniform(OBJ_SPAWN_X_MIN, OBJ_SPAWN_X_MAX))
                cy = float(rng.uniform(y_min, y_max))
                ok = all(
                    abs(cx - px) >= hx + OBJ_HALF_EXTENTS[pi][0] + OBJ_BBOX_MARGIN or
                    abs(cy - py) >= hy + OBJ_HALF_EXTENTS[pi][1] + OBJ_BBOX_MARGIN
                    for pi, (px, py) in placed[e].items()
                )
                if ok:
                    x, y = cx, cy
                    break
            if x is None:  # fallback: band centre
                x = (OBJ_SPAWN_X_MIN + OBJ_SPAWN_X_MAX) / 2.0
                y = (y_min + y_max) / 2.0
            placed[e][obj_idx] = (x, y)
            state[e, 0] = x
            state[e, 1] = y
            state[e, 2] = OBJ_SPAWN_Z
        state[:, 7:] = 0.0
        obj.write_root_state_to_sim(state)

    env.unwrapped.scene.write_data_to_sim()
    env.unwrapped.sim.forward()


# 入篮判定的 z 范围，必须与官方终止条件完全一致。
#
# 官方实现：atec_rl_lab/tasks/task_e/mdp/terminations.py 的 ObjectsInBasketDone
#     min_z = table_top_z
#     max_z = table_top_z + 0.15
#     inside_z = (z >= min_z) & (z <= max_z)
#
# 【修正历史】本文件原来写的是 `TABLE_TOP_Z + 0.1`，比官方**严了 5 cm**，
# 导致"竖立在篮内"的细高物体（object_2 中心 ≈0.93）被误判为没入篮 ——
# 用户实测确认那些物体其实好好地立在篮子里（有时还叠在 object_1 上）。
# 现改为与官方一致的 [TABLE_TOP_Z, TABLE_TOP_Z + 0.15]。
_BASKET_MIN_Z = TABLE_TOP_Z
_BASKET_MAX_Z = TABLE_TOP_Z + 0.15   # 与 terminations.py 的 max_z 保持一致

# 本工程实际配置的终止项（见 tasks/task_base/envs_base_cfg.py 的 TerminationsCfg
# 加上 task_e 自己的 basket_success）。名字写全，判定时逐个查"本步是否触发"。
_TERM_NAMES = ("basket_success", "time_out", "illegal_contact", "fall")


def fired_terms(env: ManagerBasedRLEnv) -> dict[str, bool]:
    """返回**本步真实触发**的终止项（取值来自 reset 之前）。

    ⚠️ 不要读 `termination_manager.active_terms`
    --------------------------------------------
    它是 `ManagerBase.active_terms` 的实现，TerminationManager 里直接
    `return self._term_names` —— 即"**已配置**的终止项名字"，恒含
    basket_success，与"本步有没有触发"毫无关系。用它判定会得出
    "任何终止（超时/掉落/碰撞）都是 success" 的错误结论（本项目踩过）。

    正确接口是 `termination_manager.get_term(name)` → `_term_dones[:, idx]`：
    该缓冲在 `TerminationManager.compute()` 里写入，而 compute() 发生在
    `env.step()` 内部 **reset 之前**（IsaacLab 2.3.2
    manager_based_rl_env.py: `reset_buf = termination_manager.compute()`
    在前，`self._reset_idx(reset_env_ids)` 在后），而且 reset 不会清掉它。
    所以它是"reset 前取证"最可靠的来源。
    """
    tm = getattr(env.unwrapped, "termination_manager", None)
    out: dict[str, bool] = {}
    if tm is None:
        return out
    for name in _TERM_NAMES:
        try:
            out[name] = bool(tm.get_term(name)[0].item())
        except Exception:
            out[name] = False
    return out


def termination_reason(env: ManagerBasedRLEnv) -> str:
    """本轮结束原因: "success" | "time_out" | "illegal_contact" | "fall" | "none"。

    【修正记录】原实现读 `active_terms`（= 已配置项，恒含 basket_success），
    因此**任何**终止都会被判成 "success"，失败分支是死代码。现改为读
    `fired_terms()` 的真实取值。
    """
    f = fired_terms(env)
    if f.get("basket_success"):
        return "success"
    for name in ("time_out", "illegal_contact", "fall"):
        if f.get(name):
            return name
    return "none"


def check_positions_in_basket(positions: dict[int, list[float]],
                              pick_objects: list[int]) -> tuple[bool, str]:
    """用 reset 前取证的位置做**独立交叉校验**（口径与官方 ObjectsInBasketDone 一致）。

    取证时序（已实测确认，别凭直觉）
    --------------------------------
    * 终止判定发生在 `env.step()` 内部、`_reset_idx` **之前**
      （manager_based_rl_env.step(): `termination_manager.compute()` → 奖励 →
      `_reset_idx`）。
    * 本工程**没有任何针对物体的 reset 事件**：实测把 object_3 搬进篮子、把机器人
      joint2 掰到 0.2 之后再调 `env.reset()`，物体与机器人关节**都保持改动后的值**
      （见 logs/probe_episode_reset.py）。`RigidObject.reset()` 只重置 wrench
      composer，不写回 root state。
    * 因此 `env.step()` 之后**紧接着**读 `root_pos_w`，拿到的正是"判定那一刻"的
      真实状态（场景 data 缓冲自最后一次 `scene.update()` 起就没有再刷新）——
      这就是 README §3.1 第 1 条要求的"末次状态应在 reset 前取证"。
    所以调用方要传"step 之后立刻读到的位置"，而**不是** step 之前的快照
    （step 前的快照比判定时刻早一步，临界物体会差出判定线，实测差 0.016 m）。

    返回 (是否全部入篮, 逐物体原始数值说明)。
    """
    parts: list[str] = []
    all_ok = True
    for o in sorted(pick_objects):
        p = positions.get(o)
        if p is None:
            return False, f"obj{o} 缺少取证位置"
        pos = np.asarray(p, dtype=float)
        dx = abs(float(pos[0]) - BASKET_CENTER_X)
        dy = abs(float(pos[1]) - BASKET_CENTER_Y)
        z = float(pos[2])
        ok = (dx <= BASKET_IN_X and dy <= BASKET_IN_Y
              and _BASKET_MIN_Z <= z <= _BASKET_MAX_Z)
        all_ok = all_ok and ok
        parts.append(f"obj{o} {'✓' if ok else '✗'} (Δx={dx:.3f}/{BASKET_IN_X} "
                     f"Δy={dy:.3f}/{BASKET_IN_Y} z={z:.4f}/"
                     f"{_BASKET_MIN_Z:.4f}~{_BASKET_MAX_Z:.4f})")
    return all_ok, " | ".join(parts)

def check_objects_in_basket(env: ManagerBasedRLEnv, pick_objects: list[int]) -> bool:
    """判断物体是否在篮内 —— 与官方终止条件 ObjectsInBasketDone 口径一致。

    条件（三者同时满足）：
      |x - BASKET_CENTER_X| <= BASKET_IN_X   (0.20)
      |y - BASKET_CENTER_Y| <= BASKET_IN_Y   (0.11)
      TABLE_TOP_Z <= z <= TABLE_TOP_Z+0.15  ([0.8266, 0.9766])

    注意：z 上限是 +0.15（不是 +0.10）。细高物体(object_2)竖立在篮内时
    中心约 0.93，只有用官方上限才算"入篮"。
    """
    for obj_idx in pick_objects:
        pos = env.unwrapped.scene.rigid_objects[f"object_{obj_idx}"].data.root_pos_w[0]
        if (abs(pos[0].item() - BASKET_CENTER_X) > BASKET_IN_X or
                abs(pos[1].item() - BASKET_CENTER_Y) > BASKET_IN_Y or
                not (_BASKET_MIN_Z <= pos[2].item() <= _BASKET_MAX_Z)):
            return False
    return True


def basket_report(env: ManagerBasedRLEnv, pick_objects: list[int]) -> str:
    """逐物体打印入篮判定所需的原始数值（便于人工核对，别只看 True/False）。"""
    parts = []
    for o in pick_objects:
        pos = env.unwrapped.scene.rigid_objects[f"object_{o}"].data.root_pos_w[0]
        dx = abs(pos[0].item() - BASKET_CENTER_X)
        dy = abs(pos[1].item() - BASKET_CENTER_Y)
        z = pos[2].item()
        ok = (dx <= BASKET_IN_X and dy <= BASKET_IN_Y
              and _BASKET_MIN_Z <= z <= _BASKET_MAX_Z)
        parts.append(f"obj{o} {'✓' if ok else '✗'} (Δx={dx:.3f}/{BASKET_IN_X} "
                     f"Δy={dy:.3f}/{BASKET_IN_Y} z={z:.4f}/{_BASKET_MIN_Z:.4f}~{_BASKET_MAX_Z:.4f})")
    return " | ".join(parts)


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
    image_size:  int | None = None,
    diversity:   dict | None = None,
) -> dict | None:
    """Run one full episode and return recorded data, or None on early termination.

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

    # Pre-compute grasp quaternions from actual object orientations after reset
    # 示范多样性：每条示范一套自己的各状态步数（diversity=None 时逐字沿用 STEPS）
    steps_profile = None
    if diversity:
        steps_profile = make_steps_profile(rng, tuple(diversity.get("steps_jitter")
                                                      or DIVERSITY_STEPS_JITTER))
    sm = PickPlaceStateMachine(pick_objects, device, robot=robot, steps=steps_profile)
    for obj_idx in pick_objects:
        obj_quat = env.unwrapped.scene.rigid_objects[f"object_{obj_idx}"] \
                       .data.root_state_w[0, 3:7]
        sm.set_grasp_quat(obj_idx, obj_quat)

    # Settle
    for _ in range(SETTLE_STEPS):
        _step_to(env, robot, ik_ctrl, arm_ids, gripper_ids,
                 ee_home, eq_home, g_open, default_jpos)

    ik_ctrl.reset()

    # 保险：warm-up/settle 期间若已触发任何终止项，环境会中途 auto-reset，
    # 物体被放回 init_state.pos（不是我们刚随机出来的布局），这一条必须丢弃重来，
    # 否则会录到一条"物体位置与预期不符"的轨迹。
    _early = fired_terms(env)
    if any(_early.values()):
        print(f"[WARN] warm-up/settle 期间触发终止 {_early} —— 布局已失效，丢弃本条。")
        return None

    # ---- Recording loop ---- #
    # 终止条件**保持原样，一步都不动**（README §3.1 第 1 条：
    # "应在采集端识别终止原因并保存成功轨迹，不应删除评测成功条件"）。
    #
    # 上一轮曾在这里清空 termination_manager._term_cfgs 去换"完整尾巴"，
    # 那等于把成功/超时/掉落一起关掉，属于 README 明确禁止的做法；而且它
    # 掩盖了两个真 bug（termination_reason() 用错 API；env.step() 之后读物体
    # 位置会被 auto-reset 破坏）。现在的做法：
    #   * 成功与否由**环境自己的 basket_success** 判定（get_term，reset 前取值）；
    #   * 每步 env.step 之前对三件物体做快照，作为"reset 前取证"；
    #   * 触发成功即结束本条 —— 轨迹在成功瞬间截断（缺第三件物体的
    #     OPEN/LIFT_RETRACT/RETRACT 约 7%，该动作在前两件物体上已各示范两次）。
    def _obj_xyz(i: int) -> list[float]:
        p = env.unwrapped.scene.rigid_objects[f"object_{i}"].data.root_pos_w[0]
        return [float(p[0]), float(p[1]), float(p[2])]

    meta: dict = {
        "termination": None,
        "steps": 0,
        # 本条 episode 的物体布局（沉降后），供覆盖度与"留出位置"分析
        "obj_spawn_after_settle": {i: _obj_xyz(i) for i in (1, 2, 3)},
        "obj_pos_at_termination": None,
        "obj_pos_pre_step": None,
    }

    qpos_buf, qvel_buf, ee_pos_buf, ee_quat_buf, action_buf = [], [], [], [], []
    frames_buf = [] if camera is not None else None
    _last_state = None
    _obj_z0 = {i: float(_obj_xyz(i)[2]) for i in (1, 2, 3)}   # 抓取前高度（算抬升用）
    _lift_max = {i: 0.0 for i in (1, 2, 3)}                   # 各物体最大抬升（判夹起）

    _perturb_states = tuple(diversity.get("perturb_states")
                            or DIVERSITY_PERTURB_STATES) if diversity else ()
    _perturb_sigma  = float(diversity.get("perturb_sigma", DIVERSITY_PERTURB_SIGMA)) \
                      if diversity else 0.0
    _perturb_prob   = float(diversity.get("perturb_prob", DIVERSITY_PERTURB_PROB)) \
                      if diversity else 0.0
    _n_perturb      = 0
    _perturb_log: list[dict] = []

    while not sm.done:
        # 状态轨迹（每条约 33 行）：失败时能直接看出是哪件物体、卡在哪个阶段
        _cur = f"obj{sm._obj_indices[min(sm._ptr, len(sm._obj_indices) - 1)]}/{sm.state}"
        if _cur != _last_state:
            print(f"[INFO]   step {len(qpos_buf):5d}  {_cur}", flush=True)
            _last_state = _cur

        # 示范多样性：在指定状态**刚进入时**注入一次关节扰动，让状态机
        # 自己纠回来 —— 录到的这一段就是"从偏了的地方恢复"的示范。
        # 必须放在 sm.tick() 之前、且在状态第一步，这样 tick 内部冻结的
        # 起点（LIFT/RETREAT 的 _ee_xy）反映的是扰动后的真实位形。
        if (_perturb_sigma > 0 and sm.state in _perturb_states
                and sm.at_state_start and rng.random() < _perturb_prob):
            perturb_arm(env, robot, ik_ctrl, rng, _perturb_sigma, arm_ids)
            _n_perturb += 1
            if len(_perturb_log) < 12:      # 只留前 12 次，meta 不要太大
                _perturb_log.append({
                    "step": len(qpos_buf), "state": sm.state,
                    "obj": int(sm._obj_indices[min(sm._ptr, len(sm._obj_indices) - 1)]),
                })

        obj_pos_w = env.unwrapped.scene.rigid_objects[sm.current_object_key] \
                        .data.root_pos_w[0].clone()
        ee_pos_des, ee_quat_des, gripper_cmd = sm.tick(obj_pos_w)

        if sm.home_at_retract():
            # 本轮抓取结束：直接命令回默认位形，避免把"关节顶限位"的位形
            # 带进下一件物体的接近段（实测会让末端卡在半路，
            # 见 config.HOME_AT_RETRACT 的说明）。采集时必须与验证脚本
            # 用同一套行为，否则录到的轨迹与验证过的轨迹不一致。
            arm_jpos_des = robot.data.default_joint_pos[:, arm_ids].clone()
            gripper_vals = GRIPPER_OPEN_POS
        else:
            arm_jpos_des = ik_ctrl.compute(ee_pos_des.unsqueeze(0), ee_quat_des.unsqueeze(0))
            gripper_vals = GRIPPER_OPEN_POS if gripper_cmd == "open" else GRIPPER_CLOSE_POS
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
            rgb = camera.data.output["rgb"][0][..., :3]        # (H,W,3) uint8, GPU
            if image_size:
                # 与训练侧 T.Resize((224,224), antialias=True)、推理侧
                # TF.resize(rgb,[224,224],antialias=True) 用同一种插值，
                # 训练侧对已是目标尺寸的输入是幂等的，不会二次退化。
                rgb = TF.resize(rgb.permute(2, 0, 1),
                                [int(image_size), int(image_size)],
                                antialias=True).permute(1, 2, 0)
            frames_buf.append(rgb.cpu().numpy())

        # step 前快照：只作诊断用（实测它比真正的判定时刻早一步，临界物体会差一点）
        obj_pre = {i: _obj_xyz(i) for i in (1, 2, 3)}
        for _i, _p in obj_pre.items():
            _lift_max[_i] = max(_lift_max[_i], _p[2] - _obj_z0[_i])

        _, _, terminated, truncated, _ = env.step(env_action)

        # ★ reset 前取证：终止判定发生在 env.step 内部、reset **之前**
        #   （manager_based_rl_env.step(): compute 终止 -> 算奖励 -> _reset_idx），
        #   而本工程没有任何针对物体的 reset 事件（实测：env.reset() 既不把物体
        #   放回初始布局、也不复位机器人关节），场景 data 缓冲在 reset 后不会被刷新，
        #   所以下面这一次读取拿到的是**判定那一刻**的真实位置 —— 正是 README
        #   §3.1 第 1 条要求的"末次状态应在 reset 前取证"。
        obj_at_term = {i: _obj_xyz(i) for i in (1, 2, 3)}

        if terminated.any() or truncated.any():
            # 区分成功终止与失败终止 —— 全部读环境自己的终止项真实取值：
            #   success(basket_success) -> 保留本条，轨迹在成功瞬间结束
            #   time_out / illegal_contact / fall -> 丢弃该轮
            reason = termination_reason(env)
            meta["termination"] = reason
            meta["obj_pos_at_termination"] = obj_at_term
            meta["obj_pos_pre_step"] = obj_pre      # 仅诊断：比判定早一步
            if reason == "success":
                print("[INFO] 环境判定 basket_success（reset 前取证已记录）"
                      "—— 轨迹在成功瞬间结束，保留本条。")
                break
            print(f"[WARN] Episode ended early ({reason}) — skipping demo.")
            return None

    meta["steps"] = len(qpos_buf)
    meta["obj_lift_max"] = {i: round(v, 4) for i, v in _lift_max.items()}
    # 示范多样性取证：这条示范实际用了什么步数表、加了几次扰动、加在哪
    if diversity:
        meta["diversity"] = {
            "steps_profile": sm.steps,
            "perturb_sigma": _perturb_sigma,
            "perturb_states": list(_perturb_states),
            "n_perturb": _n_perturb,
            "perturb_log": _perturb_log,
        }

    # 本条结束时的物体位置。只在**没有发生终止**（即状态机自己跑完、环境没判成功）
    # 时才有效 —— 因为一旦终止，env.step 内部已经 auto-reset，此时读到的是物体
    # 被放回桌面的位置。这条数据是判断"到底是抓取失败、还是入篮差一点"的第一手证据。
    if meta["termination"] is None:
        meta["obj_pos_final"] = {i: _obj_xyz(i) for i in (1, 2, 3)}

    result = {
        "qpos":    np.stack(qpos_buf),
        "qvel":    np.stack(qvel_buf),
        "ee_pos":  np.stack(ee_pos_buf),
        "ee_quat": np.stack(ee_quat_buf),
        "action":  np.stack(action_buf),
        # 采集元数据：终止原因、取证位置、布局、帧数（不写入 HDF5，供调用方记录）
        "meta":    meta,
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
