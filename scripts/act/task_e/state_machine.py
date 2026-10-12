"""Pick-place state machine and grasp-quaternion solver for Task E."""

import torch
from isaaclab.utils.math import matrix_from_quat, quat_from_matrix

from .config import (
    STEPS, STATE_ORDER,
    CARRY_Z, PLACE_HEIGHT,
    RETRACT_POS_X, RETRACT_POS_Y,
    GRASP_Z_OFFSET,
    BASKET_CENTER_X, BASKET_CENTER_Y,
    DEFAULT_PLACE_QUAT_W,
    GRASP_X_OFFSET,
    GRASP_Y_OFFSET,
    GRASP_OFFSET_OBJECTS,
    PRE_ROTATE_IN_PREGRASP_OBJECTS,
    PREGRASP_ROTATE_STEPS,
    SIDE_GRASP_OBJECTS,
    SIDE_GRASP_TCP_OFFSET,
    SIDE_GRASP_HEIGHT_FRAC,
    SIDE_GRASP_MARGIN,
    SIDE_GRASP_APPROACH_PITCH_DEG,
    SIDE_GRASP_APPROACH_SIGN,
    SIDE_GRASP_APPROACH_AXIS,
    TCP_COMPENSATE_TOP_DOWN,
    LONG_AXIS_GRASP_OBJECTS,
    LONG_AXIS_GRASP_TILT_DEG,
    LONG_AXIS_GRASP_ROLL_DEG,
    GRASP_X_OFFSET_DYNAMIC_OBJECTS,
    GRASP_X_OFFSET_DYN_BASE,
    GRASP_X_OFFSET_DYN_SLOPE,
    OBJ_SPAWN_X_MAX,
    LIFT_STAGE_MAX,
    LIFT_Z_CAP,
    RETREAT_POS_X,
    RETREAT_USE_LIFT_Z,
    RETREAT_POS_Z,
    RETREAT_APPLY_OBJECTS,
    CARRY_UNIFY_SIDE_GRASP,
    HOME_AT_RETRACT,
    RETRACT_HOME_STEPS,
)

# 搬运（夹着物体）阶段：这些状态下必须保持"抓取时"的末端姿态。
# 否则末端会在夹着物体的同时旋转到默认朝下姿态 —— 轻则物体在指间打滑，
# 重则旋转扫掠碰倒旁边的物体。
_CARRY_STATES = ("LIFT", "RETREAT", "TRANSPORT", "PLACE", "OPEN", "LIFT_RETRACT")


def compute_long_axis_grasp_quat(obj_quat_w: torch.Tensor, device: str,
                                 tilt_deg: float = 0.0,
                                 roll_deg: float = 0.0) -> torch.Tensor:
    """针对"长轴在水平面内"的物体，构造开合轴垂直于长轴的抓取姿态。

    设计要求（用户提出）
    --------------------
    * 开合轴（夹爪本地 Y）垂直于物体长轴；
    * 且垂直于"长轴与 Z 轴构成的平面"；
    * 允许一定侧倾 —— 纯垂直姿态会让 wrist 的 joint5 超出 ±1.22 rad 限位
      而不可达（实测误差 0.034 m、夹爪扑空）。

    构造方式（倾斜 + 滚转两个自由度）
    --------------------------------
    1. 取物体长轴在水平面内的投影 u。
    2. 末端"接近方向"从竖直向下起，朝 u 方向倾斜 tilt 度：
         tilt = 0°  -> 完全垂直向下（顶抓）
         tilt = 90° -> 完全水平（侧抓）
       倾斜方向选 u 而不是 -u，使手掌朝物体方向压下去。
    3. 在垂直于接近方向的平面内，按 roll 度旋转开合轴。
       roll = 0 时开合轴落在"垂直于长轴且水平"的方向上。

    这样开合轴恒与长轴垂直（构造保证），而 tilt 决定偏离垂直的程度，
    用 tilt>0 换取可达性。
    """
    R_obj = matrix_from_quat(obj_quat_w.unsqueeze(0)).squeeze(0)
    # 长轴 = 水平投影最强的本体轴
    best_i, best_n = 0, -1.0
    for col in range(3):
        v = R_obj[:, col].clone(); v[2] = 0.0
        n = float(v.norm())
        if n > best_n:
            best_n, best_i = n, col
    u = R_obj[:, best_i].clone(); u[2] = 0.0
    if float(u.norm()) < 1e-3:
        # 长轴几乎竖直（水平投影退化）时"垂直于长轴"没有唯一的水平解，
        # 退回默认朝下姿态，避免归一化出 NaN。
        return torch.tensor([0.0, 1.0, 0.0, 0.0], device=device)
    u = u / u.norm().clamp(min=1e-6)

    down = torch.tensor([0.0, 0.0, -1.0], device=device)
    tilt = torch.deg2rad(torch.tensor(float(tilt_deg), dtype=torch.float32, device=device))
    # 接近方向：从竖直向下朝 +u 倾斜
    approach = down * torch.cos(tilt) + u * torch.sin(tilt)
    approach = approach / approach.norm().clamp(min=1e-6)

    # 基准开合轴：垂直于长轴、且落在垂直于 approach 的平面内
    jaw0 = torch.linalg.cross(u, approach)
    if float(jaw0.norm()) < 1e-6:
        jaw0 = torch.tensor([0.0, 1.0, 0.0], device=device)
    jaw0 = jaw0 / jaw0.norm().clamp(min=1e-6)
    # 该平面内的第二根基向量
    second = torch.linalg.cross(approach, jaw0)
    second = second / second.norm().clamp(min=1e-6)

    roll = torch.deg2rad(torch.tensor(float(roll_deg), dtype=torch.float32, device=device))
    jaw = jaw0 * torch.cos(roll) + second * torch.sin(roll)
    jaw = jaw / jaw.norm().clamp(min=1e-6)
    # 正交化（数值保险）
    jaw = jaw - approach * torch.dot(jaw, approach)
    jaw = jaw / jaw.norm().clamp(min=1e-6)

    col_x = torch.linalg.cross(jaw, approach)
    return quat_from_matrix(torch.stack([col_x, jaw, approach], dim=1).unsqueeze(0)).squeeze(0)


def compute_side_grasp_quat(obj_quat_w: torch.Tensor, device: str) -> torch.Tensor:
    """横向夹取姿态：夹爪沿水平面内的短轴进给，钳口张开方向对齐长轴。

    与顶抓的区别：夹爪本地 Z 轴（进给轴）水平，而不是朝下。

    步骤
    ----
    1. 取出物体本体各轴在世界系的指向，投影到水平面，找最长/最短的水平轴。
    2. 进给轴 = 水平面内较短的那根轴（这样钳口张开方向自然落在长轴上）。
    3. 钳口张开轴（夹爪本地 Y） = 长轴方向。
    4. 夹爪本地 X = Y × Z，保证右手系且本地 Z 朝下（掌面朝下）。

    返回值是抓取姿态四元数 (w, x, y, z)。
    """
    R_obj = matrix_from_quat(obj_quat_w.unsqueeze(0)).squeeze(0)     # (3, 3)
    down = torch.tensor([0.0, 0.0, -1.0], device=device)

    axes, norms = [], []
    for col in range(3):
        a = R_obj[:, col].clone()
        a[2] = 0.0                                   # 投影到水平面
        axes.append(a)
        norms.append(float(a.norm().item()))

    long_i = int(max(range(3), key=lambda i: norms[i]))
    # 在水平投影足够强的轴里挑最短的作为进给轴
    horiz = [i for i in range(3) if norms[i] > 0.5]
    short_i = int(min(horiz, key=lambda i: norms[i])) if horiz else (long_i + 1) % 3

    long_axis = axes[long_i] / max(norms[long_i], 1e-6)      # 水平面内的长轴
    short_axis = axes[short_i] / max(norms[short_i], 1e-6)   # 水平面内的短轴

    # 进给轴 = 指向物体端面的那根轴；钳口张开 = 被夹住的那一维。
    # 默认（short）：沿短轴进给、钳口夹长轴那一维 —— 掌面朝向物体的窄端。
    if SIDE_GRASP_APPROACH_AXIS == "long":
        approach_xy, jaw_xy = long_axis, short_axis
    else:
        approach_xy, jaw_xy = short_axis, long_axis

    # 进给轴下倾 SIDE_GRASP_APPROACH_PITCH_DEG 度（0° 会卡 joint5 限位）。
    pitch = torch.deg2rad(torch.tensor(SIDE_GRASP_APPROACH_PITCH_DEG,
                                       dtype=torch.float32, device=device))
    approach = approach_xy * torch.cos(pitch) + down * torch.sin(pitch)
    approach = approach / approach.norm().clamp(min=1e-6)

    # 钳口张开方向：正交化到进给轴上，保持水平
    jaw = jaw_xy - approach * torch.dot(jaw_xy, approach)
    jaw = jaw / jaw.norm().clamp(min=1e-6)

    # 符号：把"水平分量较大的那根轴"按 SIDE_GRASP_APPROACH_SIGN 定号。
    # 例如进给主要沿 X 时，SIGN=-1 => 进给朝 -X。
    if abs(float(approach[0])) >= abs(float(approach[1])):
        if float(approach[0]) * SIDE_GRASP_APPROACH_SIGN < 0.0:
            approach = -approach
    else:
        if float(approach[1]) * SIDE_GRASP_APPROACH_SIGN < 0.0:
            approach = -approach
    approach = approach / approach.norm().clamp(min=1e-6)

    col_x = torch.linalg.cross(jaw, approach)
    return quat_from_matrix(torch.stack([col_x, jaw, approach], dim=1).unsqueeze(0)).squeeze(0)


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


class PickPlaceStateMachine:
    """Finite state machine that sequences pick-and-place for multiple objects.

    States (in order): INIT → PRE_GRASP → REACH → CLOSE → LIFT →
                       TRANSPORT → PLACE → OPEN → RETRACT → (next object or done)
    """

    def __init__(self, object_indices: list[int], device: str, robot=None,
                 steps: dict[str, int] | None = None):
        self._obj_indices = object_indices
        self._device      = device
        # 可选：用于查询末端"当前"位置。LIFT 需要保持水平位置不变，
        # 只做竖直抬升，所以必须知道末端现在在哪。
        self._robot       = robot
        # 每条示范可以带一套自己的"各状态步数"。不传时逐字沿用模块级 STEPS，
        # 行为与历史版本**完全一致**；传了就覆盖对应键（其余仍取 STEPS）。
        # 用途见 config.DIVERSITY_*：采集侧做时序抖动，让示范不再是一条
        # 固定时刻表（原实现 53 条示范的夹爪切换时刻完全相同）。
        self._steps = dict(STEPS) if steps is None else {**STEPS, **steps}
        self._grasp_quat_cache: dict[int, torch.Tensor] = {}
        self.reset()

    def reset(self) -> None:
        self._ptr             = 0
        self._state_idx       = 0
        self._count           = 0
        self.done             = False
        self._cached_obj_pos: torch.Tensor | None = None
        self._locked_quat: torch.Tensor | None = None
        self._lift_xy: torch.Tensor | None = None
        self._lift_z_from: float | None = None
        self._lift_z_to: float | None = None
        self._retreat_xy: torch.Tensor | None = None
        self._retreat_z: float | None = None
        self._grasp_quat_cache.clear()

    def set_grasp_quat(self, obj_idx: int, obj_quat_w: torch.Tensor) -> None:
        """Pre-compute and cache the grasp quaternion for one object.

        SIDE_GRASP_OBJECTS 中的物体用横向夹取姿态，其余保持原顶抓姿态。
        """
        if obj_idx in SIDE_GRASP_OBJECTS:
            self._grasp_quat_cache[obj_idx] = compute_side_grasp_quat(obj_quat_w, self._device)
        elif obj_idx in LONG_AXIS_GRASP_OBJECTS:
            self._grasp_quat_cache[obj_idx] = compute_long_axis_grasp_quat(
                obj_quat_w, self._device,
                tilt_deg=LONG_AXIS_GRASP_TILT_DEG,
                roll_deg=LONG_AXIS_GRASP_ROLL_DEG,
            )
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
        cur_obj = self._obj_indices[min(self._ptr, len(self._obj_indices) - 1)]

        # Freeze object position at start of PRE_GRASP to avoid drift during descent
        if s == "PRE_GRASP" and self._count == 0:
            self._cached_obj_pos = obj_pos.clone()
            # 抓取偏置只对 GRASP_OFFSET_OBJECTS 中的物体生效
            if cur_obj in GRASP_OFFSET_OBJECTS:
                self._cached_obj_pos[0] += GRASP_X_OFFSET
                self._cached_obj_pos[1] += GRASP_Y_OFFSET
            # 动态偏置：物体越远，抓取点越往机械臂一侧推
            if cur_obj in GRASP_X_OFFSET_DYNAMIC_OBJECTS:
                dyn = GRASP_X_OFFSET_DYN_BASE + GRASP_X_OFFSET_DYN_SLOPE * (
                    OBJ_SPAWN_X_MAX - float(obj_pos[0]))
                dyn = max(GRASP_X_OFFSET, dyn)
                self._cached_obj_pos[0] += (dyn - GRASP_X_OFFSET)
        # LIFT 必须用物体的"当前"位置：缓存的旧位置会让抬升目标带上
        # 水平分量，产生横向扫掠，实测会撞倒旁边的物体（object_1 抬臂
        # 撞倒 object_2）。REACH/CLOSE 仍用冻结位置，避免下降过程中目标漂移。
        if s in ("REACH", "CLOSE") and self._cached_obj_pos is not None:
            obj_pos = self._cached_obj_pos

        ee_pos, gripper = self._get_target_pos_gripper(s, obj_pos, d)
        ee_quat         = self._get_target_quat(s, d)

        # 提前锁定抓取姿态：PRE_GRASP 一开始就把 REACH/CLOSE 要用的姿态
        # 定下来，并在整个 PRE_GRASP 里保持不变。这样腕部在 PRE_GRASP
        # 就把"从外往内收"的旋转转完，REACH 只剩纯下降，不再边降边转 ——
        # 实测 REACH 阶段本来还会转 70°，导致爪爪落点随物体位置左右摇摆
        # （离得近卡外侧、离得远卡内侧，固定 X 偏置无法同时兼顾）。
        if s == "LIFT" and self._count == 0:
            # LIFT 起点：冻结水平位置，并记录总抬升上限。
            # 实测问题：一步抬升 25 cm 太远，IK 为够到目标会重排关节，
            # 水平位置守不住 —— base 漂 5.7 cm、指尖向 -X 扫出 13.4 cm，
            # 把前方的 object_2 撞倒。所以拆成两段，每段只抬 LIFT_STAGE_MAX。
            self._lift_xy = self._ee_xy(d)
            q0 = self._get_target_quat(s, d)
            dz0 = self._tcp_z_offset(q0, d) if TCP_COMPENSATE_TOP_DOWN else 0.0
            self._lift_z_from = float(self._ee_xy(d)[2])
            self._lift_z_to = min(CARRY_Z + dz0, LIFT_Z_CAP)
        if s == "RETREAT" and self._count == 0:
            # 后撤段起点：冻结当前水平 Y 与当前实际高度，只把末端沿 +X
            # 拉到物体生成区之外。后撤方向顺着机械臂自然伸展方向，
            # IK 不需重排关节（实测该区域 IK 误差 0.0000）。
            cur = self._ee_xy(d)
            self._retreat_xy = torch.tensor([RETREAT_POS_X, float(cur[1])], device=d)
            self._retreat_z = float(cur[2]) if RETREAT_USE_LIFT_Z else float(RETREAT_POS_Z)
        if s == "PRE_GRASP" and self._count == 0:
            self._locked_quat = self._grasp_quat_cache.get(
                cur_obj, torch.tensor(DEFAULT_PLACE_QUAT_W, dtype=torch.float32, device=d))
        if s in ("PRE_GRASP", "REACH", "CLOSE") and self._locked_quat is not None:
            ee_quat = self._locked_quat

        self._count += 1
        if self._count >= self._steps[s]:
            self._count = 0
            if s == "RETRACT":
                self._ptr          += 1
                self._cached_obj_pos = None
                self._locked_quat    = None
                self._lift_xy        = None
                self._lift_z_from    = None
                self._lift_z_to      = None
                self._retreat_xy     = None
                self._retreat_z      = None
                if self._ptr >= len(self._obj_indices):
                    self.done = True
                    return ee_pos, ee_quat, gripper
                self._state_idx = STATE_ORDER.index("PRE_GRASP")
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
    def steps(self) -> dict[str, int]:
        """本条示范实际使用的各状态步数（供采集端写进 meta 便于追溯）。"""
        return dict(self._steps)

    @property
    def at_state_start(self) -> bool:
        """当前是否处于某个状态的**第一步**（tick 之前调用）。

        采集端用它来在"状态刚进入"时注入扰动，见 config.DIVERSITY_PERTURB_STATES。
        """
        return self._count == 0

    def home_at_retract(self) -> bool:
        """RETRACT 段的**后半程**是否应该改为"让机械臂回默认位形"。

        为什么需要（实测根因）
        --------------------
        逐个抓取三件物体时，前一件物体"放进篮子"的动作会把腕部留在一个
        **关节顶在限位**的位形上。实测（logs/demo_watch.log，seed 0）：
            object_2 RETRACT   q=[...,1.745,-1.220,-2.094]  限位饱和=[4,5,6]
            object_3 PRE_GRASP q=[...,1.745,-1.177,-2.094]  限位饱和=[4,6]
            object_3 PRE_GRASP 之后末端**完全不动**（err 卡在 0.179 不收敛）
        关节 4/5/6 饱和后差速 IK 推不动，末端只能"卡在半路"，
        于是下一件物体的接近段从一个坏位形开始 —— 用户观察到的
        "机械臂没有回到原姿态，导致接下来的夹取出问题"。

        做法：在 RETRACT 后半程（用 RETRACT_HOME_STEPS 指定步数）直接命令
        各关节回到 default_joint_pos，而不是继续追笛卡尔目标。
        这样每件物体的抓取都从同一个已知的良好位形开始，与物体几号无关。

        返回 True 时，调用方应把关节目标设成 default_joint_pos（含夹爪张开），
        忽略 tick() 返回的笛卡尔目标。
        """
        if not HOME_AT_RETRACT:
            return False
        return self.state == "RETRACT" and self._count >= max(0, self._steps["RETRACT"] - RETRACT_HOME_STEPS)

    @property
    def current_object_key(self) -> str:
        return f"object_{self._obj_indices[self._ptr]}"

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _ee_xy(self, d: str) -> torch.Tensor:
        """末端(gripper_base)当前位置，含真实 Z。

        注意：调用方若只关心水平分量，自行取前两维即可；
        早期版本把 Z 恒置 0，导致 LIFT 分段抬升的基准高度取到 0。
        """
        if self._robot is not None:
            try:
                idx = self._robot.body_names.index("gripper_base")
                pos = self._robot.data.body_pos_w[0, idx]
                return torch.tensor(
                    [float(pos[0]), float(pos[1]), float(pos[2])], device=d)
            except Exception:
                pass
        return torch.zeros(3, device=d)

    @staticmethod
    def _tcp_z_offset(quat: torch.Tensor, d: str) -> float:
        """抓取姿态下，指爪接触面相对末端原点在"竖直方向"的分量。

        末端原点在 gripper_base(joint6)，指爪接触面还要沿末端本地 +Z
        再延伸 SIDE_GRASP_TCP_OFFSET。顶抓时本地 Z 朝下，所以接触面比
        末端原点低 TCP；要让接触面落在 CARRY_Z，末端原点就得在
        CARRY_Z + TCP。

        返回 -delta_z，调用方用 (目标高度 + 返回值) 得到末端原点高度。
        """
        R = matrix_from_quat(quat.unsqueeze(0)).squeeze(0)
        return -float(R[2, 2]) * SIDE_GRASP_TCP_OFFSET

    def _approach_dir(self, d: str) -> torch.Tensor:
        """当前物体抓取姿态下，夹爪本地 +Z（进给轴）在世界系的单位向量。"""
        cur = self._obj_indices[min(self._ptr, len(self._obj_indices) - 1)]
        q = self._grasp_quat_cache.get(cur)
        if q is None:
            return torch.tensor([0.0, 0.0, -1.0], device=d)
        R = matrix_from_quat(q.unsqueeze(0)).squeeze(0)
        v = R[:, 2]
        return v / v.norm().clamp(min=1e-6)

    def _side_grasp_target(self, s: str, obj_pos: torch.Tensor, d: str) -> torch.Tensor:
        """侧抓的接近段目标（PRE_GRASP / REACH / CLOSE）。

        末端原点在 gripper_base（joint6 附近），而指爪接触面还要沿夹爪本地
        +Z 方向再延伸 SIDE_GRASP_TCP_OFFSET。顶抓时这段长度朝下、影响小；
        横向夹取时它整体偏移接触点，必须显式补偿，否则夹爪会插进物体里
        或者停在物体外。

        搬运段（LIFT/RETREAT/TRANSPORT/...）不在这里处理，见 _carry_target
        —— 搬运段的水平目标不能取物体位置，否则会产生横向扫掠撞倒别的物体。
        """
        ap = self._approach_dir(d)                     # 进给方向（指向物体）
        tcp_off = ap * SIDE_GRASP_TCP_OFFSET           # 接触面相对末端原点的位移

        # 抓取点：物体中部高度。
        # SIDE_GRASP_MARGIN > 0 表示沿进给方向(ap)继续深入，即把接触面
        # 从物体中心再往里推，解决"夹太浅/夹住又掉"的问题。
        # 注意方向：ap 指向物体，所以是 + ap * MARGIN（正值=更深）。
        contact = obj_pos.clone() + ap * SIDE_GRASP_MARGIN

        if s in ("REACH", "CLOSE"):
            # 指爪接触面对准抓取点 -> 末端原点往回退一个 TCP 长度
            return contact - tcp_off
        if s == "PRE_GRASP":
            # 在抓取点上方 CARRY_Z 处、以侧抓姿态待命，随后进给到抓取点
            p = contact - tcp_off
            p[2] = CARRY_Z
            return p
        return contact

    def _carry_target(self, s: str, d: str) -> torch.Tensor:
        """搬运阶段（LIFT/RETREAT/TRANSPORT/PLACE/...）的末端目标位置。

        为什么单独抽出来：搬运阶段的水平目标**不能**再取物体位置。
        实测若沿用物体位置做水平目标，末端在追这个远目标时产生横向扫掠，
        会把旁边的 object_2 撞倒（o2 在一次 LIFT 内从 [0.96,0.19] 跳到
        [0.85,0.28]）。搬运阶段的水平位置必须由"末端当前所在位置"逐步推进。

        各状态的高度约定与本文件非侧抓分支保持一致：
          LIFT      抬到 LIFT 段预先算好并钳制过的目标高度(_lift_z_to)
          RETREAT   保持高度、只沿 +X 撤到无物体区
          TRANSPORT 横摆到篮子上方（仍用 CARRY_Z —— 该高度在"无物体区"，
                    下方没有物体可撞，够不到也不会造成碰撞）
        """
        cur = self._ee_xy(d)
        if s == "LIFT":
            z_to = self._lift_z_to if self._lift_z_to is not None else LIFT_Z_CAP
            return torch.tensor([float(cur[0]), float(cur[1]), float(z_to)], device=d)
        if s == "RETREAT":
            goal = torch.tensor([float(self._retreat_xy[0]) if self._retreat_xy is not None else float(cur[0]),
                                 float(self._retreat_xy[1]) if self._retreat_xy is not None else float(cur[1]),
                                 float(self._retreat_z) if self._retreat_z is not None else float(cur[2])],
                                device=d)
            n = max(1, self._steps["RETREAT"])
            k = min(self._count + 1, n)
            return cur + (goal - cur) * (k / float(n))
        if s in ("TRANSPORT", "PLACE", "OPEN", "LIFT_RETRACT"):
            if s in ("PLACE", "OPEN"):
                # 放置高度必须按"抓取姿态"补偿 TCP 长度。
                #
                # 为什么：PLACE_HEIGHT 表达的是"想把物体中心放到多高"，
                # 但下发的目标是**末端原点**的高度。末端原点到指爪接触面
                # 还有 SIDE_GRASP_TCP_OFFSET(0.1358 m)，方向沿夹爪本地 +Z：
                #   顶抓(ap≈[0,0,-1])：物体在末端下方 0.1358 -> 要减
                #   侧抓(ap_z≈-0.5)  ：物体在末端上方 0.0679 -> 要加
                # 不补偿时顶抓物体的实际中心比目标高 0.136，
                # 实测 PLACE 后物体仍停在 0.94~1.01，而判据上限是 0.9266
                # （success 只靠"碰到篮壁后掉落"侥幸成立）。
                ap = self._approach_dir(d)
                z = PLACE_HEIGHT - float(ap[2]) * SIDE_GRASP_TCP_OFFSET
            else:
                z = CARRY_Z
            return torch.tensor([BASKET_CENTER_X, BASKET_CENTER_Y, z], device=d)
        return cur

    @staticmethod
    def _carry_grip(s: str) -> str:
        """搬运阶段各状态的夹爪指令。

        只有 CLOSE 之后才闭合，且抬升/后撤/横摆全程必须保持闭合。
        若 LIFT 误判为 open，就会"先松爪再抬臂"，物体必然留在桌面
        （实测现象：闭合 -> 原地张开 -> 抬臂 -> 空中闭合）。
        """
        return "open" if s in ("OPEN", "LIFT_RETRACT") else "close"

    def _carry_pos(self, s: str, obj_pos: torch.Tensor, d: str) -> torch.Tensor:
        """按物体类型分派搬运阶段的目标位置。

        * CARRY_UNIFY_SIDE_GRASP=True（当前取值）：所有物体都走统一的
          _carry_target —— 因为侧抓物体原来的 LIFT 目标 CARRY_Z 不可达，
          实测会让末端水平漂 21 cm 并把瓶子甩掉。
        * False（保留开关以便回退对照）：侧抓物体在除 RETREAT 外的搬运
          状态沿用历史实现 _side_grasp_target。
        """
        cur_idx = self._obj_indices[min(self._ptr, len(self._obj_indices) - 1)]
        if (cur_idx in SIDE_GRASP_OBJECTS
                and s != "RETREAT"
                and not CARRY_UNIFY_SIDE_GRASP):
            return self._side_grasp_target(s, obj_pos, d)
        return self._carry_target(s, d)

    def _get_target_pos_gripper(
        self, s: str, obj_pos: torch.Tensor, d: str
    ) -> tuple[torch.Tensor, str]:
        # 横向夹取的物体走独立分支
        cur_side = self._obj_indices[min(self._ptr, len(self._obj_indices) - 1)] in SIDE_GRASP_OBJECTS
        if cur_side and s in ("PRE_GRASP", "REACH", "CLOSE") + _CARRY_STATES:
            if s in ("PRE_GRASP", "REACH", "CLOSE"):
                return self._side_grasp_target(s, obj_pos, d), ("close" if s == "CLOSE" else "open")
            return self._carry_pos(s, obj_pos, d), self._carry_grip(s)

        if s == "INIT":
            return torch.tensor([RETRACT_POS_X, RETRACT_POS_Y, CARRY_Z], device=d), "open"
        elif s == "PRE_GRASP":
            # 目标是指爪"接触面"的高度；末端原点还要沿抓取姿态的本地 Z
            # 再退一个 TCP 长度，所以这里必须补偿，否则末端原点被抬到
            # CARRY_Z + TCP，超出可达范围 —— 实测表现为"下降到很低的地方
            # 才开始旋转、抓不到盒子"。
            q = self._get_target_quat(s, d)
            dz = self._tcp_z_offset(q, d) if TCP_COMPENSATE_TOP_DOWN else 0.0
            p = obj_pos.clone()
            p[2] = CARRY_Z + dz
            if self._pre_rotates_in_pregrasp and self._count < PREGRASP_ROTATE_STEPS:
                return torch.tensor([RETRACT_POS_X, RETRACT_POS_Y, CARRY_Z + dz], device=d), "open"
            return p, "open"
        elif s == "REACH":
            p = obj_pos.clone(); p[2] += GRASP_Z_OFFSET
            return p, "open"
        elif s == "CLOSE":
            p = obj_pos.clone(); p[2] += GRASP_Z_OFFSET
            return p, "close"
        elif s == "LIFT":
            # 先分两段抬升：前一段只抬 LIFT_STAGE_MAX，后一段再抬到最终高度。
            # 一步抬 25 cm 太远时 IK 会为够到目标重排关节，水平位置守不住
            # （实测 base 漂 5.7 cm、指尖向 -X 扫出 13.4 cm，撞倒 object_2）。
            # 水平位置全程冻结在 LIFT 起点的"末端当前位置"（不用物体位置，
            # 否则目标带水平分量会产生横向扫掠）。
            p = self._lift_xy if self._lift_xy is not None else self._ee_xy(d)
            z_from = self._lift_z_from if self._lift_z_from is not None else float(p[2])
            z_to = self._lift_z_to if self._lift_z_to is not None else CARRY_Z
            half = self._steps["LIFT"] // 2
            if self._count < half:
                z = min(z_from + LIFT_STAGE_MAX, z_to)
            else:
                z = z_to
            p = p.clone()
            p[2] = z
            return p, "close"
        elif s == "RETREAT":
            # 后撤到无物体区：保持高度与 Y 不变，只沿 +X 拉回。
            # 中间点沿直线插值，避免一步跳过去的 IK 突变。
            # 撤到 X≈1.20 后末端已在物体生成区(X≤1.10)之外，
            # 之后的 TRANSPORT 横摆是在"下方没有物体"的位置进行。
            cur_idx = self._obj_indices[min(self._ptr, len(self._obj_indices) - 1)]
            if RETREAT_APPLY_OBJECTS and cur_idx not in RETREAT_APPLY_OBJECTS:
                # 该物体不做后撤：保持水平位置、只把 Z 抬到目标高度
                return self._carry_target("LIFT", d), "close"
            return self._carry_target(s, d), "close"
        elif s == "TRANSPORT":
            return torch.tensor([BASKET_CENTER_X, BASKET_CENTER_Y, CARRY_Z], device=d), "close"
        elif s == "PLACE":
            return torch.tensor([BASKET_CENTER_X, BASKET_CENTER_Y, PLACE_HEIGHT], device=d), "close"
        elif s == "OPEN":
            return torch.tensor([BASKET_CENTER_X, BASKET_CENTER_Y, PLACE_HEIGHT], device=d), "open"
        elif s == "LIFT_RETRACT":
            return torch.tensor([BASKET_CENTER_X, BASKET_CENTER_Y, CARRY_Z], device=d), "open"
        elif s == "RETRACT":
            return torch.tensor([RETRACT_POS_X, RETRACT_POS_Y, CARRY_Z], device=d), "open"
        else:
            raise ValueError(f"Unknown state: {s}")

    def _get_target_quat(self, s: str, d: str) -> torch.Tensor:
        default_quat = torch.tensor(DEFAULT_PLACE_QUAT_W, dtype=torch.float32, device=d)
        cur_idx = self._obj_indices[min(self._ptr, len(self._obj_indices) - 1)]

        # 横向夹取：PRE_GRASP 就必须到位并保持侧抓姿态，之后是纯水平进给，
        # 不能等到 REACH 才旋转（否则夹爪会边转边扫到瓶身）。
        if cur_idx in SIDE_GRASP_OBJECTS and s in ("PRE_GRASP", "REACH", "CLOSE", "LIFT"):
            return self._grasp_quat_cache.get(cur_idx, default_quat)

        # 需要"提前旋转开合轴"的物体：PRE_GRASP 阶段就转到抓取姿态。
        # 原代码在 REACH 下降过程中才旋转，旋转扫掠会碰倒物体。
        # 注意：只要给物体指定了专用抓取姿态（LONG_AXIS_GRASP_OBJECTS），
        # 就必须在这里一并提前旋转 —— 只加姿态、不加这个条件，
        # 会出现"姿态算了但没提前转"的情况（此前踩过）。
        if s == "PRE_GRASP" and (cur_idx in PRE_ROTATE_IN_PREGRASP_OBJECTS
                                 or cur_idx in LONG_AXIS_GRASP_OBJECTS):
            return self._grasp_quat_cache.get(cur_idx, default_quat)

        if s in _CARRY_STATES:
            return self._grasp_quat_cache.get(cur_idx, default_quat)
        return default_quat

    @property
    def _pre_rotates_in_pregrasp(self) -> bool:
        """当前物体是否需要在 PRE_GRASP 阶段提前旋转。"""
        cur_idx = self._obj_indices[min(self._ptr, len(self._obj_indices) - 1)]
        return cur_idx in PRE_ROTATE_IN_PREGRASP_OBJECTS
