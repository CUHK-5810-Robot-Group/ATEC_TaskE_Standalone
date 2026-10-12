"""Task E demo-collection constants.
"""

from atec_rl_lab.tasks.task_e.env_cfg import (
    BASKET_CENTER_X, BASKET_CENTER_Y,
    TABLE_CENTER_X, TABLE_CENTER_Y, TABLE_TOP_Z, TABLE_HALF_X,
    BASKET_EXCL_HALF_X, BASKET_EXCL_HALF_Y,
)

__all__ = [
    "BASKET_CENTER_X", "BASKET_CENTER_Y",
    "TABLE_CENTER_X", "TABLE_CENTER_Y", "TABLE_TOP_Z", "TABLE_HALF_X",
    "BASKET_EXCL_HALF_X", "BASKET_EXCL_HALF_Y",
    # robot
    "EE_BODY_NAME", "ARM_JOINT_NAMES", "GRIPPER_JOINT_NAMES",
    "GRIPPER_OPEN_POS", "GRIPPER_CLOSE_POS", "ACTION_SCALE",
    # state machine
    "STEPS", "STATE_ORDER",
    # geometry
    "PRE_GRASP_CLEARANCE", "GRASP_Z_OFFSET",
    "CARRY_Z", "PLACE_HEIGHT",
    "RETRACT_POS_X", "RETRACT_POS_Y",
    "DEFAULT_PLACE_QUAT_W",
    # success check
    "BASKET_IN_X", "BASKET_IN_Y",
    # spawn regions
    "OBJ_SPAWN_X_MIN", "OBJ_SPAWN_X_MAX", "OBJ_SPAWN_Z", "OBJ_SPAWN_Y_BANDS",
    "OBJ_HALF_EXTENTS", "OBJ_BBOX_MARGIN",
    # camera
    "CAM_POS", "CAM_ROT", "CAM_H", "CAM_W",
    # actuator
    "ACT_STIFFNESS", "ACT_DAMPING", "ACT_EFFORT_LIMIT", "ACT_VEL_LIMIT",
    # warm-up
    "WARMUP_STEPS", "SETTLE_STEPS", "LIFT_STAGE_MAX", "LIFT_Z_CAP",
    # 搬运后撤段（避免横摆时撞到物体生成区内的物体）
    "RETREAT_POS_X", "RETREAT_USE_LIFT_Z", "RETREAT_APPLY_OBJECTS",
    "CARRY_UNIFY_SIDE_GRASP",
    "HOME_AT_RETRACT", "RETRACT_HOME_STEPS",
    # object_1 专用抓取调整
    "GRASP_X_OFFSET", "GRASP_Y_OFFSET",
    "GRASP_OFFSET_OBJECTS", "PRE_ROTATE_IN_PREGRASP_OBJECTS",
    "PREGRASP_ROTATE_STEPS",
    # object_2 横向夹取
    "SIDE_GRASP_OBJECTS", "SIDE_GRASP_TCP_OFFSET",
    "SIDE_GRASP_HEIGHT_FRAC", "SIDE_GRASP_MARGIN",
    "SIDE_GRASP_APPROACH_PITCH_DEG", "SIDE_GRASP_APPROACH_SIGN",
    "SIDE_GRASP_APPROACH_AXIS",
    "LONG_AXIS_GRASP_OBJECTS", "LONG_AXIS_GRASP_TILT_DEG", "LONG_AXIS_GRASP_ROLL_DEG",
    # 示范多样性（采集侧，--diversity）
    "DIVERSITY_STEPS_JITTER", "DIVERSITY_PERTURB_SIGMA",
    "DIVERSITY_PERTURB_STATES", "DIVERSITY_PERTURB_PROB",
]

# ------------------------------------------------------------------ #
# Robot
# ------------------------------------------------------------------ #
EE_BODY_NAME        = "gripper_base"
ARM_JOINT_NAMES     = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
GRIPPER_JOINT_NAMES = ["joint7", "joint8"]
GRIPPER_OPEN_POS    = [0.035, -0.035]   # joint7, joint8
GRIPPER_CLOSE_POS   = [-0.015,  0.015]

# Must match ActionsCfg: scale=0.5, use_default_offset=True
#   env_action = (joint_target - default_joint_pos) / ACTION_SCALE
ACTION_SCALE = 0.5

# ------------------------------------------------------------------ #
# State-machine
# ------------------------------------------------------------------ #
STEPS: dict[str, int] = {
    "INIT":         100,
    "PRE_GRASP":    200,
    "REACH":        100,
    "CLOSE":         40,
    "LIFT":         160,
    "RETREAT":      120,
    "TRANSPORT":    200,
    "PLACE":        200,
    "OPEN":          60,
    "LIFT_RETRACT":  80,
    "RETRACT":       80,
}
STATE_ORDER = ["INIT", "PRE_GRASP", "REACH", "CLOSE", "LIFT", "RETREAT",
               "TRANSPORT", "PLACE", "OPEN", "LIFT_RETRACT", "RETRACT"]

# ------------------------------------------------------------------ #
# Geometry
# ------------------------------------------------------------------ #
PRE_GRASP_CLEARANCE = 0.12   # metres above object before descent
GRASP_Z_OFFSET      = 0.09   # metres: gripper approach height above object centre

CARRY_Z      = TABLE_TOP_Z + 0.40   # safe carry height

# PLACE/OPEN 阶段"想把物体中心放到的高度"（米）。
#
# 注意语义：这是**物体中心**的目标高度，不是末端指令高度。下发给 IK 的
# 末端目标还要按抓取姿态补偿 TCP 长度 0.1358（见 state_machine._carry_target
# 的 PLACE 分支）：顶抓物体在末端下方 0.1358、侧抓物体在末端上方 0.0679。
#
# 实测依据（篮子几何，见 logs/probe_basket_bbox.py）：
#   篮子世界包围盒 x[0.843,1.317] y[-0.458,-0.142] z[0.834,0.980]
#   -> 篮沿(顶) z=0.9798、开口 0.475×0.317 m（很宽，位置容差不是问题）
#   -> 成功判据上限 TABLE_TOP_Z+0.10 = 0.9266 比篮沿低 5.3 cm
#   物体静止在篮内时中心约 0.86~0.90（自由落体实测）。
# 旧值 TABLE_TOP_Z+0.15 是**末端指令**高度 0.9766（正好在篮沿高度），
# 顶抓时物体中心实际还在 0.9766 -> 物体被"举到篮沿高度松手"，
# 落点漂到篮沿上（实测静止 0.93~1.01，全部高于判据上限）。
#
# 取值 0.07 -> 0.06 的依据（轮次 C 实测残留）：
#   种子1 的 object_2（侧抓）最终停在 z≈0.933，只比判据上限 0.9266 高 0.6 cm，
#   且松手瞬间物体中心 0.979 ≈ 篮沿 0.9798 —— 明显是"卡在篮沿上"。
#   再降 1 cm 让物体在篮沿以下松手，即可自然落进篮内。
#   顶抓物体（obj1 静止 0.917、obj3 0.849）离上限还有余量，不受影响。
PLACE_HEIGHT = TABLE_TOP_Z + 0.06   # ≈0.8866：比静止高度高约 1~3 cm

RETRACT_POS_X = TABLE_CENTER_X + TABLE_HALF_X - 0.05
RETRACT_POS_Y = TABLE_CENTER_Y
DEFAULT_PLACE_QUAT_W = [0.0, 1.0, 0.0, 0.0]   # top-down orientation (w,x,y,z)

# ------------------------------------------------------------------ #
# Object spawn regions
#
#   object_1  Y ∈ [0.25, 0.29]   (top band)
#   object_2  Y ∈ [0.14, 0.20]   (middle band)
#   object_3  Y ∈ [0.03, 0.09]   (bottom band, closest to basket)
# ------------------------------------------------------------------ #
OBJ_SPAWN_X_MIN = TABLE_CENTER_X - 0.10
OBJ_SPAWN_X_MAX = TABLE_CENTER_X + 0.10
OBJ_SPAWN_Z     = TABLE_TOP_Z + 0.03

# Per-object Y-bands: {object_idx: (y_min, y_max)}
#
# 【已对齐评测分布 —— README §3.1 第 2 条要求，勿随意改回】
# README 原文："统一采集分布与评测分布……例如糖盒的 Y 采集范围是 0.20–0.25 m，
#              当前评测配置是 0.25–0.29 m。应让示范覆盖评测分布；
#              不要缩窄评测范围迁就策略。"
#
# 评测分布的权威定义在 env_cfg.py 随机化事件里的 y_bands：
#     object_1 [0.25,0.29]   object_2 [0.14,0.20]   object_3 [0.03,0.09]
# 本文件原先是 object_1 [0.20,0.25] / object_2 [0.12,0.19] / object_3 [0.03,0.10]，
# 与评测分布不一致（实测确认）：object_1 只端点相切、几乎不覆盖 ——
# 于是"在采集分布上量出来的成功率"并不代表评测表现。
# 现直接对齐到评测带，使示范数据覆盖评测分布（这是 README 授权的方向：
# 改采集去覆盖评测，而不是缩评测迁就策略）。
#
# X 方向本来就一致（两者都是 [0.90, 1.10]），无需改动。
OBJ_SPAWN_Y_BANDS = {
    1: (TABLE_CENTER_Y + 0.25, TABLE_CENTER_Y + 0.29),
    2: (TABLE_CENTER_Y + 0.14, TABLE_CENTER_Y + 0.20),
    3: (TABLE_CENTER_Y + 0.03, TABLE_CENTER_Y + 0.09),
}

# Per-object 2-D bounding-box half-extents (metres, world XY plane, scale=1).
# Used for AABB overlap rejection during randomisation.
OBJ_HALF_EXTENTS: dict[int, tuple[float, float]] = {
    1: (0.050, 0.044),   # Sugar box   (half_x, half_y)
    2: (0.050, 0.030),   # Mustard bottle
    3: (0.100, 0.040),   # Banana
}
OBJ_BBOX_MARGIN = 0.015  # extra clearance between object bounding boxes



# ------------------------------------------------------------------ #
# Success / basket bounds
# ------------------------------------------------------------------ #
BASKET_IN_X = 0.20   # ± metres in X around BASKET_CENTER_X
BASKET_IN_Y = 0.11   # ± metres in Y around BASKET_CENTER_Y

# ------------------------------------------------------------------ #
# Camera (for --save_video / --save_images)
# ------------------------------------------------------------------ #
CAM_H, CAM_W = 480, 640
CAM_POS = (TABLE_CENTER_X - 1.2, TABLE_CENTER_Y, TABLE_TOP_Z + 0.8)
CAM_ROT = (0.957, 0.0, 0.290, 0.0)   # ~34° around +Y → looks forward-down

# ------------------------------------------------------------------ #
# Actuator overrides for data collection (high stiffness for fast tracking)
# ------------------------------------------------------------------ #
ACT_STIFFNESS    = 800.0
ACT_DAMPING      = 80.0
ACT_EFFORT_LIMIT = 100.0
ACT_VEL_LIMIT    = 100.0

# ------------------------------------------------------------------ #
# Warm-up / settle steps before recording starts
# ------------------------------------------------------------------ #
WARMUP_STEPS = 150
SETTLE_STEPS = 30

# LIFT 分段抬升：单段最大抬升高度（米）。
#
# 实测问题：LIFT 一步从抓取高度抬到 CARRY_Z（约 25 cm）时，IK 为够到目标
# 会重排关节，末端水平位置守不住 —— 实测 base 在 X 上漂 5.7 cm、
# 指尖向 -X 扫出 13.4 cm，把生在前方的 object_2 撞倒。
# 拆成"先抬 LIFT_STAGE_MAX、再抬到顶"两段后，每段目标都在可达范围内。
LIFT_STAGE_MAX = 0.12

# LIFT 最终抬升高度上限（相对桌面，米）。
#
# 实测：CARRY_Z = TABLE_TOP_Z + 0.40 = 1.2266 这个高度机械臂"停不住"——
# 扫高度得到 X 误差：Z=1.02 时 +0.0001、Z=1.06 时 -0.0140、
# Z=1.10 时 -0.0302、Z=1.227 时 -0.0288 且 Z 超调 +0.057。
# 目标够不到时 IK 会重排关节，末端水平位置随之漂移，
# 实测 LIFT 期间 base 在 X 上漂 3.4~5.7 cm、指尖向 -X 扫出 11~13 cm，
# 把生在前方的 object_2 撞倒。
# 因此把抬升目标钳制到 TABLE_TOP_Z + 0.24（约 1.07），该高度 X 误差 <=1.4 cm。
LIFT_Z_CAP = TABLE_TOP_Z + 0.24

# ------------------------------------------------------------------ #
# 搬运路径：LIFT 之后先"后撤到无物体区"，再横摆去篮子
#
# 为什么必须后撤（实测数据）：
#   物体只生成在 X ∈ [0.90, 1.10]（OBJ_SPAWN_X_MIN/MAX），而机械臂底座在
#   X=1.404。抓取点 X≈0.96~1.10 在桌子远侧，从那里直接横摆向篮子
#   (1.080, -0.300) 时，末端+被夹物体要横扫整个物体生成区，实测把
#   object_2 撞出生成带（Y 从 0.126 → 0.020）。
#
# 为什么不能靠"抬更高"解决（实测数据）：
#   要让 object_1 底面高过 object_2 顶面，物体中心需 z≥1.0172、末端需
#   z≥1.153；而实测 base z=1.10 时 X 跟踪误差已达 -0.030 m、z=1.153
#   根本够不到（见 LIFT_Z_CAP 注释）。所以"垂直升更高再横摆"在本题
#   机械臂上不可行。
#
# 后撤方案为什么可行：
#   后撤方向(+X)顺着机械臂自然伸展方向，IK 不需重排关节 —— 实测
#   (1.20,0.21,1.05)/(1.25,0.21,1.07) 的 IK 误差为 0.0000，远好于横摆
#   时的 3 cm 漂移。撤到 X≈1.20 后，末端及其下方已完全离开物体生成区
#   (X≤1.10)，此后的横摆是在"无物体区"上空进行。
#
# 高度：默认沿用 LIFT 结束时的实际高度(RETREAT_USE_LIFT_Z=True)，
# 因为该高度实测可达且稳（LIFT_Z_CAP=1.0666）；后撤时不再改 Z，
# 避免 IK 在抬升+横移同时作用下重排关节。
# ------------------------------------------------------------------ #

# 后撤目标 X（世界系，米）。需 > OBJ_SPAWN_X_MAX + 物体半尺寸(0.05) + 余量。
RETREAT_POS_X = TABLE_CENTER_X + 0.20          # = 1.20

# 后撤段的最终高度。
#
# 【实测问题：原来用 LIFT 结束高度 -> 横移太矮、被篮沿碰掉】
# 用户实测发现：爪子在篮子高度的横移中被篮沿碰掉。数据证实：
#     LIFT 目标 1.0666 -> RETREAT 1.064 -> TRANSPORT 1.059
# 全程几乎不变高，而篮子 z 上界（篮沿）实测 0.9798。
# 承载物体底面要高于篮沿才不会被碰掉：
#     object_2(高0.1913, 侧抓) 末端需 >= 1.171
#     object_1/3(高0.088, 顶抓) 末端需 >= 1.124
# 后撤方向(+X)在物体生成区之外、上方没有障碍，所以**可以在后撤的同时抬到
# CARRY_Z=1.2266**，从而让之后的横移在篮沿之上进行。
#
# RETREAT_USE_LIFT_Z=True  = 沿用 LIFT 结束时的实际高度（旧行为，会碰篮沿）
# RETREAT_USE_LIFT_Z=False = 用 RETREAT_POS_Z（新行为，抬到 CARRY_Z）
RETREAT_USE_LIFT_Z = False
RETREAT_POS_Z = CARRY_Z

# 执行后撤段的物体编号。留空 = 所有物体都后撤（推荐：对所有物体都更安全）。
RETREAT_APPLY_OBJECTS = ()

# 侧抓物体(object_2)的搬运段是否也改用统一实现（_carry_target）。
#
# 【实测：必须为 True】原历史行为对 object_2 的 LIFT 用的是 CARRY_Z=1.2266，
# 而该高度实测够不到（见 LIFT_Z_CAP 注释）。实测后果（本轮 trace）：
#   object_2 CLOSE 结束时末端在 (1.013, 0.125, 0.983)，
#   LIFT 段追 1.2266 这个不可达目标时牺牲水平位置，
#   末端一路漂到 (0.793, 0.096, 0.975) —— 水平漂移 21 cm，
#   瓶子在指间被带着走然后滑落（抬升只 +0.073 m = 未夹起）。
# 改成统一实现后 LIFT 目标为可达的 LIFT_Z_CAP，不再为追高度牺牲水平位置。
#
# 副作用（已知、需实测确认）：object_2 在 TRANSPORT/PLACE 阶段的末端姿态
# 由"默认朝下"改为"侧抓姿态"（见 state_machine._CARRY_STATES）。
# 物体是水平夹住的，旋转会让它在指间打滑，所以保持抓取姿态更合理。
CARRY_UNIFY_SIDE_GRASP = True

# ------------------------------------------------------------------ #
# 每件物体抓完后是否让机械臂回"默认位形"（默认开启）
#
# 【实测根因：必须开启】逐件抓取时，前一件"放进篮子"的动作会把腕部留在
# 关节顶限位的位形上。实测（seed 0，logs/demo_watch.log）：
#   object_2 RETRACT   q=[...,1.745,-1.220,-2.094]  限位饱和=[4,5,6]
#   object_3 PRE_GRASP q=[...,1.745,-1.177,-2.094]  限位饱和=[4,6]
#   → 之后末端完全不动（误差卡在 0.179 不收敛，STEP 40 步都不变）
# 关节 4/5/6 饱和后差速 IK 推不动，下一件物体的接近只能从坏位形开始。
# 开启后：每件物体都在 RETRACT 的第 (STEPS-RETRACT_HOME_STEPS) 步起
# 直接命令回 default_joint_pos，使每次抓取的初始位形一致、与物体编号无关。
# 关掉可复现"卡住"现象：把本行改成 False。
# ------------------------------------------------------------------ #
HOME_AT_RETRACT = True

# RETRACT 段中用于"回默认位形"的步数（从该状态末尾往前数）。
RETRACT_HOME_STEPS = 45

# ------------------------------------------------------------------ #
# object_1 专用抓取调整
#
# 抓取点是物体的真值位置加上 X 方向偏置，用来把抓取点推进 IK 可达区。
# 实测（本机 Isaac Sim）：抓取点 X < 1.06 时差分 IK 稳态误差约 0.12 m，
# 夹爪停在桌面以上 12 cm 处够不到物体；X >= 1.06 时误差约 0.017 m。
#
# 只对 object_1 生效；object_2 / object_3 保持原代码行为（无偏置、
# PRE_GRASP 用默认朝下姿态）。
# ------------------------------------------------------------------ #
GRASP_X_OFFSET = 0.04
GRASP_Y_OFFSET = 0

# ------------------------------------------------------------------ #
# object_1 专用：按物体位置动态调整 X 偏置
#
# 动态规则:  offset = max(GRASP_X_OFFSET, BASE + SLOPE * (SPAWN_X_MAX - obj_x))
#            （即：物体越远(x 越小)，抓取点越往 +x / 机械臂一侧推）
#
# 【2026-10-07 根因定位 —— 这是"腕部/臂展边界"问题，不是普通调参】
# 实测失败时的末端诊断（logs/probe_obj1_offset.py 的 CLOSE 诊断）：
#   期望末端 z=0.962，实际停在 1.006~1.048（高 4.4~8.6 cm），指尖在盒子上方
#   或侧面合爪 -> 抬升 +0.000；同时 **joint5 顶在 +1.22（其上限）、joint6 顶在
#   -2.094（其下限）**。即机械臂为了"够到 + 保持朝下姿态"把腕部用到了极限，
#   位置残差随抓取点离底座的距离急剧增大。
# 真实分布上的分界（Oct-06 评测带 10 布局的真实数据）：
#   obj1 x >= 1.002 -> 夹起 7/7 ；x <= 0.965 -> 夹起 0/3（硬分界，阈值 x≈1.00）
# 因此正确的做法是"把抓取点推进可达区"，而不是加大夹持力/改姿态。
#
# 【A/B 实测（真实随机布局，每次 15 个布局，只跑 obj1 循环，无头）】
#   旧值 BASE=0.035/SLOPE=0.15（远端 0.055~0.060） -> 夹起  5/15 = 33%
#   新值 BASE=0.028/SLOPE=0.30（远端 0.070~0.088） -> 夹起 15/15 = 100%
#   新值 + 抓取姿态绕进给轴翻转 180°             -> 夹起 15/15 = 100%（无额外增益，
#        故 LONG_AXIS_GRASP_ROLL_DEG 保持 0；该旋钮在 joint6 顶限位的布局上
#        确实能把 joint6 从 -2.094 挪开，是可用的备用手段，但本批数据无需启用）
#   命令: python scripts/act/collect_demos_task_e.py --pick_objects 1 \
#           --num_demos 1 --max_attempts 15 --headless --only_success
#
# 固定偏置扫描（logs/probe_obj1_offset.py，y=0.26~0.27，每档 2 次）：
#   x=0.90 → 0.09 ✓（0.05/0.11/0.13 ✗）
#   x=0.93 → 0.07 ✓（0.055/0.085 ✗）      x=0.94 → 0.07 ✓（0.05/0.09 ✗）
#   x=0.99 → 0.00~0.10 全部 ✗（该点最敏感，见探针日志）
#   x=1.05 → 0.04 ✓(1/2)                   x=1.09 → 0.03~0.055 ✓（0.07+ ✗）
# 即远端需要 ≈0.07~0.09，近端仍要 0.04~0.055（近端给大了会把盒子推走）。
# 当前取值产生：0.90→0.088 0.93→0.079 0.96→0.070 0.99→0.061
#              1.02→0.052 1.05→0.043 1.08→0.034→(被 0.04 下限托住)
# 注意：以上都在 y∈[0.25,0.29]（评测带）内测得；"提前旋转"始终开启
# （PRE_ROTATE_IN_PREGRASP_OBJECTS=(1,) 与 LONG_AXIS_GRASP_OBJECTS=(1,) 不变）。
# ------------------------------------------------------------------ #
GRASP_X_OFFSET_DYNAMIC_OBJECTS = (1,)
GRASP_X_OFFSET_DYN_BASE  = 0.028
GRASP_X_OFFSET_DYN_SLOPE = 0.30

# 需要施加抓取偏置的物体编号（其余物体不加偏置）
GRASP_OFFSET_OBJECTS = (1,)

# 这些物体在 PRE_GRASP 阶段就提前旋转到抓取姿态。
#
# 【实测结论：留空 —— 不要启用】
# 原设想是"REACH 下降过程中才旋转，扫掠会碰倒盒子，所以把旋转提前"。
# 但 Isaac Sim 实测显示提前旋转反而降低成功率（object_1 @X=1.02：
# 原始代码 6/6 -> 提前旋转 4/6）。原因是旋转提前完成后，夹爪以固定抓取
# 姿态进入下降段，而该姿态下差分 IK 的可达范围更小；原代码在 REACH 中
# 边下降边旋转，反而更容易把末端"送"到位。
# 如需重做该实验，把 1 加回本元组即可。
PRE_ROTATE_IN_PREGRASP_OBJECTS = (1,)
# 说明：LONG_AXIS_GRASP_OBJECTS 中的物体现在会自动在 PRE_GRASP 提前旋转
# （见 _get_target_quat），这里显式写上 object_1 是为了可读性与双保险。

# PRE_GRASP 总步数(STEPS["PRE_GRASP"]=200)中，用于在 RETRACT 上方完成
# 旋转的步数；剩余步数用于带着正确姿态平移到物体上方。
# 仅在 PRE_ROTATE_IN_PREGRASP_OBJECTS 非空时生效。
PREGRASP_ROTATE_STEPS = 120

# ------------------------------------------------------------------ #
# object_2 专用：横向（侧向）夹取
#
# 背景：object_2 是细高瓶（实测世界尺寸 dx=0.096, dy=0.058, dz=0.191），
# 顶抓只能夹到瓶盖附近，容易滑脱。改为从侧面水平进给，
# 钳口张开方向对齐水平面内的长轴，抓取点取物体高度中部。
#
# 几何依据（本机实测）：
#   * 夹爪指爪接触面位于 gripper_base 局部 +Z 方向 SIDE_GRASP_TCP_OFFSET 处
#     —— 即"末端定位在 joint6 上，没算爪子长度"，顶抓时影响小，
#     横向抓取时这个长度会整体改变接触位置，必须补偿。
#   * 钳口张开间距 0.070 m（GRIPPER_OPEN_POS = ±0.035），
#     而 object_2 的短边 0.058 m，可容纳。
# ------------------------------------------------------------------ #

# 采用横向夹取的物体编号
SIDE_GRASP_OBJECTS = (2,)

# gripper_base 局部 +Z 方向上，指爪接触面到末端原点的距离（米）。
# 实测：指爪 link7/link8 几何在局部 Z 的 0.069~0.170 之间，取指根位置。
SIDE_GRASP_TCP_OFFSET = 0.1358

# 抓取点在物体高度上的比例（0.5 = 高度中部）
SIDE_GRASP_HEIGHT_FRAC = 0.5

# 抓取深度：沿进给方向从物体中心再深入的量（米）。
#
# 实测反馈：0.0 时夹爪落点太浅，夹不住或夹起后掉落
# （指爪只是勉强搭在物体前缘）。增大此值会把接触面沿进给方向
# 继续推进，让钳口更充分地抱住瓶身。
#
# 实测扫描（object_2 @X=1.02，每档 3 次，抬升 = 夹持牢固度指标）：
#   0.000 -> 夹起 3/3 入篮 2/3，抬升 +0.227/+0.355/+0.351
#            （有一次只抬到 0.227 就滑落，夹持不稳）
#   0.010 -> 夹起 3/3 入篮 3/3，抬升 +0.310/+0.364/+0.369
#   0.020 -> 夹起 3/3 入篮 3/3，抬升 +0.365/+0.371/+0.373  ← 取此值
#   0.030 -> 夹起 3/3 入篮 3/3，抬升 +0.369/+0.373/+0.364
# 0.020 与 0.030 都能全过，但 0.020 抬升更一致、手指更不易顶到瓶身另一侧。
SIDE_GRASP_MARGIN = 0.020

# 进给轴相对水平面的下倾角（度）。0 = 完全水平侧向进给。
#
# 【实测约束：不能取 0】
# 完全水平要求腕关节 joint5 转 90°，而其限位仅 ±1.22 rad(±70°)，
# 因此 0° 不可达（IK 稳态误差 0.0343 m，卡在 joint5 上限）。
# 实测可达区间（抓取点 Z=0.92）：
#   0°  -> ✗ 0.0343 m，卡 joint5
#   20° -> ✓ 0.0000 m
#   30° -> ✓ 0.0000 m   ← 取此值：倾斜最小且完美可达
#   40° -> ✓ 0.0012 m
#   50° -> ✓ 0.0195 m（临界）
#   60° -> ✗ 0.0666 m
# 30° 时钳口仍保持水平（张开方向对齐长轴），只是从斜上方切入瓶身侧面。
SIDE_GRASP_APPROACH_PITCH_DEG = 30.0

# 侧抓进给方向的水平分量符号（+1 / -1）。
#
# 为什么需要这个：接触面在 base + ap*TCP，所以 base = 物体中心 - ap*TCP。
# ap 反号会让 base 落到物体另一侧，两个解在几何上"都合法"，但可达性完全不同。
# 实测（object_2 在 X≈1.02, Y≈0.15, 抓取点 Z≈0.92）：
#   ap 水平分量 = +Y（远离机器人基座） -> IK 误差 0.0000 m，可达
#   ap 水平分量 = -Y（朝向机器人基座） -> IK 误差 0.0343 m，卡 joint5，夹爪扑空
# 因此取 +1。若把物体放到桌子另一侧，需要相应改成 -1。
SIDE_GRASP_APPROACH_SIGN = -1.0   # -1 => 沿 -X 进给（实测朝向）

# 进给轴取水平面内的哪根轴："short" | "long"
#
# 实测纠正：object_2 的正确夹法是「沿水平面内较短的轴进给、钳口夹另一根轴」。
# 该物体水平面内：长轴 0.096 m、短轴 0.058 m。
#   "short"（默认）: 进给沿短轴，钳口张开对齐长轴 -> 掌面朝向物体窄端
#   "long"          : 进给沿长轴，钳口张开对齐短轴
# 用户实测结论：朝向 -X、沿短轴方向夹取。
SIDE_GRASP_APPROACH_AXIS = "long"   # 沿长轴(X)进给、钳口夹 0.058 的短轴

# 顶抓时是否对 PRE_GRASP / LIFT 的高度做 TCP 补偿（0.1358 m）。
#
# 补偿的物理依据：末端原点在 gripper_base，指爪接触面还要沿末端本地 +Z
# 再延伸 SIDE_GRASP_TCP_OFFSET。不补偿时 PRE_GRASP 的指爪接触面会比
# 预期低 13.6 cm，机械臂为够到目标会下沉很深才开始旋转（用户实测：
# 物体1 离得远时抓不到盒子）。
#
# 【A/B 实测结论：关闭】
# 补偿=1 -> 整体成功 1/2，object_2 出现 1/2 失败，object_1/3 抬升偏大
#           (+0.547 / +0.530，疑似被"抛"起来而非稳夹)
# 补偿=0 -> 整体成功 2/2，三件物体全部 2/2 夹起且 2/2 入篮，
#           抬升 +0.412 / +0.351 / +0.370（与历史稳定值一致）
# 结论：抬高 PRE_GRASP 目标反而降低稳定性，保持原行为。
# 若日后仍遇到"离得远时下沉过深抓不到"，应从抓取姿态/可达性入手，
# 而不是抬高度。
TCP_COMPENSATE_TOP_DOWN = False

# ------------------------------------------------------------------ #
# object_1 专用：开合轴垂直于长轴的抓取姿态
#
# 目标（用户提出）：让夹爪"开合轴"提前垂直于
#   1) 物体长轴；且
#   2) 物体长轴与 Z 轴所构成的平面。
# 同时允许一定侧倾 —— 纯垂直姿态会让 wrist 的 joint5 超出 ±1.22 rad
# 限位而不可达（实测该姿态 IK 误差 0.034 m，夹爪停在物体外）。
#
# 与侧抓(object_2)的区别：这里进给方向仍以"向下"为主，只是带倾斜；
# 侧抓则是完全水平进给。object_2/3 完全不受这些开关影响。
# ------------------------------------------------------------------ #

# 采用"长轴垂直开合"姿态的物体编号
LONG_AXIS_GRASP_OBJECTS = (1,)

# 末端接近方向相对竖直向下的倾斜角（度）。
#
# 【实测结论：侧倾无助益，取 0】
# 在远端 X=0.96（原顶抓的失败区）逐档扫描，每档 2 次：
#   tilt= 0° -> 夹起 1/2（抬升 +0.408 那次成功）
#   tilt=20° -> 夹起 0/2
#   tilt=35° -> 夹起 0/2
#   tilt=50° -> 夹起 0/2（抬升 +0.089，有接触但没夹住）
# 侧倾并不能把远端位置救回来，反而让手指偏离盒体中心。
# 注意：tilt=0 时"开合轴⊥长轴、⊥长轴×Z 平面"这两条要求依然满足
# （见 compute_long_axis_grasp_quat 的构造），所以无需为几何正确性引入侧倾。
LONG_AXIS_GRASP_TILT_DEG = 0.0

# 在垂直于接近方向的平面内，开合轴的滚转角（度）。
# 0 表示开合轴取"垂直于长轴且尽量水平"的方向。
LONG_AXIS_GRASP_ROLL_DEG = 0.0


# ------------------------------------------------------------------ #
# 示范多样性（采集侧开关，--diversity）
#
# 【为什么需要 —— 2026-10-10 实测】
# 状态机是**纯步数**推进的（见 STEPS），于是：
#   * 53 条示范的夹爪跳变时刻 **全部**是 (399, 1119, 1639, 2359, 2879)；
#   * 每条**都恰好** 3431 帧；
#   * 夹爪两个动作通道跨示范的逐帧标准差**恒为 0.0000**。
# 也就是说示范只覆盖了"按同一张时刻表完美执行"这一条窄管，**完全不含
# "偏离计划之后如何回到正轨"的信息**。闭环一旦漂移就没有可学的重同步行为。
# 证据（v1 30k 策略，评测 seed 1001）：轨迹反复跑到示范关节云之外
# （最近邻距离最大 0.43 rad，而示范自身 99 分位只有 0.129），并在篮子上方
# 进入极限环 6 次以上，object_2/3 全程一动未动。
#
# 【两个旋钮】
#   * 时序抖动：每条示范给每个状态乘一个随机因子，使"何时切换状态"不再固定
#     （例如 CLOSE 会在不同的下降高度触发、抬升会在不同点分段）。
#   * 状态进入扰动：在指定状态**刚进入时**给 6 个臂关节注入高斯扰动。
#     状态机是笛卡尔目标驱动的，会自己纠回来 —— 于是录到的就是
#     "从偏了的地方恢复"的示范。这是 DAgger 式增广，**不需要额外仿真轮次**，
#     也不改动任何评测相关代码。
#
# 注意：这两个开关只影响**采集到的示范分布**，不碰评测环境、奖励或成功条件。
# ------------------------------------------------------------------ #

# 每个状态步数的乘性抖动区间（每条示范独立抽样）。
DIVERSITY_STEPS_JITTER = (0.70, 1.40)

# 注入到 6 个臂关节的高斯扰动标准差（弧度）。0 = 关闭。
DIVERSITY_PERTURB_SIGMA = 0.15

# 在哪些状态"进入时"施加扰动。
#   INIT      —— 只在第一条物体的最开头（之后状态机不再回 INIT）
#   PRE_GRASP —— 每件物体都会经过（object_2/3 是从 RETRACT 直接跳过来的），
#                是最划算的注入点：此时臂在高处、下方没有易碰倒的东西。
# 不建议默认加 REACH/CLOSE/LIFT：那几段注入扰动会直接导致夹空或掉件。
DIVERSITY_PERTURB_STATES = ("PRE_GRASP",)

# 每次进入上述状态时施加扰动的概率（1.0 = 每次都加）。
DIVERSITY_PERTURB_PROB = 1.0

# 扰动后允许状态机自行纠正的步数上限（仅用于日志/诊断，不参与控制）。
DIVERSITY_NOTE = "timing jitter + state-entry perturbation (collect-side only)"
