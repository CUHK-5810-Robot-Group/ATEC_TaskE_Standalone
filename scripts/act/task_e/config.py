"""Task E demo-collection constants.
"""

from atec_rl_lab.tasks.task_e.env_cfg import (
    BASKET_CENTER_X, BASKET_CENTER_Y,
    BASKET_SUCCESS_CENTER, BASKET_SUCCESS_HALF_X, BASKET_SUCCESS_HALF_Y,
    TABLE_CENTER_X, TABLE_CENTER_Y, TABLE_TOP_Z, TABLE_HALF_X,
    BASKET_EXCL_HALF_X, BASKET_EXCL_HALF_Y,
    OBJ_SPAWN_X_MIN, OBJ_SPAWN_X_MAX, OBJ_SPAWN_Z, OBJ_SPAWN_Y_BANDS,
    OBJ_HALF_EXTENTS as _OBJ_HALF_EXTENTS_NAMED,
    OBJ_BBOX_MARGIN,
)

__all__ = [
    "BASKET_CENTER_X", "BASKET_CENTER_Y",
    "TABLE_CENTER_X", "TABLE_CENTER_Y", "TABLE_TOP_Z", "TABLE_HALF_X",
    "BASKET_EXCL_HALF_X", "BASKET_EXCL_HALF_Y",
    # robot
    "EE_BODY_NAME", "ARM_JOINT_NAMES", "GRIPPER_JOINT_NAMES",
    "GRIPPER_OPEN_POS", "GRIPPER_CLOSE_POS", "ACTION_SCALE",
    # state machine
    "STEPS", "STATE_ORDER", "PRE_GRASP_HOME_DWELL",
    # geometry
    "PRE_GRASP_CLEARANCE", "GRASP_Z_OFFSET", "GRASP_Z_OFFSET_BY_OBJECT",
    "GRASP_XY_OFFSET_BY_OBJECT",
    "SIDE_GRASP_OBJECT_IDS", "SIDE_GRASP_Z_OFFSET",
    "SIDE_GRASP_APPROACH_DIST", "SIDE_GRASP_STANDOFF",
    "SIDE_GRASP_XY_OFFSET_BY_OBJECT",
    "CARRY_Z", "PLACE_HEIGHT",
    "RETRACT_POS_X", "RETRACT_POS_Y",
    "DEFAULT_PLACE_QUAT_W",
    # success check
    "BASKET_IN_X", "BASKET_IN_Y", "BASKET_SUCCESS_MIN_Z", "BASKET_SUCCESS_MAX_Z",
    # spawn regions
    "OBJ_SPAWN_X_MIN", "OBJ_SPAWN_X_MAX", "OBJ_SPAWN_Z", "OBJ_SPAWN_Y_BANDS",
    "OBJ_HALF_EXTENTS", "OBJ_BBOX_MARGIN",
    # camera
    "CAM_POS", "CAM_ROT", "CAM_H", "CAM_W",
    # actuator
    "ACT_STIFFNESS", "ACT_DAMPING", "ACT_EFFORT_LIMIT", "ACT_VEL_LIMIT",
    # warm-up
    "WARMUP_STEPS", "SETTLE_STEPS",
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
    "REACH":        150,
    "CLOSE":         40,
    "LIFT":         160,
    "TRANSPORT":    200,
    "PLACE":         80,
    "OPEN":          60,
    "LIFT_RETRACT":  80,
    "RETRACT":       80,
}
STATE_ORDER = ["INIT", "PRE_GRASP", "REACH", "CLOSE", "LIFT",
               "TRANSPORT", "PLACE", "OPEN", "LIFT_RETRACT", "RETRACT"]

# After object N>1, hold the home/retract pose this many PRE_GRASP steps
# (top-down only) so the arm reaches CARRY_Z before moving to the next object.
PRE_GRASP_HOME_DWELL = 120

# ------------------------------------------------------------------ #
# Geometry
# ------------------------------------------------------------------ #
PRE_GRASP_CLEARANCE = 0.12   # metres above object before descent
GRASP_Z_OFFSET      = 0.09   # default; see GRASP_Z_OFFSET_BY_OBJECT per object index

# EE target height = object_centre_z + offset (world frame). Tune while watching --save_video.
GRASP_Z_OFFSET_BY_OBJECT: dict[int, float] = {
    1: 0.07,   # sugar box — lower centre, slightly shallower approach
    2: 0.12,   # unused while object 2 is in SIDE_GRASP_OBJECT_IDS
    3: 0.07,   # banana — baseline
}

# Upright objects pinched from the side at mid-body (local Z horizontal).
# Object 2 is the standing mustard bottle; jaw opening is ~7 cm, so the
# pinch is across the shorter horizontal footprint (~6 cm).
SIDE_GRASP_OBJECT_IDS = {2}
# Pinch height of gripper_base above the rigid-body root. Origin is near the
# base of the bottle; ~0.10 m is mid-body. Tune with --save_video.
SIDE_GRASP_Z_OFFSET: dict[int, float] = {
    2: 0.08,
}
# Distance from the object root to gripper_base along the approach axis.
SIDE_GRASP_APPROACH_DIST: dict[int, float] = {
    2: 0.04,
}
# Extra retreat before REACH so the fingers clear the bottle, then move in.
SIDE_GRASP_STANDOFF: dict[int, float] = {
    2: 0.18,
}
# World-frame (dx, dy) added to the object root before side approach targets.
# Use for root vs mesh centre and jaw alignment along the bottle. Tune with --save_video.
SIDE_GRASP_XY_OFFSET_BY_OBJECT: dict[int, tuple[float, float]] = {
    2: (0.0, 0.07),
}

# World-frame (dx, dy) added to rigid-body root XY for gripper_base IK targets.
GRASP_XY_OFFSET_BY_OBJECT: dict[int, tuple[float, float]] = {
    1: (0.05, -0.025),   # sugar box — tune with --save_video
    2: (0.0, 0.0),   # unused — see SIDE_GRASP_XY_OFFSET_BY_OBJECT for object 2
    3: (0.0, 0.0),   # banana — usually OK at zero
}

CARRY_Z      = TABLE_TOP_Z + 0.40   # safe carry height
PLACE_HEIGHT = TABLE_TOP_Z + 0.15   # height at which to release into basket

RETRACT_POS_X = TABLE_CENTER_X + TABLE_HALF_X - 0.05
RETRACT_POS_Y = TABLE_CENTER_Y
DEFAULT_PLACE_QUAT_W = [0.0, 1.0, 0.0, 0.0]   # top-down orientation (w,x,y,z)

# Spawn bands imported from env_cfg (same as evaluation TaskEEnvPiperCfg).
OBJ_SPAWN_Y_BANDS = {
    1: OBJ_SPAWN_Y_BANDS["object_1"],
    2: OBJ_SPAWN_Y_BANDS["object_2"],
    3: OBJ_SPAWN_Y_BANDS["object_3"],
}

OBJ_HALF_EXTENTS: dict[int, tuple[float, float]] = {
    1: _OBJ_HALF_EXTENTS_NAMED["object_1"],
    2: _OBJ_HALF_EXTENTS_NAMED["object_2"],
    3: _OBJ_HALF_EXTENTS_NAMED["object_3"],
}

# Success check aligned with ObjectsInBasketDone / TaskETerminationsCfg.
BASKET_IN_X = BASKET_SUCCESS_HALF_X
BASKET_IN_Y = BASKET_SUCCESS_HALF_Y
BASKET_SUCCESS_MIN_Z = TABLE_TOP_Z
BASKET_SUCCESS_MAX_Z = TABLE_TOP_Z + 0.15

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

# After the state machine finishes, step physics with zero action before --only_success checks.
POST_DEMO_SETTLE_STEPS = 80
