"""Static Task E geometry and Piper model parameters.

These values describe the public task configuration and robot asset. They are
not episode object poses. The camera and FK must be checked against Isaac Sim
once on the deployment machine before using the controller for long runs.
"""

import numpy as np

TABLE_TOP_Z = 0.6613141183247961 * (0.01 / 0.008)
TABLE_CENTER_X = 1.0
ROBOT_BASE_XYZ = np.array([1.40425390256285, 0.0, TABLE_TOP_Z])
# TaskEEnvPiperCfg keeps PIPER_CFG's initial quaternion (w, x, y, z).
ROBOT_BASE_WXYZ = np.array([0.0, 0.0, 0.0, 1.0])

CAMERA_XYZ = np.array([-0.2, 0.0, TABLE_TOP_Z + 0.8])
CAMERA_WXYZ = np.array([0.957, 0.0, 0.290, 0.0])
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
FOCAL_LENGTH_MM = 24.0
HORIZONTAL_APERTURE_MM = 20.955

DEFAULT_Q = np.array([0.0, 1.2, -1.5, 0.0, 1.2, 0.0, 0.035, -0.035])
HOME_Q = np.array([0.0, 0.924525, -1.514983, 0.0, 1.219900, 0.0])
ACTION_SCALE = 0.5
GRIPPER_OPEN = np.array([0.035, -0.035])
GRIPPER_CLOSE = np.array([-0.015, 0.015])

# The six movable joint origins and axes are initialized from AgileX's Piper
# description. The local binary USD may have been converted from another
# description revision; validate FK against the simulator before rollout.
# xyz/rpy are relative to the preceding link.
JOINT_XYZ = np.array([
    [0.0, 0.0, 0.123],
    [0.0, 0.0, 0.0],
    [0.28503, 0.0, 0.0],
    [-0.021984, -0.25075, 0.0],
    [0.0, 0.0, 0.0],
    [0.000088259, -0.091, 0.0],
])
JOINT_RPY = np.array([
    [0.0, 0.0, 0.0],
    [1.5708, -0.1359, -3.1416],
    [0.0, 0.0, -1.7939],
    [1.5708, 0.0, 0.0],
    [-1.5708, 0.0, 0.0],
    [1.5708, 0.0, 0.0],
])
JOINT_AXIS = np.tile(np.array([0.0, 0.0, 1.0]), (6, 1))
JOINT_MIN = np.array([-2.618, 0.0, -2.967, -1.745, -1.22, -2.0944])
JOINT_MAX = np.array([2.618, 3.14, 0.0, 1.745, 1.22, 2.0944])

BASKET_XY = np.array([1.08, -0.30])
BASKET_SLOTS = {
    1: np.array([1.02, -0.30]),
    2: np.array([1.14, -0.30]),
    3: np.array([1.08, -0.26]),
}
CARRY_Z = TABLE_TOP_Z + 0.40
PRE_GRASP_CLEARANCE = 0.12
LIFT_CLEARANCE = 0.16
# Release above the contact-limited pose. The previous +0.15 m target
# stalled with the observed gripper base at about +0.22 m.
PLACE_Z = TABLE_TOP_Z + 0.30
RETRACT_XYZ = np.array([1.35425390256285, 0.0, CARRY_Z])

# Distinct, fixed spawn bands permit class assignment without learned weights.
OBJECT_Y_BANDS = {1: (0.25, 0.29), 2: (0.14, 0.20), 3: (0.03, 0.09)}
OBJECT_CENTER_Z = {1: TABLE_TOP_Z + 0.05, 2: TABLE_TOP_Z + 0.10, 3: TABLE_TOP_Z + 0.02}
# The sugar box and banana can enter the bottle's broad XY crop. Use points
# above both neighbours, but below the cap, to centre the bottle across Y.
BOTTLE_BODY_LOCALIZE_Z_MIN = TABLE_TOP_Z + 0.115
BOTTLE_BODY_LOCALIZE_Z_MAX = TABLE_TOP_Z + 0.180
BOTTLE_BODY_LOCALIZE_MIN_POINTS = 24
# The fixed camera sees the near face of the sugar box, biasing its visual
# centre roughly 2-3 cm toward -X. Correct that bias without targeting its
# far edge (the previous +5 cm offset put both fingers on the box lid).
# The bottle's depth-visible face biases its point median toward -X/-Y.
# The geometry check measured true minus observed at about (+0.035, +0.009) m.
GRASP_XY_OFFSETS = {
    1: np.array([0.025, 0.0]),
    2: np.array([0.035, 0.009]),
    # The robot is on +X. Move the banana grasp 1.2 cm forward toward -X
    # to sample a slightly different section of its official collision mesh.
    3: np.array([-0.006, 0.0]),
}
# Keep the bottle's separate side-entry path. The banana is again approached
# vertically; its requested wrist height is table +9 cm (centre +7 cm).
# A stalled descent closes at the measured pose instead of requiring this
# fixed height, which may sit inside the object's collision shape.
GRASP_Z_OFFSET = {1: 0.14, 3: 0.0}
# Pitch the banana fingers slightly forward while keeping their downward
# approach. Only object 3 uses this wrist orientation.
BANANA_FORWARD_TILT_DEG = 15.0
BOTTLE_SIDE_DOWN_ANGLE_DEG = 40.0
# Rotate the wrist high and outside the bottle. The former +13-16 cm,
# table+29 cm waypoint let the wrist dip 6 cm while close to the cap.
# Limit the absolute X of the high waypoint for near-arm spawns: a point too
# close to the robot base is not reachable with the side-grasp orientation.
BOTTLE_PRE_MIN_X = 0.20
BOTTLE_PRE_MAX_X = 0.26
BOTTLE_PRE_MAX_WORLD_X = 1.28
BOTTLE_PRE_GUARD_X = 0.22
BOTTLE_PRE_GUARD_Z = TABLE_TOP_Z + 0.34
BOTTLE_FAR_PRE_Z = TABLE_TOP_Z + 0.38
BOTTLE_SIDE_STAGE_X = 0.15
# Increase the outside descent clearance smoothly across the bottle spawn
# range. A grasp centre at x >= 1.00 m receives the full 3 cm increase; the
# last test at x = 1.016 m still used 0.15 m under the old 1.03 m threshold.
BOTTLE_STAGE_RAMP_START_X = 0.94
BOTTLE_STAGE_RAMP_END_X = 1.00
BOTTLE_SIDE_NEAR_STAGE_X = 0.18
# The +Y edge of the bottle spawn band lies close to the sugar box. Keep
# the open fingers farther outside the bottle while rotating/descending in
# that region. Low-Y bottles retain their previously successful entry path.
BOTTLE_SUGAR_SIDE_RAMP_START_Y = 0.17
BOTTLE_SUGAR_SIDE_RAMP_END_Y = 0.19
BOTTLE_SIDE_GRASP_X = 0.07
BOTTLE_SIDE_APPROACH_POS_TOLERANCE = 0.006
BOTTLE_SIDE_PRE_Z = TABLE_TOP_Z + 0.42
BOTTLE_SIDE_GRASP_Z = TABLE_TOP_Z + 0.18
# Keep the held bottle upright during the return. The carry frame points
# toward the robot's radial reach, only 10 degrees below horizontal.
BOTTLE_CARRY_DOWN_ANGLE_DEG = 10.0
BOTTLE_CARRY_Z = TABLE_TOP_Z + 0.38
BOTTLE_CARRY_WAYPOINTS = (
    np.array([1.08, 0.14]),
    np.array([1.22, -0.05]),
)
# Approximate horizontal offset from gripper_base to the held bottle centre.
BOTTLE_PLACE_FINGER_REACH = 0.10
BOTTLE_PLACE_Z = TABLE_TOP_Z + 0.33
OBJECT_ORDER = (1, 2, 3)
