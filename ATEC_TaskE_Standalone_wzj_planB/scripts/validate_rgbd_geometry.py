"""Compare observation-only RGB-D/FK estimates with simulator truth.

Development diagnostic only. The submitted policy does not import this file or
read any simulator object pose, camera data object, or PhysX Jacobian.
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Validate Task E RGB-D and Piper FK geometry")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--settle_steps", type=int, default=30)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from demo.rgbd_control.constants import DEFAULT_Q, ROBOT_BASE_XYZ  # noqa: E402
from demo.rgbd_control.kinematics import PiperKinematics  # noqa: E402
from demo.rgbd_control.perception import TaskPerception  # noqa: E402
from atec_rl_lab.tasks.task_e.env_cfg import TaskEEnvPiperCfg  # noqa: E402


def main():
    cfg = TaskEEnvPiperCfg(seed=args.seed)
    cfg.scene.num_envs = 1
    env = ManagerBasedRLEnv(cfg)
    try:
        obs, _ = env.reset()
        for _ in range(args.settle_steps):
            obs, _, terminated, truncated, _ = env.step(torch.zeros((1, 8), device=env.device))
            if bool(terminated[0]) or bool(truncated[0]):
                raise RuntimeError("Episode ended during geometry check")

        rgb = obs["image"]["video_rgb"][0].detach().cpu().numpy()
        depth = obs["image"]["video_depth"][0].detach().cpu().numpy()
        perception = TaskPerception()
        perception.update(rgb, depth)
        scene = env.unwrapped.scene
        env_origin = scene.env_origins[0].detach().cpu().numpy()
        print(f"depth shape={depth.shape}, finite fraction={np.isfinite(depth).mean():.3f}")
        print(f"env origin={env_origin.tolist()}")
        visual_errors = []
        for oid in (1, 2, 3):
            estimate = perception.detect(oid)
            truth = scene[f"object_{oid}"].data.root_pos_w[0, :3].detach().cpu().numpy() - env_origin
            if estimate is None:
                print(f"object_{oid}: NO DETECTION; true xyz={truth.tolist()}")
                visual_errors.append(float("inf"))
            else:
                error = np.linalg.norm(estimate.xy - truth[:2])
                visual_errors.append(float(error))
                print(f"object_{oid}: estimate_xy={estimate.xy.tolist()}, true_xy={truth[:2].tolist()}, "
                      f"xy_error_m={error:.4f}, points={estimate.point_count}")

        robot = scene["robot"]
        joint_ids, _ = robot.find_joints([f"joint{i}" for i in range(1, 7)])
        ee_ids, _ = robot.find_bodies("gripper_base")
        q_obs = obs["proprio"][0, :8].detach().cpu().numpy() + DEFAULT_Q
        q_truth = robot.data.joint_pos[0, joint_ids].detach().cpu().numpy()
        ee_truth = robot.data.body_pose_w[0, ee_ids[0], :3].detach().cpu().numpy() - env_origin
        fk_xyz, _ = PiperKinematics().forward(q_obs[:6])
        root_truth = robot.data.root_pos_w[0].detach().cpu().numpy() - env_origin
        print(f"joint_observation_error_rad={np.max(np.abs(q_obs[:6]-q_truth)):.5f}")
        print(f"robot_root_xyz={root_truth.tolist()}, configured={ROBOT_BASE_XYZ.tolist()}")
        print(f"ee_fk_xyz={fk_xyz.tolist()}, ee_true_xyz={ee_truth.tolist()}, "
              f"ee_error_m={np.linalg.norm(fk_xyz-ee_truth):.4f}")
        if max(visual_errors) > 0.04 or np.linalg.norm(fk_xyz-ee_truth) > 0.025:
            raise RuntimeError("RGB-D or FK geometry exceeds tolerance; calibrate constants before rollout")
        print("Geometry check passed")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
