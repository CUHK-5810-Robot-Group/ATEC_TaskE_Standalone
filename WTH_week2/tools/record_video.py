"""录制"完整三物体搬运"视频（README §4 交付物）。

设计
----
* 用 v2 100k 权重（`act-task-e-rgb-v2/checkpoints/final.pt`），走**标准观测推理**
  （`demo.solution.AlgSolution.predicts`，与评测完全同一条路径），不做任何特殊处理。
* 摄像机就是策略自己看的 `video_cam`（480×640）—— 录下来的正是策略的视角，
  不另设机位，避免"看起来好但不是策略看到的"。
* 候选种子取自随机留出种子测试里**已知 3/3 成功**的那批，按步数从少到多排序；
  逐局尝试，**录第一个当次也完整成功的**，录完即停（防止录到接近成功但没成功的局）。
* 每局开始都做与评测器完全相同的三件事：按评测分布重摆物体、复位机器人、
  重置策略内部状态（`env.reset()` 这三件都不做，见 HANDOFF_会话3 §3.4）。
* 视频按 50 fps 存（控制步 0.02 s → 实时速度播放）。
* 同时写一份逐帧 CSV（关节角/末端/三物体位置/阶段），便于核对。

用法
----
    python logs/record_video.py --out logs/videos/act_100k_full_run.mp4
"""
import argparse
import csv
import os
import sys
import time

ROOT = "/root/gpufree-share/Robotic2/test2/ATEC_TaskE_Standalone"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "act"))

from isaaclab.app import AppLauncher  # noqa: E402

p = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(p)
p.add_argument("--task", type=str, default="ATEC-TaskE-Piper")
p.add_argument("--seed", type=int, default=0, help="0=自动从候选里挑第一个成功的")
p.add_argument("--out", type=str,
               default=os.path.join(ROOT, "..", "logs", "videos", "act_100k_full_run.mp4"))
p.add_argument("--fps", type=int, default=50)
p.add_argument("--episode_len_s", type=float, default=150.0)
p.add_argument("--max_tries", type=int, default=6, help="最多尝试几个候选种子")
p.add_argument("--stride", type=int, default=1, help="每 N 步存一帧（1=全帧）")
a = p.parse_args()
a.enable_cameras = True
app = AppLauncher(a).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402

import atec_rl_lab.tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from atec_rl_lab.tasks.task_base.action_base import apply_safe_action_spec  # noqa: E402
from atec_rl_lab.tasks.task_e.env_cfg import (  # noqa: E402
    TABLE_CENTER_X, TABLE_CENTER_Y, TABLE_TOP_Z, OBJ_HALF_EXTENTS, OBJ_BBOX_MARGIN)
from demo.solution import AlgSolution  # noqa: E402

OBJS = (1, 2, 3)
NAMES = {1: "糖盒", 2: "芥末瓶", 3: "香蕉"}
EVAL_X_MIN, EVAL_X_MAX = TABLE_CENTER_X - 0.10, TABLE_CENTER_X + 0.10
EVAL_Y_BANDS = {1: (TABLE_CENTER_Y + 0.25, TABLE_CENTER_Y + 0.29),
                2: (TABLE_CENTER_Y + 0.14, TABLE_CENTER_Y + 0.20),
                3: (TABLE_CENTER_Y + 0.03, TABLE_CENTER_Y + 0.09)}
EVAL_Z = TABLE_TOP_Z + 0.05

# 随机留出种子测试中已知 3/3 完整成功的种子，按成功局步数从少到多排序
KNOWN_GOOD = [53841, 72100, 12768, 28480, 32897, 12664, 40387, 19292]

ckpt = os.environ.get("ATEC_TASK_E_POLICY") or os.path.join(
    ROOT, "scripts", "act", "runs", "act-task-e-rgb-v2", "checkpoints", "final.pt")
os.environ["ATEC_TASK_E_POLICY"] = ckpt
print(f"[INFO] 权重: {ckpt}", flush=True)
solution = AlgSolution()

env_cfg = parse_env_cfg(a.task, device=a.device, num_envs=1, use_fabric=True)
env_cfg.episode_length_s = a.episode_len_s
env_cfg = apply_safe_action_spec(env_cfg, solution.get_action_spec()
                                 if hasattr(solution, "get_action_spec") else None)
for _name in ("head_camera", "ee_dual_camera"):
    if hasattr(env_cfg.scene, _name):
        setattr(env_cfg.scene, _name, None)
_img = getattr(env_cfg.observations, "image", None)
if _img is not None:
    for _t in ("ee_rgb", "ee_depth", "video_depth"):
        if hasattr(_img, _t):
            setattr(_img, _t, None)

env = gym.make(a.task, cfg=env_cfg)
robot = env.unwrapped.scene.articulations["robot"]
camera = env.unwrapped.scene["video_cam"]


def place_eval_objects(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    placed, out = {}, {}
    for i in OBJS:
        hx, hy = OBJ_HALF_EXTENTS[f"object_{i}"]
        y0, y1 = EVAL_Y_BANDS[i]
        xy = None
        for _ in range(200):
            cx = float(rng.uniform(EVAL_X_MIN, EVAL_X_MAX))
            cy = float(rng.uniform(y0, y1))
            ok = all(
                abs(cx - px) >= hx + OBJ_HALF_EXTENTS[f"object_{j}"][0] + OBJ_BBOX_MARGIN or
                abs(cy - py) >= hy + OBJ_HALF_EXTENTS[f"object_{j}"][1] + OBJ_BBOX_MARGIN
                for j, (px, py) in placed.items())
            if ok:
                xy = (cx, cy)
                break
        if xy is None:
            xy = ((EVAL_X_MIN + EVAL_X_MAX) / 2, (y0 + y1) / 2)
        placed[i] = xy
        obj = env.unwrapped.scene.rigid_objects[f"object_{i}"]
        st = obj.data.default_root_state[0:1].clone()
        st[0, 0], st[0, 1], st[0, 2] = xy[0], xy[1], EVAL_Z
        st[0, 7:] = 0.0
        obj.write_root_state_to_sim(st)
        out[i] = [xy[0], xy[1], EVAL_Z]
    env.unwrapped.scene.write_data_to_sim()
    env.unwrapped.sim.forward()
    return out


def reset_robot_and_policy() -> None:
    jp = robot.data.default_joint_pos.clone()
    robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
    env.unwrapped.scene.write_data_to_sim()
    env.unwrapped.sim.forward()
    for attr, val in (("_ts", 0), ("_last_action_seq", None), ("_startup_step", 0),
                      ("_home_step", 0), ("_home_stable_steps", 0), ("_home_done", False)):
        if hasattr(solution, attr):
            setattr(solution, attr, val)
    hist = getattr(solution, "_action_history", None)
    if hist is not None and hasattr(hist, "clear"):
        hist.clear()


def in_basket(i: int) -> bool:
    pos = env.unwrapped.scene.rigid_objects[f"object_{i}"].data.root_pos_w[0]
    return (abs(pos[0].item() - (TABLE_CENTER_X + 0.08)) <= 0.20
            and abs(pos[1].item() - (TABLE_CENTER_Y - 0.30)) <= 0.11
            and TABLE_TOP_Z <= pos[2].item() <= TABLE_TOP_Z + 0.15)


obs, _ = env.reset()
env.unwrapped.scene.update(dt=env.unwrapped.physics_dt)

candidates = [a.seed] if a.seed else KNOWN_GOOD[:a.max_tries]
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
base, _ = os.path.splitext(os.path.abspath(a.out))

chosen = None
for seed in candidates:
    print(f"\n### 尝试种子 {seed} ...", flush=True)
    layout = place_eval_objects(seed)
    reset_robot_and_policy()
    env.unwrapped.scene.update(dt=env.unwrapped.physics_dt)
    for _ in range(40):
        obs, _, _, _, _ = env.step(torch.zeros((1, 8), dtype=torch.float32, device=a.device))
    env.unwrapped.scene.update(dt=env.unwrapped.physics_dt)

    frames, rows = [], []
    t0 = time.time()
    steps, score, reason = 0, 0.0, "max_steps"
    while steps < 20000:
        with torch.inference_mode():
            resp = solution.predicts(obs, score)
        act = torch.tensor(resp["action"], dtype=torch.float32, device=a.device).view(1, -1)
        obs, reward, terminated, truncated, _ = env.step(act)
        score += float(reward.mean().item() if isinstance(reward, torch.Tensor) else reward)
        steps += 1
        if steps % a.stride == 0:
            frames.append(np.asarray(camera.data.output["rgb"][0][..., :3].cpu().numpy()))
            gi = robot.body_names.index("gripper_base")
            rows.append([steps] + [float(v) for v in robot.data.joint_pos[0, :6]]
                        + [float(v) for v in robot.data.body_pos_w[0, gi]]
                        + [float(env.unwrapped.scene.rigid_objects[f"object_{i}"]
                                 .data.root_pos_w[0][k]) for i in OBJS for k in range(3)])
        if bool(terminated.item()) or bool(truncated.item()):
            reason = "time_out" if bool(truncated.item()) else "terminated"
            break

    flags = [in_basket(i) for i in OBJS]
    ok = all(flags)
    print(f"### 种子 {seed}: 入篮={flags} n_ok={sum(flags)}/3 原因={reason} "
          f"步数={steps} 耗时={time.time()-t0:.0f}s", flush=True)
    if ok:
        chosen = (seed, layout, steps, reason, frames, rows)
        break
    print("### 本局未完整成功，换下一个候选种子", flush=True)

if chosen is None:
    print("### ❌ 所有候选种子都没能完整成功，未产出视频", flush=True)
    env.close(); app.close(); raise SystemExit(1)

seed, layout, steps, reason, frames, rows = chosen
out_mp4 = base + ".mp4"
print(f"\n### 正在写视频 {out_mp4}（{len(frames)} 帧 @ {a.fps}fps ≈ {len(frames)/a.fps:.1f}s）", flush=True)
w = imageio.get_writer(out_mp4, fps=a.fps, codec="libx264", quality=8,
                       macro_block_size=None)
for f in frames:
    w.append_data(f)
w.close()

with open(base + ".csv", "w", newline="", encoding="utf-8") as fh:
    cw = csv.writer(fh)
    cw.writerow(["step"] + [f"q{i+1}" for i in range(6)] + ["ee_x", "ee_y", "ee_z"]
                + [f"o{i}_{ax}" for i in OBJS for ax in "xyz"])
    cw.writerows(rows)

print(f"### ✅ 视频: {out_mp4}  ({os.path.getsize(out_mp4)/1e6:.1f} MB)", flush=True)
print(f"### ✅ 逐帧: {base}.csv", flush=True)
print(f"### 种子={seed}  步数={steps}  终止={reason}  "
      f"耗时={steps*0.02:.1f}s(仿真)  布局={layout}", flush=True)
print(f"### 用时 {time.time()-t0:.0f}s", flush=True)
env.close()
app.close()
