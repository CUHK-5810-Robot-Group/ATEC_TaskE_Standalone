"""步骤 6：闭环评测器（每个 episode 换布局、可留出种子、结果可对比）。

为什么必须重写一个评测器（而不是直接用 logs/play_act_baseline.py）
----------------------------------------------------------------
实测（logs/probe_episode_reset.py）确认：本工程的 `env.reset()` **什么都不重置** ——
`RigidObject.reset()` 只重置 wrench、不写回 root state；Task E 也没有物体的
reset 事件；机器人关节同样停在上一局结束时的位形。因此
`play_act_baseline.py` 里"跑 N 局"实际上是**一段连续仿真**，8 局的有效样本 ≈ 1。
本评测器每局显式做三件事：
  1. 按**评测分布**重新摆放三件物体（常量取自 env_cfg，z=TABLE_TOP_Z+0.05）；
  2. 把机器人关节复位到 default_joint_pos；
  3. 重置策略内部状态（_ts / _action_history / _startup_step / _home_done …），
     否则第 2 局开头会吃到第 1 局的时序聚合残留。
**不改 demo/solution_act.py**（保持与交接包逐字节一致），重置在评测器侧完成。

留出种子
--------
`--seed_base` 之后的 `-n` 个种子即评测布局；采集用的随机布局与之独立。
同一组种子跑 baseline 与新权重 → 严格可比。建议 dev/test 分开（例如
dev 用 1000+，test 用 2000+），只在 dev 上挑 checkpoint，最终在 test 上报数。

用法
----
    # 基线（包内自带权重）
    python logs/play_act_eval.py -n 10 --seed_base 2000 --tag baseline

    # 新权重
    ATEC_TASK_E_POLICY=$PWD/scripts/act/runs/<run>/checkpoints/best_loss.pt \
      python logs/play_act_eval.py -n 10 --seed_base 2000 --tag act-new

输出：逐局明细 + 汇总（整体/逐物体成功率、终止原因、耗时、失败分类），
      并写 JSON 到 --out（默认 logs/eval_runs/<tag>_<时间>.json）。
"""
import argparse
import hashlib
import json
import os
import sys
import time

ROOT = "/root/gpufree-share/Robotic2/test2/ATEC_TaskE_Standalone"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "act"))

from isaaclab.app import AppLauncher  # noqa: E402

p = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(p)
p.add_argument("-n", "--episodes", type=int, default=10)
p.add_argument("--seed_base", type=int, default=2000, help="评测布局种子起点（留出种子）")
p.add_argument("--seeds", type=int, nargs="+", default=None,
               help="显式指定评测种子列表（覆盖 --seed_base/-n）。用于随机种子测试："
                    "种子在采集/选 checkpoint 时都没用过，才能真正当留出集报数。"
                    "脚本会把整份列表写进 JSON，保证可复现。")
p.add_argument("--task", type=str, default="ATEC-TaskE-Piper")
p.add_argument("--episode_len_s", type=float, default=240.0)
p.add_argument("--max_steps", type=int, default=20000)
p.add_argument("--tag", type=str, default="eval")
p.add_argument("--out", type=str, default=None)
p.add_argument("--progress", type=int, default=2000)
p.add_argument("--trace_every", type=int, default=200,
               help="每 N 步打印一行行为轨迹（关节角/末端/三物体位置/策略阶段），0=关")
a = p.parse_args()
a.enable_cameras = True          # 策略需要 obs["image"]["video_rgb"]
app = AppLauncher(a).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402

import atec_rl_lab.tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from atec_rl_lab.tasks.task_base.action_base import apply_safe_action_spec  # noqa: E402
from atec_rl_lab.tasks.task_e.env_cfg import (  # noqa: E402  评测分布的权威定义
    TABLE_CENTER_X, TABLE_CENTER_Y, TABLE_TOP_Z, OBJ_HALF_EXTENTS, OBJ_BBOX_MARGIN)
from demo.solution import AlgSolution  # noqa: E402

OBJS = (1, 2, 3)
NAMES = {1: "糖盒", 2: "芥末瓶", 3: "香蕉"}
# env_cfg.py:195-201 的同一套数（改那里必须同步这里；verify_dataset.py 有漂移自检）
EVAL_X_MIN, EVAL_X_MAX = TABLE_CENTER_X - 0.10, TABLE_CENTER_X + 0.10
EVAL_Y_BANDS = {1: (TABLE_CENTER_Y + 0.25, TABLE_CENTER_Y + 0.29),
                2: (TABLE_CENTER_Y + 0.14, TABLE_CENTER_Y + 0.20),
                3: (TABLE_CENTER_Y + 0.03, TABLE_CENTER_Y + 0.09)}
EVAL_Z = TABLE_TOP_Z + 0.05

print(f"[INFO] 创建策略对象（ATEC_TASK_E_POLICY={os.environ.get('ATEC_TASK_E_POLICY') or '默认包内权重'}）",
      flush=True)
solution = AlgSolution()

ckpt = os.environ.get("ATEC_TASK_E_POLICY") or os.path.join(
    ROOT, "atec_robot_model", "baseline", "act", "policy.pt")
weight_hash = "missing"
if os.path.isfile(ckpt):
    with open(ckpt, "rb") as fh:
        weight_hash = hashlib.sha256(fh.read()).hexdigest()[:16]

env_cfg = parse_env_cfg(a.task, device=a.device, num_envs=1, use_fabric=True)
env_cfg.episode_length_s = a.episode_len_s
env_cfg = apply_safe_action_spec(env_cfg, solution.get_action_spec()
                                 if hasattr(solution, "get_action_spec") else None)

# 相机裁剪：策略只读 video_cam；实测 3 个以上相机同时启用会让仿真卡在初始化
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


def place_eval_objects(seed: int) -> dict[int, list[float]]:
    """按评测环境自己的规则重新摆放三件物体（与 env_cfg.__post_init__ 同规则）。

    规则：x ~ U[中心±0.10]、y ~ 各自 Y 带、z = 桌面+0.05；AABB 互不重叠
    （任一轴分开即视为不重叠，与 env_cfg / 采集脚本逐字一致）。
    """
    rng = np.random.default_rng(seed)
    placed: dict[int, tuple[float, float]] = {}
    out: dict[int, list[float]] = {}
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
    """复位机器人关节 + 策略内部状态（env.reset() 不会做这两件事）。"""
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

SEEDS = list(a.seeds) if a.seeds else [a.seed_base + ep for ep in range(a.episodes)]
a.episodes = len(SEEDS)

print(f"\n### 闭环评测: tag={a.tag} 权重={os.path.basename(ckpt)} "
      f"hash={weight_hash} 局数={a.episodes} "
      f"种子={'显式列表' if a.seeds else f'{a.seed_base}..{a.seed_base + a.episodes - 1}'} "
      f"单局上限={a.episode_len_s:.0f}s", flush=True)
print(f"### 种子列表: {SEEDS}", flush=True)

results = []
for ep, seed in enumerate(SEEDS):
    layout = place_eval_objects(seed)
    reset_robot_and_policy()
    env.unwrapped.scene.update(dt=env.unwrapped.physics_dt)
    # 让物体从 5cm 落下并稳定（策略前 25 步发零动作，这里额外给 40 步）
    # 注意：必须接住 env.step 返回的 obs，否则本局第一次 predicts 会用到
    # 摆放之前的旧观测（旧图像 + 旧状态）。
    for _ in range(40):
        obs, _, _, _, _ = env.step(torch.zeros((1, 8), dtype=torch.float32, device=a.device))
    env.unwrapped.scene.update(dt=env.unwrapped.physics_dt)

    t0 = time.time()
    steps, score = 0, 0.0
    reason = "max_steps"
    while steps < a.max_steps:
        with torch.inference_mode():
            resp = solution.predicts(obs, score)
        if resp.get("giveup"):
            reason = "giveup"
            break
        act = torch.tensor(resp["action"], dtype=torch.float32, device=a.device).view(1, -1)
        obs, reward, terminated, truncated, _ = env.step(act)
        score += float(reward.mean().item() if isinstance(reward, torch.Tensor) else reward)
        steps += 1
        if bool(terminated.item()) or bool(truncated.item()):
            reason = "time_out" if bool(truncated.item()) else "success_or_terminated"
            if all(in_basket(i) for i in OBJS):
                reason = "basket_success"
            break
        if a.progress and steps % a.progress == 0:
            print(f"###   第{ep + 1}局 {steps} 步 ({steps * 0.02:.0f}s)", flush=True)
        if a.trace_every and steps % a.trace_every == 0:
            q = [round(float(v), 2) for v in robot.data.joint_pos[0, :6]]
            gi = robot.body_names.index("gripper_base")
            ee = [round(float(v), 3) for v in robot.data.body_pos_w[0, gi]]
            ob = {i: [round(float(v), 3) for v in
                      env.unwrapped.scene.rigid_objects[f"object_{i}"].data.root_pos_w[0]]
                  for i in OBJS}
            phase = "home" if not getattr(solution, "_home_done", True) else "policy"
            print(f"###   [{steps:5d}] q={q} ee={ee} o1={ob[1]} o2={ob[2]} o3={ob[3]} 阶段={phase}",
                  flush=True)

    flags = [in_basket(i) for i in OBJS]
    if reason == "success_or_terminated" and not all(flags):
        reason = "terminated_other"
    rec = {"ep": ep, "seed": seed, "layout": layout, "in_basket": flags,
           "n_ok": int(sum(flags)), "reason": reason, "steps": steps,
           "elapsed_s": round(time.time() - t0, 1)}
    results.append(rec)
    print(f"### 第 {ep + 1}/{a.episodes} 局 seed={seed}: 完成物体 {rec['n_ok']}/3 "
          f"入篮={flags} 原因={reason} 步数={steps} 耗时={rec['elapsed_s']}s", flush=True)

def wilson_ci(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """二项比例的 Wilson 95% 置信区间（小样本比正态近似可靠）。

    报"成功率 = 6/12"时必须同时给区间 —— 12 局的 50% 与 50 局的 50%
    可信度完全不同。
    """
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z / d * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (max(0.0, c - h), min(1.0, c + h))


n = len(results)
full = sum(1 for r in results if r["n_ok"] == 3)
lo, hi = wilson_ci(full, n)
per_obj = {i: sum(1 for r in results if r["in_basket"][i - 1]) for i in OBJS}
from collections import Counter
reasons = dict(Counter(r["reason"] for r in results))
print("\n### ================= 汇总 ================= ###", flush=True)
print(f"### tag={a.tag} 权重hash={weight_hash}", flush=True)
print(f"### 三物体全部入篮 = {full}/{n} ({100.0 * full / max(n, 1):.1f}%)"
      f"   95% CI = [{100*lo:.1f}%, {100*hi:.1f}%]   (Wilson)", flush=True)
for i in OBJS:
    print(f"###   object_{i}({NAMES[i]}): 入篮 {per_obj[i]}/{n}", flush=True)
print(f"### 终止原因分布: {reasons}", flush=True)
print(f"### 平均耗时 {np.mean([r['elapsed_s'] for r in results]):.0f}s/局", flush=True)

out = a.out or os.path.join(ROOT, "..", "logs", "eval_runs",
                            f"{a.tag}_{time.strftime('%Y%m%d_%H%M%S')}.json")
out = os.path.abspath(out)
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8") as fh:
    json.dump({"tag": a.tag, "weight": ckpt, "weight_hash": weight_hash,
               "episodes": a.episodes, "seed_base": a.seed_base,
               "seeds": SEEDS,
               "episode_len_s": a.episode_len_s, "summary": {
                   "full_success": full, "n": n,
                   "success_rate": full / n if n else 0.0,
                   "wilson95_ci": [lo, hi],
                   "per_object": per_obj,
                   "reasons": reasons},
               "results": results}, fh, ensure_ascii=False, indent=2)
print(f"### 明细已写 → {out}", flush=True)
env.close()
app.close()
