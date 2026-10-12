"""Scripted oracle data collection for Task E (pick-and-place).

Usage
-----
# NOTE: only for object 3 now, if you want to pick objects 1 and 2, please modify the distance accordingly.
python scripts/act/collect_demos_task_e.py --pick_objects 3 --num_demos 50 --headless

"""

import argparse
import hashlib
import os
import shutil
import sys
import time

# sys.path.insert(0, os.path.dirname(__file__))

from isaaclab.app import AppLauncher
from cli_args import add_collect_demo_args

parser = argparse.ArgumentParser(description="Collect Task E demonstrations for ACT.")
add_collect_demo_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.save_video or args_cli.save_images:
    args_cli.enable_cameras = True

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

import h5py
import json
import numpy as np

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import CameraCfg
import isaaclab.sim as sim_utils

from atec_rl_lab.tasks.task_e.env_cfg import TaskEEnvPiperCfg
from atec_rl_lab.utils import CartesianController

from task_e.config import (
    EE_BODY_NAME, ARM_JOINT_NAMES, GRIPPER_JOINT_NAMES,
    ACT_STIFFNESS, ACT_DAMPING, ACT_EFFORT_LIMIT, ACT_VEL_LIMIT,
    CAM_H, CAM_W, CAM_POS, CAM_ROT,
)
from task_e.collector import (
    check_positions_in_basket, collect_one_demo, fired_terms,
)


# ------------------------------------------------------------------ #
# 采集记录（README §4：每次记录代码版本、场景配置/种子、完成物体数、
# 成功终止与否、耗时、失败阶段）
# ------------------------------------------------------------------ #

def code_fingerprint() -> dict[str, str]:
    """采集所用的代码指纹（src 侧 5 个文件的 sha256 前 16 位）。

    本项目不是 git 仓库，用指纹代替"代码版本"，保证每批数据都能对上
    产生它的那一版专家代码。
    """
    base = os.path.dirname(os.path.abspath(__file__))
    out: dict[str, str] = {}
    for rel in ("task_e/config.py", "task_e/state_machine.py", "task_e/collector.py",
                "collect_demos_task_e.py", "cli_args.py"):
        p = os.path.join(base, rel)
        try:
            with open(p, "rb") as fh:
                out[rel] = hashlib.sha256(fh.read()).hexdigest()[:16]
        except OSError:
            out[rel] = "missing"
    return out


def _read_meta(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            pass
    return {}


def _write_meta(path: str, blob: dict) -> None:
    """原子写：先写 .tmp 再 replace，中途被杀不会留下半个 JSON。"""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def free_gb(path: str) -> float:
    """输出目录所在文件系统的剩余空间（GB）。

    为什么需要：HDF5 边写边追加，磁盘写满会让文件处于不一致状态、整批数据报废。
    采集前检查余量，宁可少采几条也不要写坏文件。
    """
    try:
        return shutil.disk_usage(path).free / (1024 ** 3)
    except OSError:
        return float("inf")


def append_meta(path: str, record: dict) -> None:
    blob = _read_meta(path)
    blob.setdefault("records", []).append(record)
    _write_meta(path, blob)


def build_env(pick_objects: list[int], need_camera: bool) -> ManagerBasedRLEnv:
    import time
    cfg = TaskEEnvPiperCfg()
    cfg.seed              = int(time.time_ns() % (2**31))   # random seed each call
    cfg.scene.num_envs    = 1
    cfg.episode_length_s  = 40.0 * len(pick_objects) + 10.0
    cfg.scene.robot.actuators["default"] = ImplicitActuatorCfg(
        joint_names_expr=[".*"],
        effort_limit=ACT_EFFORT_LIMIT,
        velocity_limit=ACT_VEL_LIMIT,
        stiffness=ACT_STIFFNESS,
        damping=ACT_DAMPING,
    )
    # ---- 相机裁剪（必须，否则仿真初始化卡死）----
    #
    # 实测问题：默认配置带 4 个相机（head/ee/ee_dual/video）。只要 3 个以上
    # 同时启用，仿真就会卡在 "Starting the simulation..." 之后不再前进
    # （实测 4.5 分钟无输出、CPU 266%、显存 16.7 GB）。
    # 对照（logs/probe_cam_combo.py 逐组合实测）：
    #     全开            -> 卡死
    #     只留 video_cam  -> 4.2 s 建好环境、5.4 s 跑完 30 步 ✓
    # 本采集脚本只读 video_cam（见下方 camera = ...["video_cam"]），
    # 另外三个相机及其观测项（ee_rgb/ee_depth）用不到，故一律关闭。
    # 注意：禁用相机后必须同步关掉引用它的观测项，否则建环境会报
    #      "The scene entity 'ee_camera' does not exist"。
    for _cam in ("head_camera", "ee_camera", "ee_dual_camera"):
        if hasattr(cfg.scene, _cam):
            setattr(cfg.scene, _cam, None)
    _img = getattr(cfg.observations, "image", None)
    if _img is not None:
        for _t in ("ee_rgb", "ee_depth"):
            if hasattr(_img, _t):
                setattr(_img, _t, None)

    if not need_camera:
        # 本次运行不需要图像（例如"只统计成功率"的测量跑）：video_cam 也一并关掉。
        # 否则 Isaac Sim 会直接报
        #   "A camera was spawned without the --enable_cameras flag"
        # （即使补上 --enable_cameras，也会白白付出每步渲染开销）。
        # 需要 --save_images/--save_video 的正式采集不受影响。
        if hasattr(cfg.scene, "video_cam"):
            setattr(cfg.scene, "video_cam", None)
        if getattr(cfg.observations, "image", None) is not None:
            cfg.observations.image = None
        print("[INFO] 未请求保存图像 → 已关闭全部相机与 image 观测（测量跑加速）",
              flush=True)

    # if need_camera:
    #     cfg.scene.video_cam = CameraCfg(
    #         prim_path="{ENV_REGEX_NS}/video_cam",
    #         update_period=0.0,
    #         height=CAM_H, width=CAM_W,
    #         data_types=["rgb"],
    #         spawn=sim_utils.PinholeCameraCfg(
    #             focal_length=24.0, focus_distance=400.0,
    #             horizontal_aperture=20.955, clipping_range=(0.1, 100.0),
    #         ),
    #         offset=CameraCfg.OffsetCfg(pos=CAM_POS, rot=CAM_ROT, convention="world"),
    #     )
    return ManagerBasedRLEnv(cfg)


def init_output(output_dir: str) -> tuple[str, str]:
    """Create output directory, wipe any existing trajectory.hdf5, write JSON metadata."""
    os.makedirs(output_dir, exist_ok=True)
    traj_path = os.path.join(output_dir, "trajectory.hdf5")
    json_path = os.path.join(output_dir, "trajectory.json")
    with h5py.File(traj_path, "w"):   # truncate / create fresh
        pass
    with open(json_path, "w") as fh:
        json.dump({"env_info": {"env_kwargs": {"control_mode": "pd_joint_pos"}}}, fh)
    return traj_path, json_path


def save_traj(traj_path: str, traj_idx: int, data: dict,
              save_images: bool) -> None:
    """Append one trajectory group to the consolidated HDF5."""
    with h5py.File(traj_path, "a") as f:
        grp = f.create_group(f"traj_{traj_idx}")
        grp.create_dataset("obs",     data=data["qpos"],    compression="gzip")
        grp.create_dataset("actions", data=data["action"],  compression="gzip")
        grp.create_dataset("qvel",    data=data["qvel"],    compression="gzip")
        grp.create_dataset("ee_pos",  data=data["ee_pos"],  compression="gzip")
        grp.create_dataset("ee_quat", data=data["ee_quat"], compression="gzip")
        if save_images and "frames" in data:
            grp.create_group("images").create_dataset(
                "rgb", data=data["frames"], compression="gzip"
            )


def main() -> None:
    pick_objects = sorted(set(args_cli.pick_objects))
    need_camera = args_cli.save_video or args_cli.save_images

    env    = build_env(pick_objects, need_camera)
    dev    = env.unwrapped.device
    camera = env.unwrapped.scene["video_cam"] if need_camera else None

    robot = env.unwrapped.scene.articulations["robot"]
    arm_ids,     _ = robot.find_joints(ARM_JOINT_NAMES)
    gripper_ids, _ = robot.find_joints(GRIPPER_JOINT_NAMES)
    ik_ctrl = CartesianController(
        robot=robot, ee_body_name=EE_BODY_NAME,
        arm_joint_names=ARM_JOINT_NAMES,
        num_envs=1, device=dev,
        command_type="pose",
        lambda_val=0.05,
        max_joint_delta=0.2,
    )
    default_jpos = robot.data.default_joint_pos.clone()

    video_dir = None
    imageio   = None
    if args_cli.save_video:
        video_dir = args_cli.video_dir or os.path.join(args_cli.output_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)
        import imageio as _io
        imageio = _io

    traj_path, _ = init_output(args_cli.output_dir)
    rng = np.random.default_rng()   # unseeded → different positions every run

    n_ok = 0
    attempt = 0
    max_att = int(getattr(args_cli, "max_attempts", 0) or 0)

    meta_path = os.path.join(args_cli.output_dir, "collection_meta.json")
    _write_meta(meta_path, {
        "code_fingerprint": code_fingerprint(),
        "pick_objects": pick_objects,
        "output_dir": os.path.abspath(args_cli.output_dir),
        "num_demos_target": args_cli.num_demos,
        "max_attempts": max_att,
        "only_success": bool(args_cli.only_success),
        "image_saved": bool(args_cli.save_images),
        "image_size": int(getattr(args_cli, "image_size", 0) or 0),
        "min_free_gb": float(getattr(args_cli, "min_free_gb", 0.0) or 0.0),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "success_criterion": ("环境自身的 basket_success 终止项（reset 前 get_term 取值）；"
                              "脚本侧 check_positions_in_basket 仅作交叉校验"),
        "records": [],
    })

    min_free_gb = float(getattr(args_cli, "min_free_gb", 0.0) or 0.0)
    print(f"[INFO] 输出目录: {os.path.abspath(args_cli.output_dir)}  "
          f"剩余空间 {free_gb(args_cli.output_dir):.2f} GB "
          f"(低于 {min_free_gb} GB 会停止)", flush=True)

    while n_ok < args_cli.num_demos:
        _free = free_gb(args_cli.output_dir)
        if min_free_gb and _free < min_free_gb:
            print(f"[WARN] 输出目录所在磁盘仅剩 {_free:.2f} GB（阈值 {min_free_gb} GB）"
                  f"—— 停止采集以免写坏 HDF5。已收集 {n_ok}/{args_cli.num_demos} 条；"
                  f"清理空间或换 output_dir 后可续采。")
            append_meta(meta_path, {"attempt": attempt + 1, "saved": False,
                                    "termination": "aborted_low_disk",
                                    "free_gb": round(_free, 2)})
            break
        if max_att and attempt >= max_att:
            print(f"[WARN] 达到尝试上限 {max_att} 次，停止（已收集 {n_ok}/"
                  f"{args_cli.num_demos} 条）。可换 output_dir 或加大 --max_attempts 继续。")
            break
        attempt += 1
        print(f"\n[INFO] Demo {n_ok + 1}/{args_cli.num_demos}  (attempt {attempt})")
        t_start = time.time()

        data = collect_one_demo(
            env, robot, ik_ctrl,
            arm_ids, gripper_ids,
            pick_objects, dev,
            default_jpos=default_jpos,
            rng=rng,
            camera=camera,
            image_size=(args_cli.image_size or None),
            diversity=({
                "steps_jitter": tuple(args_cli.steps_jitter) if args_cli.steps_jitter else None,
                "perturb_sigma": args_cli.perturb_sigma,
                "perturb_states": tuple(args_cli.perturb_states) if args_cli.perturb_states else None,
            } if getattr(args_cli, "diversity", False) else None),
        )
        elapsed = time.time() - t_start
        rec = {
            "attempt": attempt,
            "traj_index": n_ok,
            "elapsed_s": round(elapsed, 1),
            "saved": False,
            "termination": None,
            "steps": None,
            "pick_objects": pick_objects,
            "obj_spawn_after_settle": None,
            "obj_pos_at_termination": None,
            "obj_pos_pre_step": None,
            "obj_pos_final": None,
            "final_check_ok": None,
            "cross_check_ok": None,
            "cross_check": None,
        }

        if data is None:
            print(f"[WARN] Early termination — skipping. ({elapsed:.0f}s)")
            append_meta(meta_path, rec)
            continue

        # ---- 成功与否：以**环境自己的** basket_success 为准（README §3.1 第 1 条）----
        dmeta = data.get("meta", {})
        rec["termination"] = dmeta.get("termination")
        rec["steps"] = dmeta.get("steps")
        rec["obj_spawn_after_settle"] = dmeta.get("obj_spawn_after_settle")
        rec["obj_pos_at_termination"] = dmeta.get("obj_pos_at_termination")
        rec["obj_pos_pre_step"] = dmeta.get("obj_pos_pre_step")
        rec["obj_lift_max"] = dmeta.get("obj_lift_max")
        # 示范多样性取证（只有 --diversity 时才有）
        if dmeta.get("diversity"):
            rec["diversity"] = dmeta["diversity"]
            _dv = dmeta["diversity"]
            print(f"[INFO] 多样性: 扰动 {_dv['n_perturb']} 次 "
                  f"(sigma={_dv['perturb_sigma']}, states={_dv['perturb_states']})；"
                  f"本条的步数表 obj周期≈"
                  f"{sum(_dv['steps_profile'][k] for k in ('PRE_GRASP','REACH','CLOSE','LIFT','RETREAT','TRANSPORT','PLACE','OPEN','LIFT_RETRACT','RETRACT'))} 步")
        if rec["obj_lift_max"]:
            print("[INFO] 各物体最大抬升（>0.10 记为夹起）: "
                  + " ".join(f"obj{i}={v:+.3f}" for i, v in sorted(rec["obj_lift_max"].items())))

        if rec["termination"] != "success":
            print(f"[WARN] 环境未判定 basket_success（termination="
                  f"{rec['termination']}）—— 丢弃本条。({elapsed:.0f}s)")
            # 诊断：状态机跑完但环境没判成功时，物体最后停在哪？
            fin = dmeta.get("obj_pos_final") or {}
            if fin:
                rec["obj_pos_final"] = fin
                ok_fin, rep_fin = check_positions_in_basket(fin, pick_objects)
                rec["final_check_ok"] = bool(ok_fin)
                print(f"[INFO] 本条结束时物体位置（官方口径判定）: {rep_fin}")
                print(f"[INFO]   物体最终坐标: "
                      + " ".join(f"obj{i}=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})"
                                 for i, p in sorted(fin.items())))
            append_meta(meta_path, rec)
            continue

        # ---- 交叉校验：用 reset 前取证的位置按官方口径重算一遍（只做诊断）----
        ok, report = check_positions_in_basket(
            dmeta.get("obj_pos_at_termination") or {}, pick_objects)
        rec["cross_check_ok"] = bool(ok)
        rec["cross_check"] = report
        print(f"[INFO] 交叉校验（reset 前取证=判定时刻，官方口径）: {report}")
        if not ok and args_cli.only_success:
            print("[WARN] 环境判成功、脚本口径判失败 —— 两者不一致会掩盖真实问题，"
                  "请排查后放量（本条仍按环境判定保留）。")

        save_traj(traj_path, n_ok, data, args_cli.save_images)

        T     = len(data["qpos"])
        notes = [f"{T} steps", f"{elapsed:.0f}s"]
        if args_cli.save_video and "frames" in data:
            vp = os.path.join(video_dir, f"demo_{n_ok:04d}.mp4")
            imageio.mimwrite(vp, data["frames"], fps=50, quality=7)
            notes.append(f"video → {vp}")
        if args_cli.save_images and "frames" in data:
            notes.append("images saved")
        print(f"[INFO] traj_{n_ok}: {', '.join(notes)}"
              f"  终止={rec['termination']}  布局={rec['obj_spawn_after_settle']}")
        rec["saved"] = True
        append_meta(meta_path, rec)
        n_ok += 1

    print(f"\n[INFO] Collected {n_ok} demos → {traj_path}"
          f"（尝试 {attempt} 次，成功率 {n_ok}/{attempt}）")
    print(f"[INFO] 结束时剩余空间 {free_gb(args_cli.output_dir):.2f} GB")
    print(f"[INFO] 采集记录 → {meta_path}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
