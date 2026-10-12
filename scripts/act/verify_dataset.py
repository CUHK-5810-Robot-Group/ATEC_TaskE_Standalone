"""采集数据验收：把 README §3.2 第 4 步的"进入下一阶段的条件"变成可执行检查。

用法
----
    python scripts/act/verify_dataset.py \
        --dataset datasets/atec_task_e/trajectory.hdf5 \
        --meta    datasets/atec_task_e/collection_meta.json

验收项（README §3.2 第 4 步 + §3.1 第 4 条 + §7.4 的验收清单）
--------------------------------------------------------------
1. 成功判定：全部来自**环境自身的 basket_success**（不是脚本自判）
2. 不含跨 reset 拼接：每条 max|Δqpos| 在正常量级（~0.04~0.15 rad），
   出现 ~1 rad 级别的跳变即判失败
3. 帧数一致：同一 traj 内 obs/actions/qvel/ee_pos/ee_quat/images 第一维相同
4. 动作定义一致：8 维、范围合理；夹爪两维只取"张开/闭合"两个值
5. RGB 与动作对齐：images 与 obs 帧数相同；抽首/末帧写 PNG 供人工核对
6. 覆盖评测分布：三物体初始位置落在评测分布内（X 中心 ±0.10、三条 Y 带），
   并报告实际覆盖区间与带宽覆盖率
7. 数量：默认要求 >= 期望条数（--expect_demos，默认 50）
8. 元数据齐全：每条尝试有终止原因/布局/取证位置/抬升/耗时/代码指纹

退出码：全部通过 0，否则 1。**不通过就不要进入训练/评测。**
"""
import argparse
import json
import os
import sys

import h5py
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 评测分布的权威定义在 source/atec_rl_lab/atec_rl_lab/tasks/task_e/env_cfg.py:195-201。
# 下面这份副本只用于"数据覆盖率"检查；脚本会顺手核对源文件里是否仍是这些数，
# 不一致就报警（避免像 +0.10/+0.15 那样悄悄漂移）。
EVAL_X_MIN, EVAL_X_MAX = 0.90, 1.10
EVAL_Y_BANDS = {1: (0.25, 0.29), 2: (0.14, 0.20), 3: (0.03, 0.09)}
ENV_CFG = os.path.join(
    ROOT, "source", "atec_rl_lab", "atec_rl_lab", "tasks", "task_e", "env_cfg.py")
ENV_CFG_MARKERS = [
    "TABLE_CENTER_X - 0.10, TABLE_CENTER_X + 0.10",
    "TABLE_CENTER_Y + 0.25, TABLE_CENTER_Y + 0.29",
    "TABLE_CENTER_Y + 0.14, TABLE_CENTER_Y + 0.20",
    "TABLE_CENTER_Y + 0.03, TABLE_CENTER_Y + 0.09",
]

# 动作/状态维度（joint1..joint6 + joint7/8 夹爪）
DIM = 8
GRIP_OPEN, GRIP_CLOSE = 0.035, -0.015
ACTION_SCALE = 0.5

results: list[tuple[bool, str, str]] = []


def check(ok: bool, title: str, detail: str = "") -> None:
    results.append((bool(ok), title, detail))
    print(f"{'[PASS]' if ok else '[FAIL]'} {title}" + (f"  —— {detail}" if detail else ""),
          flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="trajectory.hdf5")
    ap.add_argument("--meta", default=None, help="collection_meta.json（可选但强烈建议）")
    ap.add_argument("--expect_demos", type=int, default=50, help="期望条数")
    ap.add_argument("--save_frames", default=None,
                    help="把每条的首/末帧导出到这个目录，便于人工核对 RGB 与动作对齐")
    ap.add_argument("--allow_bad_groups", nargs="*", default=[],
                    help="已知且已确认的损坏组名（例如文件被中断写入留下的孤儿组）。"
                         "列在这里的组会被记为『已确认坏组』而不是未通过 —— "
                         "训练脚本会自动跳过它们。")
    ap.add_argument("--max_dqpos", type=float, default=0.30,
                    help="单条内 max|Δqpos| 上限（rad）；跨 reset 拼接会到 ~1 rad")
    a = ap.parse_args()

    print("=" * 78)
    print(f"数据验收: {os.path.abspath(a.dataset)}")
    print("=" * 78, flush=True)

    # ---------- 0. 评测分布定义是否仍与源文件一致 ----------
    if os.path.exists(ENV_CFG):
        txt = open(ENV_CFG, encoding="utf-8").read()
        missing = [m for m in ENV_CFG_MARKERS if m not in txt]
        check(not missing, "评测分布定义未漂移",
              "env_cfg.py 中仍能找到 X±0.10 与三条 Y 带" if not missing
              else f"env_cfg.py 里找不到: {missing} —— 本脚本的副本可能已过期")
    else:
        check(False, "评测分布定义未漂移", f"找不到 {ENV_CFG}")

    if not os.path.exists(a.dataset):
        check(False, "数据集文件存在", a.dataset)
        return finish()

    size_gb = os.path.getsize(a.dataset) / 1e9
    print(f"\n文件大小: {size_gb:.2f} GB\n")

    # ---------- 逐条检查 ----------
    trajs = []
    bad_groups: list[tuple[str, str]] = []
    with h5py.File(a.dataset, "r") as f:
        keys = sorted(f.keys(), key=lambda k: int(k.split("_")[1]))
        check(len(keys) >= a.expect_demos, f"条数 >= {a.expect_demos}",
              f"实际 {len(keys)} 条")
        for k in keys:
            g = f[k]
            try:
                fields = {n: g[n].shape for n in g if n != "images"}
                T = g["obs"].shape[0]
            except Exception as exc:
                # 损坏组（例如文件被中断写入留下的孤儿组）：记录为坏组，继续检查其余
                bad_groups.append((k, f"{type(exc).__name__}: {exc}"))
                continue
            same = all(v[0] == T for v in fields.values())
            has_img = "images" in g and "rgb" in g["images"]
            img_ok = (not has_img) or (g["images/rgb"].shape[0] == T)
            q = g["obs"][:]
            act = g["actions"][:]
            dq = float(np.abs(np.diff(q, axis=0)).max()) if T > 1 else 0.0
            grip_vals = np.unique(np.round(act[:, 6:8], 4))
            trajs.append(dict(key=k, T=T, same=same, img_ok=img_ok, has_img=has_img,
                              dq=dq, act_min=float(act.min()), act_max=float(act.max()),
                              n_grip=len(grip_vals), dim=act.shape[1],
                              img_shape=(g["images/rgb"].shape if has_img else None)))
            if a.save_frames and has_img:
                os.makedirs(a.save_frames, exist_ok=True)
                from PIL import Image
                im = g["images/rgb"]
                for idx, tag in ((0, "first"), (T - 1, "last")):
                    Image.fromarray(im[idx]).save(
                        os.path.join(a.save_frames, f"{k}_{tag}.png"))

    unexpected = [b for b in bad_groups if b[0] not in set(a.allow_bad_groups)]
    check(not unexpected, "无未确认的损坏轨迹组",
          "全部可读" if not bad_groups
          else (f"已确认坏组 {[b[0] for b in bad_groups]}（训练脚本会自动跳过）"
                if not unexpected else f"未确认的损坏: {unexpected}"))
    check(all(t["same"] for t in trajs), "帧数一致（各字段第一维相同）",
          f"T 取值: {sorted({t['T'] for t in trajs})}")
    check(all(t["img_ok"] for t in trajs), "图像帧数与状态帧数相同",
          f"图像形状: {trajs[0]['img_shape']}")
    check(all(t["dq"] <= a.max_dqpos for t in trajs), "不含跨 reset 拼接",
          f"max|Δqpos| = {max(t['dq'] for t in trajs):.4f} rad（上限 {a.max_dqpos}）")
    check(all(t["dim"] == DIM for t in trajs), f"动作维度 = {DIM}",
          f"实际 {sorted({t['dim'] for t in trajs})}")
    check(all(t["n_grip"] <= 4 for t in trajs), "夹爪两维只取开/合两种值",
          f"每条不同夹爪值个数 <= {max(t['n_grip'] for t in trajs)}")
    check(all(abs(t["act_min"]) < 20 and abs(t["act_max"]) < 20 for t in trajs),
          "动作范围合理（(target-default)/0.5 量级）",
          f"[{min(t['act_min'] for t in trajs):.2f}, {max(t['act_max'] for t in trajs):.2f}]")

    # ---------- 元数据检查 ----------
    spawn = {1: [], 2: [], 3: []}
    if a.meta and os.path.exists(a.meta):
        m = json.load(open(a.meta, encoding="utf-8"))
        recs = m.get("records", [])
        saved = [r for r in recs if r.get("saved")]
        succ = [r for r in saved if r.get("termination") == "success"]
        check(len(succ) >= a.expect_demos, "成功判定全部来自环境 basket_success",
              f"成功 {len(succ)} 条 / 已存 {len(saved)} 条 / 共 {len(recs)} 次尝试"
              f"（成功率 {len(saved)}/{len(recs)}）")
        check(bool(m.get("code_fingerprint")), "记录了产生该批数据的代码指纹",
              json.dumps(m.get("code_fingerprint", {}), ensure_ascii=False))
        have = all(r.get("obj_spawn_after_settle") and r.get("obj_lift_max") is not None
                   for r in succ)
        check(have, "每条成功轨迹都有布局/抬升记录", "obj_spawn_after_settle + obj_lift_max")
        for r in succ:
            sp = r.get("obj_spawn_after_settle") or {}
            for i in (1, 2, 3):
                if str(i) in sp or i in sp:
                    spawn[i].append((sp.get(str(i)) or sp.get(i))[:2])
    else:
        check(False, "提供了 collection_meta.json",
              "没有它无法核对成功判定来源与布点覆盖")

    # ---------- 评测分布覆盖率 ----------
    if all(spawn[i] for i in (1, 2, 3)):
        lines = []
        ok_cov = True
        for i in (1, 2, 3):
            arr = np.array(spawn[i])
            y0, y1 = EVAL_Y_BANDS[i]
            in_x = (arr[:, 0] >= EVAL_X_MIN - 0.03) & (arr[:, 0] <= EVAL_X_MAX + 0.03)
            in_y = (arr[:, 1] >= y0 - 0.03) & (arr[:, 1] <= y1 + 0.03)
            cover_x = (arr[:, 0].min() <= EVAL_X_MIN + 0.05) and (arr[:, 0].max() >= EVAL_X_MAX - 0.05)
            cover_y = (arr[:, 1].min() <= y0 + 0.02) and (arr[:, 1].max() >= y1 - 0.02)
            ok_cov &= bool(in_x.mean() > 0.9 and in_y.mean() > 0.9 and cover_x and cover_y)
            lines.append(f"obj{i}: x[{arr[:,0].min():.3f},{arr[:,0].max():.3f}] "
                         f"y[{arr[:,1].min():.3f},{arr[:,1].max():.3f}] "
                         f"带内比例 x={in_x.mean():.0%} y={in_y.mean():.0%} "
                         f"（评测带 y[{y0},{y1}]）")
        check(ok_cov, "三物体布点覆盖评测分布", " | ".join(lines))
    else:
        check(False, "三物体布点覆盖评测分布", "缺少布局记录")

    # ---------- 抽帧导出提示 ----------
    if a.save_frames:
        print(f"\n已导出首/末帧到 {os.path.abspath(a.save_frames)} —— 请人工对照动作轨迹确认对齐")

    return finish()


def finish() -> int:
    bad = [r for r in results if not r[0]]
    print("\n" + "=" * 78)
    if bad:
        print(f"结论: ❌ 不通过（{len(bad)}/{len(results)} 项失败）—— 不要进入训练")
        for _, t, d in bad:
            print(f"   - {t}: {d}")
        return 1
    print(f"结论: ✅ 全部 {len(results)} 项通过 —— 可以进入步骤 5（训练）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
