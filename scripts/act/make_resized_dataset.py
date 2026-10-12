"""把采集数据预处理成目标分辨率（默认 224），供训练快速加载。

为什么需要
----------
训练脚本 `train_task_e.py` 会把**每条轨迹的图像整段读进内存再 resize 到 224**：
  源数据 480×640、33.5 GB → 实测加载耗时 **约 36 分钟**
  （瓶颈是 gzip 解压 + CPU resize，磁盘读只有 17 MB/s，不是 IO 受限）
预处理成 224 后：
  体积 ≈ 3.6 GB、加载约 6 分钟；而且**训练结果完全等价** ——
  训练侧 `T.Resize((224,224), antialias=True)` 对已是 224 的输入是幂等的，
  用的也是同一种双线性+抗锯齿插值（见 collector 里 --image_size 的说明）。
每次训练都能省 ~30 分钟，做 2 次以上训练就回本。

用法
----
    python scripts/act/make_resized_dataset.py \
      --src /root/gpufree-data/VibeCo_proj/Robotic/test2_data/atec_task_e/trajectory.hdf5 \
      --dst /root/gpufree-data/VibeCo_proj/Robotic/test2_data/atec_task_e_224/trajectory.hdf5 \
      --size 224 --chunk 128

注意
----
* 逐条流式处理（不会把整条轨迹的图像同时驻留内存），中途被杀只影响当前这条；
* 不可读的轨迹组（例如文件被中断写入留下的孤儿组）会被跳过并告警；
* 其余字段（obs/actions/qvel/ee_pos/ee_quat）逐位原样复制，不做任何改动。
"""
import argparse
import os
import time

import h5py
import numpy as np
import torch
import torchvision.transforms.functional as TF


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--chunk", type=int, default=128, help="每次处理的帧数（控制内存）")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    if os.path.exists(a.dst) and not a.overwrite:
        print(f"[ERROR] 目标已存在：{a.dst}（加 --overwrite 覆盖）")
        return 1
    os.makedirs(os.path.dirname(os.path.abspath(a.dst)), exist_ok=True)

    t_all = time.time()
    n_ok = n_skip = 0
    with h5py.File(a.src, "r") as fs, h5py.File(a.dst, "w") as fd:
        src_keys = sorted(fs.keys(), key=lambda k: int(k.split("_")[1]))
        print(f"来源 {len(src_keys)} 组 → 目标 {a.dst}（{a.size}×{a.size}）", flush=True)
        for k in src_keys:
            t0 = time.time()
            try:
                gs = fs[k]
                obs = gs["obs"][:]
                actions = gs["actions"][:]
                T = obs.shape[0]
                gd = fd.create_group(k)
                gd.create_dataset("obs", data=obs, compression="gzip")
                gd.create_dataset("actions", data=actions, compression="gzip")
                for name in ("qvel", "ee_pos", "ee_quat"):
                    if name in gs:
                        gd.create_dataset(name, data=gs[name][:], compression="gzip")
                img_src = gs["images/rgb"]
                # 分块交给 h5py 自动决定：实测自动分块 = (215,14,28,1)（84 KB/块），
                # gzip 压缩率约 4 倍（20.4 KB/帧 → 单条约 70 MB）；
                # 若显式写 chunks=(64,224,224,3)（9.6 MB/块）压缩率只有约 2.3 倍
                # （66 KB/帧 → 单条 227 MB，53 条要 12 GB，会顶爆数据盘）。
                # 这也与采集器 --image_size 224 直接存出来的文件布局一致。
                img_dst = gd.create_group("images").create_dataset(
                    "rgb", shape=(T, a.size, a.size, 3), dtype=np.uint8,
                    compression="gzip")
                for s in range(0, T, a.chunk):
                    e = min(s + a.chunk, T)
                    batch = torch.from_numpy(np.asarray(img_src[s:e]))      # (n,H,W,3) uint8
                    out = TF.resize(batch.permute(0, 3, 1, 2),              # (n,3,H,W)
                                    [a.size, a.size], antialias=True)
                    img_dst[s:e] = out.permute(0, 2, 3, 1).numpy()
                n_ok += 1
                print(f"  {k}: {T} 帧 → {a.size}² 用时 {time.time()-t0:.1f}s", flush=True)
            except Exception as exc:
                n_skip += 1
                print(f"  [WARN] 跳过不可读的 {k}: {type(exc).__name__}: {exc}", flush=True)
    dt = time.time() - t_all
    size_gb = os.path.getsize(a.dst) / 1e9
    print(f"\n完成：{n_ok} 条（跳过 {n_skip}）→ {a.dst}")
    print(f"体积 {size_gb:.2f} GB，总耗时 {dt/60:.1f} 分钟")
    print(f"提示：训练时把 --demo_path 指向这个文件即可，结果与用原始 480×640 等价。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
