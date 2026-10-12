"""把一次采集的 traj_* **就地追加**到已有数据集（不复制整份数据）。

为什么需要
----------
采集脚本的 `init_output()` 会用 `mode="w"` **清空**输出目录里已有的
trajectory.hdf5，所以"跑满尝试上限但条数不够"时不能原地续采。
如果改用"合并成新文件"的常规做法，需要同时容纳两份数据：
    已有 28.6 GB + 新增 8 GB + 合并结果 36 GB > 数据盘 48 GB  ✗
因此本脚本只做**追加**：打开目标文件（mode="a"），把源文件里的
traj_* 依次接在已有最大编号之后，磁盘开销 = 新增数据本身。

用法
----
    python scripts/act/append_trajs.py \
        --into datasets/atec_task_e/trajectory.hdf5 \
        --from datasets/atec_task_e_part2/trajectory.hdf5 \
        --meta_into datasets/atec_task_e/collection_meta.json \
        --meta_from datasets/atec_task_e_part2/collection_meta.json

注意
----
* 目标文件正在被采集进程写入时**不要**运行（同一文件两个写者会互相破坏）。
* 追加是逐条进行的，中途被杀只影响当前这条；已追加的组保留。
"""
import argparse
import json
import os
import shutil

import h5py


def max_index(f: h5py.File) -> int:
    idx = [int(k.split("_")[1]) for k in f.keys() if k.startswith("traj_")]
    return max(idx) + 1 if idx else 0


def copy_group(src_file: h5py.File, src_path: str, dst_file: h5py.File,
               dst_path: str) -> None:
    """用 HDF5 原生对象复制（H5Ocopy）。

    ⚠️ 教训：**不要**写成 `dst.create_dataset(data=src[...][()], compression=...)`
    —— 那会先把数据解压、再用 gzip 重新压缩，5.7 GB 要跑很久；一旦中途被杀，
    目标文件会留下"根组链接已写、EOA 未更新"的不一致状态（本项目真实踩过，
    需要 h5clear/手工补 EOA 才能救回）。H5Ocopy 直接搬运压缩块，又快又安全。
    """
    src_file.copy(src_path, dst_file, name=dst_path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--into", required=True, help="目标 trajectory.hdf5（就地追加）")
    ap.add_argument("--from", dest="src", required=True, help="来源 trajectory.hdf5")
    ap.add_argument("--meta_into", default=None)
    ap.add_argument("--meta_from", default=None)
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()

    with h5py.File(a.into, "a") as fd, h5py.File(a.src, "r") as fs:
        src_keys = sorted(fs.keys(), key=lambda k: int(k.split("_")[1]))
        start = max_index(fd)
        print(f"目标已有 traj: {start} 条；来源 {len(src_keys)} 条 → "
              f"追加为 traj_{start} .. traj_{start + len(src_keys) - 1}")
        if a.dry_run:
            print("（dry-run，未写入）")
            return 0
        for i, k in enumerate(src_keys):
            dst = f"traj_{start + i}"
            if dst in fd:
                print(f"[WARN] {dst} 已存在，跳过")
                continue
            copy_group(fs, k, fd, dst)
            print(f"  {k} → {dst}", flush=True)

    if a.meta_into and a.meta_from and os.path.exists(a.meta_from):
        tgt = json.load(open(a.meta_into, encoding="utf-8")) if os.path.exists(a.meta_into) else {}
        src = json.load(open(a.meta_from, encoding="utf-8"))
        recs = tgt.setdefault("records", [])
        for r in src.get("records", []):
            r = dict(r)
            r["appended_from"] = os.path.abspath(a.meta_from)
            if r.get("saved") and isinstance(r.get("traj_index"), int):
                r["traj_index"] = start + r["traj_index"]
            recs.append(r)
        tgt.setdefault("appends", []).append({
            "from": os.path.abspath(a.src),
            "added": len(src.get("records", [])),
            "code_fingerprint": src.get("code_fingerprint"),
        })
        tmp = a.meta_into + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(tgt, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, a.meta_into)
        print(f"meta 已合并 → {a.meta_into}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
