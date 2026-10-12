"""CLI argument definitions for Task E demo collection.
"""


def add_collect_demo_args(parser) -> None:
    parser.add_argument(
        "--num_demos", type=int, default=50,
        help="Number of successful demos to collect.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./datasets/atec_task_e",
        help="Directory to save trajectory.hdf5 and trajectory.json.",
    )
    parser.add_argument(
        "--pick_objects", type=int, nargs="+", default=[1, 2, 3],
        metavar="N",
        help="Which objects to pick (subset of {1,2,3}). Default: all three.",
    )
    parser.add_argument(
        "--save_video", action="store_true", default=False,
        help="Save an MP4 per demo for visualization "
             "(requires: pip install imageio imageio-ffmpeg).",
    )
    parser.add_argument(
        "--video_dir", type=str, default=None,
        help="Output directory for MP4 files. Defaults to <output_dir>/videos/.",
    )
    parser.add_argument(
        "--save_images", action="store_true", default=False,
        help="Save raw RGB frames into HDF5 under traj_N/images/rgb (T,H,W,3) "
             "for ACT RGBD training. Shares the camera with --save_video.",
    )
    parser.add_argument(
        "--min_free_gb", type=float, default=3.0,
        help="输出目录所在磁盘剩余空间低于此值(GB)时停止采集（0=不检查）。"
             "防止把磁盘写满导致 HDF5 写坏：单条成功轨迹约 0.07 GB(224) ~ 0.6 GB(原生)。",
    )
    parser.add_argument(
        "--image_size", type=int, default=0,
        help="存图前把 RGB 缩放到的正方形边长（0=存相机原生分辨率）。"
             "训练与推理两侧都会把图缩到 224x224，所以 224 不损失任何策略"
             "会用到的信息；而 480x640 原生每条约 572 MB，50 条约 29 GB，"
             "在有磁盘配额的环境里放不下。建议 --image_size 224。",
    )
    parser.add_argument(
        "--max_attempts", type=int, default=0,
        help="尝试次数上限（0=不限）。原实现只有 num_demos 这一个条件，"
             "若成功率异常低会无限循环（README §7 已警告）。"
             "按 60%% 成功率、num_demos=50 估算：设 120 可留足余量。",
    )
    parser.add_argument(
        "--diversity", action="store_true", default=False,
        help="示范多样性（采集侧，默认关）。开启后每条示范会：(1) 给各状态步数乘一个"
             "随机因子（时序抖动）；(2) 在 --perturb_states 指定的状态刚进入时给 6 个"
             "臂关节注入高斯扰动，让状态机自己纠回来，从而录到「从偏了的地方恢复」的"
             "示范。原实现 53 条示范的夹爪切换时刻完全相同、示范里没有任何重同步信息"
             "（实测：夹爪通道跨示范标准差恒为 0），闭环漂移后无法恢复。"
             "只影响采集到的示范分布，不碰评测环境/奖励/成功条件。",
    )
    parser.add_argument(
        "--steps_jitter", type=float, nargs=2, default=None, metavar=("LO", "HI"),
        help="各状态步数的乘性抖动区间，默认 0.70 1.40（需配合 --diversity）。",
    )
    parser.add_argument(
        "--perturb_sigma", type=float, default=0.15,
        help="注入到 6 个臂关节的高斯扰动标准差（弧度），默认 0.15；0=不加扰动。",
    )
    parser.add_argument(
        "--perturb_states", type=str, nargs="+", default=None,
        help="在哪些状态「刚进入时」施加扰动，默认 PRE_GRASP（每件物体都会经过，"
             "且此时臂在高处、下方没有易碰倒的东西）。不建议加 REACH/CLOSE/LIFT。",
    )
    parser.add_argument(
        "--only_success", action="store_true", default=False,
        help="Discard demos where not all picked objects ended up in the basket.",
    )
