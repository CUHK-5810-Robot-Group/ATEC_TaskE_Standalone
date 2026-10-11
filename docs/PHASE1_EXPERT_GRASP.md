# 阶段 1：单物体专家抓取与数据管道修复

本文档配合 README 第 3.1 节，指导你完成**物体 1（糖盒）与物体 2（芥末瓶）**的专家调试，并说明已在代码中落地的算法修复。

## 已完成的代码修复（3.1 节 1、2、4 的部分）

| 问题 | 修改位置 | 说明 |
| --- | --- | --- |
| 成功终止误删轨迹 | `scripts/act/task_e/collector.py` | 环境因三物体入篮 `terminated` 时**保留**已录制帧；非成功终止仍丢弃 |
| 采集与评测分布不一致 | `env_cfg.py` + `config.py` | 物体 XY/Z 出生带与 `TaskEEnvPiperCfg` **同一套常量** |
| 入篮检查与评测 Z 不一致 | `collector.py` | `check_objects_in_basket` 使用与 `ObjectsInBasketDone` 相同的 XY 半宽与 Z 区间 |
| 物体 1/2 抓取高度 | `config.py` → `GRASP_Z_OFFSET_BY_OBJECT` | 按物体索引调节 REACH/CLOSE 高度 |
| 采集死循环 | `cli_args.py` + `collect_demos_task_e.py` | 新增 `--max_attempts`（默认 500） |

**说明**：单物体采集时，评测终止条件仍要求**三件**物体同时在篮内，因此通常**不会**触发 `terminated`；状态机在 `sm.done` 后正常结束即可。三物体全任务采集时，成功终止修复才会明显生效。

---

## 推荐操作顺序

### 步骤 A：确认环境与 PYTHONPATH（与阶段 0 相同）

```bash
conda activate <你的 Isaac Lab 环境>
cd /path/to/ATEC_TaskE_Standalone
export PYTHONPATH="$PWD/source/atec_rl_lab:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python scripts/check_handoff.py
```

### 步骤 B：调试物体 1（糖盒）— 带视频、要入篮

使用**独立输出目录**，避免覆盖其它实验数据：

```bash
python scripts/act/collect_demos_task_e.py \
  --pick_objects 1 \
  --num_demos 5 \
  --output_dir ./datasets/atec_task_e/obj1_debug \
  --save_video \
  --only_success \
  --headless --enable_cameras
```

视频默认在：`datasets/atec_task_e/obj1_debug/videos/demo_XXXX.mp4`。

**验收**：连续多次 attempt 中，多数 episode 能在 `--only_success` 下被保存（终端出现 `traj_N: ... steps`）。

### 步骤 C：调试物体 2（芥末瓶）

```bash
python scripts/act/collect_demos_task_e.py \
  --pick_objects 2 \
  --num_demos 5 \
  --output_dir ./datasets/atec_task_e/obj2_debug \
  --save_video \
  --only_success \
  --headless --enable_cameras
```

### 步骤 D：根据视频调参（循环 B/C 直到稳定）

主要旋钮在 `scripts/act/task_e/config.py`：

| 现象 | 建议调整 |
| --- | --- |
| 夹爪在物体**上方过高**，CLOSE 时夹空 | 减小该物体的 `GRASP_Z_OFFSET_BY_OBJECT[id]`（如 1: `0.07→0.05`） |
| **高度/开口对，但 XY 偏一侧**（贴边夹空） | 调 `GRASP_XY_OFFSET_BY_OBJECT[id]` 的 `(dx, dy)`（世界坐标米），每次 ±0.005~0.01 |
| 夹爪**碰桌或推飞**物体 | 略增大 offset，或增加 `STEPS["REACH"]` 让步数更慢 |
| CLOSE 时滑脱 | 增大 `STEPS["CLOSE"]`；物体 2 可略增 offset 抓瓶身中部 |
| LIFT 时物体仍留桌面 | 检查 CLOSE 阶段 gripper 是否用到 `GRIPPER_CLOSE_POS`；略增 CLOSE 步数 |
| TRANSPORT/PLACE 掉落 | 先确认 LIFT 后物体随动；再查 `CARRY_Z` 是否够高 |

次要旋钮：`scripts/act/task_e/state_machine.py` 中抓取四元数逻辑一般无需改，除非物体姿态极端。

### 步骤 E：记录通过标准（阶段 1 第一个目标）

对物体 **1** 和 **2** 分别满足：

1. `--only_success` 下 **≥3/5** 条 demo 被保存（或你自定更严标准）。
2. 目视视频：失败案例能归类到 PRE_GRASP / REACH / CLOSE / LIFT / PLACE 某一阶段。
3. 终端无大量 `Episode terminated (non-success)` 或 `timed out before state machine finished`。

### 步骤 F：（可选）固定种子复现

采集脚本对物体位置使用未固定 RNG。若需同一布局反复试参，可在 `collect_demos_task_e.py` 里临时给 `rng = np.random.default_rng(42)`（调试完再改回）。

---

## 三物体采集前检查清单

在进入 `--pick_objects 1 2 3` 大规模采集前：

- [ ] 物体 1、2、3 单物体 `--only_success` 均基本稳定  
- [ ] 理解 `trajectory.hdf5` 会被 `--output_dir` 内**覆盖**  
- [ ] 全任务时若日志出现 `Evaluation success termination — keeping recorded trajectory`，表示 3.1(1) 修复生效  

---

## 相关文件速查

```text
scripts/act/collect_demos_task_e.py   # 入口
scripts/act/task_e/collector.py       # 录制与成功判定
scripts/act/task_e/state_machine.py   # 抓放阶段
scripts/act/task_e/config.py          # 抓取高度、步数、出生带（与 env 同步）
source/.../tasks/task_e/env_cfg.py    # 评测场景物体分布（权威来源）
```
