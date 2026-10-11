# ATEC Task E：OYWC 第二周改进包

这是一个本地生成的独立代码文件夹，按原仓库的 `demo/`、`scripts/`、`docs/` 和 `source/` 组织。它不连接远端，也不修改原仓库的 `main`。

## 本周目标

原始方案 A 是 ACT 模仿学习：使用 RGB 和关节状态预测绝对 8 维动作。它适合作为可复现基线，但固定动作序列不能根据瓶身位置和朝向修正抓取。

本包新增方案 B：

```text
RGB-D → 实例 mask → 有效深度 → 主轴/窄边 → 物体三维中心
      → 相机到 base 变换 → 瓶身中部横向抓取
      → 接近/闭合/抬升/搬运/释放/撤回
      → 物体随动检查 → 有限重试 → 下一个物体
```

最关键的修改是：芥末瓶的抓取点位于瓶身主轴中部，夹爪闭合方向来自 mask 的 minor axis。这样不是把一个固定 TCP 姿态硬套到所有位置，而是每次用新观测重新计算目标位置、yaw 和公制开口宽度。

## 文件说明

- `demo/oywc_rgbd_grasp.py`：RGB-D 几何规划、深度离群点剔除、时间滤波和有限状态机。
- `demo/solution_rgbd.py`：方案 B 的八维动作适配器；segmenter 与 Piper IK 通过依赖注入提供。
- `scripts/validate_oywc_grasp.py`：不依赖 Isaac Lab 的离线几何测试。
- `scripts/compare_strategies.py`：同 seed、同终止条件的 A/B 结果比较。
- `scripts/analyze_task_e_videos.py`：视频元数据检查。
- `docs/EXPERIMENT_PROTOCOL.md`：相对原工程的修改点、验收步骤和真实仿真接入要求。
- `docs/video_analysis.md`：视频观察与代码改动的对应关系。

## 运行离线检查

```bash
python3 scripts/validate_oywc_grasp.py
python3 -m compileall demo scripts
```

## 接入真实工程

`ObservationOnlyRGBDController` 不读取 Isaac 物体真值、PhysX Jacobian 或奖励。真实接入时必须提供：

- `segmenter(payload)`：从外部 RGB-D 生成 `ObjectObservation`；
- `ik_adapter(target_base, yaw, opening, obs)`：调用 Piper 的 FK/阻尼 IK，并转换成 8 维环境动作；
- 实际相机内参、相机到 base 外参和篮子坐标。

仍然要使用原 README 的动作定义：

```text
joint_target = default_joint_pos + 0.5 × action
action = (joint_target - default_joint_pos) / 0.5
```

## 验收要求

三件物体必须在同一个 episode 中全部放入篮子，并使用原环境的完整成功条件。方案 A 和方案 B 必须使用相同的 seed、位置范围、控制频率、超时和终止定义。记录完整成功率、每件物体抓取成功率、掉落/空抓次数、完成时间、重试次数和视觉延迟。

离线测试通过只说明几何和状态机逻辑正确，不能替代 Isaac Sim 闭环评测。本地当前没有评分系统，因此本包不会虚构“满分 18 分”结果。
