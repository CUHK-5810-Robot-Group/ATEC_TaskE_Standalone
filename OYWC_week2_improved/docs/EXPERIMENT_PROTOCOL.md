# OYWC 第二周实验协议

这份本地包相对原始 `main` 的改动集中在策略层，保留原来的 Task E 奖励、终止条件、动作顺序和场景资产。

## 改动痕迹

- `demo/oywc_rgbd_grasp.py`：用 mask 内有效深度、PCA 主次轴和公制夹爪宽度估计抓取目标。
- `demo/solution_rgbd.py`：新增方案 B 的观测驱动控制器，按 `sugar_box -> mustard_bottle -> banana` 执行阶段闭环。
- `scripts/compare_strategies.py`：新增公平的方案 A/B 对比，拒绝不同种子或不同终止条件的比较。
- `scripts/validate_oywc_grasp.py`：新增瓶身中部、横向闭合轴、深度离群点、跳变门控和有限重试检查。
- `docs/video_analysis.md`：将视频中固定 TCP 姿态的问题转成可验证的工程假设。

## 抓取思路

瓶子不是从瓶口或肩部硬夹，而是在实例 mask 的主轴中间 50% 取点，并用 minor axis 作为夹爪闭合方向。竖直瓶子的 minor axis 是水平轴，因此夹爪横向穿过瓶身中部。目标位置还要经过时间滤波和跳变门控，单帧误检不会直接驱动机械臂。

方案 B 需要两个工程适配器：

1. `segmenter(payload)`：只从 `payload["rgb"]` 和 `payload["depth"]` 返回三类 `ObjectObservation`。
2. `ik_adapter(target_base, yaw, opening, obs)`：把 base 坐标目标转换为 Piper 的 8 维环境动作，并保留 `action=(joint_target-default)/0.5` 的动作语义。

适配器没有接上时，控制器保持当前动作，不会假装已经完成视觉抓取。

## 验收顺序

```bash
cd OYWC_week2_improved
python3 scripts/validate_oywc_grasp.py
python3 -m compileall demo scripts
```

然后在同一组未参与采集的 seed 上运行：

```bash
python3 scripts/compare_strategies.py act_results.json rgbd_results.json \
  --output comparison.json
```

每条 episode 至少记录 `seed`、`complete`、`objects_in_basket`、`time_s` 和 `retries`。`complete` 必须沿用 Task E 的“三件物体同时在篮内”判定，不能用部分奖励替代。

## 不能在本地伪造的部分

本文件夹不包含 Isaac Lab 运行时、评分系统或远端提交凭据，因此不能在没有仿真运行器的情况下声称达到 18 分。真实验收还需加载原项目资产，确认相机深度单位、外参和 Piper IK，并在固定评测条件下记录完整三物体成功率。
