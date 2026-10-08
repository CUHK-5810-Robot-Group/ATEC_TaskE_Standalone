# OYWC：Task E 的 RGB-D 自适应抓取方案

## 目标

现有 ACT 基线从示范中学习固定动作序列，遇到瓶身位置或接触高度偏差时没有可解释的恢复路径。尤其是芥末瓶不能用一个固定 TCP 姿态从上方或瓶口附近硬夹；实际线下赛应在瓶身中部沿横向窄边夹住，夹爪的两个指尖才会有稳定的支撑长度。

OYWC 保留 `AlgSolution.predicts(obs, current_score)` 的 8 维动作接口，把方案 B 做成可单测的感知和控制层，并把方案 A 作为可复现对照：

```text
外部 RGB-D → 类别 mask → 有效深度/离群点筛选 → mask PCA/中心线
          → 相机到 base 的几何变换 → 窄边夹爪朝向
          → 接近/闭合/抬升/搬运/释放/撤回 → 物体随动与有限重试
```

## 关键设计

1. **抓取点不是 bbox 中心。** 深度在 mask 内按 10–90% 分位裁剪，沿主轴取中间 50% 的像素，得到瓶身中部而不是瓶盖、瓶肩或背景污染的中心。
2. **夹爪方向来自窄边。** PCA 的 minor axis 作为夹爪闭合轴；竖直瓶身的主轴是图像竖直方向，因此 minor axis 自动变成水平夹取方向。糖盒和香蕉也会跟随各自观测到的平面方向。
3. **目标需要新鲜并且连续。** `TemporalPoseFilter` 对置信度、位置跳变和深度有效性设门控；相机暂时丢帧时复用短时目标，超过帧数则安全保持，不会沿最后一次固定位姿无限运动。夹爪开口由重建点云沿窄边的公制跨度估计，避免用像素宽度造成距离变化时的饱和。
4. **执行是有限状态机。** `GraspStateMachine` 明确区分 approach、close、lift、transport、release、retreat，并在抬升后检查物体是否随动；空抓或滑脱只能进入有限次数 retry，且每次 retry 都必须刷新视觉目标并回到 approach，不能直接把动作块结束当成成功。
5. **没有 oracle。** 几何模块只接受 mask、RGB-D 和标定参数，不能读取 Isaac 物体真值、PhysX Jacobian 或奖励。Piper 的 FK/阻尼 IK 通过提交工程的适配器注入，便于和原有动作尺度 `action=(target-default)/0.5` 对齐。

## 与方案 A 的公平比较

两种策略应在同一组未参与采集的 reset seed 和物体位置上运行，保持奖励、终止条件、控制频率和超时一致。至少记录：三物体完整成功率、每件物体的抓取成功率、掉落/空抓次数、平均完成时间、视觉更新延迟和重试次数。训练 loss 或单次把一个物体放入篮子不能代替完整 episode 成功。

建议先以 ACT 建立基线，再只切换策略层到 RGB-D；若 RGB-D 未提供可用的专用分割器或 Piper IK 适配器，入口默认保持 ACT，并显式通过 `ATEC_TASK_E_STRATEGY=rgbd` 才启用新路径，避免未配置的实验悄悄改变评分。

## 运行检查

```bash
python scripts/validate_oywc_grasp.py
python -m compileall demo scripts
```

实际仿真提交前还必须用本仓库的真实相机内参、外参、深度单位和 Piper IK 适配器运行闭环评测。该离线包不会伪造 Isaac Sim 成功率。
