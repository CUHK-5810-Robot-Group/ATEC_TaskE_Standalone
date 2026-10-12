"""EMAModel —— 从 diffusers 原样内联（只为绕开依赖冲突，语义逐行一致）。

为什么要内联，而不是 import diffusers
------------------------------------
train_task_e.py 里只用到 `diffusers.training_utils.EMAModel` 一个工具类，但：

* 本环境装的是 **huggingface_hub 1.6.0**，而 **transformers 5.3.0 依赖它**；
* 镜像上最新的 **diffusers 0.41.0** 却仍然 `from huggingface_hub import
  resolve_revision`（hub 1.x 已移除该 API）→ `import diffusers` 直接 ImportError；
* 降级 hub 会把 transformers 5.3 弄坏（README §4 方案 D 还要用它做 SmolVLA）。

所以把 EMAModel 按 diffusers 0.41.0 的实现原样内联，**不改动环境里任何包的版本**。

语义忠实性（重要，不能凭感觉写）
--------------------------------
train_task_e.py 调用形式是：
    ema = EMAModel(parameters=agent.parameters(), power=0.75)
    ema.step(agent.parameters())
    ema.copy_to(ema_agent.parameters())
这对应 diffusers **≥0.15 的参数式 API**（0.11 时代是 `EMAModel(model)`，签名不同）。
在该 API 下 `use_ema_warmup` 默认 **False** —— 也就是说 **`power=0.75` 本来就不生效**，
实际调度是（已对照本机安装的 0.41.0 源码逐行核对）：

    step = max(0, optimization_step - update_after_step - 1)
    step <= 0            -> decay = 0.0
    use_ema_warmup=False -> decay = (1 + step) / (10 + step)
    decay = clamp(min(decay, 0.9999), min_decay)

即 decay 从 0 起步、约 1e5 步逼近上限 0.9999。本文件复刻的正是这条路径
（非 foreach 分支），`foreach`/deepspeed/`state_dict` 等本项目用不到的
分支一律不带，避免引入额外依赖。
"""

from __future__ import annotations

from typing import Iterable

import torch


class EMAModel:
    """Exponential Moving Average of model weights（diffusers 0.41.0 语义）。"""

    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        decay: float = 0.9999,
        min_decay: float = 0.0,
        update_after_step: int = 0,
        use_ema_warmup: bool = False,
        inv_gamma: float | int = 1.0,
        power: float | int = 2 / 3,
        **kwargs,
    ):
        parameters = list(parameters)
        self.shadow_params = [p.clone().detach() for p in parameters]

        self.decay = decay
        self.min_decay = min_decay
        self.update_after_step = update_after_step
        self.use_ema_warmup = use_ema_warmup
        self.inv_gamma = inv_gamma
        self.power = power
        self.optimization_step = 0
        self.cur_decay_value = None  # set in `step()`

    # ------------------------------------------------------------------ #

    def get_decay(self, optimization_step: int) -> float:
        """Compute the decay factor for the exponential moving average."""
        step = max(0, optimization_step - self.update_after_step - 1)
        if step <= 0:
            return 0.0
        if self.use_ema_warmup:
            cur_decay_value = 1 - (1 + step / self.inv_gamma) ** -self.power
        else:
            cur_decay_value = (1 + step) / (10 + step)
        cur_decay_value = min(cur_decay_value, self.decay)
        cur_decay_value = max(cur_decay_value, self.min_decay)
        return cur_decay_value

    @torch.no_grad()
    def step(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        parameters = list(parameters)

        self.optimization_step += 1

        decay = self.get_decay(self.optimization_step)
        self.cur_decay_value = decay
        one_minus_decay = 1 - decay

        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                s_param.sub_(one_minus_decay * (s_param - param))
            else:
                s_param.copy_(param)

    def copy_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """Copy current averaged parameters into given collection of parameters."""
        parameters = list(parameters)
        for s_param, param in zip(self.shadow_params, parameters):
            param.data.copy_(s_param.to(param.device).data)

    # 便于排查：当前 decay（原实现同样维护 cur_decay_value）
    @property
    def current_decay(self) -> float | None:
        return self.cur_decay_value
