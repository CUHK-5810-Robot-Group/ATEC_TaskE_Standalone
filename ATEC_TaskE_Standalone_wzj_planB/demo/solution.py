"""Task E submission entry point. Set ATEC_TASK_E_CONTROLLER=act for baseline."""

import os

_controller = os.environ.get("ATEC_TASK_E_CONTROLLER", "rgbd").lower()
if _controller == "rgbd":
    from .solution_rgbd import AlgSolution
elif _controller == "act":
    from .solution_act import AlgSolution
else:
    raise ValueError("ATEC_TASK_E_CONTROLLER must be 'rgbd' or 'act'")

__all__ = ["AlgSolution"]
