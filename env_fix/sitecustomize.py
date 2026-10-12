"""ATEC Task E 环境补丁：预加载 libtorch 的延迟 CUDA linalg 库。

背景
----
Isaac Lab 的 DifferentialIKController 使用 torch.inverse / torch.linalg.inv。
PyTorch 对 libtorch_cuda_linalg.so 采用延迟 dlopen()，在 Isaac Sim 的
"rendering" experience（--enable_cameras）下该 dlopen 会在运行时解析失败：

    RuntimeError: Error in dlopen: libtorch_cuda_linalg.so:
                  cannot open shared object file: No such file or directory

文件实际存在于 torch/lib 下，提前用绝对路径以 RTLD_GLOBAL 打开即可解决。
本模块通过 PYTHONPATH 注入，在解释器启动最早期执行。

禁用方式：设置环境变量 ATEC_SKIP_TORCH_LINALG_PRELOAD=1
"""
import os

if os.environ.get("ATEC_SKIP_TORCH_LINALG_PRELOAD") != "1":
    def _preload() -> None:
        try:
            import ctypes
            import importlib.util

            spec = importlib.util.find_spec("torch")
            if spec is None or not spec.submodule_search_locations:
                return
            lib = os.path.join(
                list(spec.submodule_search_locations)[0], "lib", "libtorch_cuda_linalg.so"
            )
            if os.path.exists(lib):
                ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
        except Exception:
            # 预加载失败不应阻断解释器启动；后续会在使用处报出原始错误
            pass

    _preload()
