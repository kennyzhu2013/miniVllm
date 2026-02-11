import os
import torch


def get_gpu_memory():
    """Return (total, used, free) GPU memory in bytes.

    Prefer NVML; fallback to torch.cuda.mem_get_info when NVML is unavailable.
    """
    torch.cuda.synchronize()
    try:
        from pynvml import nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
        nvmlInit()
        # 最多支持8卡.
        visible_device = list(map(int, os.getenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(',')))
        cuda_device_idx = torch.cuda.current_device()
        cuda_device_idx = visible_device[cuda_device_idx]
        handle = nvmlDeviceGetHandleByIndex(cuda_device_idx)
        mem_info = nvmlDeviceGetMemoryInfo(handle)
        total_memory = mem_info.total
        used_memory = mem_info.used
        free_memory = mem_info.free
        nvmlShutdown()
        return total_memory, used_memory, free_memory
    except Exception:
        # Fallback: torch.cuda.mem_get_info returns (free, total)
        free, total = torch.cuda.mem_get_info()
        # Approximate used as (total - free)
        used = total - free
        return total, used, free
