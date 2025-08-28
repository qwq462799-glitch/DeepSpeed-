import os
import json
import psutil
import yaml
from pathlib import Path
from typing import Any, Optional, Callable

def has_cuda() -> bool:
    """检查是否有CUDA可用"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def try_import(module_name: str) -> Any:
    """安全导入模块，不存在时返回None"""
    try:
        return __import__(module_name)
    except ImportError:
        return None

def safe_getattr(obj: Any, attr: str, default: Any = None) -> Any:
    """安全获取对象属性，避免AttributeError"""
    try:
        return getattr(obj, attr, default)
    except Exception:
        return default

def load_model_config(config_path: str = "models/model_config.yaml") -> dict:
    """加载模型配置文件"""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_json(data: dict, path: str) -> None:
    """保存JSON数据"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def show_metrics() -> str:
    """展示系统资源信息"""
    metrics = []
    # CPU信息
    cpu_percent = psutil.cpu_percent(interval=0.5)
    metrics.append(f"CPU使用率: {cpu_percent}%")
    
    # 内存信息
    mem = psutil.virtual_memory()
    mem_used = mem.used / (1024 **3)
    mem_total = mem.total / (1024** 3)
    metrics.append(f"内存使用: {mem_used:.2f}GB / {mem_total:.2f}GB ({mem.percent}%)")
    
    # GPU信息（如果有）
    if has_cuda():
        pynvml = try_import("pynvml")
        if pynvml:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                mem_used_gpu = mem_info.used / (1024 **3)
                mem_total_gpu = mem_info.total / (1024** 3)
                metrics.append(f"GPU {i} 使用率: {gpu_util}% | 显存: {mem_used_gpu:.2f}GB / {mem_total_gpu:.2f}GB")
            pynvml.nvmlShutdown()
        else:
            metrics.append("GPU信息: 需安装pynvml (pip install pynvml)")
    else:
        metrics.append("GPU: 无可用CUDA设备")
    
    return "\n".join(metrics)
