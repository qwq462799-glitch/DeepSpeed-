import os
from typing import Optional
from app.utils import try_import, has_cuda  # 新增CUDA检测

class Compressor:
    """
    压缩器：提供三类可用路径
    1) FP16/FP32重新保存（通用、稳定）
    2) 8bit/4bit加载配置导出（需bitsandbytes，导出配置用于推理加载）
    3) 记录ZeroQuant/FP6计划（提示与日志，不在本示例直接执行硬件/系统级算法）
    """

    def __init__(self, model_name: str, output_dir: str = "compressed"):
        self.model_name = model_name
        self.output_dir = output_dir

    def export_fp16(self) -> str:
        transformers = try_import("transformers")
        torch = try_import("torch")
        if transformers is None:
            raise RuntimeError("需要安装transformers才能导出模型：pip install transformers")
        dtype = torch.float16 if (torch is not None and has_cuda()) else None  # 仅CUDA用FP16
        model = transformers.AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True, torch_dtype=dtype)
        tok = transformers.AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        os.makedirs(self.output_dir, exist_ok=True)
        model.save_pretrained(self.output_dir)
        tok.save_pretrained(self.output_dir)
        return f"已导出模型到 {self.output_dir}（{'FP16' if dtype else '原精度'}）"

    def export_8bit_4bit_config(self, use_int4: bool = False, use_int8: bool = True) -> str:
        bitsandbytes = try_import("bitsandbytes")
        if bitsandbytes is None:
            raise RuntimeError("生成8bit/4bit配置需要安装bitsandbytes：pip install bitsandbytes>=0.43.0")
        if not has_cuda():
            raise RuntimeError("8bit/4bit量化仅支持CUDA环境")
            
        os.makedirs(self.output_dir, exist_ok=True)
        cfg_path = os.path.join(self.output_dir, "quantization_hint.txt")
        with open(cfg_path, "w", encoding="utf-8") as f:
            f.write("# 使用bitsandbytes在推理时加载8bit/4bit模型示例\n")
            f.write("from transformers import AutoModelForCausalLM, AutoTokenizer\n")
            f.write("import torch\n")
            if use_int4:
                f.write("model = AutoModelForCausalLM.from_pretrained('"
                        + self.model_name +
                        "', load_in_4bit=True, torch_dtype=torch.float16, device_map='auto', trust_remote_code=True)\n")
            elif use_int8:
                f.write("model = AutoModelForCausalLM.from_pretrained('"
                        + self.model_name +
                        "', load_in_8bit=True, torch_dtype=torch.float16, device_map='auto', trust_remote_code=True)\n")
            f.write("tok = AutoTokenizer.from_pretrained('" + self.model_name + "', trust_remote_code=True)\n")
        return f"已生成量化加载提示文件：{cfg_path}（需要bitsandbytes支持）"

    def record_plan_zeroquant_fp6(self, enable_zeroquant: bool, enable_fp6: bool) -> str:
        # 记录计划与说明，提示用户使用对应工程流程（此处不直接执行，以避免复杂硬件/系统依赖）
        msg = []
        if enable_zeroquant:
            msg.append("ZeroQuant计划：建议参照对应论文/实现进行离线PTQ；本控制台导出8bit/4bit加载提示。")
        if enable_fp6:
            msg.append("FP6计划：需特定推理/编译栈；本控制台暂不直接执行。")
        if not msg:
            msg.append("未启用ZeroQuant/FP6计划。")
        return " | ".join(msg)

    def run(
        self,
        export_fp16: bool = True,
        use_int8: bool = False,
        use_int4: bool = False,
        plan_zeroquant: bool = False,
        plan_fp6: bool = False
    ) -> str:
        logs = []
        if export_fp16:
            logs.append(self.export_fp16())
        if use_int4 or use_int8:
            logs.append(self.export_8bit_4bit_config(use_int4=use_int4, use_int8=use_int8))
        logs.append(self.record_plan_zeroquant_fp6(plan_zeroquant, plan_fp6))
        return "\n".join(logs)
