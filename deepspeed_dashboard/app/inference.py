from typing import Any, Union, List
from app.utils import try_import, safe_getattr, load_model_config  # 新增配置加载

class InferenceEngine:
    """
    推理引擎：优先使用DeepSpeed MII；不可用时回退到transformers.pipeline。
    单例模式确保全局唯一实例
    """
    _instance = None  # 单例实例

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_name: Optional[str] = None, device_map: str = "auto"):
        # 若未指定模型名，从配置文件加载
        config = load_model_config()
        self.model_name = model_name or config["model_name"]
        self.device_map = device_map
        self._backend = None           # "mii" or "hf"
        self._pipe = None
        self._load_backend()

    def _load_backend(self):
        # 先尝试MII
        mii = try_import("mii")
        if mii is not None:
            try:
                self._pipe = mii.pipeline(
                    task="text-generation",
                    model=self.model_name,
                    deployment_name=self._norm_deploy_name(self.model_name)
                )
                self._backend = "mii"
                return
            except Exception:
                self._pipe = None

        # 回退到HF transformers
        transformers = try_import("transformers")
        torch = try_import("torch")
        if transformers is None:
            raise RuntimeError("未安装transformers或mii，无法初始化推理后端。")

        dtype = None
        if torch is not None and hasattr(torch, "float16"):
            dtype = torch.float16

        try:
            self._pipe = transformers.pipeline(
                "text-generation",
                model=self.model_name,
                torch_dtype=dtype,
                device_map=self.device_map,
                trust_remote_code=True
            )
            self._backend = "hf"
        except Exception as e:
            raise RuntimeError(f"初始化HF推理失败：{e}")

    def _norm_deploy_name(self, name: str) -> str:
        return name.split("/")[-1].replace(".", "-").replace("_", "-")

    def reload(self, model_name: str, device_map: str = "auto"):
        self.model_name = model_name
        self.device_map = device_map
        self.close()
        self._load_backend()
        return {"backend": self._backend, "model": self.model_name}

    def backend(self) -> str:
        return self._backend or "unknown"

    def close(self):
        self._pipe = None

    def _normalize_output(self, result: Any) -> str:
        if result is None:
            return ""
        gen_text = safe_getattr(result, "generated_text", None)
        if gen_text is not None:
            return gen_text if isinstance(gen_text, str) else str(gen_text)
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], dict) and "generated_text" in result[0]:
                return result[0]["generated_text"]
            if isinstance(result[0], str):
                return result[0]
        if isinstance(result, dict) and "generated_text" in result:
            return result["generated_text"]
        return str(result)

    def generate(
        self,
        prompt: Union[str, List[str]],
        *,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        max_new_tokens: int = 256,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        num_return_sequences: int = 1,** kwargs
    ) -> str:
        if self._pipe is None:
            raise RuntimeError("推理后端未初始化。")

        if self._backend == "mii":
            result = self._pipe(
                prompt,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                **kwargs
            )
            return self._normalize_output(result)

        result = self._pipe(
            prompt,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences
        )
        return self._normalize_output(result)
