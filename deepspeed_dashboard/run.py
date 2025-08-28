import sys
import asyncio
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from app.inference import InferenceEngine  # 导入统一推理引擎
from app.utils import has_cuda, load_model_config  # 加载配置

# 日志配置
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# 初始化 FastAPI
app = FastAPI(title="DeepSpeed 推理 API")

# 加载配置文件
config = load_model_config()

# 全局推理引擎实例（单例）
engine = InferenceEngine(
    model_name=config["model_name"],
    device_map=config["device_map"]
)

# 请求模型
class GenerateRequest(BaseModel):
    prompt: str
    temperature: float = config["default_generation"]["temperature"]
    top_k: int = config["default_generation"]["top_k"]
    top_p: float = config["default_generation"]["top_p"]
    max_new_tokens: int = config["default_generation"]["max_new_tokens"]
    repetition_penalty: float = config["default_generation"]["repetition_penalty"]
    do_sample: bool = True
    num_return_sequences: int = 1

class ReloadRequest(BaseModel):
    model_name: str
    device_map: str = "auto"

# API 路由
@app.get("/status")
async def get_status():
    return {
        "ready": engine._pipe is not None,
        "model": engine.model_name,
        "backend": engine.backend(),
        "cuda_available": has_cuda()
    }

@app.post("/generate")
async def generate(req: GenerateRequest):
    if engine._pipe is None:
        raise HTTPException(status_code=400, detail="模型未加载")
    try:
        result = engine.generate(
            req.prompt,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            max_new_tokens=req.max_new_tokens,
            repetition_penalty=req.repetition_penalty,
            do_sample=req.do_sample,
            num_return_sequences=req.num_return_sequences
        )
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reload_model")
async def reload_model(req: ReloadRequest):
    try:
        info = engine.reload(req.model_name, req.device_map)
        return {"reloaded": True, **info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 启动函数
def start_api(host: str = "0.0.0.0", port: int = 8000):
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_api()
