import time
import requests
import gradio as gr
from app.utils import show_metrics, load_model_config  # 加载配置
from run import start_api  # 导入API启动函数

# 加载配置文件默认值
config = load_model_config()
DEFAULT_MODEL = config["model_name"]
DEFAULT_GEN_PARAMS = config["default_generation"]

# 全局变量：记录API线程
_uvicorn_thread = {"th": None}

def human_time(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds*1000:.1f} 毫秒"
    return f"{seconds:.2f} 秒"

def reload_model(model_name: str, device_map: str) -> str:
    try:
        # 调用API重新加载模型
        response = requests.post(
            "http://localhost:8000/reload_model",
            json={"model_name": model_name, "device_map": device_map},
            timeout=120
        )
        response.raise_for_status()
        return f"模型加载成功：{response.json()['model']}（后端：{response.json()['backend']}）"
    except Exception as e:
        return f"加载失败：{str(e)}"

def do_infer(prompt: str, temperature: float, top_k: int, top_p: float, 
             max_new_tokens: int, repetition_penalty: float, do_sample: bool, num_return_sequences: int):
    start = time.time()
    try:
        response = requests.post(
            "http://localhost:8000/generate",
            json={
                "prompt": prompt,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "max_new_tokens": int(max_new_tokens),
                "repetition_penalty": repetition_penalty,
                "do_sample": do_sample,
                "num_return_sequences": int(num_return_sequences)
            },
            timeout=60
        )
        response.raise_for_status()
        result = response.json()["result"]
        return result, time.time() - start
    except Exception as e:
        return f"推理失败：{str(e)}", time.time() - start

def start_api_service(host: str, port: int) -> str:
    global _uvicorn_thread
    if _uvicorn_thread["th"] is None:
        import threading
        _uvicorn_thread["th"] = threading.Thread(
            target=start_api, args=(host, port), daemon=True
        )
        _uvicorn_thread["th"].start()
        return f"API服务已启动：http://{host}:{port}"
    return "API服务已在运行"

def stop_api_service():
    global _uvicorn_thread
    if _uvicorn_thread["th"]:
        _uvicorn_thread["th"].join(timeout=1.0)  # 优雅关闭
        _uvicorn_thread["th"] = None
        return "API已停止"
    return "API未运行"

def run_training(model_t: str, dataset_file, epochs: int, batch_size: int, lr: float, 
                 use_lora: bool, use_zero: bool, max_length: int) -> str:
    from app.training import Trainer
    if not dataset_file:
        return "请上传训练数据集（JSON格式）"
    try:
        trainer = Trainer(
            model_name=model_t,
            dataset_path=dataset_file.name,
            output_dir="checkpoints"
        )
        result = trainer.train(
            epochs=int(epochs),
            batch_size=int(batch_size),
            lr=lr,
            use_lora=use_lora,
            use_zero=use_zero,
            max_length=int(max_length)
        )
        return result
    except Exception as e:
        return f"训练失败：{str(e)}"

def run_compress(model_c: str, export_fp16: bool, use_int8: bool, use_int4: bool, 
                 plan_zeroquant: bool, plan_fp6: bool) -> str:
    from app.compression import Compressor
    try:
        compressor = Compressor(model_name=model_c, output_dir="compressed")
        result = compressor.run(
            export_fp16=export_fp16,
            use_int8=use_int8,
            use_int4=use_int4,
            plan_zeroquant=plan_zeroquant,
            plan_fp6=plan_fp6
        )
        return result
    except Exception as e:
        return f"压缩失败：{str(e)}"

def launch_ui():
    with gr.Blocks(title="DeepSpeed中文控制台", theme=gr.themes.Soft()) as demo:
        gr.Markdown("## 🚀 DeepSpeed中文控制台（本地）")
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 推理控制台")
                model_name_box = gr.Textbox(label="模型名称/路径", value=DEFAULT_MODEL)
                device_map_box = gr.Dropdown(choices=["auto", "cpu"], value=config["device_map"], label="设备映射")
                reload_btn = gr.Button("加载/切换模型")
                reload_log = gr.Textbox(label="加载日志", lines=2)
                reload_btn.click(fn=reload_model, inputs=[model_name_box, device_map_box], outputs=reload_log)

                prompt = gr.Textbox(label="输入文本", placeholder="例如：写一首关于秋天的诗", lines=5)
                with gr.Row():
                    temperature = gr.Slider(0.0, 2.0, value=DEFAULT_GEN_PARAMS["temperature"], step=0.05, label="温度")
                    top_k = gr.Slider(0, 200, value=DEFAULT_GEN_PARAMS["top_k"], step=1, label="Top-K")
                    top_p = gr.Slider(0.0, 1.0, value=DEFAULT_GEN_PARAMS["top_p"], step=0.01, label="Top-P")
                    repetition_penalty = gr.Slider(0.8, 2.0, value=DEFAULT_GEN_PARAMS["repetition_penalty"], step=0.05, label="重复惩罚")
                with gr.Row():
                    max_new_tokens = gr.Slider(1, 2048, value=DEFAULT_GEN_PARAMS["max_new_tokens"], step=1, label="最大生成长度")
                    do_sample = gr.Checkbox(value=True, label="采样")
                    num_return_sequences = gr.Slider(1, 4, value=1, step=1, label="返回序列数")

                gen_btn = gr.Button("开始推理")
                output = gr.Textbox(label="模型输出", lines=8)
                latency_box = gr.Textbox(label="推理耗时")
                
                @gr.render(inputs=[prompt, temperature, top_k, top_p, max_new_tokens, repetition_penalty, do_sample, num_return_sequences], triggers=[gen_btn.click])
                def _render_infer(p, t, k, pp, mnt, rp, ds, nrs):
                    text, lat = do_infer(p, t, k, pp, mnt, rp, ds, nrs)
                    return [output.update(value=text), latency_box.update(value=human_time(lat))]

            with gr.Column(scale=1):
                gr.Markdown("### 系统监控")
                metrics_box = gr.Textbox(label="资源信息", lines=20, interactive=False)
                refresh_btn = gr.Button("刷新资源信息")
                refresh_btn.click(fn=show_metrics, inputs=None, outputs=metrics_box)

        gr.Markdown("---")
        with gr.Tab("训练模块"):
            with gr.Row():
                model_t = gr.Textbox(label="模型名称/路径", value=DEFAULT_MODEL)
                dataset_file = gr.File(label="上传训练数据（JSON列表，含input/output）", file_types=[".json", ".jsonl"])  # 支持jsonl
            with gr.Row():
                epochs = gr.Slider(1, 3, value=1, step=1, label="训练轮数")
                batch_size = gr.Slider(1, 8, value=1, step=1, label="批次大小（显存紧张请用1）")
                max_length = gr.Slider(128, 2048, value=1024, step=64, label="最大序列长度")

            with gr.Row():
                lr = gr.Number(label="学习率", value=5e-5, precision=6)
                use_lora = gr.Checkbox(label="启用LoRA", value=False)
                use_zero = gr.Checkbox(label="启用DeepSpeed ZeRO（简化）", value=False)
            train_btn = gr.Button("开始训练")
            train_log = gr.Textbox(label="训练日志/结果", lines=8)
            train_btn.click(
                fn=run_training,
                inputs=[model_t, dataset_file, epochs, batch_size, lr, use_lora, use_zero, max_length],
                outputs=train_log
            )

        with gr.Tab("压缩模块"):
            model_c = gr.Textbox(label="模型名称/路径", value=DEFAULT_MODEL)
            export_fp16 = gr.Checkbox(label="导出FP16/原精度权重副本", value=True)
            use_int8 = gr.Checkbox(label="生成8bit加载提示（bitsandbytes）", value=False)
            use_int4 = gr.Checkbox(label="生成4bit加载提示（bitsandbytes）", value=False)
            plan_zeroquant = gr.Checkbox(label="记录ZeroQuant计划", value=False)
            plan_fp6 = gr.Checkbox(label="记录FP6计划", value=False)
            compress_btn = gr.Button("执行压缩/导出")
            compress_log = gr.Textbox(label="压缩日志", lines=10)
            compress_btn.click(
                fn=run_compress,
                inputs=[model_c, export_fp16, use_int8, use_int4, plan_zeroquant, plan_fp6],
                outputs=compress_log
            )

        with gr.Tab("API服务"):
            host = gr.Textbox(label="Host", value="0.0.0.0")
            port = gr.Number(label="Port", value=8000, precision=0)
            api_btn = gr.Button("启动本地API（FastAPI）")
            stop_api_btn = gr.Button("停止API服务")  # 新增停止按钮
            api_log = gr.Textbox(label="状态", lines=3)
            
            api_btn.click(fn=start_api_service, inputs=[host, port], outputs=api_log)
            stop_api_btn.click(fn=stop_api_service, inputs=None, outputs=api_log)  # 绑定停止函数

        gr.Markdown("提示：推理默认优先使用DeepSpeed MII，不可用时自动回退到Transformers。训练为轻量示例，复杂场景请按需扩展。")

    demo.launch()

if __name__ == "__main__":
    launch_ui()
