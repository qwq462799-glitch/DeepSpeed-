import os
import json
import math
from typing import Optional
from app.utils import try_import, save_json

class Trainer:
    """
    轻量训练器：基于Hugging Face Trainer，支持可选LoRA、可选DeepSpeed Zero。
    注意：真实大模型训练需充足GPU资源与更完整配置。
    """

    def __init__(self, model_name: str, dataset_path: str, output_dir: str = "checkpoints"):
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.output_dir = output_dir

    def _load_dataset(self):
        # 支持JSON和JSONL格式
        if self.dataset_path.endswith((".json", ".jsonl")):
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                if self.dataset_path.endswith(".jsonl"):
                    data = [json.loads(line) for line in f]  # JSONL按行解析
                else:
                    data = json.load(f)  # JSON整体解析
        else:
            # 支持HuggingFace数据集格式
            datasets = try_import("datasets")
            if datasets is None:
                raise RuntimeError("需要安装datasets才能加载非JSON数据集")
            ds = datasets.load_dataset("json", data_files=self.dataset_path) if self.dataset_path.endswith(".json") else datasets.load_dataset(self.dataset_path)
            data = ds["train"].to_list()

        texts = []
        for item in data:
            inp = item.get("input", "")
            out = item.get("output", "")
            # 简单指令拼接
            texts.append(f"用户：{inp}\n助手：{out}")
        return texts

    def _maybe_build_lora(self, model, lora_r=8, lora_alpha=16, lora_dropout=0.05):
        peft = try_import("peft")
        if peft is None:
            raise RuntimeError("启用LoRA需要安装peft：pip install peft>=0.11.0")
        LoraConfig = peft.LoraConfig
        get_peft_model = peft.get_peft_model
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # 常见投影层
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, config)
        return model

    def _maybe_build_deepspeed_config(self, use_zero: bool, save_path: str) -> Optional[str]:
        if not use_zero:
            return None
        ds_cfg = {
            "zero_optimization": {
                "stage": 2,
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "reduce_scatter": True,
                "reduce_bucket_size": 2e8,
                "overlap_comm": True,
                "contiguous_gradients": True
            },
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 4,
            "fp16": {"enabled": True}
        }
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_json(ds_cfg, save_path)
        return save_path

    def train(
        self,
        epochs: int = 1,
        batch_size: int = 1,
        lr: float = 5e-5,
        use_lora: bool = False,
        use_zero: bool = False,
        max_length: int = 1024
    ) -> str:
        transformers = try_import("transformers")
        datasets = try_import("datasets")
        torch = try_import("torch")

        if transformers is None or datasets is None:
            raise RuntimeError("需要安装transformers和datasets才能训练：pip install transformers datasets")

        AutoTokenizer = transformers.AutoTokenizer
        AutoModelForCausalLM = transformers.AutoModelForCausalLM
        Trainer = transformers.Trainer
        TrainingArguments = transformers.TrainingArguments

        texts = self._load_dataset()
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        def encode(example):
            out = tokenizer(
                example["text"],
                truncation=True,
                max_length=max_length,
                padding="max_length"
            )
            out["labels"] = out["input_ids"].copy()
            return out

        ds = datasets.Dataset.from_dict({"text": texts}).map(encode, batched=False, remove_columns=["text"])

        # 加载基础模型
        model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True)
        if use_lora:
            model = self._maybe_build_lora(model)

        # DeepSpeed配置（可选）
        deepspeed_config_path = self._maybe_build_deepspeed_config(use_zero, os.path.join(self.output_dir, "ds_config.json"))

        args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=lr,
            logging_steps=10,
            save_steps=100,
            save_total_limit=2,
            fp16=True,
            deepspeed=deepspeed_config_path,
            gradient_accumulation_steps=1,
            report_to=[],
            optim="adamw_torch"
        )

        trainer = Trainer(model=model, args=args, train_dataset=ds)
        trainer.train()
        model.save_pretrained(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)
        return f"训练完成：输出目录 = {self.output_dir}"
