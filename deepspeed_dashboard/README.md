# DeepSpeed 推理控制台

一个基于 DeepSpeed 和 Hugging Face 生态的大模型推理、训练与压缩一体化控制台，提供 API 服务和可视化界面。

![项目架构](https://picsum.photos/id/0/800/400)

## 功能特点

- **多后端推理**：支持 DeepSpeed MII 和 Transformers 后端，自动选择最优方案
- **可视化界面**：通过 Gradio 提供直观的操作界面，支持推理、训练和模型压缩
- **API 服务**：内置 FastAPI 服务，方便集成到其他系统
- **灵活配置**：通过配置文件集中管理模型参数，支持动态调整
- **资源监控**：实时显示 CPU、内存和 GPU 使用情况
- **模型管理**：支持加载、切换不同模型，支持 8bit/4bit 量化

## 项目结构
deepspeed_dashboard/
├── app/                  # 核心功能模块
│   ├── inference.py      # 推理引擎（单例模式）
│   ├── training.py       # 模型训练模块
│   ├── compression.py    # 模型压缩与量化
│   └── utils.py          # 工具函数
├── interface/
│   └── gradio_ui.py      # 可视化界面
├── models/
│   └── model_config.yaml # 模型配置文件
├── run.py                # API 服务入口
├── requirements.txt      # 项目依赖
└── README.md             # 项目说明
## 安装指南

### 前提条件

- Python 3.8+
- CUDA 11.6+（推荐，用于 GPU 加速）
- 足够的磁盘空间（模型文件可能较大）

### 安装步骤

1. 克隆仓库（如果适用）：
   ```bash
   git clone https://github.com/qwq462799-glitch/DeepSpeed-.git
   cd deepspeed_dashboard
   ```

2. 创建并激活虚拟环境：
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. 安装依赖：
   ```bash
   # 安装核心依赖
   pip install -r requirements.txt
   
   # 可选：安装量化支持（8bit/4bit）
   pip install bitsandbytes>=0.43.0
   
   # 可选：安装LoRA训练支持
   pip install peft>=0.11.0
   ```

## 使用方法

### 1. 启动可视化界面
python interface/gradio_ui.py
界面将自动在浏览器中打开，包含以下功能区：
- 推理控制台：加载模型并进行文本生成
- 系统监控：查看CPU、内存和GPU使用情况
- 训练模块：微调模型（支持LoRA）
- 压缩模块：模型量化和优化
- API服务：启动/停止FastAPI服务

### 2. 启动独立API服务
python run.py
API服务默认运行在 `http://0.0.0.0:8000`，提供以下端点：
- `GET /status`：查看服务状态
- `POST /generate`：文本生成
- `POST /reload_model`：加载/切换模型

API文档可通过 `http://localhost:8000/docs` 访问。

### 3. 配置文件说明

`models/model_config.yaml` 包含系统默认配置：model_name: "EleutherAI/gpt-neox-20b"  # 默认模型
device_map: "auto"                     # 设备映射策略
default_generation:
  temperature: 0.7
  top_k: 50
  top_p: 0.9
  max_new_tokens: 256
  repetition_penalty: 1.0
可根据需要修改默认模型和生成参数。

## 示例

### 文本生成

1. 在可视化界面的"推理控制台"中，输入提示词：
   ```
   写一首关于秋天的诗
   ```

2. 调整生成参数（可选），点击"开始推理"

3. 查看生成结果

### 模型训练

1. 准备训练数据（JSON格式，包含input和output字段）
2. 在"训练模块"上传数据，设置训练参数
3. 点击"开始训练"，查看训练进度
4. 训练结果将保存到`checkpoints`目录

## 常见问题

1. **模型加载失败**：
   - 检查网络连接（首次加载需要下载模型）
   - 确认有足够的磁盘空间和内存
   - 对于大型模型，尝试使用"auto"设备映射或8bit量化

2. **CUDA内存不足**：
   - 减少`max_new_tokens`参数
   - 启用8bit/4bit量化
   - 尝试更小的模型

3. **API服务无法访问**：
   - 检查端口是否被占用
   - 确认防火墙设置
   - 尝试修改`host`参数为"0.0.0.0"

## 扩展与定制

- 如需添加新模型，只需在配置文件中指定模型名称或本地路径
- 可通过修改`app/inference.py`扩展推理后端
- 训练和压缩模块可根据需求调整参数或添加新功能

## 许可证

[MIT](LICENSE)


