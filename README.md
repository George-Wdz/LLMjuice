# LLMjuice PDF文献处理系统

一个基于Flask的Web应用，专门用于PDF文献处理、OCR识别、数据切分、问答对生成以及LLM模型微调效果评测。

## 🚀 功能特性

### 核心功能
- **PDF文献处理**: 支持批量上传PDF文件，自动进行OCR识别
- **OCR文字识别**: 集成MinerU API，将PDF转换为Markdown格式
- **智能数据切分**: 自动提取关键信息并进行语义切分
- **问答对生成**: 基于文献内容生成高质量的训练问答对
- **模型评测**: 专业的LLM微调效果对比评测系统

### 评测系统特性
- **多模型对比**: 支持微调模型 vs 基座模型的性能对比
- **智能裁判**: 使用Judge模型进行自动评分和评判
- **实时监控**: 完整的进度监控和日志显示
- **结果分析**: 详细的评分统计和结果展示
- **防偏见设计**: 采用位置随机化避免评判偏见

## 📦 安装部署

### 环境要求
- Python 3.8+
- pip
- 虚拟环境（推荐）

### 快速开始

1. **克隆项目**
```bash
git clone <repository-url>
cd LLMjuice_tashan
```

2. **创建虚拟环境**
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **配置环境变量**
创建 `.env` 文件并配置以下变量：

```env
# MinerU OCR服务配置
MinerU_KEY=your_mineru_api_key

# 问答对生成模型配置
API_KEY=your_api_key
BASE_URL=https://api.example.com/v1
MODEL_NAME=your_model_name

# 评测系统配置（可选）
FT_API_URL=https://api.example.com/v1
FT_API_KEY=your_finetuned_model_key
FT_MODEL_NAME=your_finetuned_model_name
BASE_API_URL=https://api.example.com/v1
BASE_API_KEY=your_base_model_key
BASE_MODEL_NAME=your_base_model_name
```

5. **启动应用**
```bash
python app.py
```

6. **访问系统**
打开浏览器访问: `http://localhost:5000`

## 📖 使用指南

### 文献处理流程

1. **上传PDF文件**
   - 访问主页
   - 点击"选择文件"上传PDF（支持批量上传）
   - 单个文件最大50MB

2. **配置API密钥**
   - 进入"配置"页面
   - 填写MinerU API Key和生成模型的API信息
   - 保存配置

3. **开始处理**
   - 返回主页
   - 点击"开始处理"按钮
   - 系统将自动执行：OCR识别 → 数据切分 → 问答对生成

4. **下载结果**
   - 处理完成后，可下载生成的`train_final.jsonl`文件

### 模型评测使用

1. **进入评测页面**
   - 点击导航栏中的"模型评测"

2. **配置三个模型**
   - **微调后的模型**: 填写API URL、API Key、Model Name
   - **基座模型**: 填写API URL、API Key、Model Name
   - **Judge模型**: 填写API URL、API Key、Model Name

3. **设置评测参数**
   - 确认数据集文件路径（默认：`./data/train_data/train_final.jsonl`）
   - 设置抽样数量N（建议5-20条测试，50-100条正式评估）

4. **开始评测**
   - 点击"开始评估"
   - 实时监控进度和日志
   - 等待评测完成

5. **查看结果**
   - 查看两个模型的平均分对比
   - 浏览详细的评估结果表格
   - 点击每行可查看详细信息

## 🔧 API接口说明

### 文献处理相关

- `POST /upload`: 上传PDF文件
- `GET /files`: 获取文件列表
- `POST /process`: 开始处理流程
- `GET /status`: 获取处理状态
- `GET /download/<path:filename>`: 下载文件

### 评测系统相关

- `GET /api/dataset_info`: 获取数据集信息
- `POST /api/start_evaluation`: 启动评测任务
- `GET /api/evaluation_progress/<task_id>`: 获取评测进度
- `GET /api/evaluation_results/<task_id>`: 获取评测结果
- `GET /api/recent_evaluation_results`: 获取最近评测结果
- `POST /api/save_evaluation_config`: 保存评测配置

## 📁 项目结构

```
LLMjuice_tashan/
├── app.py                      # Flask主应用
├── batch_ocr.py               # OCR批处理脚本
├── data_split.py              # 数据切分脚本
├── data_generatefinal.py      # 问答对生成脚本
├── evaluation_engine.py       # 评测引擎（完整版）
├── evaluation_engine_simple.py # 评测引擎（简化版）
├── model_evaluate.py          # 模型评测脚本
├── requirements.txt           # Python依赖
├── .env                       # 环境变量配置文件
├── templates/                 # HTML模板
│   ├── base.html             # 基础模板
│   ├── index.html            # 主页模板
│   ├── config.html           # 配置页面模板
│   ├── generation_config.html # 生成配置模板
│   ├── evaluation.html       # 评测页面模板
│   ├── 404.html              # 404错误页面
│   └── 500.html              # 500错误页面
├── data/                      # 数据目录
│   ├── pdf/                  # 原始PDF文件
│   ├── markdown/             # OCR识别结果
│   ├── split/                # 切分后数据
│   ├── train_data/           # 训练数据
│   └── evaluate/             # 评测结果
└── static/                    # 静态资源
    └── uploads/              # 上传文件
```

## ⚙️ 配置说明

### OCR配置（MinerU）
- 获取API Key: 访问 https://mineru.net/apiManage/docs
- 在`.env`文件中配置`MinerU_KEY`

### 生成模型配置
- 支持OpenAI兼容格式的API
- 需要配置`API_KEY`、`BASE_URL`、`MODEL_NAME`

### 评测系统配置
所有模型名称现在均为用户手动填写，支持任意模型：

- **微调模型**: 经过微调的模型
- **基座模型**: 微调前的原始模型
- **Judge模型**: 用于改写Query和评分的裁判模型

## 🔍 评测算法说明

### 评测流程
1. **数据准备**: 从数据集中随机抽取N条原始问答对
2. **Query改写**: 使用Judge模型对原始Query进行语义改写
3. **双盲推理**: 并发调用微调模型和基座模型生成回答
4. **位置随机化**: 随机交换两个模型的回答顺序以避免位置偏见
5. **裁判打分**: Judge模型对两个回答进行0-10分评分
6. **结果映射**: 将评分结果映射回正确的模型身份

### 防偏见机制
- 采用双盲测试，模型身份对Judge模型不可见
- 随机交换回答位置，避免位置偏见
- 强制JSON格式输出，确保评分结构化

## 🛠️ 开发说明

### 添加新功能
1. 在`app.py`中添加路由
2. 在`templates/`目录添加HTML模板
3. 如需后端处理，编写相应的Python脚本

### 自定义评测指标
修改`evaluation_engine_simple.py`中的评分逻辑和Prompt模板

### 扩展数据源
修改数据处理脚本以支持更多输入格式

## 📝 更新日志

### v1.0.0
- 实现PDF文献处理全流程
- 集成MinerU OCR服务
- 实现问答对自动生成
- 添加专业的模型评测系统
- 支持实时进度监控和结果展示

### 最新更新
- ✅ 所有Model Name字段改为用户手动填写
- ✅ 支持任意模型名称输入
- ✅ 优化用户界面和交互体验

## 🐛 故障排除

### 常见问题

1. **OCR识别失败**
   - 检查MinerU API Key是否正确
   - 确认PDF文件格式和大小

2. **问答对生成失败**
   - 检查生成模型的API配置
   - 确认网络连接正常

3. **评测任务失败**
   - 检查三个模型的API配置是否完整
   - 确认数据集文件存在

4. **文件上传失败**
   - 检查文件大小是否超过50MB限制
   - 确认文件格式为PDF

### 日志查看
- 应用日志: `webapp.log`
- 评测日志: `./data/evaluate/evaluation_log_*.jsonl`

## 📄 许可证

本项目遵循MIT许可证。

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进项目！

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交Issue: [GitHub Issues](link-to-issues)
- 邮箱: [kimistar@foxmail.com]

---

**注意**: 请妥善保管API密钥，不要将其提交到版本控制系统。