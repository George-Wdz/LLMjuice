# LLMjuice Web Application

一个用于PDF文献处理和问答对生成的Web应用程序，集成OCR识别、数据处理和训练数据生成功能。

## 🚀 功能特性

### 核心功能
- **PDF文件上传**: 支持拖拽上传，批量处理PDF文件
- **OCR识别**: 集成MinerU API，高质量PDF文字识别
- **智能切分**: 自动过滤无关信息，切分有意义的文本片段
- **问答对生成**: 基于文本片段生成高质量训练数据
- **实时监控**: 实时显示处理进度和状态
- **文件管理**: 可视化管理PDF文件和处理结果

### 技术特性
- **现代UI**: 响应式设计，支持桌面和移动设备
- **实时更新**: 自动刷新处理状态和文件信息
- **配置管理**: Web界面管理API密钥和配置
- **错误处理**: 完善的错误提示和异常处理
- **安全上传**: 文件类型验证和大小限制

## 📋 系统要求

- Python 3.7+
- 现代浏览器 (Chrome, Firefox, Safari, Edge)
- 至少2GB可用内存
- 稳定的网络连接

## 🛠️ 安装和配置

### 1. 环境准备

```bash
# 克隆项目
cd LLMjuice_tashan

# 激活虚拟环境
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置API密钥

#### MinerU OCR API
1. 访问 [MinerU官网](https://mineru.net/apiManage/docs)
2. 注册账号并创建API密钥
3. 在Web界面的配置页面填入密钥

#### AI模型API
1. 选择AI服务提供商 (如DeepSeek, OpenAI等)
2. 获取API密钥
3. 在配置页面填入相关信息

### 3. 启动应用

```bash
# 方式1: 使用启动脚本 (推荐)
python run_web.py

# 方式2: 直接启动Flask应用
python app.py
```

### 4. 访问应用

打开浏览器访问: http://localhost:5000

## 📖 使用指南

### 首次使用

1. **配置API密钥**
   - 访问 http://localhost:5000/config
   - 填入MinerU API Key和AI API Key
   - 点击"保存配置"

2. **上传PDF文件**
   - 在主页拖拽PDF文件到上传区域
   - 或点击选择文件按钮
   - 支持批量上传

3. **开始处理**
   - 确认文件已上传成功
   - 点击"开始处理"按钮
   - 等待处理完成

4. **下载结果**
   - 处理完成后下载生成的训练数据文件
   - 可在页面查看处理统计信息

### 处理流程

```
PDF上传 → OCR识别 → 数据切分 → 问答对生成 → 完成下载
```

- **OCR识别**: 使用MinerU将PDF转换为Markdown
- **数据切分**: 过滤无关信息，切分有意义的文本片段
- **问答对生成**: 基于片段生成对话式训练数据

## 🏗️ 项目结构

```
LLMjuice_tashan/
├── app.py                 # Flask主应用
├── run_web.py            # 启动脚本
├── requirements.txt      # 依赖包列表
├── .env                  # 环境变量配置
├── templates/            # HTML模板
│   ├── base.html        # 基础模板
│   ├── index.html       # 主页模板
│   ├── config.html      # 配置页模板
│   ├── 404.html         # 404错误页
│   └── 500.html         # 500错误页
├── static/              # 静态资源
│   ├── css/
│   │   └── style.css    # 自定义样式
│   └── js/
│       └── main.js      # 主要JavaScript
├── data/                # 数据目录
│   ├── pdf/             # 上传的PDF文件
│   ├── markdown/        # OCR识别结果
│   ├── split/           # 切分后的数据
│   └── train_data/      # 训练数据
├── batch_ocr.py         # OCR处理脚本
├── data_split.py        # 数据切分脚本
└── data_generatefinal.py # 问答对生成脚本
```

## ⚙️ 配置说明

### 环境变量 (.env)

```bash
# MinerU OCR配置
MinerU_KEY="your_mineru_api_key"

# AI模型配置
API_KEY="your_ai_api_key"
BASE_URL="https://api.deepseek.com"
MODEL_NAME="deepseek-chat"
```

### 处理参数

可在 `data_generatefinal.py` 中调整以下参数:

- `num_chat_to_generate`: 生成的对话对数量
- `language`: 生成语言 ("zh" 或 "en")
- `assistant_word_count`: 回答的字数要求
- `human_word_count`: 提问的字数要求
- `num_turn_ratios`: 对话轮次比例

## 🔧 故障排除

### 常见问题

1. **上传失败**
   - 检查文件格式是否为PDF
   - 确认文件大小小于50MB
   - 检查网络连接

2. **OCR识别失败**
   - 验证MinerU API密钥是否正确
   - 确认API额度是否充足
   - 检查PDF文件质量

3. **问答对生成失败**
   - 验证AI API密钥是否正确
   - 确认模型名称和Base URL正确
   - 检查API服务是否正常

4. **处理卡住**
   - 刷新页面检查状态
   - 查看浏览器控制台错误
   - 检查服务器日志

### 日志文件

- `webapp.log`: Web应用日志
- `markdown_processor_final.log`: 数据处理日志

## 📊 性能优化

### 建议配置

- **内存**: 至少4GB，推荐8GB+
- **CPU**: 多核处理器，处理速度更快
- **网络**: 稳定的互联网连接

### 处理建议

- 单次处理PDF文件数量: 建议1-5个
- 单个PDF文件大小: 建议<20MB
- 处理时间: 根据文件大小，通常需要5-30分钟

## 🔒 安全注意事项

- API密钥请妥善保管，不要泄露
- 定期更新API密钥
- 注意API调用频率限制
- 建议在安全的环境中使用

## 📝 更新日志

### v1.0.0 (2025-12-06)
- ✅ 完整的Web界面
- ✅ PDF文件上传功能
- ✅ OCR处理集成
- ✅ 数据处理集成
- ✅ 问答对生成集成
- ✅ 实时状态监控
- ✅ 配置管理界面
- ✅ 响应式设计

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目。

## 📄 许可证

本项目采用MIT许可证。

## 🆘 支持

如果遇到问题，请:

1. 查看故障排除部分
2. 检查日志文件
3. 提交Issue描述问题
4. 提供详细的错误信息和环境信息

---

**享受使用LLMjuice PDF文献处理系统！** 🎉