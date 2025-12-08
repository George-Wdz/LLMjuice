#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLMjuice Web Application
一个用于PDF文献处理和问答对生成的Web界面
集成OCR、数据处理和训练数据生成功能
"""

import os
import json
import threading
import time
import subprocess
import shutil
import asyncio
import random
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash
from werkzeug.utils import secure_filename
import logging
from dotenv import load_dotenv, set_key

# 初始化Flask应用
app = Flask(__name__)
app.secret_key = 'llmjuice_secret_key_2025'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('webapp.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 全局变量
processing_status = {
    'current_step': '',
    'progress': 0,
    'total_steps': 0,
    'message': '',
    'is_processing': False,
    'start_time': None,
    'files_processed': [],
    'error': None
}

# 确保目录存在
def ensure_directories():
    """创建必要的目录"""
    directories = [
        'data/pdf',
        'data/markdown',
        'data/split',
        'data/train_data',
        'uploads',
        'static/uploads'
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def get_processing_steps():
    """获取处理步骤"""
    return [
        {
            'id': 'upload',
            'name': 'PDF上传',
            'description': '上传PDF文件到处理队列'
        },
        {
            'id': 'ocr',
            'name': 'OCR识别',
            'description': '使用MinerU进行OCR文字识别'
        },
        {
            'id': 'split',
            'name': '数据切分',
            'description': '将识别结果切分成小片段'
        },
        {
            'id': 'generate',
            'name': '生成问答对',
            'description': '基于片段生成训练数据 (最耗时)'
        },
        {
            'id': 'complete',
            'name': '处理完成',
            'description': '所有处理步骤完成'
        }
    ]

def get_generation_params():
    """获取问答对生成参数"""
    # 强制重新加载环境变量，覆盖已存在的环境变量
    load_dotenv(override=True)

    # 获取片段总数用于计算最大生成数量
    current_max_count = 1
    try:
        processed_data_path = Path('data/split/processed_data.jsonl')
        if processed_data_path.exists():
            with open(processed_data_path, 'r', encoding='utf-8') as f:
                current_max_count = sum(1 for line in f if line.strip())
    except:
        current_max_count = 1

    # 获取生成模式，默认为手动模式
    generation_mode = os.getenv('GENERATION_MODE', 'manual').lower()

    # 获取用户设置的生成数量
    user_num_chat = os.getenv('NUM_CHAT_TO_GENERATE')

    # 根据模式确定生成数量和显示的最大值
    if generation_mode == 'auto':
        # 自动模式：承诺处理所有切片，使用特殊值表示动态最大值
        num_chat = -1  # -1 表示"处理所有切片"
        display_max_count = 999999  # 前端显示的大数值
        display_text = "自动最大化 (将处理所有切片)"
    else:
        # 手动模式：基于当前实际切片数量
        display_max_count = current_max_count
        display_text = f"当前最大值: {current_max_count}"

        # 解析用户设置的数量
        if user_num_chat and user_num_chat.lower() == 'max':
            num_chat = current_max_count
        else:
            try:
                num_chat = int(user_num_chat) if user_num_chat else current_max_count
            except (ValueError, TypeError):
                num_chat = current_max_count

    return {
        'max_requests_per_minute': int(os.getenv('MAX_REQUESTS_PER_MINUTE', '30')),
        'num_chat_to_generate': num_chat,
        'max_chat_to_generate': display_max_count,
        'current_max_chat_to_generate': current_max_count,  # 实际当前切片数
        'generation_mode': generation_mode,
        'display_text': display_text,
        'num_turn_ratios': [1, 0, 0, 0, 0]  # 固定1轮对话
    }

def save_generation_params(params):
    """保存问答对生成参数到环境变量"""
    try:
        env_file = Path('.env')
        if not env_file.exists():
            env_file.touch()

        # 验证参数
        required_keys = ['max_requests_per_minute', 'num_chat_to_generate', 'max_chat_to_generate']
        for key in required_keys:
            if key not in params:
                logger.error(f"缺少必需参数: {key}")
                return False

        # 只保存用户可配置的参数
        set_key('.env', 'MAX_REQUESTS_PER_MINUTE', str(params['max_requests_per_minute']))

        # 保存生成模式
        generation_mode = params.get('generation_mode', 'manual')
        set_key('.env', 'GENERATION_MODE', generation_mode)

        # 根据模式保存生成数量
        if generation_mode == 'auto':
            # 自动模式：保存为'max'，表示处理所有切片
            set_key('.env', 'NUM_CHAT_TO_GENERATE', 'max')
        else:
            # 手动模式：保存实际值或'max'
            if params['num_chat_to_generate'] == -1:
                set_key('.env', 'NUM_CHAT_TO_GENERATE', 'max')
            elif params['num_chat_to_generate'] == params['max_chat_to_generate']:
                set_key('.env', 'NUM_CHAT_TO_GENERATE', 'max')
            else:
                set_key('.env', 'NUM_CHAT_TO_GENERATE', str(params['num_chat_to_generate']))

        # 固定1轮对话比例
        set_key('.env', 'NUM_TURN_RATIOS', '1,0,0,0,0')

        # 重新加载环境变量以立即生效，强制覆盖已存在的环境变量
        load_dotenv(override=True)
        logger.info(f"成功保存生成参数: max_requests={params['max_requests_per_minute']}, num_chat={params['num_chat_to_generate']}")
        return True

    except Exception as e:
        logger.error(f"保存生成参数失败: {e}")
        logger.error(f"参数详情: {params}")
        return False

def get_env_config():
    """获取环境变量配置"""
    load_dotenv()
    return {
        'MinerU_KEY': os.getenv('MinerU_KEY', ''),
        'API_KEY': os.getenv('API_KEY', ''),
        'BASE_URL': os.getenv('BASE_URL', ''),
        'MODEL_NAME': os.getenv('MODEL_NAME', '')
    }

def save_env_config(config):
    """保存环境变量配置到.env文件"""
    try:
        env_file = Path('.env')
        if not env_file.exists():
            env_file.touch()

        for key, value in config.items():
            set_key('.env', key, value)

        # 重新加载环境变量
        load_dotenv()
        return True
    except Exception as e:
        logger.error(f"保存配置失败: {e}")
        return False

def get_file_list(directory, file_type='pdf'):
    """获取指定目录下的文件列表"""
    try:
        dir_path = Path(f'data/{directory}')
        if not dir_path.exists():
            return []

        files = []
        for file_path in dir_path.rglob(f'*.{file_type}'):
            stat = file_path.stat()
            files.append({
                'name': file_path.name,
                'path': str(file_path),
                'size': f"{stat.st_size / 1024 / 1024:.2f} MB",
                'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                'relative_path': str(file_path.relative_to('data'))
            })

        return sorted(files, key=lambda x: x['modified'], reverse=True)
    except Exception as e:
        logger.error(f"获取文件列表失败: {e}")
        return []

def get_processing_results():
    """获取处理结果文件"""
    results = {
        'markdown_files': get_file_list('markdown', 'md'),
        'split_files': get_file_list('split', 'jsonl'),
        'train_files': get_file_list('train_data', 'jsonl')
    }

    # 检查是否有最终的训练数据文件
    train_final_path = Path('data/train_data/train_final.jsonl')
    if train_final_path.exists():
        stat = train_final_path.stat()
        results['train_final'] = {
            'name': 'train_final.jsonl',
            'path': str(train_final_path),
            'size': f"{stat.st_size / 1024 / 1024:.2f} MB",
            'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
            'download_url': '/download/train_data/train_final.jsonl'
        }

    return results

def run_processing_script(script_name, *args):
    """运行处理脚本"""
    try:
        script_path = Path(f"{script_name}")
        if not script_path.exists():
            raise FileNotFoundError(f"脚本文件不存在: {script_name}")

        cmd = ['python', str(script_path)] + list(args)
        logger.info(f"执行命令: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=3600  # 1小时超时
        )

        if result.returncode != 0:
            error_msg = f"脚本执行失败: {result.stderr}"
            logger.error(error_msg)
            raise Exception(error_msg)

        logger.info(f"脚本执行成功: {result.stdout}")
        return True

    except subprocess.TimeoutExpired:
        error_msg = f"脚本执行超时: {script_name}"
        logger.error(error_msg)
        raise Exception(error_msg)
    except Exception as e:
        logger.error(f"执行脚本时发生错误: {e}")
        raise

def processing_worker():
    """后台处理工作线程"""
    try:
        processing_status['is_processing'] = True
        processing_status['start_time'] = datetime.now()
        processing_status['error'] = None

        steps = get_processing_steps()
        total_steps = len(steps) - 1  # 减去complete步骤
        current_step_index = 0

        # 检查配置
        config = get_env_config()
        if not all([config['MinerU_KEY'], config['API_KEY']]):
            raise Exception("请先配置API密钥")

        # 获取生成参数
        gen_params = get_generation_params()
        logger.info(f"使用生成参数: {gen_params}")

        # 步骤1: OCR处理
        processing_status['current_step'] = 'ocr'
        processing_status['progress'] = (current_step_index / total_steps) * 100
        processing_status['message'] = '正在进行OCR识别...'

        run_processing_script('batch_ocr.py')
        current_step_index += 1

        # 步骤2: 数据切分
        processing_status['current_step'] = 'split'
        processing_status['progress'] = (current_step_index / total_steps) * 100
        processing_status['message'] = '正在进行数据切分...'

        run_processing_script('data_split.py')
        current_step_index += 1

        # 步骤3: 生成问答对 (使用固定参数)
        processing_status['current_step'] = 'generate'
        processing_status['progress'] = (current_step_index / total_steps) * 100
        processing_status['message'] = f'正在生成问答对 (并发: {gen_params["max_requests_per_minute"]}/分钟, 生成数量: {gen_params["num_chat_to_generate"]})...'

        # 构建命令参数 - 使用固定参数
        cmd_args = [
            '--reference_filepaths', './data/split/processed_data.jsonl',
            '--save_filepath', './data/train_data/train_final.jsonl',
            '--num_chat_to_generate', str(gen_params['num_chat_to_generate']),
            '--language', 'zh',
            '--num_turn_ratios', '1', '0', '0', '0', '0'
        ]

        # 只有用户设置了并发数才添加并发参数
        if gen_params['max_requests_per_minute'] != 30:  # 30是新的默认值
            cmd_args.extend([
                '--max_requests_per_minute', str(gen_params['max_requests_per_minute'])
            ])

        run_processing_script('data_generatefinal.py', *cmd_args)
        current_step_index += 1

        # 完成
        processing_status['current_step'] = 'complete'
        processing_status['progress'] = 100
        processing_status['message'] = '🎉 所有处理步骤完成！'
        processing_status['is_processing'] = False

        logger.info("所有处理步骤完成")

    except Exception as e:
        logger.error(f"处理过程中发生错误: {e}")
        processing_status['error'] = str(e)
        processing_status['message'] = f'❌ 处理失败: {str(e)}'
        processing_status['is_processing'] = False

# 路由定义
@app.route('/')
def index():
    """主页"""
    return render_template('index.html',
                         steps=get_processing_steps(),
                         config=get_env_config(),
                         generation_config=get_generation_params(),
                         results=get_processing_results())

@app.route('/config', methods=['GET', 'POST'])
def config():
    """配置页面"""
    if request.method == 'POST':
        config_data = {
            'MinerU_KEY': request.form.get('MinerU_KEY', ''),
            'API_KEY': request.form.get('API_KEY', ''),
            'BASE_URL': request.form.get('BASE_URL', ''),
            'MODEL_NAME': request.form.get('MODEL_NAME', '')
        }

        if save_env_config(config_data):
            flash('配置保存成功！', 'success')
        else:
            flash('配置保存失败！', 'error')

        return redirect(url_for('config'))

    return render_template('config.html', config=get_env_config())

@app.route('/generation_config', methods=['GET', 'POST'])
def generation_config():
    """问答对生成参数配置页面"""
    if request.method == 'POST':
        try:
            # 获取表单数据
            max_requests = int(request.form.get('max_requests_per_minute', 30))
            generation_mode = request.form.get('generation_mode', 'manual').lower()
            num_chat = request.form.get('num_chat_to_generate', 'max')

            # 获取当前最大生成数量
            current_params = get_generation_params()
            max_chat = current_params['max_chat_to_generate']

            # 根据生成模式确定生成数量
            if generation_mode == 'auto':
                # 自动模式：使用特殊值 -1 表示"处理所有切片"
                num_chat_value = -1
            else:
                # 手动模式：解析用户输入的数量
                if num_chat == 'max':
                    num_chat_value = max_chat
                else:
                    num_chat_value = int(num_chat)
                    if num_chat_value > max_chat:
                        num_chat_value = max_chat
                    elif num_chat_value < 1:
                        num_chat_value = 1

            # 构建参数字典
            params = {
                'max_requests_per_minute': max_requests,
                'num_chat_to_generate': num_chat_value,
                'max_chat_to_generate': max_chat,
                'generation_mode': generation_mode,
                'num_turn_ratios': [1, 0, 0, 0, 0]  # 固定1轮对话
            }

            if save_generation_params(params):
                flash('生成参数保存成功！', 'success')
            else:
                flash('生成参数保存失败！', 'error')

        except ValueError as e:
            flash(f'参数格式错误: {str(e)}', 'error')
        except Exception as e:
            flash(f'保存失败: {str(e)}', 'error')

        return redirect(url_for('generation_config'))

    return render_template('generation_config.html', params=get_generation_params())

@app.route('/upload', methods=['POST'])
def upload_file():
    """文件上传"""
    if 'files' not in request.files:
        return jsonify({'error': '没有选择文件'}), 400

    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({'error': '没有选择文件'}), 400

    uploaded_files = []
    pdf_dir = Path('data/pdf')
    pdf_dir.mkdir(parents=True, exist_ok=True)

    for file in files:
        if file and file.filename.lower().endswith('.pdf'):
            filename = secure_filename(file.filename)
            file_path = pdf_dir / filename

            # 避免文件名冲突
            counter = 1
            original_name = filename
            while file_path.exists():
                name_parts = original_name.rsplit('.', 1)
                if len(name_parts) == 2:
                    filename = f"{name_parts[0]}_{counter}.{name_parts[1]}"
                else:
                    filename = f"{original_name}_{counter}"
                file_path = pdf_dir / filename
                counter += 1

            file.save(file_path)
            uploaded_files.append(filename)

    return jsonify({
        'success': True,
        'files': uploaded_files,
        'message': f'成功上传 {len(uploaded_files)} 个PDF文件'
    })

@app.route('/files')
def get_files():
    """获取文件列表"""
    pdf_files = get_file_list('pdf', 'pdf')
    return jsonify({'files': pdf_files})

@app.route('/process', methods=['POST'])
def start_processing():
    """开始处理"""
    if processing_status['is_processing']:
        return jsonify({'error': '正在处理中，请等待当前处理完成'}), 400

    # 检查是否有PDF文件
    pdf_files = get_file_list('pdf', 'pdf')
    if not pdf_files:
        return jsonify({'error': '没有找到PDF文件，请先上传PDF文件'}), 400

    # 检查配置
    config = get_env_config()
    if not all([config['MinerU_KEY'], config['API_KEY']]):
        return jsonify({'error': '请先配置API密钥'}), 400

    # 重置状态
    processing_status.update({
        'current_step': '',
        'progress': 0,
        'message': '开始处理...',
        'is_processing': True,
        'start_time': None,
        'files_processed': [],
        'error': None
    })

    # 启动后台处理线程
    thread = threading.Thread(target=processing_worker)
    thread.daemon = True
    thread.start()

    return jsonify({'success': True, 'message': '处理已开始'})

@app.route('/status')
def get_status():
    """获取处理状态"""
    status = processing_status.copy()
    if status['start_time']:
        status['elapsed_time'] = str(datetime.now() - status['start_time']).split('.')[0]

    # 添加文件信息
    status['files'] = {
        'pdf_count': len(get_file_list('pdf', 'pdf')),
        'results': get_processing_results()
    }

    return jsonify(status)

@app.route('/download/<path:filename>')
def download_file(filename):
    """下载文件"""
    try:
        file_path = Path('data') / filename
        if not file_path.exists():
            logger.error(f"文件不存在: {file_path}")
            return jsonify({'error': f'文件不存在: {filename}'}), 404

        logger.info(f"下载文件: {file_path}")
        return send_file(file_path, as_attachment=True, download_name=file_path.name)
    except Exception as e:
        logger.error(f"下载文件失败: {e}")
        return jsonify({'error': '下载失败'}), 500

@app.route('/delete_file/<path:filename>', methods=['POST'])
def delete_file(filename):
    """删除文件"""
    try:
        file_path = Path('data') / filename
        if file_path.exists():
            file_path.unlink()
            return jsonify({'success': True, 'message': '文件删除成功'})
        else:
            return jsonify({'error': '文件不存在'}), 404
    except Exception as e:
        logger.error(f"删除文件失败: {e}")
        return jsonify({'error': '删除失败'}), 500

@app.route('/api/generation_config')
def api_generation_config():
    """获取生成参数配置API"""
    try:
        return jsonify(get_generation_params())
    except Exception as e:
        logger.error(f"获取生成参数失败: {e}")
        return jsonify({'error': '获取参数失败'}), 500

# 错误处理
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': '文件太大，请上传小于50MB的文件'}), 413

@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500

# 评估页面路由
@app.route('/evaluation')
def evaluation():
    """LLM微调效果对比评测页面"""
    return render_template('evaluation.html')

# 评估相关API
@app.route('/api/dataset_info')
def get_dataset_info():
    """获取数据集信息"""
    try:
        dataset_path = Path('./data/train_data/train_final.jsonl')
        if not dataset_path.exists():
            return jsonify({'error': '数据集文件不存在'})

        with open(dataset_path, 'r', encoding='utf-8') as f:
            total_pairs = sum(1 for line in f if line.strip())

        return jsonify({'total_pairs': total_pairs})
    except Exception as e:
        logger.error(f"获取数据集信息失败: {e}")
        return jsonify({'error': str(e)}), 500

# 评估任务管理
evaluation_tasks = {}

@app.route('/api/start_evaluation', methods=['POST'])
def start_evaluation():
    """启动评估任务"""
    try:
        config = request.json
        logger.info(f"收到评估启动请求: {config}")

        # 验证配置
        required_fields = ['ft_model', 'base_model', 'judge_model', 'sample_count']
        for field in required_fields:
            if field not in config:
                logger.error(f"缺少必需字段: {field}")
                return jsonify({'success': False, 'message': f'缺少必需字段: {field}'})

        # 生成任务ID
        task_id = f"eval_{int(time.time())}"
        logger.info(f"生成任务ID: {task_id}")

        # 创建任务
        from evaluation_engine import EvaluationTask
        task = EvaluationTask(task_id, config)
        evaluation_tasks[task_id] = task
        logger.info(f"任务已创建，当前任务数量: {len(evaluation_tasks)}")

        # 异步启动任务
        import threading

        def run_task():
            try:
                logger.info(f"开始执行任务: {task_id}")
                # 在新线程中创建事件循环
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(task.run())
                logger.info(f"任务执行完成: {task_id}")
            except Exception as e:
                logger.error(f"评估任务执行失败: {e}")
                task.status = 'failed'
                task.error = str(e)

        thread = threading.Thread(target=run_task)
        thread.daemon = True
        thread.start()
        logger.info(f"任务线程已启动: {task_id}")

        return jsonify({'success': True, 'task_id': task_id})

    except Exception as e:
        logger.error(f"启动评估失败: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/evaluation_progress/<task_id>')
def get_evaluation_progress(task_id):
    """获取评估进度"""
    try:
        logger.info(f"查询任务进度: {task_id}, 当前任务列表: {list(evaluation_tasks.keys())}")

        if task_id not in evaluation_tasks:
            logger.error(f"任务不存在: {task_id}")
            return jsonify({'error': '任务不存在'}), 404

        task = evaluation_tasks[task_id]
        progress_data = {
            'status': task.status,
            'progress': task.progress,
            'current_step': task.current_step,
            'processed': task.processed,
            'total': task.total,
            'log_message': task.get_latest_log()
        }
        logger.info(f"返回进度数据: {progress_data}")
        return progress_data

    except Exception as e:
        logger.error(f"获取进度失败: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/evaluation_results/<task_id>')
def get_evaluation_results(task_id):
    """获取评估结果"""
    try:
        if task_id not in evaluation_tasks:
            return jsonify({'error': '任务不存在'}), 404

        task = evaluation_tasks[task_id]
        if task.status != 'completed':
            return jsonify({'error': '任务尚未完成'}), 400

        return jsonify({
            'success': True,
            'results': task.results,
            'statistics': task.statistics
        })

    except Exception as e:
        logger.error(f"获取结果失败: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/recent_evaluation_results')
def get_recent_evaluation_results():
    """获取最近的评估结果"""
    try:
        # 查找最近的评估结果文件
        evaluate_dir = Path('./data/evaluate')
        if not evaluate_dir.exists():
            return jsonify({'success': False, 'message': '没有找到评估结果目录'})

        # 查找最新的日志文件
        log_files = list(evaluate_dir.glob('evaluation_log_*.jsonl'))
        if not log_files:
            return jsonify({'success': False, 'message': '没有找到评估结果文件'})

        latest_file = max(log_files, key=lambda f: f.stat().st_mtime)

        # 读取结果
        results = []
        with open(latest_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))

        return jsonify({
            'success': True,
            'results': results,
            'file_name': latest_file.name
        })

    except Exception as e:
        logger.error(f"获取最近结果失败: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/evaluation_config', methods=['GET'])
def get_evaluation_config():
    """获取评估配置（已保存的）"""
    try:
        load_dotenv()

        # 从环境变量读取已保存的配置信息（不包括API Key）
        config = {
            'ft_model': {
                'api_url': os.getenv('FT_API_URL', ''),
                'model_name': os.getenv('FT_MODEL_NAME', '')
            },
            'base_model': {
                'api_url': os.getenv('BASE_API_URL', ''),
                'model_name': os.getenv('BASE_MODEL_NAME', '')
            },
            'judge_model': {
                'api_url': os.getenv('JUDGE_API_URL', os.getenv('BASE_URL', '')),  # 复用主配置作为后备
                'model_name': os.getenv('JUDGE_MODEL_NAME', os.getenv('MODEL_NAME', ''))
            }
        }
        return jsonify({'config': config})
    except Exception as e:
        logger.error(f"获取评估配置失败: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/env_config', methods=['GET'])
def api_get_env_config():
    """获取.env配置"""
    try:
        load_dotenv()

        # 读取.env中的配置，支持多种环境变量名称
        api_url = os.getenv('BASE_URL') or os.getenv('API_URL', '')
        api_key = os.getenv('API_KEY', '')
        model_name = os.getenv('MODEL_NAME', '')

        config = {
            'success': True,
            'api_url': api_url,
            'api_key': api_key,
            'model_name': model_name
        }

        return jsonify(config)
    except Exception as e:
        logger.error(f"获取.env配置失败: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/save_evaluation_config', methods=['POST'])
def save_evaluation_config():
    """保存评估配置到.env文件"""
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'message': '无效的请求数据'}), 400

        # 读取当前.env文件内容
        env_path = '.env'
        env_content = ''
        if os.path.exists(env_path):
            with open(env_path, 'r', encoding='utf-8') as f:
                env_content = f.read()

        # 保存所有模型配置
        lines = env_content.split('\n')
        new_lines = []

        # 更新或添加所有模型配置
        for line in lines:
            if line.startswith('FT_API_URL='):
                new_lines.append(f"FT_API_URL={data.get('ft_model', {}).get('api_url', '')}")
            elif line.startswith('FT_API_KEY='):
                new_lines.append(f"FT_API_KEY={data.get('ft_model', {}).get('api_key', '')}")
            elif line.startswith('FT_MODEL_NAME='):
                new_lines.append(f"FT_MODEL_NAME={data.get('ft_model', {}).get('model_name', '')}")
            elif line.startswith('BASE_API_URL='):
                new_lines.append(f"BASE_API_URL={data.get('base_model', {}).get('api_url', '')}")
            elif line.startswith('BASE_API_KEY='):
                new_lines.append(f"BASE_API_KEY={data.get('base_model', {}).get('api_key', '')}")
            elif line.startswith('BASE_MODEL_NAME='):
                new_lines.append(f"BASE_MODEL_NAME={data.get('base_model', {}).get('model_name', '')}")
            elif line.startswith('JUDGE_API_URL='):
                new_lines.append(f"JUDGE_API_URL={data.get('judge_model', {}).get('api_url', '')}")
            elif line.startswith('JUDGE_API_KEY='):
                new_lines.append(f"JUDGE_API_KEY={data.get('judge_model', {}).get('api_key', '')}")
            elif line.startswith('JUDGE_MODEL_NAME='):
                new_lines.append(f"JUDGE_MODEL_NAME={data.get('judge_model', {}).get('model_name', '')}")
            else:
                new_lines.append(line)

        # 如果没有找到相关配置行，在文件末尾添加
        config_keys = ['FT_API_URL', 'FT_API_KEY', 'FT_MODEL_NAME',
                      'BASE_API_URL', 'BASE_API_KEY', 'BASE_MODEL_NAME',
                      'JUDGE_API_URL', 'JUDGE_API_KEY', 'JUDGE_MODEL_NAME']
        existing_keys = []

        for line in new_lines:
            if '=' in line:
                existing_keys.append(line.split('=')[0])

        for key in config_keys:
            if key not in existing_keys:
                if key == 'FT_API_URL':
                    new_lines.append(f"FT_API_URL={data.get('ft_model', {}).get('api_url', '')}")
                elif key == 'FT_API_KEY':
                    new_lines.append(f"FT_API_KEY={data.get('ft_model', {}).get('api_key', '')}")
                elif key == 'FT_MODEL_NAME':
                    new_lines.append(f"FT_MODEL_NAME={data.get('ft_model', {}).get('model_name', '')}")
                elif key == 'BASE_API_URL':
                    new_lines.append(f"BASE_API_URL={data.get('base_model', {}).get('api_url', '')}")
                elif key == 'BASE_API_KEY':
                    new_lines.append(f"BASE_API_KEY={data.get('base_model', {}).get('api_key', '')}")
                elif key == 'BASE_MODEL_NAME':
                    new_lines.append(f"BASE_MODEL_NAME={data.get('base_model', {}).get('model_name', '')}")
                elif key == 'JUDGE_API_URL':
                    new_lines.append(f"JUDGE_API_URL={data.get('judge_model', {}).get('api_url', '')}")
                elif key == 'JUDGE_API_KEY':
                    new_lines.append(f"JUDGE_API_KEY={data.get('judge_model', {}).get('api_key', '')}")
                elif key == 'JUDGE_MODEL_NAME':
                    new_lines.append(f"JUDGE_MODEL_NAME={data.get('judge_model', {}).get('model_name', '')}")

        # 写入.env文件
        with open(env_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_lines))

        logger.info("评估配置已保存到.env文件")
        return jsonify({'success': True, 'message': '配置已保存'})

    except Exception as e:
        logger.error(f"保存评估配置失败: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


# 初始化
if __name__ == '__main__':
    ensure_directories()
    logger.info("LLMjuice Web Application 启动")
    logger.info("访问地址: http://localhost:5000")

    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)