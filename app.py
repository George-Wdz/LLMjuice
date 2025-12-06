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
            'description': '基于片段生成训练数据'
        },
        {
            'id': 'complete',
            'name': '处理完成',
            'description': '所有处理步骤完成'
        }
    ]

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
        current_step_index = 0

        # 检查配置
        config = get_env_config()
        if not all([config['MinerU_KEY'], config['API_KEY']]):
            raise Exception("请先配置API密钥")

        # 步骤1: OCR处理
        processing_status['current_step'] = 'ocr'
        processing_status['progress'] = (current_step_index + 1) / len(steps) * 100
        processing_status['message'] = '正在进行OCR识别...'

        run_processing_script('batch_ocr.py')
        current_step_index += 1

        # 步骤2: 数据切分
        processing_status['current_step'] = 'split'
        processing_status['progress'] = (current_step_index + 1) / len(steps) * 100
        processing_status['message'] = '正在进行数据切分...'

        run_processing_script('data_split.py')
        current_step_index += 1

        # 步骤3: 生成问答对
        processing_status['current_step'] = 'generate'
        processing_status['progress'] = (current_step_index + 1) / len(steps) * 100
        processing_status['message'] = '正在生成问答对...'

        run_processing_script('data_generatefinal.py',
                            '--reference_filepaths', './data/split/processed_data.jsonl',
                            '--save_filepath', './data/train_data/train_final.jsonl',
                            '--num_chat_to_generate', '1',
                            '--language', 'zh',
                            '--num_turn_ratios', '1', '0', '0', '0', '0')
        current_step_index += 1

        # 完成
        processing_status['current_step'] = 'complete'
        processing_status['progress'] = 100
        processing_status['message'] = '处理完成！'
        processing_status['is_processing'] = False

        logger.info("所有处理步骤完成")

    except Exception as e:
        logger.error(f"处理过程中发生错误: {e}")
        processing_status['error'] = str(e)
        processing_status['message'] = f'处理失败: {str(e)}'
        processing_status['is_processing'] = False

# 路由定义
@app.route('/')
def index():
    """主页"""
    return render_template('index.html',
                         steps=get_processing_steps(),
                         config=get_env_config(),
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

# 初始化
if __name__ == '__main__':
    ensure_directories()
    logger.info("LLMjuice Web Application 启动")
    logger.info("访问地址: http://localhost:5000")

    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)