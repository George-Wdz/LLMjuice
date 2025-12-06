#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试Web应用程序的功能
"""

import sys
import os
from pathlib import Path

def test_imports():
    """测试必要的包导入"""
    print("🔍 测试包导入...")

    try:
        import flask
        print("   ✅ Flask已安装")
    except ImportError:
        print("   ❌ Flask未安装")
        return False

    try:
        import requests
        print("   ✅ Requests已安装")
    except ImportError:
        print("   ❌ Requests未安装")
        return False

    try:
        from dotenv import load_dotenv
        print("   ✅ python-dotenv已安装")
    except ImportError:
        print("   ❌ python-dotenv未安装")
        return False

    return True

def test_app_structure():
    """测试应用程序结构"""
    print("\n🔍 测试应用程序结构...")

    required_files = [
        'app.py',
        'templates/base.html',
        'templates/index.html',
        'templates/config.html',
        'static/css/style.css',
        'static/js/main.js'
    ]

    for file_path in required_files:
        if Path(file_path).exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} 缺失")
            return False

    return True

def test_data_directories():
    """测试数据目录"""
    print("\n🔍 测试数据目录...")

    directories = [
        'data/pdf',
        'data/markdown',
        'data/split',
        'data/train_data'
    ]

    for directory in directories:
        dir_path = Path(directory)
        if dir_path.exists():
            print(f"   ✅ {directory} 存在")
        else:
            print(f"   ⚠️  {directory} 不存在 (将自动创建)")

    return True

def test_config():
    """测试配置文件"""
    print("\n🔍 测试配置...")

    env_file = Path('.env')
    if env_file.exists():
        print("   ✅ .env 文件存在")

        from dotenv import load_dotenv
        load_dotenv()

        mineru_key = os.getenv('MinerU_KEY')
        api_key = os.getenv('API_KEY')

        if mineru_key:
            print("   ✅ MinerU_KEY 已配置")
        else:
            print("   ⚠️  MinerU_KEY 未配置")

        if api_key:
            print("   ✅ API_KEY 已配置")
        else:
            print("   ⚠️  API_KEY 未配置")
    else:
        print("   ⚠️  .env 文件不存在")

    return True

def test_train_data():
    """测试训练数据文件"""
    print("\n🔍 测试训练数据...")

    train_final = Path('data/train_data/train_final.jsonl')
    if train_final.exists():
        size = train_final.stat().st_size
        print(f"   ✅ train_final.jsonl 存在 ({size} bytes)")

        try:
            with open(train_final, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                print(f"   ✅ 包含 {len(lines)} 行训练数据")

                if lines:
                    import json
                    first_item = json.loads(lines[0])
                    if 'messages' in first_item:
                        print("   ✅ 数据格式正确")
                    else:
                        print("   ❌ 数据格式错误")
        except Exception as e:
            print(f"   ❌ 读取文件失败: {e}")
    else:
        print("   ⚠️  train_final.jsonl 不存在")

    return True

def main():
    """主测试函数"""
    print("=" * 60)
    print("🧪 LLMjuice Web Application 测试")
    print("=" * 60)

    tests = [
        ("包导入测试", test_imports),
        ("应用结构测试", test_app_structure),
        ("数据目录测试", test_data_directories),
        ("配置文件测试", test_config),
        ("训练数据测试", test_train_data)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"   ❌ {test_name} 失败")
        except Exception as e:
            print(f"   ❌ {test_name} 异常: {e}")

    print("\n" + "=" * 60)
    print(f"📊 测试结果: {passed}/{total} 通过")

    if passed == total:
        print("🎉 所有测试通过！Web应用程序准备就绪。")
        print("\n🚀 启动命令:")
        print("   python run_web.py")
        print("   或")
        print("   python app.py")
        print("\n🌐 访问地址: http://localhost:5000")
    else:
        print("⚠️  部分测试失败，请检查上述问题。")

    print("=" * 60)

if __name__ == '__main__':
    main()