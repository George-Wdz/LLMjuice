#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM微调效果对比评测引擎
支持异步并发处理、实时进度监控和结果导出
"""

import asyncio
import json
import random
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import aiohttp
import re

logger = logging.getLogger(__name__)


class EvaluationTask:
    """评估任务类"""

    def __init__(self, task_id: str, config: Dict[str, Any]):
        self.task_id = task_id
        self.config = config
        self.status = 'pending'  # pending, running, completed, failed, stopped
        self.progress = 0
        self.current_step = '准备开始...'
        self.processed = 0
        self.total = 0
        self.results = []
        self.statistics = {}
        self.error = None
        self.log_messages = []
        self.start_time = datetime.now()
        self.output_file = None

        # 创建评估目录
        self.evaluate_dir = Path('./data/evaluate')
        self.evaluate_dir.mkdir(parents=True, exist_ok=True)

    def add_log(self, message: str, level: str = 'info'):
        """添加日志消息"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        self.log_messages.append(log_entry)
        logger.info(f"Task {self.task_id}: {message}")

    def get_latest_log(self) -> str:
        """获取最新的日志消息"""
        return self.log_messages[-1] if self.log_messages else ""

    async def run(self):
        """执行评估任务"""
        try:
            self.status = 'running'
            self.add_log("开始执行评估任务")

            # 步骤1: 数据准备
            await self.prepare_data()

            # 步骤2: 执行评估
            await self.run_evaluation()

            # 步骤3: 完成处理
            await self.finalize_results()

            self.status = 'completed'
            self.add_log("评估任务完成")

        except asyncio.CancelledError:
            self.status = 'stopped'
            self.add_log("任务被停止")
        except Exception as e:
            self.status = 'failed'
            self.error = str(e)
            self.add_log(f"任务失败: {e}", 'error')
            logger.error(f"Evaluation task {self.task_id} failed: {e}")

    async def prepare_data(self):
        """准备数据：从数据集中随机抽取N条原始(Question, Answer)对"""
        self.current_step = "准备数据..."
        self.add_log("开始准备数据")

        dataset_path = Path('./data/train_data/train_final.jsonl')
        if not dataset_path.exists():
            raise FileNotFoundError("数据集文件不存在")

        # 读取数据集
        dataset = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        if 'messages' in data and len(data['messages']) >= 2:
                            # 提取question和answer
                            messages = data['messages']
                            question = None
                            answer = None

                            for msg in messages:
                                if msg['role'] == 'user':
                                    question = msg['content']
                                elif msg['role'] == 'assistant':
                                    answer = msg['content']

                            if question and answer:
                                dataset.append({
                                    'original_query': question,
                                    'standard_answer': answer
                                })
                    except json.JSONDecodeError:
                        continue

        if not dataset:
            raise ValueError("数据集为空或格式不正确")

        # 随机抽样
        sample_count = min(self.config['sample_count'], len(dataset))
        self.sampled_data = random.sample(dataset, sample_count)
        self.total = sample_count

        self.add_log(f"从 {len(dataset)} 条数据中随机抽取了 {sample_count} 条")

    async def run_evaluation(self):
        """执行评估流程"""
        self.current_step = "执行评估..."
        self.add_log(f"开始评估 {self.total} 条数据")

        # 创建输出文件
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_file = self.evaluate_dir / f'evaluation_log_{timestamp}.jsonl'

        # 并发处理每个样本
        semaphore = asyncio.Semaphore(3)  # 限制并发数

        tasks = []
        for i, sample in enumerate(self.sampled_data):
            task = self.process_sample(i + 1, sample, semaphore)
            tasks.append(task)

        # 等待所有任务完成
        await asyncio.gather(*tasks, return_exceptions=True)

    async def process_sample(self, index: int, sample: Dict, semaphore: asyncio.Semaphore):
        """处理单个样本"""
        async with semaphore:
            try:
                self.add_log(f"处理第 {index}/{self.total} 条数据")

                # 步骤1: Query改写
                rewritten_query = await self.rewrite_query(sample['original_query'])

                # 步骤2: 双盲推理
                ft_response, base_response = await self.blind_inference(rewritten_query)

                # 步骤3: 裁判打分
                evaluation_result = await self.evaluate_responses(
                    rewritten_query, sample['standard_answer'],
                    ft_response, base_response
                )

                # 构建结果
                result = {
                    'index': index,
                    'original_query': sample['original_query'],
                    'rewritten_query': rewritten_query,
                    'standard_answer': sample['standard_answer'],
                    'ft_model_response': ft_response,
                    'base_model_response': base_response,
                    'ft_model_score': evaluation_result['ft_model_score'],
                    'base_model_score': evaluation_result['base_model_score'],
                    'reasoning': evaluation_result['reasoning'],
                    'timestamp': datetime.now().isoformat()
                }

                # 实时落盘
                await self.save_result(result)

                # 更新进度
                self.results.append(result)
                self.processed = index
                self.progress = int((self.processed / self.total) * 100)

                self.add_log(f"完成第 {index}/{self.total} 条数据")

            except Exception as e:
                error_msg = f"处理第 {index} 条数据失败: {e}"
                self.add_log(error_msg, 'error')
                logger.error(error_msg)

    async def rewrite_query(self, original_query: str) -> str:
        """使用Judge模型改写Query"""
        prompt = f"""请将以下问题进行语义改写，保持原意但改变问法和表达方式：

原始问题：{original_query}

要求：
1. 保持问题的核心含义不变
2. 改变问题的表达方式和句式结构
3. 可以适当增减一些修饰词，但不能改变问题的本质
4. 改写后的问题应该自然流畅
5. 只返回改写后的问题，不要包含其他解释

改写后的问题："""

        return await self.call_api(
            self.config['judge_model'],
            prompt,
            max_tokens=500,
            temperature=0.7
        )

    async def blind_inference(self, query: str) -> tuple:
        """双盲推理：并发调用微调模型和基座模型"""

        # 创建并发任务
        ft_task = self.call_api(
            self.config['ft_model'],
            query,
            max_tokens=1000,
            temperature=0.3
        )

        base_task = self.call_api(
            self.config['base_model'],
            query,
            max_tokens=1000,
            temperature=0.3
        )

        # 并发执行
        ft_response, base_response = await asyncio.gather(
            ft_task, base_task, return_exceptions=True
        )

        # 处理异常
        if isinstance(ft_response, Exception):
            ft_response = f"微调模型调用失败: {str(ft_response)}"
        if isinstance(base_response, Exception):
            base_response = f"基座模型调用失败: {str(base_response)}"

        return ft_response, base_response

    async def evaluate_responses(self, query: str, standard_answer: str,
                                ft_response: str, base_response: str) -> Dict[str, Any]:
        """裁判打分：使用Judge模型评估两个回答的质量"""

        # 随机决定顺序以消除位置偏见
        swap_order = random.random() < 0.5

        if swap_order:
            response_a = base_response
            response_b = ft_response
            model_a = "基座模型"
            model_b = "微调模型"
        else:
            response_a = ft_response
            response_b = base_response
            model_a = "微调模型"
            model_b = "基座模型"

        prompt = f"""你是一个专业的AI回答质量评估专家。请对以下两个AI模型的回答进行评分。

问题：{query}

标准答案：{standard_answer}

模型A的回答：{response_a}

模型B的回答：{response_b}

请从以下几个方面评估两个回答的质量：
1. 准确性：回答是否准确，是否符合标准答案
2. 完整性：回答是否完整，是否覆盖了问题的关键点
3. 清晰性：回答是否清晰易懂，逻辑是否清楚
4. 实用性：回答是否有实际价值，是否能解决用户的问题

请以JSON格式返回评分结果，格式如下：
{{
    "ft_model_score": 0-10的分数,
    "base_model_score": 0-10的分数,
    "reasoning": "详细的评估理由"
}}

注意：
- 分数为0-10的整数，0表示最差，10表示最好
- ft_model_score对应微调模型，base_model_score对应基座模型
- 请仔细分辨哪个回答来自哪个模型，确保评分正确

评估结果："""

        # 调用Judge模型，增加重试机制
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await self.call_api(
                    self.config['judge_model'],
                    prompt,
                    max_tokens=800,
                    temperature=0.1
                )

                # 解析JSON响应
                result = self.parse_evaluation_json(response, swap_order)

                if result:
                    self.add_log(f"Judge模型评分完成 (尝试 {attempt + 1})")
                    return result
                else:
                    if attempt < max_retries - 1:
                        self.add_log(f"JSON解析失败，重试中... (尝试 {attempt + 1})")
                        await asyncio.sleep(1)

            except Exception as e:
                if attempt < max_retries - 1:
                    self.add_log(f"Judge模型调用失败，重试中...: {e} (尝试 {attempt + 1})")
                    await asyncio.sleep(2)
                else:
                    raise e

        # 如果所有重试都失败，返回默认值
        return {
            'ft_model_score': 5,
            'base_model_score': 5,
            'reasoning': 'Judge模型评分失败，使用默认分数'
        }

    def parse_evaluation_json(self, response: str, swap_order: bool) -> Optional[Dict[str, Any]]:
        """解析Judge模型的JSON响应"""
        try:
            # 提取JSON部分
            json_match = re.search(r'\{[^}]*\}', response, re.DOTALL)
            if not json_match:
                # 尝试查找更完整的JSON
                json_match = re.search(r'\{.*?\}', response, re.DOTALL)

            if not json_match:
                self.add_log(f"未找到JSON格式响应: {response[:100]}...")
                return None

            json_str = json_match.group(0)
            result = json.loads(json_str)

            # 验证必要字段
            required_fields = ['ft_model_score', 'base_model_score', 'reasoning']
            if not all(field in result for field in required_fields):
                self.add_log(f"JSON响应缺少必要字段: {result}")
                return None

            # 验证分数范围
            ft_score = result.get('ft_model_score', 0)
            base_score = result.get('base_model_score', 0)

            if not (0 <= ft_score <= 10) or not (0 <= base_score <= 10):
                self.add_log(f"分数超出范围: ft={ft_score}, base={base_score}")
                return None

            # 确保分数是数值类型
            result['ft_model_score'] = int(float(ft_score))
            result['base_model_score'] = int(float(base_score))

            return result

        except json.JSONDecodeError as e:
            self.add_log(f"JSON解析失败: {e}")
            return None
        except Exception as e:
            self.add_log(f"解析响应时出错: {e}")
            return None

    async def call_api(self, model_config: Dict[str, str], prompt: str,
                      max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """调用LLM API"""

        # 构建请求URL
        api_url = model_config['api_url'].rstrip('/')
        if not api_url.endswith('/chat/completions'):
            api_url += '/chat/completions'

        headers = {
            'Authorization': f"Bearer {model_config['api_key']}",
            'Content-Type': 'application/json'
        }

        payload = {
            'model': model_config['model_name'],
            'messages': [
                {'role': 'user', 'content': prompt}
            ],
            'max_tokens': max_tokens,
            'temperature': temperature
        }

        timeout = aiohttp.ClientTimeout(total=60)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.post(api_url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result['choices'][0]['message']['content']
                    else:
                        error_text = await response.text()
                        raise Exception(f"API调用失败: {response.status} - {error_text}")

            except asyncio.TimeoutError:
                raise Exception("API调用超时")
            except Exception as e:
                raise Exception(f"API调用失败: {e}")

    async def save_result(self, result: Dict[str, Any]):
        """实时保存结果到文件"""
        try:
            with open(self.output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        except Exception as e:
            self.add_log(f"保存结果失败: {e}", 'error')

    async def finalize_results(self):
        """完成结果处理"""
        self.current_step = "完成统计..."
        self.add_log("正在计算统计数据")

        if not self.results:
            self.statistics = {
                'total_count': 0,
                'ft_model_avg_score': 0,
                'base_model_avg_score': 0,
                'ft_model_win_count': 0,
                'base_model_win_count': 0,
                'tie_count': 0
            }
            return

        # 计算统计数据
        ft_scores = [r['ft_model_score'] for r in self.results if r['ft_model_score'] is not None]
        base_scores = [r['base_model_score'] for r in self.results if r['base_model_score'] is not None]

        ft_avg = sum(ft_scores) / len(ft_scores) if ft_scores else 0
        base_avg = sum(base_scores) / len(base_scores) if base_scores else 0

        # 计算胜负统计
        ft_wins = 0
        base_wins = 0
        ties = 0

        for r in self.results:
            if r['ft_model_score'] is not None and r['base_model_score'] is not None:
                if r['ft_model_score'] > r['base_model_score']:
                    ft_wins += 1
                elif r['base_model_score'] > r['ft_model_score']:
                    base_wins += 1
                else:
                    ties += 1

        self.statistics = {
            'total_count': len(self.results),
            'ft_model_avg_score': round(ft_avg, 2),
            'base_model_avg_score': round(base_avg, 2),
            'ft_model_win_count': ft_wins,
            'base_model_win_count': base_wins,
            'tie_count': ties,
            'execution_time': str(datetime.now() - self.start_time)
        }

        self.add_log(f"统计完成 - 微调模型平均分: {ft_avg:.2f}, 基座模型平均分: {base_avg:.2f}")
        self.add_log(f"胜负统计 - 微调模型胜: {ft_wins}, 基座模型胜: {base_wins}, 平局: {ties}")