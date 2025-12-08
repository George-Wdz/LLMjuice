#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM微调效果对比评测引擎 (简化版)
使用requests库而不是aiohttp，便于测试
"""

import asyncio
import json
import random
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import requests
from openai import OpenAI
import os

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

        # 处理每个样本
        for i, sample in enumerate(self.sampled_data):
            try:
                self.add_log(f"处理第 {i + 1}/{self.total} 条数据")

                # 模拟处理步骤
                await self.process_sample(i + 1, sample)

                # 更新进度
                self.processed = i + 1
                self.progress = int((self.processed / self.total) * 100)

                self.add_log(f"完成第 {i + 1}/{self.total} 条数据")

            except Exception as e:
                error_msg = f"处理第 {i + 1} 条数据失败: {e}"
                self.add_log(error_msg, 'error')
                logger.error(error_msg)

    async def process_sample(self, index: int, sample: Dict):
        """处理单个样本（真实API调用版本）"""
        try:
            # 步骤1：使用Judge模型改写Query
            rewritten_query = await self.rewrite_query(sample['original_query'])

            # 步骤2：并发调用微调模型和基座模型
            ft_response, base_response = await asyncio.gather(
                self.call_model_api(self.config['ft_model'], rewritten_query),
                self.call_model_api(self.config['base_model'], rewritten_query)
            )

            # 步骤3：使用Judge模型进行双盲评估
            ft_score, base_score, reasoning = await self.evaluate_responses(
                original_query=sample['original_query'],
                rewritten_query=rewritten_query,
                standard_answer=sample['standard_answer'],
                response_a=ft_response,
                response_b=base_response
            )

            # 构建结果
            result = {
                'index': index,
                'original_query': sample['original_query'],
                'rewritten_query': rewritten_query,
                'standard_answer': sample['standard_answer'],
                'ft_model_response': ft_response,
                'base_model_response': base_response,
                'ft_model_score': ft_score,
                'base_model_score': base_score,
                'reasoning': reasoning,
                'timestamp': datetime.now().isoformat()
            }

            # 实时落盘
            await self.save_result(result)
            self.results.append(result)

        except Exception as e:
            error_msg = f"处理第 {index} 条样本失败: {e}"
            self.add_log(error_msg, 'error')
            logger.error(error_msg)

            # 保存错误记录
            error_result = {
                'index': index,
                'original_query': sample['original_query'],
                'rewritten_query': sample['original_query'],  # 如果改写失败，使用原始问题
                'standard_answer': sample['standard_answer'],
                'ft_model_response': f"API调用失败: {str(e)}",
                'base_model_response': f"API调用失败: {str(e)}",
                'ft_model_score': None,
                'base_model_score': None,
                'reasoning': f"评估失败: {str(e)}",
                'timestamp': datetime.now().isoformat()
            }

            await self.save_result(error_result)
            self.results.append(error_result)

    async def rewrite_query(self, original_query: str) -> str:
        """使用Judge模型改写Query"""
        try:
            prompt = f"""请将以下问题改写为更清晰、更具体的形式，保持原意不变：

原始问题：{original_query}

改写后的问题："""

            response = await self.call_model_api(self.config['judge_model'], prompt)
            return response.strip() if response else original_query

        except Exception as e:
            logger.warning(f"Query改写失败，使用原始问题: {e}")
            return original_query

    async def call_model_api(self, model_config: Dict, query: str) -> str:
        """调用模型API获取回答（支持阿里云思考模型和流式输出）"""
        try:
            # 创建OpenAI客户端
            client = OpenAI(
                api_key=model_config['api_key'],
                base_url=model_config['api_url']
            )

            messages = [{"role": "user", "content": query}]

            # 判断是否是思考型模型（基于模型名称）
            thinking_models = ['qwen3-8b', 'qwen3-32b', 'qwen3-72b', 'qwen3-plus', 'qwen3-turbo']
            is_thinking_model = any(model in model_config['model_name'].lower() for model in thinking_models)

            # 准备调用参数
            kwargs = {
                'model': model_config['model_name'],
                'messages': messages,
                'stream': True
            }

            # 如果是思考型模型，启用思考模式
            if is_thinking_model:
                kwargs['extra_body'] = {"enable_thinking": True}

            # 调用模型API
            completion = client.chat.completions.create(**kwargs)

            # 处理流式输出
            is_answering = False
            thinking_content = ""
            answer_content = ""

            for chunk in completion:
                delta = chunk.choices[0].delta

                # 处理思考内容
                if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                    thinking_content += delta.reasoning_content

                # 处理正式回答内容
                if hasattr(delta, "content") and delta.content:
                    if not is_answering:
                        is_answering = True
                    answer_content += delta.content

            # 如果有思考内容，组合返回
            if thinking_content and answer_content:
                return f"思考过程：{thinking_content}\n\n回答：{answer_content}"
            elif answer_content:
                return answer_content
            elif thinking_content:
                return thinking_content
            else:
                return "模型没有返回任何内容"

        except Exception as e:
            error_msg = f"API调用异常: {str(e)}"
            logger.error(error_msg)
            return f"API调用失败: {error_msg}"

    async def evaluate_responses(self, original_query: str, rewritten_query: str,
                                standard_answer: str, response_a: str, response_b: str) -> tuple:
        """使用Judge模型评估两个回答的质量"""
        try:
            # 随机决定A/B的顺序，消除位置偏差
            is_swapped = random.choice([True, False])

            if is_swapped:
                # A是基座模型，B是微调模型（交换）
                response_a_desc = "回答A（基座模型）"
                response_b_desc = "回答B（微调模型）"
                eval_response_a = response_b  # 微调模型的回答作为回答B
                eval_response_b = response_a  # 基座模型的回答作为回答A
            else:
                # A是微调模型，B是基座模型（正常顺序）
                response_a_desc = "回答A（微调模型）"
                response_b_desc = "回答B（基座模型）"
                eval_response_a = response_a  # 微调模型的回答作为回答A
                eval_response_b = response_b  # 基座模型的回答作为回答B

            prompt = f"""请作为专业评委，评估以下两个AI模型对问题的回答质量。

原始问题：{original_query}
改写后的问题：{rewritten_query}
标准答案参考：{standard_answer}

{response_a_desc}：
{eval_response_a}

{response_b_desc}：
{eval_response_b}

评估说明：
- 如果回答包含"思考过程"，请综合考虑其思考深度和最终回答质量
- 思考过程体现了模型的推理能力，是评分的重要参考
- 重点评估回答的准确性、深度、逻辑性和实用性

请从以下维度评分（1-10分）：
1. 回答的准确性和相关性（是否正确理解问题，回答是否准确）
2. 回答的完整性和深度（是否全面，是否有深入分析）
3. 思考过程和逻辑性（如果有思考过程，其逻辑是否清晰合理）
4. 语言表达和实用性（表达是否清晰，对用户是否有帮助）

请按以下格式输出：
回答A得分：[1-10的整数分数]
回答B得分：[1-10的整数分数]
评判理由：[详细的评判说明，包含对思考过程（如有）的评价]"""

            judge_response = await self.call_model_api(self.config['judge_model'], prompt)

            # 解析评判结果
            ft_score, base_score, reasoning = self.parse_judge_response(
                judge_response, is_swapped
            )

            return ft_score, base_score, reasoning

        except Exception as e:
            logger.error(f"评判过程失败: {e}")
            return None, None, f"评判失败: {str(e)}"

    def parse_judge_response(self, judge_response: str, is_swapped: bool) -> tuple:
        """解析Judge模型的评判结果"""
        try:
            import re

            # 提取回答A的分数
            a_score_match = re.search(r'回答A[^：:]*得分[：:]\s*(\d+)', judge_response)
            # 提取回答B的分数
            b_score_match = re.search(r'回答B[^：:]*得分[：:]\s*(\d+)', judge_response)

            # 提取评判理由
            reasoning_match = re.search(r'评判理由[：:]\s*(.+)', judge_response, re.DOTALL)

            if a_score_match and b_score_match:
                a_score = int(a_score_match.group(1))
                b_score = int(b_score_match.group(1))
                reasoning = reasoning_match.group(1).strip() if reasoning_match else "无详细理由"

                # 如果顺序被交换了，需要交换回来
                # 如果is_swapped=True，那么回答A是基座模型，回答B是微调模型
                if is_swapped:
                    ft_score = b_score    # 回答B是微调模型
                    base_score = a_score  # 回答A是基座模型
                else:
                    ft_score = a_score    # 回答A是微调模型
                    base_score = b_score  # 回答B是基座模型

                return ft_score, base_score, reasoning
            else:
                # 解析失败，返回默认分数
                return 5, 5, f"解析评判结果失败，原始回复：{judge_response}"

        except Exception as e:
            logger.error(f"解析评判结果失败: {e}")
            return 5, 5, f"解析评判结果异常: {str(e)}"

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