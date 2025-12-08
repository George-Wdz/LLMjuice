#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM微调效果对比评测引擎
实现完整的评估流程：数据准备、Query改写、双盲推理、裁判打分
"""

import os
import json
import asyncio
import random
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import logging
from openai import AsyncOpenAI
import re

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ModelClient:
    """统一的模型客户端，支持OpenAI兼容格式"""

    def __init__(self, api_url: str, api_key: str, model_name: str):
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.model_name = model_name
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=api_url
        )

    async def chat_completion(self, messages: List[Dict], **kwargs) -> str:
        """异步调用聊天完成API"""
        try:
            # 对于非流式调用，需要通过 extra_body 设置 enable_thinking=false
            extra_body = {"enable_thinking": False}

            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                extra_body=extra_body,
                **kwargs
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"模型调用失败: {e}")
            raise


class EvaluationTask:
    """评估任务类"""

    def __init__(self, task_id: str, config: Dict[str, Any]):
        self.task_id = task_id
        self.config = config

        # 状态管理
        self.status = 'pending'  # pending, running, completed, failed, stopped
        self.progress = 0
        self.current_step = '准备开始...'
        self.processed = 0
        self.total = 0
        self.error = None

        # 结果存储
        self.results = []
        self.statistics = {}
        self.logs = []

        # 模型客户端
        self.ft_client = None
        self.base_client = None
        self.judge_client = None

        # 输出文件
        self.output_file = None

        # 停止标志
        self.should_stop = False

    def add_log(self, message: str, level: str = 'info'):
        """添加日志"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message
        }
        self.logs.append(log_entry)
        logger.info(f"[{self.task_id}] {message}")

    def get_latest_log(self) -> Optional[str]:
        """获取最新日志消息"""
        if not self.logs:
            return None
        return self.logs[-1]['message']

    async def initialize_clients(self):
        """初始化模型客户端"""
        try:
            self.add_log("初始化模型客户端...")

            # 微调模型客户端
            ft_config = self.config['ft_model']
            self.ft_client = ModelClient(
                ft_config['api_url'],
                ft_config['api_key'],
                ft_config['model_name']
            )

            # 基座模型客户端
            base_config = self.config['base_model']
            self.base_client = ModelClient(
                base_config['api_url'],
                base_config['api_key'],
                base_config['model_name']
            )

            # Judge模型客户端
            judge_config = self.config['judge_model']
            self.judge_client = ModelClient(
                judge_config['api_url'],
                judge_config['api_key'],
                judge_config['model_name']
            )

            self.add_log("模型客户端初始化完成")

        except Exception as e:
            self.add_log(f"客户端初始化失败: {str(e)}", 'error')
            raise

    def load_dataset(self) -> List[Dict[str, Any]]:
        """加载数据集"""
        try:
            dataset_path = Path('./data/train_data/train_final.jsonl')
            if not dataset_path.exists():
                raise FileNotFoundError(f"数据集文件不存在: {dataset_path}")

            data = []
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            item = json.loads(line)
                            # 提取query和answer
                            if 'messages' in item and len(item['messages']) >= 3:
                                messages = item['messages']
                                query = messages[1]['content']  # user消息
                                answer = messages[2]['content']  # assistant消息
                                data.append({
                                    'original_query': query,
                                    'standard_answer': answer,
                                    'line_number': line_num
                                })
                        except json.JSONDecodeError as e:
                            logger.warning(f"跳过无效JSON行 {line_num}: {e}")
                            continue

            if not data:
                raise ValueError("数据集为空或格式不正确")

            self.add_log(f"成功加载 {len(data)} 条训练数据")
            return data

        except Exception as e:
            self.add_log(f"加载数据集失败: {str(e)}", 'error')
            raise

    def sample_data(self, data: List[Dict[str, Any]], sample_count: int) -> List[Dict[str, Any]]:
        """随机抽样数据"""
        try:
            if sample_count > len(data):
                sample_count = len(data)

            # 使用当前时间作为随机种子，确保每次都有不同的随机结果
            random.seed(time.time())
            sampled = random.sample(data, sample_count)

            # 记录抽样的数据索引，用于调试
            sampled_indices = [item.get('line_number', 'unknown') for item in sampled]
            self.add_log(f"从 {len(data)} 条数据中随机抽取 {len(sampled)} 条，索引: {sampled_indices[:10]}{'...' if len(sampled_indices) > 10 else ''}")
            return sampled

        except Exception as e:
            self.add_log(f"数据抽样失败: {str(e)}", 'error')
            raise

    async def rewrite_query(self, original_query: str) -> str:
        """使用Judge模型改写Query"""
        try:
            messages = [
                {
                    "role": "system",
                    "content": "你是一个专业的Query改写专家。请将用户的原始查询进行语义改写，保持原意但改变问法。要求：\n1. 保持原始问题的核心意图不变\n2. 使用不同的表达方式和句式结构\n3. 保持问题的清晰性和可理解性\n4. 只返回改写后的问题，不要添加任何解释"
                },
                {
                    "role": "user",
                    "content": f"原始问题：{original_query}\n\n请改写这个问题："
                }
            ]

            rewritten = await self.judge_client.chat_completion(messages)
            return rewritten.strip()

        except Exception as e:
            logger.error(f"Query改写失败: {e}")
            # 如果改写失败，返回原始查询
            return original_query

    async def get_model_response(self, client: ModelClient, query: str) -> str:
        """获取模型回答"""
        try:
            messages = [
                {
                    "role": "system",
                    "content": "你是一个有用的AI助手，请根据用户的问题提供准确、有帮助的回答。"
                },
                {
                    "role": "user",
                    "content": query
                }
            ]

            response = await client.chat_completion(messages)
            return response

        except Exception as e:
            logger.error(f"获取模型回答失败: {e}")
            return f"模型调用失败: {str(e)}"

    async def judge_responses(self, query: str, standard_answer: str,
                           response_a: str, response_b: str,
                           model_a_name: str, model_b_name: str) -> Dict[str, Any]:
        """使用Judge模型对两个回答进行评分"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # 随机决定顺序以消除位置偏见
                swap_order = random.random() < 0.5

                if swap_order:
                    response_1 = response_b
                    response_2 = response_a
                    model_1_name = model_b_name
                    model_2_name = model_a_name
                else:
                    response_1 = response_a
                    response_2 = response_b
                    model_1_name = model_a_name
                    model_2_name = model_b_name

                    messages = [
                        {
                            "role": "system",
                            "content": """你是一位精通大模型技术原理的评估专家。你的任务是根据【标准答案】，对比评估两个模型回答的质量。

                    ### 核心评估维度（按重要性排序）：

                    1. **上下文精准度（Context Precision） - 权重最高**
                    - **判定原则**：必须严格判断模型是否在讨论标准答案限定的特定领域。
                    - **陷阱识别**：如果是关于“AI模型训练”的问题，模型如果回答了“推荐系统用户冷启动”或“APP启动加载”，即使回答再详细、格式再精美，也属于**严重跑题（Domain Mismatch）**，应给予极低分。
                    - **优势判定**：如果模型锁定在“模型训练/学习”领域，即使细节不完美，也优于跑题的模型。

                    2. **信息特异性（Information Specificity）**
                    - **标准**：回答是否包含了标准答案中的“特有逻辑”？
                    - **对比**：
                        - 高特异性：提到了标准答案中特有的机制（如：结合两种数据、提示语策略、特定训练阶段）。
                        - 低特异性：回答的是互联网上的通用科普知识（如：泛泛而谈的冷启动定义）。
                        - **加分项**：如果模型使用了贴切的类比（如将训练数据比作教材/操作手册）来解释特定机制，视为理解了核心逻辑。

                    3. **表达效率与信噪比（Efficiency & Signal-to-Noise Ratio）**
                    - **原则**：奖励“切中要害”的回答，惩罚“看似专业实则无关”的冗长废话。
                    - **判定**：
                        - 模型1如果用较短的篇幅讲清楚了逻辑，得分应高于模型2用长篇大论讲无关内容。
                        - **严禁**仅因为回答长、有Markdown格式（表格、加粗）就给高分。

                    ### 评分标准（0-10分）：
                    - **8-10分**：精确命中标准答案的核心机制（如：工具调用+推理），且解释通俗易懂，无跑题。
                    - **5-7分**：**锁定正确领域（训练策略）**，虽然遗漏了部分核心细节（如忘了说工具调用），但逻辑自洽，且没有发生领域偏移（没扯到推荐系统去）。
                    - **3-4分**：内容看似丰富，但**领域完全跑题**（如讲的是推荐系统冷启动），或者包含大量通用废话。
                    - **0-2分**：完全无法理解，或严重事实错误。

                    请按照以下JSON格式返回结果：
                    {
                        "model_1_score": 0-10的分数,
                        "model_2_score": 0-10的分数,
                        "reasoning": "1. 首先进行【领域判定】：检查每个模型是在讲'模型训练'还是'推荐系统/通用软件'。跑题者直接降级。\n2. 评估【特异性】：对比模型是否触及了标准答案中的特定逻辑（如数据混合策略）。\n3. 评估【效率】：评论回答的信噪比。"
                    }"""
                        },
                        {
                            "role": "user",
                            "content": f"""请基于以下信息进行评估：

                    【用户问题】：{query}

                    【标准答案（绝对基准）】：{standard_answer}

                    【模型1（微调模型）回答】：{response_1}

                    【模型2（基座模型）回答】：{response_2}

                    ### 评估重点提醒：
                    1. **警惕通用回答**：模型2如果回答了推荐系统（User/Item Cold Start），请务必指出这是跑题，分数不能超过锁定正确领域的模型。
                    2. **重视微调价值**：模型1如果更聚焦于“训练/数据”层面，即使解释不够完美，也比完全跑题的模型更有价值。
                    3. **长度不代表质量**：请忽略模型2的排版优势，只看内容是否对齐标准答案的意图。

                    请返回JSON结果："""
                        }
                    ]
                result_text = await self.judge_client.chat_completion(messages)

                # 尝试解析JSON
                try:
                    # 清理可能的markdown格式
                    json_text = result_text.strip()
                    if json_text.startswith('```json'):
                        json_text = json_text[7:]
                    if json_text.endswith('```'):
                        json_text = json_text[:-3]
                    json_text = json_text.strip()

                    result = json.loads(json_text)

                    # 验证JSON结构
                    if 'model_1_score' not in result or 'model_2_score' not in result:
                        raise ValueError("JSON缺少必需的score字段")

                    # 映射回正确的模型
                    if swap_order:
                        ft_score = result.get('model_2_score')
                        base_score = result.get('model_1_score')
                    else:
                        ft_score = result.get('model_1_score')
                        base_score = result.get('model_2_score')

                    return {
                        'ft_model_score': self._validate_score(ft_score),
                        'base_model_score': self._validate_score(base_score),
                        'reasoning': result.get('reasoning', '无评分理由')
                    }

                except json.JSONDecodeError as e:
                    logger.warning(f"JSON解析失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                    logger.warning(f"原始文本: {result_text}")
                    if attempt == max_retries - 1:
                        # 最后一次尝试失败，返回默认分数
                        return {
                            'ft_model_score': 5.0,
                            'base_model_score': 5.0,
                            'reasoning': f"Judge模型输出格式错误，无法解析JSON: {str(e)}"
                        }
                    await asyncio.sleep(1)  # 重试前等待
                    continue

            except Exception as e:
                logger.error(f"Judge模型调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return {
                        'ft_model_score': 5.0,
                        'base_model_score': 5.0,
                        'reasoning': f"Judge模型调用失败: {str(e)}"
                    }
                await asyncio.sleep(1)  # 重试前等待
                continue

    def _validate_score(self, score: Any) -> float:
        """验证并标准化分数"""
        try:
            if isinstance(score, str):
                score = float(score)
            elif isinstance(score, (int, float)):
                score = float(score)
            else:
                return 5.0  # 默认分数

            # 确保分数在0-10范围内
            score = max(0.0, min(10.0, score))
            return round(score, 1)

        except:
            return 5.0  # 解析失败时返回默认分数

    def setup_output_file(self):
        """设置输出文件"""
        try:
            # 确保评估目录存在
            eval_dir = Path('./data/evaluate')
            eval_dir.mkdir(parents=True, exist_ok=True)

            # 创建带时间戳的输出文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_file = eval_dir / f'evaluation_log_{timestamp}.jsonl'

            self.add_log(f"创建输出文件: {self.output_file}")

        except Exception as e:
            self.add_log(f"创建输出文件失败: {str(e)}", 'error')
            raise

    def save_result(self, result: Dict[str, Any]):
        """实时保存单条评估结果"""
        try:
            if self.output_file:
                with open(self.output_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"保存结果失败: {e}")

    def update_progress(self, step: str, progress: Optional[int] = None):
        """更新进度"""
        self.current_step = step
        if progress is not None:
            self.progress = progress
        else:
            # 根据已处理数量计算进度
            if self.total > 0:
                self.progress = int((self.processed / self.total) * 100)

    def calculate_statistics(self):
        """计算统计信息"""
        if not self.results:
            self.statistics = {}
            return

        ft_scores = [r['ft_model_score'] for r in self.results if r.get('ft_model_score') is not None]
        base_scores = [r['base_model_score'] for r in self.results if r.get('base_model_score') is not None]

        self.statistics = {
            'total_evaluated': len(self.results),
            'ft_model': {
                'avg_score': sum(ft_scores) / len(ft_scores) if ft_scores else 0,
                'max_score': max(ft_scores) if ft_scores else 0,
                'min_score': min(ft_scores) if ft_scores else 0,
                'count': len(ft_scores)
            },
            'base_model': {
                'avg_score': sum(base_scores) / len(base_scores) if base_scores else 0,
                'max_score': max(base_scores) if base_scores else 0,
                'min_score': min(base_scores) if base_scores else 0,
                'count': len(base_scores)
            }
        }

    async def run(self):
        """运行完整的评估流程"""
        try:
            self.status = 'running'
            self.add_log("开始评估流程...")

            # 1. 初始化
            await self.initialize_clients()
            self.setup_output_file()

            # 2. 数据准备
            self.update_progress("准备数据...")
            full_data = self.load_dataset()
            sampled_data = self.sample_data(full_data, self.config['sample_count'])
            self.total = len(sampled_data)

            # 3. 开始逐条处理
            self.add_log(f"开始处理 {self.total} 条数据...")

            for i, data_item in enumerate(sampled_data):
                if self.should_stop:
                    self.status = 'stopped'
                    self.add_log("评估被用户停止")
                    return

                self.processed = i + 1
                self.update_progress(f"处理第 {self.processed}/{self.total} 条数据...")

                try:
                    original_query = data_item['original_query']
                    standard_answer = data_item['standard_answer']

                    self.add_log(f"开始处理第 {self.processed} 条: {original_query[:50]}...")

                    # 4. Query改写
                    self.update_progress(f"改写Query ({self.processed}/{self.total})...")
                    rewritten_query = await self.rewrite_query(original_query)

                    # 5. 并发获取两个模型的回答
                    self.update_progress(f"获取模型回答 ({self.processed}/{self.total})...")
                    ft_response_task = self.get_model_response(self.ft_client, rewritten_query)
                    base_response_task = self.get_model_response(self.base_client, rewritten_query)

                    ft_response, base_response = await asyncio.gather(
                        ft_response_task,
                        base_response_task,
                        return_exceptions=True
                    )

                    # 处理可能的异常
                    if isinstance(ft_response, Exception):
                        ft_response = f"微调模型调用失败: {str(ft_response)}"
                    if isinstance(base_response, Exception):
                        base_response = f"基座模型调用失败: {str(base_response)}"

                    # 6. Judge模型评分
                    self.update_progress(f"Judge模型评分 ({self.processed}/{self.total})...")
                    scoring_result = await self.judge_responses(
                        rewritten_query, standard_answer,
                        ft_response, base_response,
                        self.config['ft_model']['model_name'],
                        self.config['base_model']['model_name']
                    )

                    # 7. 构建结果
                    result = {
                        'index': i + 1,
                        'original_query': original_query,
                        'rewritten_query': rewritten_query,
                        'standard_answer': standard_answer,
                        'ft_model_response': ft_response,
                        'base_model_response': base_response,
                        'ft_model_score': scoring_result['ft_model_score'],
                        'base_model_score': scoring_result['base_model_score'],
                        'reasoning': scoring_result['reasoning'],
                        'timestamp': datetime.now().isoformat()
                    }

                    # 8. 实时保存结果
                    self.save_result(result)
                    self.results.append(result)

                    self.add_log(f"完成第 {self.processed} 条数据评估")

                except Exception as e:
                    self.add_log(f"处理第 {self.processed} 条数据失败: {str(e)}", 'error')
                    # 即使失败也要保存一条记录
                    error_result = {
                        'index': i + 1,
                        'original_query': data_item.get('original_query', ''),
                        'rewritten_query': '',
                        'standard_answer': data_item.get('standard_answer', ''),
                        'ft_model_response': '',
                        'base_model_response': '',
                        'ft_model_score': None,
                        'base_model_score': None,
                        'reasoning': f'处理失败: {str(e)}',
                        'timestamp': datetime.now().isoformat()
                    }
                    self.save_result(error_result)
                    self.results.append(error_result)

            # 9. 计算统计信息
            self.update_progress("计算统计信息...", 100)
            self.calculate_statistics()

            # 10. 完成
            self.status = 'completed'
            self.add_log(f"评估完成！共处理 {len(self.results)} 条数据")
            self.add_log(f"微调模型平均分: {self.statistics.get('ft_model', {}).get('avg_score', 0):.2f}")
            self.add_log(f"基座模型平均分: {self.statistics.get('base_model', {}).get('avg_score', 0):.2f}")

        except Exception as e:
            self.status = 'failed'
            self.error = str(e)
            self.add_log(f"评估失败: {str(e)}", 'error')
            raise

    def stop(self):
        """停止评估"""
        self.should_stop = True