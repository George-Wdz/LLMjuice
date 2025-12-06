#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Markdown文件处理脚本（2024最终版）
读取 data/markdown 文件夹及其子文件夹下所有的.md文件，
按标题层级进行分片，过滤非核心内容，
最终保存为jsonl格式到 data/split 文件夹。
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional

# --- 日志配置 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('markdown_processor_final.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MarkdownProcessor:
    """
    Markdown文件处理器
    集成了对参考文献、问卷、出版社信息、目录、声明、数据表格及附录式列表的识别逻辑。
    """

    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        # parents=True 确保如果中间目录不存在也会自动创建
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # --- 定义各类关键词 ---
        self.reference_keywords = [
            'references', 'reference', '参考文献', '参考资料', 'bibliography',
            'works cited', 'citations', '引用文献', '文献', '参考书目'
        ]
        self.questionnaire_keywords = ['个人信息', '问卷', '调查表', '基本情况', '填写', '答案']
        self.publishing_keywords = ['出版社', '出版', '作者', '译者', 'isbn', 'editor', 'author', 'publisher']
        self.toc_keywords = ['目录', 'contents', '索引', 'index', 'table of contents']
        self.disclaimer_keywords = [
            '版权声明', '免责声明', '法律声明', '重要提示',
            'copyright ©', 'all rights reserved', 'disclaimer', 'legal notice'
        ]
        self.data_table_keywords = ['评分表', '评分标准', '指数', '量表', 'scoring table', 'rating scale']
        self.appendix_keywords = ['video', 'figure', 'table', 'appendix', 'supplementary', '附录', '图', '表']

        # --- 正则表达式 ---
        self.header_pattern = re.compile(r'^(#{1,6})\s+(.+)', re.MULTILINE)

    # --- 高级内容识别方法 ---

    def is_reference_section(self, text: str) -> bool:
        text_lower = text.lower().strip()
        for keyword in self.reference_keywords:
            if keyword in text_lower: return True
        lines = text.strip().split('\n')
        if not lines: return False
        reference_line_count = sum(1 for line in lines if re.match(r'^\[\d+\]', line.strip()) or re.match(r'^\d+\.\s', line.strip()))
        if len(lines) > 2 and reference_line_count / len(lines) > 0.7: return True
        if reference_line_count > 0 and (re.search(r'\(\d{4}\)', text) or re.search(r'\d{4}年', text) or "et al" in text_lower): return True
        return False

    def is_questionnaire_section(self, text: str) -> bool:
        text_lower = text.lower().strip()
        for keyword in self.questionnaire_keywords:
            if keyword in text_lower: return True
        symbol_count = sum(text.count(s) for s in ['：', ':', '_', '□', '☐'])
        if len(text) > 50 and symbol_count / len(text) > 0.02: return True
        return False

    def is_publishing_info(self, text: str) -> bool:
        text_lower = text.lower().strip()
        for keyword in self.publishing_keywords:
            if keyword in text_lower: return True
        return False

    def is_table_of_contents(self, title: str, content: str) -> bool:
        for keyword in self.toc_keywords:
            if keyword in title.lower() or keyword in content.lower().split('\n')[0]: return True
        lines = content.strip().split('\n')
        if not lines or len(lines) < 3: return False
        match_count = sum(1 for line in lines if re.search(r'[\.·\s-]{5,}\s*\d+$', line.strip()) or (re.search(r'\s+\d+$', line.strip()) and len(line.strip().split()) > 1))
        if match_count / len(lines) > 0.6: return True
        return False

    def is_disclaimer_section(self, title: str, content: str) -> bool:
        text_to_check_lower = (title + ' ' + content[:150]).lower()
        for keyword in self.disclaimer_keywords:
            if keyword in text_to_check_lower: return True
        return False

    def is_data_table_section(self, title: str, content: str) -> bool:
        combined_text_lower = (title + ' ' + content[:100]).lower()
        for keyword in self.data_table_keywords:
            if keyword in combined_text_lower: return True
        if len(content) < 150: return False
        digit_count = sum(c.isdigit() for c in content)
        symbol_count = sum(content.count(s) for s in ['~', '～', '≤', '≥', '<', '>', '±', '-', '.'])
        density = (digit_count + symbol_count) / len(content) if len(content) > 0 else 0
        if density > 0.25: return True
        return False

    def is_list_like_appendix(self, title: str, content: str) -> bool:
        if any(keyword in title.lower() for keyword in self.appendix_keywords): return True
        lines = content.strip().split('\n')
        if len(lines) < 4: return False
        list_line_count = 0
        total_words = 0
        for line in lines:
            line_strip = line.strip()
            if re.match(r'^\s*(\d+\.|\*|-|[a-zA-Z]\.|\bVideo\b)', line_strip, re.IGNORECASE):
                list_line_count += 1
            total_words += len(line_strip.split())
        if len(lines) > 0 and list_line_count / len(lines) > 0.8: return True
        if total_words > 0 and len(lines) > 0 and (total_words / len(lines)) < 8: return True
        return False

    # --- 核心处理流程方法 ---
    
    def clean_text(self, text: str) -> str:
        """清理文本，去除图片、HTML标签、多余空行和首尾空白"""
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        lines = [line.strip() for line in text.split('\n')]
        return '\n'.join(lines).strip()

    def extract_sections(self, content: str) -> List[Dict]:
        """根据标题层级提取文档段落，并应用高级过滤规则"""
        sections = []
        lines = content.split('\n')
        current_section = {'level': 0, 'title': '', 'content': []}
        first_section_processed = False

        def process_and_add_section(section: Dict, is_first: bool):
            """内部函数，用于处理和过滤单个章节"""
            if not section['title'] and not section['content']:
                return

            title = section['title']
            content_text = '\n'.join(section['content']).strip()

            # --- 优先级过滤链 ---
            reason = None
            if is_first and not title and self.is_publishing_info(content_text):
                reason = "出版社信息"
            elif self.is_table_of_contents(title, content_text):
                reason = "目录"
            elif self.is_disclaimer_section(title, content_text):
                reason = "声明类内容"
            elif self.is_list_like_appendix(title, content_text):
                reason = "附录式列表"
            elif self.is_data_table_section(title, content_text):
                reason = "数据表格/评分表"
            elif self.is_questionnaire_section(content_text):
                reason = "问卷/调查表"
            elif self.is_reference_section(title + ' ' + content_text):
                reason = "参考文献"
            
            if reason:
                logger.info(f"  - 跳过章节 '{title}': 识别为 {reason}。")
                return

            # --- 清洗和最终检查 ---
            cleaned_content = self.clean_text(content_text)
            if len(cleaned_content) < 50:
                logger.info(f"  - 跳过章节 '{title}': 清洗后内容过短 (<50字符)。")
                return

            sections.append({
                'level': section['level'],
                'title': title,
                'content': cleaned_content
            })

        for line in lines:
            header_match = self.header_pattern.match(line)
            if header_match:
                process_and_add_section(current_section, not first_section_processed)
                first_section_processed = True
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                current_section = {'level': level, 'title': title, 'content': []}
            else:
                current_section['content'].append(line)
        
        # 处理文件末尾的最后一个章节
        process_and_add_section(current_section, not first_section_processed)
        
        return sections

    def split_long_content(self, content: str, max_length: int = 800) -> List[str]:
        """将过长的内容按句子分割成较小的片段"""
        if len(content) <= max_length:
            return [content]
            
        chunks, current_chunk = [], ""
        # 改进的句子分割，保留分隔符
        sentences = re.split(r'([。！？\.!\?])', content)
        temp_sentences = [sentences[i] + sentences[i+1] for i in range(0, len(sentences) - 1, 2)]
        if len(sentences) % 2 != 0: temp_sentences.append(sentences[-1])

        for sentence in temp_sentences:
            sentence = sentence.strip()
            if not sentence: continue
            if len(current_chunk) + len(sentence) > max_length and current_chunk:
                chunks.append(current_chunk)
                current_chunk = sentence
            else:
                current_chunk += (" " + sentence) if current_chunk else sentence
        
        if current_chunk: chunks.append(current_chunk)
        return chunks

    def process_file(self, file_path: Path) -> List[Dict]:
        """处理单个Markdown文件"""
        logger.info(f"处理文件: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"读取文件失败 {file_path}: {e}")
            return []
            
        sections = self.extract_sections(content)
        results = []
        
        for section in sections:
            desc_text = f"{section['title']}: {section['content']}" if section['title'] else section['content']
            chunks = self.split_long_content(desc_text)
            
            for chunk in chunks:
                if chunk:
                    results.append({
                        'desc': chunk,
                        'source_file': str(file_path.relative_to(self.data_dir)),
                        'level': section['level'],
                        'title': section['title']
                    })
                    
        logger.info(f"从 {file_path.name} 提取了 {len(results)} 个有效片段")
        return results

    def process_all_files(self) -> None:
        """处理所有markdown文件并保存结果"""
        logger.info(f"开始处理 {self.data_dir} 目录下的所有.md文件（包含子目录）...")
        
        all_results = []
        processed_count = 0
        
        # rglob 支持递归查找所有子文件夹
        for md_file in self.data_dir.rglob("*.md"):
            try:
                results = self.process_file(md_file)
                all_results.extend(results)
                processed_count += 1
            except Exception as e:
                logger.error(f"处理文件时发生严重错误 {md_file}: {e}")
                continue
        
        # --- 保存结果 ---
        output_file = self.output_dir / "processed_data.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in all_results:
                json_line = {"desc": result['desc']}
                f.write(json.dumps(json_line, ensure_ascii=False) + '\n')
        
        detail_file = self.output_dir / "processed_data_with_metadata.jsonl"
        with open(detail_file, 'w', encoding='utf-8') as f:
            for result in all_results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        logger.info("=" * 50)
        logger.info("处理完成！")
        logger.info(f"共处理 {processed_count} 个文件")
        logger.info(f"共生成 {len(all_results)} 个文本片段")
        logger.info(f"主要结果已保存到: {output_file}")
        logger.info(f"带元数据的结果已保存到: {detail_file}")
        logger.info("=" * 50)


def main():
    """主函数"""
    # 修改了这里：输入路径
    data_dir = "data/markdown"
    # 修改了这里：输出路径
    output_dir = "data/split"
    
    # 检查输入目录是否存在，避免报错
    if not os.path.exists(data_dir):
        logger.error(f"错误: 输入目录 '{data_dir}' 不存在，请检查路径。")
        return

    processor = MarkdownProcessor(data_dir, output_dir)
    processor.process_all_files()


if __name__ == "__main__":
    main()