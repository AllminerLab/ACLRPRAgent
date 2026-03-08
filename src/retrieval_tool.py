"""
_____________________________
  Author: Claude Assistant
   Time : 2025/12/28
   File : medical_text_retrieval.py
   Desc : 医学临床文本/检查报告检索模块（优化版 - 合并LLM调用）
______________________________
"""
import os
import re
import json
import glob
import time
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field

import pandas as pd
import numpy as np


# 尝试导入项目依赖，如果不存在则使用占位符
try:
    from logger import init_logger
    from time_util import TimeTracker, ThinkTracker, TokenTracker
    import llm_api_config
    from streaming_llm_handler import stream_openai_response, deepseek_stream_calling, qwen_stream_completion
except ImportError:
    # 如果导入失败，创建占位符
    init_logger = None
    TimeTracker = None
    ThinkTracker = None
    TokenTracker = None
    llm_api_config = None
    deepseek_stream_calling = None

# 尝试导入OpenAI客户端
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# 设置日志
base_dir = os.path.split(os.path.realpath(__file__))[0]
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)

if not logger.handlers:
    if init_logger:
        logger = init_logger('../data/log')
    else:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )


def create_dir_if_not_exist(dir_path: str) -> str:
    """如果目录不存在则创建"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path



@dataclass
class RetrievalResult:
    """检索结果数据类"""
    success: bool
    data: List[Dict[str, Any]] = field(default_factory=list)
    field_empty: bool = True  # 标记是否因未找到字段而为空
    error_message: str = ""
    # thinking: str = ""
    source_file: str = ""
    source_documents: List[str] = field(default_factory=list)
    document_selection_reason: str = ""
    field_extraction_reason: str = ""


# ==================== 数据库检索工具类 ====================

class DataPathRetriever:
    """
    数据库路径检索工具类
    用于在多级目录结构中查找患者数据、文档和缺失文件
    
    目录结构假设:
    - data_dir/
        - center1/
            - patient1/
                - doc1.csv
                - doc2.pdf
            - patient2/
                ...
        - center2/
            ...
    """

    def __init__(
        self,
        raw_dir: str = "../data/raw",
        raw_processed_dir: str = "../data/raw_processed",
        raw_pdf_extract_dir: str = "../data/raw_pdf_extract",
        interim_dir: str = "../data/interim"
    ):
        """
        初始化数据路径检索器

        Args:
            raw_dir: 原始数据目录（包含PDF等原始文件）
            raw_processed_dir: 处理后数据目录（清洗后的CSV）
            raw_pdf_extract_dir: PDF提取数据目录
            interim_dir: 中间数据目录
        """
        self.raw_dir = Path(raw_dir)
        self.raw_processed_dir = Path(raw_processed_dir)
        self.raw_pdf_extract_dir = Path(raw_pdf_extract_dir)
        self.interim_dir = Path(interim_dir)

    # ==================== 基础目录操作 ====================

    def get_data_centers(self, base_dir: Path = None) -> List[str]:
        """
        获取数据中心列表
        
        Args:
            base_dir: 基础目录，默认使用 raw_processed_dir
            
        Returns:
            数据中心名称列表
        """
        if base_dir is None:
            base_dir = self.raw_processed_dir
        
        if isinstance(base_dir, str):
            base_dir = Path(base_dir)
            
        if not base_dir.exists():
            return []
        return [d.name for d in base_dir.iterdir() if d.is_dir()]

    def get_patients_in_center(self, center: str, base_dir: Path = None) -> List[str]:
        """
        获取指定数据中心中的所有患者
        
        Args:
            center: 数据中心名称
            base_dir: 基础目录，默认使用 raw_processed_dir
            
        Returns:
            患者名称列表
        """
        if base_dir is None:
            base_dir = self.raw_processed_dir
            
        if isinstance(base_dir, str):
            base_dir = Path(base_dir)
            
        center_path = base_dir / center
        if not center_path.exists():
            return []
        return [p.name for p in center_path.iterdir() if p.is_dir()]

    def get_all_patients(self, base_dir: Path = None) -> Dict[str, List[str]]:
        """
        获取所有数据中心的所有患者
        
        Args:
            base_dir: 基础目录，默认使用 raw_processed_dir
            
        Returns:
            字典 {center: [patient1, patient2, ...], ...}
        """
        result = {}
        centers = self.get_data_centers(base_dir)
        for center in centers:
            patients = self.get_patients_in_center(center, base_dir)
            if patients:
                result[center] = patients
        return result

    # ==================== 患者查找方法 ====================

    def find_patient_in_center(
        self, 
        base_dir: Path, 
        center: str, 
        patient_name: str
    ) -> Optional[Tuple[str, Path]]:
        """
        在指定的数据中心中查找患者
        
        Args:
            base_dir: 基础目录
            center: 数据中心名称
            patient_name: 患者姓名
            
        Returns:
            (center, patient_path) 或 None
        """
        if isinstance(base_dir, str):
            base_dir = Path(base_dir)
            
        center_path = base_dir / center
        if not center_path.exists():
            logger.warning(f"数据中心 {center} 不存在于 {base_dir}")
            return None
            
        patient_path = center_path / patient_name
        if patient_path.exists() and patient_path.is_dir():
            logger.info(f"在 {center} 找到患者 {patient_name}")
            return (center, patient_path)
        return None

    def find_patient_in_dir(
        self, 
        base_dir: Path, 
        patient_name: str,
        center: str = None
    ) -> Optional[Tuple[str, Path]]:
        """
        在目录中查找患者（可指定或遍历所有中心）
        
        Args:
            base_dir: 基础目录
            patient_name: 患者姓名
            center: 指定数据中心（可选）
            
        Returns:
            (center, patient_path) 或 None
        """
        if isinstance(base_dir, str):
            base_dir = Path(base_dir)
            
        if center:
            return self.find_patient_in_center(base_dir, center, patient_name)

        data_centers = self.get_data_centers(base_dir)
        for c in data_centers:
            center_path = base_dir / c
            patient_path = center_path / patient_name
            if patient_path.exists() and patient_path.is_dir():
                logger.info(f"在 {c} 找到患者 {patient_name}")
                return (c, patient_path)
        return None

    def find_patient_in_raw(
        self, 
        patient_name: str,
        center: str = None
    ) -> Optional[Tuple[str, Path]]:
        """
        在 raw 目录中查找患者
        
        Args:
            patient_name: 患者姓名
            center: 指定数据中心（可选）
            
        Returns:
            (center, patient_path) 或 None
        """
        return self.find_patient_in_dir(self.raw_dir, patient_name, center)

    def find_patient_in_processed(
        self, 
        patient_name: str,
        center: str = None
    ) -> Optional[Tuple[str, Path]]:
        """
        在 raw_processed 目录中查找患者
        
        Args:
            patient_name: 患者姓名
            center: 指定数据中心（可选）
            
        Returns:
            (center, patient_path) 或 None
        """
        return self.find_patient_in_dir(self.raw_processed_dir, patient_name, center)

    def find_patient_in_pdf_extract(
        self, 
        patient_name: str,
        center: str = None
    ) -> Optional[Tuple[str, Path]]:
        """
        在 raw_pdf_extract 目录中查找患者
        
        Args:
            patient_name: 患者姓名
            center: 指定数据中心（可选）
            
        Returns:
            (center, patient_path) 或 None
        """
        return self.find_patient_in_dir(self.raw_pdf_extract_dir, patient_name, center)

    def find_patient_across_dirs(
        self, 
        patient_name: str,
        center: str = None
    ) -> Dict[str, Optional[Tuple[str, Path]]]:
        """
        在所有数据目录中查找患者
        
        Args:
            patient_name: 患者姓名
            center: 指定数据中心（可选）
            
        Returns:
            {
                'raw': (center, path) or None,
                'raw_processed': (center, path) or None,
                'raw_pdf_extract': (center, path) or None
            }
        """
        return {
            'raw': self.find_patient_in_raw(patient_name, center),
            'raw_processed': self.find_patient_in_processed(patient_name, center),
            'raw_pdf_extract': self.find_patient_in_pdf_extract(patient_name, center)
        }

    # ==================== 文档操作方法 ====================

    def get_doc_names_from_dir(
        self, 
        dir_path: Path, 
        extension: str = None
    ) -> set:
        """
        获取目录中指定扩展名的文档基础名（不含扩展名）
        
        Args:
            dir_path: 目录路径
            extension: 文件扩展名（如 '.pdf', '.csv'），为 None 则获取所有文件
            
        Returns:
            文档基础名集合
        """
        if isinstance(dir_path, str):
            dir_path = Path(dir_path)
            
        if not dir_path.exists():
            return set()

        names = set()
        for f in dir_path.iterdir():
            if f.is_file():
                if extension is None or f.suffix.lower() == extension.lower():
                    names.add(f.stem)
        return names

    def get_documents_in_patient_dir(
        self, 
        patient_path: Path,
        extensions: List[str] = None
    ) -> Dict[str, List[str]]:
        """
        获取患者目录中的所有文档，按扩展名分类
        
        Args:
            patient_path: 患者目录路径
            extensions: 要查找的扩展名列表，默认为 ['.csv', '.pdf', '.PDF']
            
        Returns:
            {extension: [doc_names], ...}
        """
        if isinstance(patient_path, str):
            patient_path = Path(patient_path)
            
        if extensions is None:
            extensions = ['.csv', '.pdf', '.PDF', '.xlsx', '.xls']
            
        result = {}
        for ext in extensions:
            doc_names = list(self.get_doc_names_from_dir(patient_path, ext))
            if doc_names:
                result[ext] = doc_names
        return result

    def find_missing_docs(
        self, 
        raw_path: Path, 
        processed_path: Path,
        progress_note_split: bool = True
    ) -> List[str]:
        """
        查找在 raw 目录中存在但在 processed 目录中缺失的文档
        
        Args:
            raw_path: raw 目录中患者的路径
            processed_path: processed 目录中患者的路径
            progress_note_split: 是否考虑病程记录的术前/术后拆分
            
        Returns:
            缺失的文档名列表
        """
        if isinstance(raw_path, str):
            raw_path = Path(raw_path)
        if isinstance(processed_path, str):
            processed_path = Path(processed_path)

        # 获取 raw 目录中的所有文档（PDF和CSV）
        raw_pdf_names = self.get_doc_names_from_dir(raw_path, '.pdf')
        raw_pdf_names.update(self.get_doc_names_from_dir(raw_path, '.PDF'))
        raw_pdf_names.update(self.get_doc_names_from_dir(raw_path, '.csv'))
        
        # 获取 processed 目录中的 CSV 文档
        processed_csv_names = self.get_doc_names_from_dir(processed_path, '.csv')

        missing_docs = set()
        for doc_name in raw_pdf_names:
            if doc_name in processed_csv_names:
                continue
                
            # 特殊处理：病程记录 与 术前病程记录/术后病程记录 视为一致
            if progress_note_split and "病程记录" in doc_name:
                pre_op_name = f"术前{doc_name}"
                post_op_name = f"术后{doc_name}"
                if pre_op_name in processed_csv_names or post_op_name in processed_csv_names:
                    logger.info(f"'{doc_name}' 已有术前/术后拆分版本，跳过")
                    continue
                    
            missing_docs.add(doc_name)

        if missing_docs:
            logger.info(f"发现 {len(missing_docs)} 个缺失的文档: {missing_docs}")

        return list(missing_docs)

    def check_patient_data_completeness(
        self, 
        patient_name: str,
        center: str = None
    ) -> Dict[str, Any]:
        """
        检查患者数据的完整性
        
        Args:
            patient_name: 患者姓名
            center: 数据中心（可选）
            
        Returns:
            {
                'raw_exists': bool,
                'processed_exists': bool,
                'pdf_extract_exists': bool,
                'missing_docs': List[str],
                'raw_docs': List[str],
                'processed_docs': List[str]
            }
        """
        result = {
            'raw_exists': False,
            'processed_exists': False,
            'pdf_extract_exists': False,
            'missing_docs': [],
            'raw_docs': [],
            'processed_docs': []
        }
        
        # 查找患者在各目录的位置
        locations = self.find_patient_across_dirs(patient_name, center)
        
        if locations['raw']:
            result['raw_exists'] = True
            _, raw_path = locations['raw']
            raw_docs = self.get_documents_in_patient_dir(raw_path)
            result['raw_docs'] = [
                doc for docs in raw_docs.values() for doc in docs
            ]
            
        if locations['raw_processed']:
            result['processed_exists'] = True
            _, processed_path = locations['raw_processed']
            processed_docs = self.get_documents_in_patient_dir(processed_path, ['.csv'])
            result['processed_docs'] = processed_docs.get('.csv', [])
            
        if locations['raw_pdf_extract']:
            result['pdf_extract_exists'] = True
            
        # 计算缺失文档
        if locations['raw'] and locations['raw_processed']:
            _, raw_path = locations['raw']
            _, processed_path = locations['raw_processed']
            result['missing_docs'] = self.find_missing_docs(raw_path, processed_path)
            
        return result

    # ==================== 目录列表方法 ====================

    def list_all_centers(self) -> Dict[str, List[str]]:
        """
        列出所有数据目录中的数据中心
        
        Returns:
            {
                'raw': [centers],
                'raw_processed': [centers],
                'raw_pdf_extract': [centers]
            }
        """
        return {
            'raw': self.get_data_centers(self.raw_dir),
            'raw_processed': self.get_data_centers(self.raw_processed_dir),
            'raw_pdf_extract': self.get_data_centers(self.raw_pdf_extract_dir)
        }

    def get_patient_path(
        self,
        patient_name: str,
        center: str = None,
        data_type: str = 'processed'
    ) -> Optional[Path]:
        """
        获取患者数据目录路径
        
        Args:
            patient_name: 患者姓名
            center: 数据中心（可选）
            data_type: 数据类型 ('raw', 'processed', 'pdf_extract')
            
        Returns:
            患者目录路径或 None
        """
        if data_type == 'raw':
            result = self.find_patient_in_raw(patient_name, center)
        elif data_type == 'processed':
            result = self.find_patient_in_processed(patient_name, center)
        elif data_type == 'pdf_extract':
            result = self.find_patient_in_pdf_extract(patient_name, center)
        else:
            logger.warning(f"未知的数据类型: {data_type}")
            return None
            
        if result:
            _, path = result
            return path
        return None


class MedicalRetriever:
    """医学临床文本/检查报告检索类"""

    # Token估算常量（中文约1.5-2个字符/token，英文约4个字符/token）
    CHARS_PER_TOKEN_CN = 1.8
    CHARS_PER_TOKEN_EN = 4.0

    # 默认最大token限制（留出输出空间）
    DEFAULT_MAX_INPUT_TOKENS = 6000

    # 字段值截断长度
    MAX_FIELD_VALUE_LENGTH = 100

    # 提示词基础token开销（不含文档内容）
    PROMPT_BASE_TOKENS = 800

    # 默认文件扩展名
    DEFAULT_EXTENSION = '.csv'

    def __init__(
        self,
        data_dir: str = "../data/raw_processed",
        output_dir: str = "../data/interim",
        supported_extensions: List[str] = None,
        print_thinking: bool = True,
        max_input_tokens: int = None,
        default_extension: str = None
    ):
        """
        初始化检索器

        Args:
            data_dir: 数据根目录，包含各个数据中心
            output_dir: 输出目录
            supported_extensions: 支持的文件扩展名列表
            print_thinking: 是否打印大模型思考过程
            max_input_tokens: 最大输入token数，防止超限
            default_extension: 默认文件扩展名（当文档名无扩展名时使用）
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.supported_extensions = supported_extensions or ['.csv', '.txt', '.json', '.xlsx', '.xls', '.md']
        self.token_recorder = None
        self.print_thinking = print_thinking
        self.max_input_tokens = max_input_tokens or self.DEFAULT_MAX_INPUT_TOKENS
        self.default_extension = default_extension or self.DEFAULT_EXTENSION

    # ==================== Token估算方法 ====================

    def _estimate_tokens(self, text: str) -> int:
        """
        估算文本的token数量

        Args:
            text: 输入文本

        Returns:
            估算的token数量
        """
        if not text:
            return 0

        # 分别统计中文和非中文字符
        cn_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        other_chars = len(text) - cn_chars

        # 估算token数
        cn_tokens = cn_chars / self.CHARS_PER_TOKEN_CN
        other_tokens = other_chars / self.CHARS_PER_TOKEN_EN

        return int(cn_tokens + other_tokens)

    def _truncate_to_token_limit(
        self,
        content_parts: List[Tuple[str, str, int]],  # [(doc_name, content, priority)]
        max_tokens: int
    ) -> List[Tuple[str, str]]:
        """
        根据token限制截断内容

        Args:
            content_parts: [(文档名, 内容, 优先级)] 优先级越小越重要
            max_tokens: 最大token数

        Returns:
            截断后的[(文档名, 内容)]列表
        """
        # 按优先级排序
        sorted_parts = sorted(content_parts, key=lambda x: x[2])

        result = []
        current_tokens = 0

        for doc_name, content, _ in sorted_parts:
            content_tokens = self._estimate_tokens(content)

            if current_tokens + content_tokens <= max_tokens:
                result.append((doc_name, content))
                current_tokens += content_tokens
            else:
                # 计算剩余可用token
                remaining_tokens = max_tokens - current_tokens
                if remaining_tokens > 100:  # 至少保留100个token才值得截断
                    # 按比例截断内容
                    truncate_ratio = remaining_tokens / content_tokens
                    truncate_len = int(len(content) * truncate_ratio * 0.9)  # 留10%余量
                    truncated_content = content[:truncate_len] + "\n... [内容已截断]"
                    result.append((doc_name, truncated_content))
                    logger.warning(f"文档 {doc_name} 内容已截断以适应token限制")
                break

        return result

    def _split_documents_into_batches(
        self,
        doc_summaries: List[Dict[str, Any]],
        max_tokens_per_batch: int
    ) -> List[List[Dict[str, Any]]]:
        """
        简单分批：按顺序将文档分成多批，超过token限制则开新批次

        Args:
            doc_summaries: 文档摘要列表
            max_tokens_per_batch: 每批最大token数

        Returns:
            分批后的文档摘要列表 [[batch1_docs], [batch2_docs], ...]
        """
        # 可用于文档内容的token数（减去提示词基础开销）
        available_tokens = max_tokens_per_batch - self.PROMPT_BASE_TOKENS

        batches = []
        current_batch = []
        current_tokens = 0

        for doc in doc_summaries:
            doc_text = json.dumps(doc, ensure_ascii=False)
            doc_tokens = self._estimate_tokens(doc_text)

            # 如果加入当前文档会超限，开新批次
            if current_tokens + doc_tokens > available_tokens:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [doc]
                current_tokens = doc_tokens
            else:
                current_batch.append(doc)
                current_tokens += doc_tokens

        # 别忘了最后一批
        if current_batch:
            batches.append(current_batch)

        if len(batches) > 1:
            logger.info(f"文档已分为 {len(batches)} 批处理，各批文档数: {[len(b) for b in batches]}")

        return batches

    def _truncate_single_doc_summary(
        self,
        doc: Dict[str, Any],
        max_tokens: int
    ) -> Dict[str, Any]:
        """
        截断单个文档摘要以适应token限制
        """
        truncated_doc = {
            "document_name": doc["document_name"],
            "fields": []
        }

        if "record_count" in doc:
            truncated_doc["record_count"] = doc["record_count"]

        current_tokens = self._estimate_tokens(json.dumps(truncated_doc, ensure_ascii=False))

        for field in doc.get("fields", []):
            field_tokens = self._estimate_tokens(json.dumps(field, ensure_ascii=False))

            if current_tokens + field_tokens <= max_tokens * 0.9:  # 留10%余量
                truncated_doc["fields"].append(field)
                current_tokens += field_tokens
            else:
                # 尝试截断字段值
                truncated_field = {
                    "field_name": field["field_name"],
                    "value_preview": field.get("value_preview", "")[:30] + "..."
                }
                truncated_doc["fields"].append(truncated_field)
                break

        remaining = len(doc.get("fields", [])) - len(truncated_doc["fields"])
        if remaining > 0:
            truncated_doc["truncated"] = f"还有{remaining}个字段未显示"
            logger.warning(f"文档 {doc['document_name']} 字段已截断，保留{len(truncated_doc['fields'])}个")

        return truncated_doc

    # ==================== 目录扫描方法 ====================

    def get_centers(self) -> List[str]:
        """获取所有数据中心列表"""
        if not os.path.exists(self.data_dir):
            logger.error(f"数据目录不存在: {self.data_dir}")
            return []

        centers = [
            d for d in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, d))
        ]
        return centers

    def get_patients(self, center: str) -> List[str]:
        """获取指定中心下的所有患者（人名）列表"""
        center_path = os.path.join(self.data_dir, center)
        if not os.path.exists(center_path):
            logger.error(f"数据中心不存在: {center_path}")
            return []

        patients = [
            d for d in os.listdir(center_path)
            if os.path.isdir(os.path.join(center_path, d))
        ]
        return patients

    def get_documents(self, center: str, patient: str) -> List[str]:
        """获取指定患者下的所有文档列表"""
        patient_path = os.path.join(self.data_dir, center, patient)
        if not os.path.exists(patient_path):
            logger.error(f"患者目录不存在: {patient_path}")
            return []

        documents = []
        for ext in self.supported_extensions:
            pattern = os.path.join(patient_path, f"*{ext}")
            documents.extend([os.path.basename(f) for f in glob.glob(pattern)])

        return documents

    def _has_extension(self, filename: str) -> bool:
        """检查文件名是否有扩展名"""
        _, ext = os.path.splitext(filename)
        return bool(ext)

    def _normalize_document_name(self, document_name: str) -> str:
        """
        规范化文档名，如果没有扩展名则添加默认扩展名

        Args:
            document_name: 原始文档名

        Returns:
            规范化后的文档名
        """
        if not self._has_extension(document_name):
            return document_name + self.default_extension
        return document_name

    def get_document_path(self, center: str, patient: str, document_name: str) -> Optional[str]:
        """
        获取文档的完整路径

        Args:
            center: 数据中心名称
            patient: 患者姓名
            document_name: 文档名（如果没有扩展名，默认添加.csv）

        Returns:
            文档完整路径，找不到则返回None
        """
        patient_path = os.path.join(self.data_dir, center, patient)

        # 1. 先尝试原始文档名
        doc_path = os.path.join(self.data_dir, center, patient, document_name)
        if os.path.exists(doc_path):
            return doc_path

        # 2. 如果没有扩展名，尝试添加默认扩展名(.csv)
        if not self._has_extension(document_name):
            doc_path_with_ext = os.path.join(self.data_dir, center, patient, document_name + self.default_extension)
            if os.path.exists(doc_path_with_ext):
                return doc_path_with_ext

            # 3. 尝试其他支持的扩展名
            for ext in self.supported_extensions:
                doc_path_with_ext = os.path.join(self.data_dir, center, patient, document_name + ext)
                if os.path.exists(doc_path_with_ext):
                    return doc_path_with_ext

        # 4. 尝试模糊匹配
        if os.path.exists(patient_path):
            for file in os.listdir(patient_path):
                # 去掉扩展名后比较
                file_base = os.path.splitext(file)[0]
                doc_base = os.path.splitext(document_name)[0]
                if doc_base.lower() in file_base.lower() or file_base.lower() in doc_base.lower():
                    return os.path.join(patient_path, file)

        return None

    def get_documents_by_pattern(
        self,
        center: str,
        patient: str,
        pattern: str
    ) -> List[Tuple[str, str]]:
        """
        根据模式匹配获取文档列表

        Args:
            center: 数据中心名称
            patient: 患者姓名
            pattern: 匹配模式（支持部分匹配）

        Returns:
            List[(文档名, 文档路径)]
        """
        all_docs = self.get_documents(center, patient)
        matched = []

        # 去掉pattern的扩展名进行匹配
        pattern_base = os.path.splitext(pattern)[0]

        for doc_name in all_docs:
            doc_base = os.path.splitext(doc_name)[0]
            if pattern_base.lower() in doc_base.lower():
                doc_path = self.get_document_path(center, patient, doc_name)
                if doc_path:
                    matched.append((doc_name, doc_path))

        return matched

    # ==================== 文档读取方法 ====================

    def read_document(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        读取文档内容

        Args:
            file_path: 文档路径

        Returns:
            (原始文本内容, 结构化数据字典)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        raw_text = ""
        structured_data = {}

        try:
            if ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    raw_text = f.read()
                structured_data = self._parse_medical_text(raw_text)

            elif ext == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    structured_data = json.load(f)
                raw_text = json.dumps(structured_data, ensure_ascii=False, indent=2)

            elif ext == '.csv':
                df = pd.read_csv(file_path, encoding='utf-8')
                structured_data = df.to_dict(orient='records')
                raw_text = df.to_string()

            elif ext in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
                structured_data = df.to_dict(orient='records')
                raw_text = df.to_string()

            elif ext == '.md':
                with open(file_path, 'r', encoding='utf-8') as f:
                    raw_text = f.read()
                structured_data = self._parse_markdown_text(raw_text)

            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    raw_text = f.read()
                structured_data = {"raw_content": raw_text}

        except Exception as e:
            logger.error(f"读取文件失败 {file_path}: {str(e)}")
            raise

        return raw_text, structured_data

    def _parse_medical_text(self, text: str) -> Dict[str, Any]:
        """
        解析医学文本，提取字段和值
        """
        result = {}
        lines = text.split('\n')
        current_field = None
        current_value = []

        field_pattern = re.compile(r'^([^：:]+)[：:](.*)$')

        for line in lines:
            line_stripped = line.strip()

            if not line_stripped:
                if current_field and current_value:
                    result[current_field] = self._format_value(current_value)
                    current_field = None
                    current_value = []
                continue

            match = field_pattern.match(line_stripped)
            if match:
                if current_field and current_value:
                    result[current_field] = self._format_value(current_value)

                current_field = match.group(1).strip()
                value = match.group(2).strip()
                current_value = [value] if value else []

            elif current_field:
                if line.startswith(' ') or line.startswith('\t'):
                    current_value.append(line_stripped)
                else:
                    current_value.append(line_stripped)

        if current_field and current_value:
            result[current_field] = self._format_value(current_value)

        return result

    def _parse_markdown_text(self, text: str) -> Dict[str, Any]:
        """解析Markdown格式的医学文本"""
        result = {}
        sections = re.split(r'^#+\s+', text, flags=re.MULTILINE)

        for section in sections:
            if not section.strip():
                continue

            lines = section.strip().split('\n')
            if lines:
                title = lines[0].strip()
                content = '\n'.join(lines[1:]).strip()
                result[title] = content

        return result

    def _format_value(self, value_list: List[str]) -> Union[str, List[str]]:
        """格式化字段值"""
        if len(value_list) == 1:
            return value_list[0]
        elif len(value_list) > 1:
            return value_list
        return ""

    # ==================== 精确检索方法 ====================

    def retrieve_by_field(
        self,
        center: str,
        patient: str,
        document_name: str,
        field_names: Union[str, List[str]]
    ) -> RetrievalResult:
        """
        精确检索：根据指定的center、人名、文档名、字段名检索内容

        Args:
            center: 数据中心名称
            patient: 患者姓名
            document_name: 文档名（如果没有扩展名，默认添加.csv）
            field_names: 字段名或字段名列表
        """
        if isinstance(field_names, str):
            field_names = [field_names]

        doc_path = self.get_document_path(center, patient, document_name)
        if not doc_path:
            return RetrievalResult(
                success=False,
                error_message=f"找不到文档: center={center}, patient={patient}, document={document_name}"
            )

        try:
            raw_text, structured_data = self.read_document(doc_path)
        except Exception as e:
            return RetrievalResult(
                success=False,
                error_message=f"读取文档失败: {str(e)}"
            )

        result_data = []
        not_found_fields = []

        # 获取实际的文档名（从路径中提取）
        actual_doc_name = os.path.basename(doc_path)

        if isinstance(structured_data, dict):
            # 键值对数据：每个字段一条记录
            for field_name in field_names:
                if field_name in structured_data:
                    result_data.append({
                        'field': field_name,
                        'value': structured_data[field_name]
                    })
                else:
                    not_found_fields.append(field_name)

        elif isinstance(structured_data, list):
            # 表格数据：保留记录结构，筛选指定字段
            found_fields = set()
            for record in structured_data:
                if isinstance(record, dict):
                    filtered_record = {}
                    for field_name in field_names:
                        if field_name in record:
                            filtered_record[field_name] = record[field_name]
                            found_fields.add(field_name)
                    if filtered_record:
                        result_data.append(filtered_record)

            not_found_fields = [f for f in field_names if f not in found_fields]

        if result_data:
            return RetrievalResult(
                success=True,
                data=result_data,
                field_empty=False,  # 成功且有数据
                source_file=doc_path,
                source_documents=[actual_doc_name],
                error_message=f"未找到字段: {not_found_fields}" if not_found_fields else ""
            )
        else:
            # 修改重点：Success=True, 但 field_empty=True
            return RetrievalResult(
                success=True,  # 执行成功
                data=[],
                field_empty=True,  # 结果为空
                error_message=f"在文档中未找到任何指定字段: {field_names}",  # 这里的msg仅作为提示
                source_file=doc_path,
                source_documents=[os.path.basename(doc_path)]
            )

    def retrieve_all_fields(
        self,
        center: str,
        patient: str,
        document_name: str
    ) -> RetrievalResult:
        """
        检索文档的所有字段

        Args:
            center: 数据中心名称
            patient: 患者姓名
            document_name: 文档名（如果没有扩展名，默认添加.csv）
        """
        doc_path = self.get_document_path(center, patient, document_name)
        if not doc_path:
            return RetrievalResult(
                success=False,
                error_message=f"找不到文档: center={center}, patient={patient}, document={document_name}"
            )

        try:
            raw_text, structured_data = self.read_document(doc_path)
            print(structured_data)
            actual_doc_name = os.path.basename(doc_path)

            # 转换为records格式
            if isinstance(structured_data, dict):
                # 键值对数据
                result_data = [{'field': k, 'value': v} for k, v in structured_data.items()]
            elif isinstance(structured_data, list):
                # 已经是records格式，直接使用
                result_data = structured_data
            else:
                result_data = []

            return RetrievalResult(
                success=True,
                data=result_data,
                source_file=doc_path,
                source_documents=[actual_doc_name]
            )
        except Exception as e:
            return RetrievalResult(
                success=False,
                error_message=f"读取文档失败: {str(e)}"
            )

    # ==================== 智能检索方法（基于LLM）====================

    def retrieve_by_llm(
        self,
        center: str,
        patient: str,
        task_goal: str,
        search_condition: Optional[str] = None,
        document_names: Optional[Union[str, List[str]]] = None,
        max_retries: int = 3
    ) -> RetrievalResult:
        """
        智能检索：调用大模型检索相关字段和内容（优化版 - 合并调用）

        两种检索模式：
        1. 按search_condition筛选文档：根据条件找到相关文档，再提取字段
        2. 按document_names指定文档：直接在指定文档中提取字段

        Args:
            center: 数据中心名称
            patient: 患者姓名
            task_goal: 任务目标（用于字段提取）
            search_condition: 搜索条件（如"术前"、"入院"等）
            document_names: 指定的文档名称（单个或列表，如果没有扩展名默认添加.csv）
            max_retries: 最大重试次数

        Returns:
            RetrievalResult: 检索结果
        """
        if isinstance(document_names, str):
            document_names = [document_names]

        all_doc_names = self.get_documents(center, patient)
        if not all_doc_names:
            return RetrievalResult(
                success=False,
                error_message=f"未找到任何文档: center={center}, patient={patient}"
            )

        # ========== 阶段1：文档筛选 ==========
        if document_names:
            selected_docs = []
            for doc_name in document_names:
                doc_path = self.get_document_path(center, patient, doc_name)
                if doc_path:
                    actual_doc_name = os.path.basename(doc_path)
                    selected_docs.append((actual_doc_name, doc_path))
                else:
                    matched = self.get_documents_by_pattern(center, patient, doc_name)
                    selected_docs.extend(matched)

            if not selected_docs:
                return RetrievalResult(
                    success=False,
                    error_message=f"找不到指定的文档: {document_names}"
                )

            doc_selection_reason = f"使用指定的文档: {[d[0] for d in selected_docs]}"

        elif search_condition:
            selected_docs, doc_selection_reason = self._select_documents_by_condition(
                center=center,
                patient=patient,
                all_doc_names=all_doc_names,
                search_condition=search_condition,
                max_retries=max_retries
            )

            if not selected_docs:
                # 修改重点：如果按条件没找到文档，这不算系统错误，而是“结果为空”
                return RetrievalResult(
                    success=True,  # 流程是通的
                    data=[],
                    field_empty=True,  # 没文档自然没字段，标记为空
                    error_message=f"根据条件'{search_condition}'未找到相关文档",
                    document_selection_reason=doc_selection_reason
                )
        else:
            return RetrievalResult(
                success=False,
                error_message="必须指定search_condition或document_names之一"
            )

        # ========== 阶段2：读取文档并提取字段（合并调用）==========
        all_content = {}
        for doc_name, doc_path in selected_docs:
            try:
                raw_text, structured_data = self.read_document(doc_path)
                all_content[doc_name] = {
                    "raw_text": raw_text,
                    "structured_data": structured_data,
                    "path": doc_path
                }
            except Exception as e:
                logger.warning(f"读取文档失败 {doc_name}: {str(e)}")
                continue

        if not all_content:
            return RetrievalResult(
                success=False,
                error_message="无法读取任何选中的文档内容",
                source_documents=[d[0] for d in selected_docs],
                document_selection_reason=doc_selection_reason
            )

        # 合并调用LLM提取字段
        for attempt in range(max_retries):
            try:
                result = self._extract_fields_by_llm_batch(
                    all_content=all_content,
                    task_goal=task_goal,
                    search_condition=search_condition
                )

                if result.success:
                    validated_result = self._validate_llm_result(result, all_content)
                    if validated_result.success:
                        validated_result.source_documents = list(all_content.keys())
                        validated_result.document_selection_reason = doc_selection_reason
                        return validated_result
                    else:
                        logger.warning(f"第{attempt + 1}次检索结果验证失败: {validated_result.error_message}")
                else:
                    logger.warning(f"第{attempt + 1}次LLM检索失败: {result.error_message}")

            except Exception as e:
                logger.error(f"第{attempt + 1}次LLM调用异常: {str(e)}")

        return RetrievalResult(
            success=False,
            error_message=f"经过{max_retries}次尝试后仍无法成功检索",
            source_documents=list(all_content.keys()),
            document_selection_reason=doc_selection_reason
        )

    def _select_documents_by_condition(
        self,
        center: str,
        patient: str,
        all_doc_names: List[str],
        search_condition: str,
        max_retries: int = 3
    ) -> Tuple[List[Tuple[str, str]], str]:
        """根据搜索条件选择相关文档"""
        # 首先尝试简单的关键词匹配
        # keyword_matched = []
        # for doc_name in all_doc_names:
        #     if search_condition.lower() in doc_name.lower():
        #         doc_path = self.get_document_path(center, patient, doc_name)
        #         if doc_path:
        #             keyword_matched.append((doc_name, doc_path))
        #
        # if keyword_matched:
        #     reason = f"通过关键词'{search_condition}'直接匹配到文档: {[d[0] for d in keyword_matched]}"
        #     return keyword_matched, reason

        # 如果关键词匹配失败，调用LLM进行智能选择
        for attempt in range(max_retries):
            try:
                result = self._llm_select_documents(
                    all_doc_names=all_doc_names,
                    search_condition=search_condition
                )

                if result and result.get('selected_documents'):
                    selected = []
                    for doc_name in result['selected_documents']:
                        doc_path = self.get_document_path(center, patient, doc_name)
                        if doc_path:
                            selected.append((doc_name, doc_path))

                    if selected:
                        reason = result.get('reason', f"LLM根据条件'{search_condition}'选择了文档")
                        return selected, reason

            except Exception as e:
                logger.info(f"第{attempt + 1}次文档选择异常: {str(e)}")

        return [], f"无法找到与条件'{search_condition}'相关的文档"

    def _llm_select_documents(
        self,
        all_doc_names: List[str],
        search_condition: str,
        model: str = 'qwen3-max',
    ) -> Optional[Dict[str, Any]]:
        """调用LLM选择相关文档"""

        prompt = f"""你是一位资深医学数据分析专家，精通临床医学和医学信息学。

请根据搜索条件，从以下文档列表中选择最相关的文档。

搜索条件: {search_condition}

可用文档列表:
{json.dumps(all_doc_names, ensure_ascii=False, indent=2)}

请分析每个文档名称，选择与搜索条件最相关的文档。

常见的医学文档类型参考:
- 术前: 术前检查、术前评估、术前记录、术前检验
- 术后: 术后记录、术后随访、术后检查
- 入院: 入院记录、入院评估、首次病程记录
- 出院: 出院记录、出院小结、出院诊断
- 手术: 手术记录、手术同意书、麻醉记录
- 检验: 血常规、生化、凝血功能、感染指标
- 影像: CT报告、MRI报告、X线报告、超声报告
- 病程: 病程记录、护理记录、查房记录

请以JSON格式返回结果:
{{
    "reason": "选择这些文档的理由",
    "selected_documents": ["文档1", "文档2"]
}}

如果没有找到相关文档，请返回:
{{
    "reason": "说明为什么没有找到相关文档",
    "selected_documents": []
}}
"""
        return self._call_llm_api(
            prompt=prompt,
            print_thinking=self.print_thinking,
            model =model,
            task_name=f"文档选择 - {search_condition}"
        )

    # ==================== 批量字段提取方法 ====================

    def _extract_fields_by_llm_batch(
        self,
        all_content: Dict[str, Any],
        task_goal: str,
        search_condition: Optional[str] = None
    ) -> RetrievalResult:
        """
        批量调用LLM从多个文档中提取相关字段（简化版 - 顺序分批）

        Args:
            all_content: 文档内容字典 {doc_name: {raw_text, structured_data, path}}
            task_goal: 任务目标
            search_condition: 可选的搜索条件
        """
        # 构建优化的文档摘要
        doc_summaries, total_tokens = self._build_optimized_doc_summary(all_content)

        # 计算可用于文档内容的token数
        available_tokens = self.max_input_tokens - self.PROMPT_BASE_TOKENS

        # 检查是否需要分批
        if total_tokens <= available_tokens:
            # 单批处理
            return self._process_single_batch(
                doc_summaries=doc_summaries,
                all_content=all_content,
                task_goal=task_goal,
                search_condition=search_condition
            )
        else:
            # 分批处理
            logger.info(f"文档总token({total_tokens})超过限制({available_tokens})，启用分批处理")
            return self._process_multiple_batches(
                doc_summaries=doc_summaries,
                all_content=all_content,
                task_goal=task_goal,
                search_condition=search_condition
            )

    def _process_single_batch(
        self,
        doc_summaries: List[Dict[str, Any]],
        all_content: Dict[str, Any],
        task_goal: str,
        search_condition: Optional[str]
    ) -> RetrievalResult:
        """处理单批文档"""
        prompt = self._build_batch_extraction_prompt(
            doc_summaries=doc_summaries,
            task_goal=task_goal,
            search_condition=search_condition
        )

        llm_response = self._call_llm_api(
            prompt=prompt,
            print_thinking=self.print_thinking,
            task_name=f"字段提取 - {task_goal}..."
        )

        if llm_response is None:
            return RetrievalResult(
                success=False,
                error_message="LLM API调用失败"
            )

        return self._parse_extraction_response(llm_response, all_content)

    def _process_multiple_batches(
        self,
        doc_summaries: List[Dict[str, Any]],
        all_content: Dict[str, Any],
        task_goal: str,
        search_condition: Optional[str]
    ) -> RetrievalResult:
        """
        处理多批文档，合并结果
        """
        # 简单顺序分批
        batches = self._split_documents_into_batches(
            doc_summaries=doc_summaries,
            max_tokens_per_batch=self.max_input_tokens
        )

        total_batches = len(batches)
        logger.info(f"共分为 {total_batches} 批处理")

        if self.print_thinking:
            print(f"\n📊 分批信息: 总文档数 {len(doc_summaries)}，分为 {total_batches} 批")

        # 处理每个批次
        all_selected_fields = []
        all_reasons = []

        for i, batch_docs in enumerate(batches):
            if self.print_thinking:
                print(f"\n{'='*50}")
                print(f"📦 处理批次 {i+1}/{total_batches} ({len(batch_docs)}个文档)")
                print(f"{'='*50}")

            prompt = self._build_batch_extraction_prompt(
                doc_summaries=batch_docs,
                task_goal=task_goal,
                search_condition=search_condition,
                batch_info=f"(批次 {i+1}/{total_batches})" if total_batches > 1 else ""
            )

            llm_response = self._call_llm_api(
                prompt=prompt,
                print_thinking=self.print_thinking,
                task_name=f"批次{i+1}/{total_batches}"
            )

            if llm_response:
                selected_fields = llm_response.get('selected_fields', [])
                reason = llm_response.get('reason', '')

                if selected_fields and selected_fields != 'nan':
                    all_selected_fields.extend(selected_fields)
                    all_reasons.append(f"批次{i+1}: {reason}")
            else:
                logger.warning(f"批次 {i+1} 处理失败")

        # 合并结果
        if not all_selected_fields:
            # 只有当确实进行过尝试（all_reasons不为空），才视为“结果为空”
            # 如果是因为API全挂了导致的空，应该在外部捕获为False
            if all_reasons:
                return RetrievalResult(
                    success=True,  # API调用成功了
                    data=[],
                    field_empty=True,  # 但LLM没找到数据
                    error_message="所有批次均未找到相关字段",
                    field_extraction_reason="; ".join(all_reasons)
                )
            else:
                return RetrievalResult(
                    success=False,  # 可能是API全挂了
                    error_message="API调用失败或无响应"
                )

        # 去重
        unique_fields = self._deduplicate_fields(all_selected_fields)

        # 从原始数据中提取对应字段的值
        result_data = self._extract_fields_from_content(
            all_content=all_content,
            selected_fields=unique_fields
        )

        combined_reason = "; ".join(all_reasons) if len(all_reasons) > 1 else (all_reasons[0] if all_reasons else "")

        if self.print_thinking:
            print(f"\n✅ 所有批次处理完成，总计选择字段: {len(unique_fields)}个，成功提取: {len(result_data)}条记录")

        return RetrievalResult(
            success=True,
            data=result_data,
            field_extraction_reason=combined_reason
        )

    def _deduplicate_fields(
        self,
        fields: List[Union[str, Dict[str, str]]]
    ) -> List[Dict[str, str]]:
        """
        去重字段列表
        """
        seen = set()
        unique = []

        for field_spec in fields:
            if isinstance(field_spec, dict):
                key = (field_spec.get('document', ''), field_spec.get('field', ''))
            else:
                key = ('', field_spec)

            if key not in seen:
                seen.add(key)
                if isinstance(field_spec, dict):
                    unique.append(field_spec)
                else:
                    unique.append({'document': '', 'field': field_spec})

        return unique

    def _parse_extraction_response(
        self,
        llm_response: Dict[str, Any],
        all_content: Dict[str, Any]
    ) -> RetrievalResult:
        """解析LLM提取响应"""
        try:
            selected_fields = llm_response.get('selected_fields', [])
            thinking = llm_response.get('reason', '')

            # 情况1：LLM明确返回空列表或'nan' -> 成功但为空
            if not selected_fields or selected_fields == 'nan':
                return RetrievalResult(
                    success=True,
                    data=[],
                    field_empty=True,  # <--- 标记
                    error_message="LLM未找到相关字段",
                    field_extraction_reason=thinking
                )

            # 情况2：LLM返回了字段，去原始内容提取
            result_data = self._extract_fields_from_content(
                all_content=all_content,
                selected_fields=selected_fields
            )

            # 情况3：LLM产生了幻觉，返回了不存在的字段 -> 提取结果为空 ->
            if len(result_data) == 0:
                return RetrievalResult(
                    success=False,
                    data=[],
                    field_empty=True,  # <--- 动态判断
                    error_message="LLM返回了字段但在文档中未找到对应值",
                    field_extraction_reason=thinking
                )
            else:
                return RetrievalResult(
                    success=True,
                    data=result_data,
                    field_empty=False,  # <--- 动态判断
                    error_message="",
                    field_extraction_reason=thinking
                )

        except Exception as e:
            return RetrievalResult(
                success=False,
                field_empty=True,
                error_message=f"解析LLM响应异常: {str(e)}"
            )

    def _build_optimized_doc_summary(
        self,
        all_content: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        构建优化的文档摘要，包含字段列表和简短值预览

        Returns:
            (文档摘要列表, 估算的总token数)
        """
        doc_summaries = []
        total_text = ""

        for doc_name, content in all_content.items():
            structured_data = content.get('structured_data', {})

            doc_summary = {
                "document_name": doc_name,
                "fields": []
            }

            if isinstance(structured_data, dict):
                for field_name, value in structured_data.items():
                    # 截断过长的值
                    value_str = str(value)
                    if len(value_str) > self.MAX_FIELD_VALUE_LENGTH:
                        value_preview = value_str[:self.MAX_FIELD_VALUE_LENGTH] + "..."
                    else:
                        value_preview = value_str

                    doc_summary["fields"].append({
                        "field_name": field_name,
                        "value_preview": value_preview
                    })

            elif isinstance(structured_data, list):
                # 对于表格数据，提取字段名和样例
                if structured_data and isinstance(structured_data[0], dict):
                    fields = list(structured_data[0].keys())
                    doc_summary["fields"] = [{"field_name": f, "value_preview": "[表格数据]"} for f in fields]
                    doc_summary["record_count"] = len(structured_data)

            doc_summaries.append(doc_summary)

            # 累计文本用于token估算
            total_text += doc_name + " " + json.dumps(doc_summary, ensure_ascii=False)

        total_tokens = self._estimate_tokens(total_text)

        return doc_summaries, total_tokens

    def _build_batch_extraction_prompt(
        self,
        doc_summaries: List[Dict[str, Any]],
        task_goal: str,
        search_condition: Optional[str] = None,
        batch_info: str = ""
    ) -> str:
        """
        构建批量字段提取的提示词

        Args:
            doc_summaries: 文档摘要列表
            task_goal: 任务目标
            search_condition: 搜索条件
            batch_info: 批次信息（用于多批处理时显示）
        """
        condition_text = f"\n附加筛选条件: {search_condition}" if search_condition else ""
        batch_text = f" {batch_info}" if batch_info else ""

        # 格式化文档摘要
        doc_text_parts = []
        for doc in doc_summaries:
            doc_text = f"\n【文档: {doc['document_name']}】"
            if "record_count" in doc:
                doc_text += f" (共{doc['record_count']}条记录)"
            doc_text += "\n字段列表:"

            for field in doc.get("fields", []):
                field_name = field["field_name"]
                value_preview = field.get("value_preview", "")
                doc_text += f"\n  • {field_name}: {value_preview}"

            if doc.get("truncated"):
                doc_text += f"\n  ... {doc['truncated']}"

            doc_text_parts.append(doc_text)

        all_docs_text = "\n".join(doc_text_parts)

        prompt = f"""你是一位资深的运动医学专家。

请根据给定的条件文档和任务目标，从以下文档中选择任务高度相关的字段。\n
在关于{condition_text}的文档下，执行任务目标: {task_goal} 。\n

===== 文档内容 =====
\n
{all_docs_text} 
\n
===== 任务要求 =====
1. 分析任务目标，确定需要哪些数据\n
2. 从所有文档中识别与任务目标相关的字段\n
3. 字段名必须与文档中的原始字段名完全一致\n
4. 如果同一指标在多个文档中出现，都应该选择\n

请以JSON格式返回结果:\n
{{
    "reason": "选择这些字段的理由（简要说明）",
    "selected_fields": [
        {{"document": "文档名1", "field": "字段名1"}},
        {{"document": "文档名1", "field": "字段名2"}},
        {{"document": "文档名2", "field": "字段名3"}}
    ]
}}\n

如果没有找到符合条件的字段，请返回:\n
{{
    "reason": "说明为什么没有找到相关字段",
    "selected_fields": []
}}
"""
        return prompt

    def _extract_fields_from_content(
        self,
        all_content: Dict[str, Any],
        selected_fields: List[Union[str, Dict[str, str]]]
    ) -> List[Dict[str, Any]]:
        """
        从原始内容中提取选中字段的值，返回records格式

        对于表格数据：保留记录结构，筛选指定字段
        对于键值对数据：返回 [{'field': 字段名, 'value': 值}, ...]
        """
        # 按文档分组需要提取的字段
        doc_fields_map = {}  # {doc_name: [field1, field2, ...]}
        global_fields = []   # 未指定文档的字段

        for field_spec in selected_fields:
            if isinstance(field_spec, dict):
                doc_name = field_spec.get('document', '')
                field_name = field_spec.get('field', '')
            else:
                doc_name = ''
                field_name = field_spec

            if doc_name:
                if doc_name not in doc_fields_map:
                    doc_fields_map[doc_name] = []
                if field_name not in doc_fields_map[doc_name]:
                    doc_fields_map[doc_name].append(field_name)
            else:
                if field_name not in global_fields:
                    global_fields.append(field_name)

        result = []

        for content_doc_name, content in all_content.items():
            structured_data = content.get('structured_data', {})

            # 确定该文档需要提取的字段
            fields_to_extract = []
            if content_doc_name in doc_fields_map:
                fields_to_extract.extend(doc_fields_map[content_doc_name])
            fields_to_extract.extend(global_fields)

            if not fields_to_extract:
                continue

            if isinstance(structured_data, dict):
                # 键值对数据
                for field_name in fields_to_extract:
                    # 精确匹配
                    if field_name in structured_data:
                        result.append({
                            'document': content_doc_name,
                            'field': field_name,
                            'value': structured_data[field_name]
                        })


            elif isinstance(structured_data, list):
                # 表格数据：保留记录结构，筛选指定字段
                for record in structured_data:
                    if isinstance(record, dict):
                        filtered_record = {'文档名': content_doc_name}
                        has_field = False

                        for field_name in fields_to_extract:
                            # 精确匹配
                            if field_name in record:
                                filtered_record[field_name] = record[field_name]
                                has_field = True

                        if has_field:
                            result.append(filtered_record)

        return result

    def _validate_llm_result(
            self,
            result: RetrievalResult,
            all_content: Dict[str, Any]
    ) -> RetrievalResult:
        """
        验证LLM检索结果：
        1. 仅验证数据来源的文档是否存在。
        2. 只要字段在文档中存在（无论值是否为空/NaN），都予以保留。
        """

        # 如果结果本来就是空的，直接返回成功（由上层逻辑决定空是否合理）
        if not result.data:
            return RetrievalResult(
                success=True,
                data=[],
                field_extraction_reason=result.field_extraction_reason,
                error_message="检索结果为空"
            )

        valid_data = []

        for record in result.data:
            # 1. 识别文档来源
            # 兼容两种格式：{'_document': 'Name', ...} 或 {'document': 'Name', 'field': ...}
            doc_name = record.get('文档名') or record.get('document')

            # 2. 核心验证：只判断文档来源是否真实存在
            # 字段名的存在性已在 _extract_fields_from_content 中通过 if field in structured_data 保证
            if doc_name and doc_name in all_content:
                valid_data.append(record)
            else:
                # 只有当文档名是幻觉（不存在于all_content）时才丢弃
                logger.warning(f"丢弃无效数据: 文档 '{doc_name}' 不在原始内容列表中")

        # 3. 构造返回结果
        # 即使 valid_data 为空（例如文档名全是错的），也视为 Success=True (只要流程没崩)
        if len(valid_data) == 0:
            return RetrievalResult(
                success=True,
                field_empty=True,
                data=valid_data,
                field_extraction_reason=result.field_extraction_reason,
                error_message=""
            )
        else:
            return RetrievalResult(
                success=True,
                field_empty=False,
                data=valid_data,
                field_extraction_reason=result.field_extraction_reason,
                error_message="")

    # ==================== 便捷检索方法 ====================

    def retrieve_by_condition(
        self,
        center: str,
        patient: str,
        search_condition: str,
        task_goal: str,
        max_retries: int = 3
    ) -> RetrievalResult:
        """根据搜索条件检索（便捷方法）"""
        return self.retrieve_by_llm(
            center=center,
            patient=patient,
            task_goal=task_goal,
            search_condition=search_condition,
            max_retries=max_retries
        )

    def retrieve_by_documents(
        self,
        center: str,
        patient: str,
        document_names: Union[str, List[str]],
        task_goal: str,
        max_retries: int = 3
    ) -> RetrievalResult:
        """根据指定文档检索（便捷方法）"""
        return self.retrieve_by_llm(
            center=center,
            patient=patient,
            task_goal=task_goal,
            document_names=document_names,
            max_retries=max_retries
        )

    # ==================== LLM API 调用方法 ====================

    def _call_llm_api(
        self,
        prompt: str,
        temperature: float = 0.2,
        print_thinking: bool = True,
        task_name: str = "LLM检索",
        model: str = 'qwen3-max'
    ) -> Optional[Dict[str, Any]]:
        """调用LLM API"""

        if llm_api_config is None or deepseek_stream_calling is None:
            logger.warning("LLM配置未加载")
            return None

        api_key_list = [llm_api_config.QWEN_KEY_API]

        for api_key in api_key_list:
            try:
                if print_thinking:
                    print(f"🤖 [{task_name}] 大模型思考中...")

                result = qwen_stream_completion(model=model, api_key=api_key, content=prompt, temperature=temperature)

                if result:
                    parsed_result = result #self._parse_api_response(result)
                    if parsed_result:
                        if print_thinking:
                            print(f"\n{'='*50}")
                            print(f"✅ [{task_name}] 思考完成")
                            if 'reason' in parsed_result:
                                print(f"\n📝 理由: {parsed_result['reason']}")
                            if 'selected_documents' in parsed_result:
                                docs = parsed_result['selected_documents']
                                if docs:
                                    print(f"📁 选择文档: {docs}")
                            if 'selected_fields' in parsed_result:
                                fields = parsed_result['selected_fields']
                                if fields:
                                    print(f"📋 选择字段: {len(fields)}个")
                                    for f in fields:  # 只显示前5个
                                        if isinstance(f, dict):
                                            print(f"   - {f.get('document', '')}: {f.get('field', '')}")
                            print(f"{'='*50}\n")
                        return parsed_result
                    else:
                        logger.info('返回内容解析失败，尝试换用API KEY')
                        continue
                else:
                    logger.info('调用失败，尝试换用API KEY')
                    continue

            except Exception as e:
                logger.error(f'API调用异常: {str(e)}')
                continue

        logger.error('所有API KEY均调用失败')
        return None

    def _parse_api_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """解析API响应"""
        cleaned_text = response_text.strip()

        try:
            return json.loads(cleaned_text)
        except json.JSONDecodeError:
            pattern = r'\{.*\}'
            json_match = re.search(pattern, cleaned_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(), strict=False)
                except:
                    return None
            return None

    # ==================== 批量检索方法 ====================

    def batch_retrieve_by_field(
        self,
        center: str,
        patients: List[str],
        document_name: str,
        field_names: Union[str, List[str]]
    ) -> Dict[str, RetrievalResult]:
        """批量检索多个患者的指定字段"""
        results = {}
        for patient in patients:
            logger.info(f"正在检索患者 {patient} 的数据...")
            results[patient] = self.retrieve_by_field(
                center=center,
                patient=patient,
                document_name=document_name,
                field_names=field_names
            )
        return results

    def batch_retrieve_by_condition(
        self,
        center: str,
        patients: List[str],
        search_condition: str,
        task_goal: str
    ) -> Dict[str, RetrievalResult]:
        """批量根据条件检索多个患者的相关数据"""
        results = {}
        for patient in patients:
            logger.info(f"正在智能检索患者 {patient} 的数据...")
            results[patient] = self.retrieve_by_condition(
                center=center,
                patient=patient,
                search_condition=search_condition,
                task_goal=task_goal
            )
        return results

    def batch_retrieve_by_documents(
        self,
        center: str,
        patients: List[str],
        document_names: Union[str, List[str]],
        task_goal: str
    ) -> Dict[str, RetrievalResult]:
        """批量根据指定文档检索多个患者的相关数据"""
        results = {}
        for patient in patients:
            logger.info(f"正在检索患者 {patient} 的指定文档...")
            results[patient] = self.retrieve_by_documents(
                center=center,
                patient=patient,
                document_names=document_names,
                task_goal=task_goal
            )
        return results


# ==================== 使用示例 ====================

def example_usage():
    """使用示例"""

    # 初始化检索器（可配置最大token限制和默认扩展名）
    retriever = MedicalRetriever(
        data_dir="../data/raw_processed",
        output_dir="../data/interim",
        print_thinking=True,
        max_input_tokens=8096,
        default_extension=".csv"  # 默认扩展名
    )

    # # 示例1: 精确检索（文档名可以不带扩展名，默认添加.csv）
    print("\n" + "="*60)
    print("示例1: 精确检索（默认.csv扩展名）")
    print("="*60)
    result = retriever.retrieve_by_field(
        center="中山二院",
        patient="严振刚",
        document_name="入院记录",  # 自动补全为 入院记录.csv
        field_names=["性别", "年龄", "科室"]
    )
    print(f"检索成功: {result.success}")
    print(f"来源文档: {result.source_documents}")
    print(f"检索数据 (records格式):")
    for record in result.data:
        print(f"  {record}")

    # 示例2: 按搜索条件检索（自动分批）
    # print("\n" + "="*60)
    # print("示例2: 按搜索条件检索（自动分批）")
    # print("="*60)
    # result = retriever.retrieve_by_condition(
    #     center="中山二院",
    #     patient="严振刚",
    #     search_condition="术前",
    #     task_goal="现在要预测患者术后ACL-RSI的评分,请帮我找出与之相关的字段，辅助进行预测"
    # )
    # print(f"检索成功: {result.success}")
    # print(f"来源文档: {result.source_documents}")
    # print(f"文档选择理由: {result.document_selection_reason}")
    # print(f"字段提取理由: {result.field_extraction_reason}")
    # print(f"检索记录数: {len(result.data)}")

    # 示例3: 按指定多个文档检索（文档名可以不带扩展名）
    print("\n" + "="*60)
    print("示例3: 按指定文档检索（文档名自动补全扩展名）")
    print("="*60)
    result = retriever.retrieve_by_documents(
        center="中心A",
        patient="张三",
        document_names=[
            "入院记录",      # 自动补全为 入院记录.csv
            "血常规报告",    # 自动补全为 血常规报告.csv
            "生化检查",      # 自动补全为 生化检查.csv
            "凝血功能",
            "心电图报告",
            "超声心动图"
        ],
        task_goal="提取患者的感染指标、心功能和凝血相关数据"
    )
    # print(f"检索成功: {result.success}")
    # print(f"来源文档: {result.source_documents}")
    # print(f"检索记录数: {len(result.data)}")
    #
    # # 展示data可以直接转为DataFrame
    # if result.data:
    #     print("\n转为DataFrame:")
    #     df = pd.DataFrame(result.data)
    #     print(df.head())
    #
    # # 示例4: 批量检索
    # print("\n" + "="*60)
    # print("示例4: 批量检索多患者")
    # print("="*60)
    # patients = ["张三", "李四", "王五"]
    # results = retriever.batch_retrieve_by_condition(
    #     center="中心A",
    #     patients=patients,
    #     search_condition="术前",
    #     task_goal="提取术前血常规和凝血功能指标"
    # )
    # for patient, result in results.items():
    #     print(f"{patient}: 文档={result.source_documents}, 记录数={len(result.data)}, 成功={result.success}")


if __name__ == "__main__":
    example_usage()