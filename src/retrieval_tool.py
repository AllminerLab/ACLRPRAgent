from prompt_templates import render_prompt, system_prompt, retrieval_goal, RETRIEVAL_TASKS
"Medical record and examination report retrieval with batched LLM calls. Original module: medical_text_retrieval.py, 2025-12-28."
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


# Import project dependencies, using placeholders if unavailable.
try:
    from logger import init_logger
    from time_util import TimeTracker, ThinkTracker, TokenTracker
    import llm_api_config
    from streaming_llm_handler import stream_openai_response, deepseek_stream_calling, qwen_stream_completion
except ImportError:
    # Use placeholders when imports fail.
    init_logger = None
    TimeTracker = None
    ThinkTracker = None
    TokenTracker = None
    llm_api_config = None
    deepseek_stream_calling = None

# Import the OpenAI client if available.
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# Configure logging.
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
    "Create the directory if it does not exist."
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path



@dataclass
class RetrievalResult:
    "Container for retrieval results."
    success: bool
    data: List[Dict[str, Any]] = field(default_factory=list)
    field_empty: bool = True  # Indicate an empty result caused by missing fields.
    error_message: str = ""
    # thinking: str = ""
    source_file: str = ""
    source_documents: List[str] = field(default_factory=list)
    document_selection_reason: str = ""
    field_extraction_reason: str = ""


# Data-directory retrieval utilities

class DataPathRetriever:
    "Locate patient data, documents and missing files in a hierarchy of data_dir/center/patient/document.csv or document.pdf."

    def __init__(
        self,
        raw_dir: str = "../data/raw",
        raw_processed_dir: str = "../data/raw_processed",
        raw_pdf_extract_dir: str = "../data/raw_pdf_extract",
        interim_dir: str = "../data/interim"
    ):
        "Initialize source, cleaned, extracted and interim data paths using raw_dir, raw_processed_dir, raw_pdf_extract_dir and interim_dir."
        self.raw_dir = Path(raw_dir)
        self.raw_processed_dir = Path(raw_processed_dir)
        self.raw_pdf_extract_dir = Path(raw_pdf_extract_dir)
        self.interim_dir = Path(interim_dir)

    # Basic directory operations

    def get_data_centers(self, base_dir: Path = None) -> List[str]:
        "Return center names under base_dir, defaulting to raw_processed_dir."
        if base_dir is None:
            base_dir = self.raw_processed_dir
        
        if isinstance(base_dir, str):
            base_dir = Path(base_dir)
            
        if not base_dir.exists():
            return []
        return [d.name for d in base_dir.iterdir() if d.is_dir()]

    def get_patients_in_center(self, center: str, base_dir: Path = None) -> List[str]:
        "Return patient names in center under base_dir, defaulting to raw_processed_dir."
        if base_dir is None:
            base_dir = self.raw_processed_dir
            
        if isinstance(base_dir, str):
            base_dir = Path(base_dir)
            
        center_path = base_dir / center
        if not center_path.exists():
            return []
        return [p.name for p in center_path.iterdir() if p.is_dir()]

    def get_all_patients(self, base_dir: Path = None) -> Dict[str, List[str]]:
        "Return {center: [patient, ...]} for all centers under base_dir, defaulting to raw_processed_dir."
        result = {}
        centers = self.get_data_centers(base_dir)
        for center in centers:
            patients = self.get_patients_in_center(center, base_dir)
            if patients:
                result[center] = patients
        return result

    # Patient lookup methods

    def find_patient_in_center(
        self, 
        base_dir: Path, 
        center: str, 
        patient_name: str
    ) -> Optional[Tuple[str, Path]]:
        "Find patient_name in center under base_dir. Return (center, patient_path), or None."
        if isinstance(base_dir, str):
            base_dir = Path(base_dir)
            
        center_path = base_dir / center
        if not center_path.exists():
            logger.warning(f"""Data center {center} does not exist in {base_dir}""")
            return None
            
        patient_path = center_path / patient_name
        if patient_path.exists() and patient_path.is_dir():
            logger.info(f"""Found patient {patient_name} in {center}""")
            return (center, patient_path)
        return None

    def find_patient_in_dir(
        self, 
        base_dir: Path, 
        patient_name: str,
        center: str = None
    ) -> Optional[Tuple[str, Path]]:
        "Find patient_name under base_dir, searching the specified center or all centers. Return (center, patient_path), or None."
        if isinstance(base_dir, str):
            base_dir = Path(base_dir)
            
        if center:
            return self.find_patient_in_center(base_dir, center, patient_name)

        data_centers = self.get_data_centers(base_dir)
        for c in data_centers:
            center_path = base_dir / c
            patient_path = center_path / patient_name
            if patient_path.exists() and patient_path.is_dir():
                logger.info(f"""Found patient {patient_name} in {c}""")
                return (c, patient_path)
        return None

    def find_patient_in_raw(
        self, 
        patient_name: str,
        center: str = None
    ) -> Optional[Tuple[str, Path]]:
        "Find patient_name in raw, optionally restricted to center. Return (center, patient_path), or None."
        return self.find_patient_in_dir(self.raw_dir, patient_name, center)

    def find_patient_in_processed(
        self, 
        patient_name: str,
        center: str = None
    ) -> Optional[Tuple[str, Path]]:
        "Find patient_name in raw_processed, optionally restricted to center. Return (center, patient_path), or None."
        return self.find_patient_in_dir(self.raw_processed_dir, patient_name, center)

    def find_patient_in_pdf_extract(
        self, 
        patient_name: str,
        center: str = None
    ) -> Optional[Tuple[str, Path]]:
        "Find patient_name in raw_pdf_extract, optionally restricted to center. Return (center, patient_path), or None."
        return self.find_patient_in_dir(self.raw_pdf_extract_dir, patient_name, center)

    def find_patient_across_dirs(
        self, 
        patient_name: str,
        center: str = None
    ) -> Dict[str, Optional[Tuple[str, Path]]]:
        "Find patient_name across data directories, optionally restricted to center. Return a dictionary with raw, raw_processed and raw_pdf_extract entries, each containing (center, path) or None."
        return {
            'raw': self.find_patient_in_raw(patient_name, center),
            'raw_processed': self.find_patient_in_processed(patient_name, center),
            'raw_pdf_extract': self.find_patient_in_pdf_extract(patient_name, center)
        }

    # Document operations

    def get_doc_names_from_dir(
        self, 
        dir_path: Path, 
        extension: str = None
    ) -> set:
        "Return document stems in dir_path for extension, or all file types when extension is None."
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
        "Group document names in patient_path by extensions, defaulting to .csv, .pdf and .PDF. Return {extension: [document_names]}."
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
        "Return document names present in raw_path but absent from processed_path. progress_note_split determines whether preoperative/postoperative split notes count as present."
        if isinstance(raw_path, str):
            raw_path = Path(raw_path)
        if isinstance(processed_path, str):
            processed_path = Path(processed_path)

        # List all PDF and CSV documents in raw.
        raw_pdf_names = self.get_doc_names_from_dir(raw_path, '.pdf')
        raw_pdf_names.update(self.get_doc_names_from_dir(raw_path, '.PDF'))
        raw_pdf_names.update(self.get_doc_names_from_dir(raw_path, '.csv'))
        
        # List CSV documents in processed data.
        processed_csv_names = self.get_doc_names_from_dir(processed_path, '.csv')

        missing_docs = set()
        for doc_name in raw_pdf_names:
            if doc_name in processed_csv_names:
                continue
                
            # Treat split preoperative/postoperative progress notes as an available progress-note document.
            if progress_note_split and "Progress notes" in doc_name:
                pre_op_name = f"Preoperative_{doc_name}"
                post_op_name = f"Postoperative_{doc_name}"
                if pre_op_name in processed_csv_names or post_op_name in processed_csv_names:
                    logger.info(f"""'{doc_name}' already has preoperative/postoperative split versions; skipping""")
                    continue
                    
            missing_docs.add(doc_name)

        if missing_docs:
            logger.info(f"""Found {len(missing_docs)} missing documents: {missing_docs}""")

        return list(missing_docs)

    def check_patient_data_completeness(
        self, 
        patient_name: str,
        center: str = None
    ) -> Dict[str, Any]:
        "Check completeness for patient_name and optional center. Return existence flags raw_exists, processed_exists and pdf_extract_exists, and lists missing_docs, raw_docs and processed_docs."
        result = {
            'raw_exists': False,
            'processed_exists': False,
            'pdf_extract_exists': False,
            'missing_docs': [],
            'raw_docs': [],
            'processed_docs': []
        }
        
        # Locate the patient in each data directory.
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
            
        # Identify missing documents.
        if locations['raw'] and locations['raw_processed']:
            _, raw_path = locations['raw']
            _, processed_path = locations['raw_processed']
            result['missing_docs'] = self.find_missing_docs(raw_path, processed_path)
            
        return result

    # Directory listing methods

    def list_all_centers(self) -> Dict[str, List[str]]:
        "Return center lists for raw, raw_processed and raw_pdf_extract."
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
        "Return the patient directory for patient_name, optional center and data_type (raw, processed or pdf_extract), or None if absent."
        if data_type == 'raw':
            result = self.find_patient_in_raw(patient_name, center)
        elif data_type == 'processed':
            result = self.find_patient_in_processed(patient_name, center)
        elif data_type == 'pdf_extract':
            result = self.find_patient_in_pdf_extract(patient_name, center)
        else:
            logger.warning(f"""Unknown data type: {data_type}""")
            return None
            
        if result:
            _, path = result
            return path
        return None


class MedicalRetriever:
    "Retrieve clinical records and examination reports."

    # Token estimates: approximately 1.5-2 CJK characters or 4 English characters per token.
    CHARS_PER_TOKEN_CN = 1.8
    CHARS_PER_TOKEN_EN = 4.0

    # Default token limit, reserving space for the response
    DEFAULT_MAX_INPUT_TOKENS = 6000

    # Maximum field-value preview length
    MAX_FIELD_VALUE_LENGTH = 100

    # Base prompt token overhead, excluding document content
    PROMPT_BASE_TOKENS = 800

    # Default file extension
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
        "Initialize retrieval using data_dir, output_dir, supported_extensions, print_thinking, max_input_tokens and default_extension. The token limit constrains model input; the default extension applies to names without one."
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.supported_extensions = supported_extensions or ['.csv', '.txt', '.json', '.xlsx', '.xls', '.md']
        self.token_recorder = None
        self.print_thinking = print_thinking
        self.max_input_tokens = max_input_tokens or self.DEFAULT_MAX_INPUT_TOKENS
        self.default_extension = default_extension or self.DEFAULT_EXTENSION

    # Token estimation methods

    def _estimate_tokens(self, text: str) -> int:
        "Return an estimated token count for text."
        if not text:
            return 0

        # Count CJK and other characters separately.
        cn_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        other_chars = len(text) - cn_chars

        # Estimate the token count.
        cn_tokens = cn_chars / self.CHARS_PER_TOKEN_CN
        other_tokens = other_chars / self.CHARS_PER_TOKEN_EN

        return int(cn_tokens + other_tokens)

    def _truncate_to_token_limit(
        self,
        content_parts: List[Tuple[str, str, int]],  # [(doc_name, content, priority)]
        max_tokens: int
    ) -> List[Tuple[str, str]]:
        "Truncate content_parts, a list of (document_name, content, priority), to max_tokens. Lower priority values take precedence. Return (document_name, content) pairs."
        # Sort by priority.
        sorted_parts = sorted(content_parts, key=lambda x: x[2])

        result = []
        current_tokens = 0

        for doc_name, content, _ in sorted_parts:
            content_tokens = self._estimate_tokens(content)

            if current_tokens + content_tokens <= max_tokens:
                result.append((doc_name, content))
                current_tokens += content_tokens
            else:
                # Calculate the remaining token budget.
                remaining_tokens = max_tokens - current_tokens
                if remaining_tokens > 100:  # Only truncate if at least 100 tokens remain.
                    # Truncate content proportionally.
                    truncate_ratio = remaining_tokens / content_tokens
                    truncate_len = int(len(content) * truncate_ratio * 0.9)  # Reserve a 10% margin.
                    truncated_content = content[:truncate_len] + "\n... [Content truncated]"
                    result.append((doc_name, truncated_content))
                    logger.warning(f"""Truncated document {doc_name} to fit the token limit""")
                break

        return result

    def _split_documents_into_batches(
        self,
        doc_summaries: List[Dict[str, Any]],
        max_tokens_per_batch: int
    ) -> List[List[Dict[str, Any]]]:
        "Partition doc_summaries sequentially into batches within max_tokens_per_batch. Return a list of document-summary batches."
        # Document token budget after subtracting prompt overhead
        available_tokens = max_tokens_per_batch - self.PROMPT_BASE_TOKENS

        batches = []
        current_batch = []
        current_tokens = 0

        for doc in doc_summaries:
            doc_text = json.dumps(doc, ensure_ascii=False)
            doc_tokens = self._estimate_tokens(doc_text)

            # Start a new batch if adding this document would exceed the limit.
            if current_tokens + doc_tokens > available_tokens:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [doc]
                current_tokens = doc_tokens
            else:
                current_batch.append(doc)
                current_tokens += doc_tokens

        # Include the final batch.
        if current_batch:
            batches.append(current_batch)

        if len(batches) > 1:
            logger.info(f"""Split documents into {len(batches)} batches; documents per batch: {[len(b) for b in batches]}""")

        return batches

    def _truncate_single_doc_summary(
        self,
        doc: Dict[str, Any],
        max_tokens: int
    ) -> Dict[str, Any]:
        "Truncate one document summary to fit the token limit."
        truncated_doc = {
            "document_name": doc["document_name"],
            "fields": []
        }

        if "record_count" in doc:
            truncated_doc["record_count"] = doc["record_count"]

        current_tokens = self._estimate_tokens(json.dumps(truncated_doc, ensure_ascii=False))

        for field in doc.get("fields", []):
            field_tokens = self._estimate_tokens(json.dumps(field, ensure_ascii=False))

            if current_tokens + field_tokens <= max_tokens * 0.9:  # Reserve a 10% margin.
                truncated_doc["fields"].append(field)
                current_tokens += field_tokens
            else:
                # Truncate field values if needed.
                truncated_field = {
                    "field_name": field["field_name"],
                    "value_preview": field.get("value_preview", "")[:30] + "..."
                }
                truncated_doc["fields"].append(truncated_field)
                break

        remaining = len(doc.get("fields", [])) - len(truncated_doc["fields"])
        if remaining > 0:
            truncated_doc["truncated"] = f"""{remaining} additional fields not shown"""
            logger.warning(f"""Truncated fields in {doc['document_name']}; retained {len(truncated_doc['fields'])}""")

        return truncated_doc

    # Directory scanning methods

    def get_centers(self) -> List[str]:
        "List all data centers."
        if not os.path.exists(self.data_dir):
            logger.error(f"""Data directory does not exist: {self.data_dir}""")
            return []

        centers = [
            d for d in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, d))
        ]
        return centers

    def get_patients(self, center: str) -> List[str]:
        "List patients in the specified center."
        center_path = os.path.join(self.data_dir, center)
        if not os.path.exists(center_path):
            logger.error(f"""Data center does not exist: {center_path}""")
            return []

        patients = [
            d for d in os.listdir(center_path)
            if os.path.isdir(os.path.join(center_path, d))
        ]
        return patients

    def get_documents(self, center: str, patient: str) -> List[str]:
        "List documents for the specified patient."
        patient_path = os.path.join(self.data_dir, center, patient)
        if not os.path.exists(patient_path):
            logger.error(f"""Patient directory does not exist: {patient_path}""")
            return []

        documents = []
        for ext in self.supported_extensions:
            pattern = os.path.join(patient_path, f"*{ext}")
            documents.extend([os.path.basename(f) for f in glob.glob(pattern)])

        return documents

    def _has_extension(self, filename: str) -> bool:
        "Check whether a filename includes an extension."
        _, ext = os.path.splitext(filename)
        return bool(ext)

    def _normalize_document_name(self, document_name: str) -> str:
        "Normalize document_name by adding the default extension when absent."
        if not self._has_extension(document_name):
            return document_name + self.default_extension
        return document_name

    def get_document_path(self, center: str, patient: str, document_name: str) -> Optional[str]:
        "Return the document path for center, patient and document_name, adding .csv when the extension is absent. Return None if no file matches."
        patient_path = os.path.join(self.data_dir, center, patient)

        # 1. Try the original document name.
        doc_path = os.path.join(self.data_dir, center, patient, document_name)
        if os.path.exists(doc_path):
            return doc_path

        # 2. Add the default .csv extension when absent.
        if not self._has_extension(document_name):
            doc_path_with_ext = os.path.join(self.data_dir, center, patient, document_name + self.default_extension)
            if os.path.exists(doc_path_with_ext):
                return doc_path_with_ext

            # 3. Try other supported extensions.
            for ext in self.supported_extensions:
                doc_path_with_ext = os.path.join(self.data_dir, center, patient, document_name + ext)
                if os.path.exists(doc_path_with_ext):
                    return doc_path_with_ext

        # 4. Try partial-name matching.
        if os.path.exists(patient_path):
            for file in os.listdir(patient_path):
                # Compare names without extensions.
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
        "Return (document_name, path) pairs matching pattern within center and patient. Partial-name matching is supported."
        all_docs = self.get_documents(center, patient)
        matched = []

        # Remove the pattern extension before matching.
        pattern_base = os.path.splitext(pattern)[0]

        for doc_name in all_docs:
            doc_base = os.path.splitext(doc_name)[0]
            if pattern_base.lower() in doc_base.lower():
                doc_path = self.get_document_path(center, patient, doc_name)
                if doc_path:
                    matched.append((doc_name, doc_path))

        return matched

    # Document reading methods

    def read_document(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        "Read file_path and return (raw_text, structured_data)."
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"""File not found: {file_path}""")

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
            logger.error(f"""Failed to read file {file_path}: {str(e)}""")
            raise

        return raw_text, structured_data

    def _parse_medical_text(self, text: str) -> Dict[str, Any]:
        "Parse clinical text into fields and values."
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
        "Parse clinical text in Markdown format."
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
        "Format a field value."
        if len(value_list) == 1:
            return value_list[0]
        elif len(value_list) > 1:
            return value_list
        return ""

    # Exact field retrieval methods

    def retrieve_by_field(
        self,
        center: str,
        patient: str,
        document_name: str,
        field_names: Union[str, List[str]]
    ) -> RetrievalResult:
        "Retrieve field_names, either one name or a list, from document_name for center and patient. Add the default .csv extension when absent."
        if isinstance(field_names, str):
            field_names = [field_names]

        doc_path = self.get_document_path(center, patient, document_name)
        if not doc_path:
            return RetrievalResult(
                success=False,
                error_message=f"""Document not found: center={center}, patient={patient}, document={document_name}"""
            )

        try:
            raw_text, structured_data = self.read_document(doc_path)
        except Exception as e:
            return RetrievalResult(
                success=False,
                error_message=f"""Failed to read document: {str(e)}"""
            )

        result_data = []
        not_found_fields = []

        # Use the actual document name from its path.
        actual_doc_name = os.path.basename(doc_path)

        if isinstance(structured_data, dict):
            # Key-value data: one record per field.
            for field_name in field_names:
                if field_name in structured_data:
                    result_data.append({
                        'field': field_name,
                        'value': structured_data[field_name]
                    })
                else:
                    not_found_fields.append(field_name)

        elif isinstance(structured_data, list):
            # Tabular data: retain record structure and select requested fields.
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
                field_empty=False,  # Successful retrieval with data
                source_file=doc_path,
                source_documents=[actual_doc_name],
                error_message=f"""Fields not found: {not_found_fields}""" if not_found_fields else ""
            )
        else:
            # Return success=True and field_empty=True when no fields match.
            return RetrievalResult(
                success=True,  # The operation completed successfully.
                data=[],
                field_empty=True,  # The result is empty.
                error_message=f"""None of the specified fields were found: {field_names}""",  # This message is informational.
                source_file=doc_path,
                source_documents=[os.path.basename(doc_path)]
            )

    def retrieve_all_fields(
        self,
        center: str,
        patient: str,
        document_name: str
    ) -> RetrievalResult:
        "Retrieve every field from document_name for center and patient, adding the default .csv extension when absent."
        doc_path = self.get_document_path(center, patient, document_name)
        if not doc_path:
            return RetrievalResult(
                success=False,
                error_message=f"""Document not found: center={center}, patient={patient}, document={document_name}"""
            )

        try:
            raw_text, structured_data = self.read_document(doc_path)
            print(structured_data)
            actual_doc_name = os.path.basename(doc_path)

            # Convert to records format.
            if isinstance(structured_data, dict):
                # Key-value data
                result_data = [{'field': k, 'value': v} for k, v in structured_data.items()]
            elif isinstance(structured_data, list):
                # Data are already in records format.
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
                error_message=f"""Failed to read document: {str(e)}"""
            )

    # LLM-assisted semantic retrieval methods

    def retrieve_by_llm(
        self,
        center: str,
        patient: str,
        task_goal: str,
        search_condition: Optional[str] = None,
        document_names: Optional[Union[str, List[str]]] = None,
        max_retries: int = 3
    ) -> RetrievalResult:
        "Retrieve fields with an LLM. Select documents by search_condition or use document_names directly, then select fields according to task_goal. center and patient identify the data; max_retries limits attempts. Return a RetrievalResult."
        if isinstance(document_names, str):
            document_names = [document_names]

        all_doc_names = self.get_documents(center, patient)
        if not all_doc_names:
            return RetrievalResult(
                success=False,
                error_message=f"""No documents found: center={center}, patient={patient}"""
            )

        # Stage 1: Document selection
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
                    error_message=f"""Specified documents not found: {document_names}"""
                )

            doc_selection_reason = f"""Using specified documents: {[d[0] for d in selected_docs]}"""

        elif search_condition:
            selected_docs, doc_selection_reason = self._select_documents_by_condition(
                center=center,
                patient=patient,
                all_doc_names=all_doc_names,
                search_condition=search_condition,
                max_retries=max_retries
            )

            if not selected_docs:
                # No matching documents is an empty result, not a system error.
                return RetrievalResult(
                    success=True,  # The retrieval operation completed.
                    data=[],
                    field_empty=True,  # No documents means no fields; mark the result empty.
                    error_message=f"""No documents matched condition '{search_condition}'""",
                    document_selection_reason=doc_selection_reason
                )
        else:
            return RetrievalResult(
                success=False,
                error_message="Specify either search_condition or document_names"
            )

        # Stage 2: Read documents and select fields in a combined call.
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
                logger.warning(f"""Failed to read document {doc_name}: {str(e)}""")
                continue

        if not all_content:
            return RetrievalResult(
                success=False,
                error_message="Unable to read any selected document",
                source_documents=[d[0] for d in selected_docs],
                document_selection_reason=doc_selection_reason
            )

        # Select fields with a combined LLM call.
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
                        logger.warning(f"""Retrieval attempt {attempt + 1} failed validation: {validated_result.error_message}""")
                else:
                    logger.warning(f"""LLM retrieval attempt {attempt + 1} failed: {result.error_message}""")

            except Exception as e:
                logger.error(f"""LLM call attempt {attempt + 1} raised an exception: {str(e)}""")

        return RetrievalResult(
            success=False,
            error_message=f"""Retrieval failed after {max_retries} attempts""",
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
        "Select relevant documents using a search condition."
        # First try simple keyword matching.
        # keyword_matched = []
        # for doc_name in all_doc_names:
        #     if search_condition.lower() in doc_name.lower():
        #         doc_path = self.get_document_path(center, patient, doc_name)
        #         if doc_path:
        #             keyword_matched.append((doc_name, doc_path))
        #
        # if keyword_matched:
        # A keyword match can provide a direct document-selection rationale.
        #     return keyword_matched, reason

        # Use LLM selection when keyword matching fails.
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
                        reason = result.get('reason', f"""LLM selected documents for condition '{search_condition}'""")
                        return selected, reason

            except Exception as e:
                logger.info(f"""Document selection attempt {attempt + 1} raised an exception: {str(e)}""")

        return [], f"""No documents found for condition '{search_condition}'"""

    def _llm_select_documents(
        self,
        all_doc_names: List[str],
        search_condition: str,
        model: str = 'qwen3-max',
    ) -> Optional[Dict[str, Any]]:
        "Call the LLM to select relevant records."

        prompt = render_prompt("record_selection", **{"available curated record list": json.dumps(all_doc_names, ensure_ascii=False, indent=2), "record-level search condition": search_condition})
        return self._call_llm_api(
            prompt=prompt,
            print_thinking=self.print_thinking,
            model =model,
            task_name=f"""Record selection - {search_condition}"""
        )

    # Batched field selection methods

    def _extract_fields_by_llm_batch(
        self,
        all_content: Dict[str, Any],
        task_goal: str,
        search_condition: Optional[str] = None
    ) -> RetrievalResult:
        "Select fields from all_content in sequential LLM batches. all_content maps document names to raw_text, structured_data and path; task_goal specifies the field-selection objective; search_condition is optional."
        # Build compact document summaries.
        doc_summaries, total_tokens = self._build_optimized_doc_summary(all_content)

        # Calculate the document token budget.
        available_tokens = self.max_input_tokens - self.PROMPT_BASE_TOKENS

        # Determine whether batching is needed.
        if total_tokens <= available_tokens:
            # Single-batch processing
            return self._process_single_batch(
                doc_summaries=doc_summaries,
                all_content=all_content,
                task_goal=task_goal,
                search_condition=search_condition
            )
        else:
            # Multiple-batch processing
            logger.info(f"""Document tokens ({total_tokens}) exceed the limit ({available_tokens}); using batches""")
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
        "Process one document batch."
        prompt = self._build_batch_extraction_prompt(
            doc_summaries=doc_summaries,
            task_goal=task_goal,
            search_condition=search_condition
        )

        llm_response = self._call_llm_api(
            prompt=prompt,
            print_thinking=self.print_thinking,
            task_name=f"""Field selection - {task_goal}..."""
        )

        if llm_response is None:
            return RetrievalResult(
                success=False,
                error_message="LLM API call failed"
            )

        return self._parse_extraction_response(llm_response, all_content)

    def _process_multiple_batches(
        self,
        doc_summaries: List[Dict[str, Any]],
        all_content: Dict[str, Any],
        task_goal: str,
        search_condition: Optional[str]
    ) -> RetrievalResult:
        "Process multiple document batches and merge results."
        # Partition documents sequentially.
        batches = self._split_documents_into_batches(
            doc_summaries=doc_summaries,
            max_tokens_per_batch=self.max_input_tokens
        )

        total_batches = len(batches)
        logger.info(f"""Split into {total_batches} batches""")

        if self.print_thinking:
            print(f"""\nBatch details: {len(doc_summaries)} documents in {total_batches} batches""")

        # Process each batch.
        all_selected_fields = []
        all_reasons = []

        for i, batch_docs in enumerate(batches):
            if self.print_thinking:
                print(f"\n{'='*50}")
                print(f"""Processing batch {i+1}/{total_batches} ({len(batch_docs)} documents)""")
                print(f"{'='*50}")

            prompt = self._build_batch_extraction_prompt(
                doc_summaries=batch_docs,
                task_goal=task_goal,
                search_condition=search_condition,
                batch_info=f"""(batch {i+1}/{total_batches})""" if total_batches > 1 else ""
            )

            llm_response = self._call_llm_api(
                prompt=prompt,
                print_thinking=self.print_thinking,
                task_name=f"""batch {i+1}/{total_batches}"""
            )

            if llm_response:
                selected_fields = llm_response.get('selected_fields', [])
                reason = llm_response.get('reason', '')

                if selected_fields and selected_fields != 'nan':
                    all_selected_fields.extend(selected_fields)
                    all_reasons.append(f"""Batch {i+1}: {reason}""")
            else:
                logger.warning(f"""Batch {i+1} failed""")

        # Merge results.
        if not all_selected_fields:
            # Treat the result as empty only if at least one API attempt succeeded.
            # Complete API failure should be returned as success=False.
            if all_reasons:
                return RetrievalResult(
                    success=True,  # The API call succeeded.
                    data=[],
                    field_empty=True,  # The LLM found no matching data.
                    error_message="No relevant fields found in any batch",
                    field_extraction_reason="; ".join(all_reasons)
                )
            else:
                return RetrievalResult(
                    success=False,  # All API calls may have failed.
                    error_message="API call failed or returned no response"
                )

        # Remove duplicates.
        unique_fields = self._deduplicate_fields(all_selected_fields)

        # Extract selected field values from the original records.
        result_data = self._extract_fields_from_content(
            all_content=all_content,
            selected_fields=unique_fields
        )

        combined_reason = "; ".join(all_reasons) if len(all_reasons) > 1 else (all_reasons[0] if all_reasons else "")

        if self.print_thinking:
            print(f"""\nAll batches complete: {len(unique_fields)} fields selected, {len(result_data)} records extracted""")

        return RetrievalResult(
            success=True,
            data=result_data,
            field_extraction_reason=combined_reason
        )

    def _deduplicate_fields(
        self,
        fields: List[Union[str, Dict[str, str]]]
    ) -> List[Dict[str, str]]:
        "Remove duplicate field selections."
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
        "Parse the LLM field-selection response."
        try:
            selected_fields = llm_response.get('selected_fields', [])
            thinking = llm_response.get('reason', '')

            # Case 1: An empty list or 'nan' means successful retrieval with no fields.
            if not selected_fields or selected_fields == 'nan':
                return RetrievalResult(
                    success=True,
                    data=[],
                    field_empty=True,  # Mark the result as empty.
                    error_message="LLM found no relevant fields",
                    field_extraction_reason=thinking
                )

            # Case 2: Resolve the selected fields against the original records.
            result_data = self._extract_fields_from_content(
                all_content=all_content,
                selected_fields=selected_fields
            )

            # Case 3: Nonexistent field names yield an empty extraction result.
            if len(result_data) == 0:
                return RetrievalResult(
                    success=False,
                    data=[],
                    field_empty=True,  # Determine emptiness from the extracted data.
                    error_message="LLM returned field names, but no corresponding values were found in the documents",
                    field_extraction_reason=thinking
                )
            else:
                return RetrievalResult(
                    success=True,
                    data=result_data,
                    field_empty=False,  # Determine emptiness from the extracted data.
                    error_message="",
                    field_extraction_reason=thinking
                )

        except Exception as e:
            return RetrievalResult(
                success=False,
                field_empty=True,
                error_message=f"""Exception parsing LLM response: {str(e)}"""
            )

    def _build_optimized_doc_summary(
        self,
        all_content: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], int]:
        "Build document summaries with field names and short value previews. Return (summaries, estimated_total_tokens)."
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
                    # Truncate long values.
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
                # For tabular data, collect column names and example values.
                if structured_data and isinstance(structured_data[0], dict):
                    fields = list(structured_data[0].keys())
                    doc_summary["fields"] = [{"field_name": f, "value_preview": "[Tabular data]"} for f in fields]
                    doc_summary["record_count"] = len(structured_data)

            doc_summaries.append(doc_summary)

            # Accumulate text for token estimation.
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
        "Build a field-selection prompt from doc_summaries, task_goal and optional search_condition. batch_info labels a batch for progress reporting."
        condition_text = f"""\nAdditional selection condition: {search_condition}""" if search_condition else ""
        batch_text = f" {batch_info}" if batch_info else ""

        # Format document summaries.
        doc_text_parts = []
        for doc in doc_summaries:
            doc_text = f"""\n[Document: {doc['document_name']}]"""
            if "record_count" in doc:
                doc_text += f""" ({doc['record_count']} records)"""
            doc_text += "\nFields:"

            for field in doc.get("fields", []):
                field_name = field["field_name"]
                value_preview = field.get("value_preview", "")
                doc_text += f"\n  • {field_name}: {value_preview}"

            if doc.get("truncated"):
                doc_text += f"\n  ... {doc['truncated']}"

            doc_text_parts.append(doc_text)

        all_docs_text = "\n".join(doc_text_parts)

        prompt = render_prompt("field_selection", **{"record-level search condition or specified target records": search_condition or ", ".join(doc["document_name"] for doc in doc_summaries), "field-level task goal": task_goal, "field-level summaries from candidate records": all_docs_text})
        return prompt

    def _extract_fields_from_content(
        self,
        all_content: Dict[str, Any],
        selected_fields: List[Union[str, Dict[str, str]]]
    ) -> List[Dict[str, Any]]:
        "Extract selected values from the original records. Preserve row structure for tables; return field/value records for key-value data."
        # Group requested fields by document.
        doc_fields_map = {}  # {doc_name: [field1, field2, ...]}
        global_fields = []   # Fields without a specified document

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

            # Determine which fields to extract from this document.
            fields_to_extract = []
            if content_doc_name in doc_fields_map:
                fields_to_extract.extend(doc_fields_map[content_doc_name])
            fields_to_extract.extend(global_fields)

            if not fields_to_extract:
                continue

            if isinstance(structured_data, dict):
                # Key-value data
                for field_name in fields_to_extract:
                    # Exact matching
                    if field_name in structured_data:
                        result.append({
                            'document': content_doc_name,
                            'field': field_name,
                            'value': structured_data[field_name]
                        })


            elif isinstance(structured_data, list):
                # Tabular data: retain record structure and select requested fields.
                for record in structured_data:
                    if isinstance(record, dict):
                        filtered_record = {"document_name": content_doc_name}
                        has_field = False

                        for field_name in fields_to_extract:
                            # Exact matching
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
        "Validate the source documents of retrieval results. Retain fields present in the source even if their values are empty or NaN."

        # Return a successful empty result; callers decide whether emptiness is acceptable.
        if not result.data:
            return RetrievalResult(
                success=True,
                data=[],
                field_extraction_reason=result.field_extraction_reason,
                error_message="Retrieval result is empty"
            )

        valid_data = []

        for record in result.data:
            # 1. Identify the source document.
            # Support both {'_document': 'Name', ...} and {'document': 'Name', 'field': ...}.
            doc_name = record.get("document_name") or record.get('document')

            # 2. Verify that the source document exists.
            # Field existence is checked against structured_data in _extract_fields_from_content.
            if doc_name and doc_name in all_content:
                valid_data.append(record)
            else:
                # Discard a record only when its source document is absent from all_content.
                logger.warning(f"""Discarding invalid data: document '{doc_name}' is absent from the original content""")

        # 3. Construct the return value.
        # An empty valid_data list still represents a completed retrieval operation.
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

    # Convenience retrieval methods

    def retrieve_by_condition(
        self,
        center: str,
        patient: str,
        search_condition: str,
        task_goal: str,
        max_retries: int = 3
    ) -> RetrievalResult:
        "Retrieve using a search condition."
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
        "Retrieve from specified documents."
        return self.retrieve_by_llm(
            center=center,
            patient=patient,
            task_goal=task_goal,
            document_names=document_names,
            max_retries=max_retries
        )

    # LLM API methods

    def _call_llm_api(
        self,
        prompt: str,
        temperature: float = 0.2,
        print_thinking: bool = True,
        task_name: str = "LLM retrieval",
        model: str = 'qwen3-max'
    ) -> Optional[Dict[str, Any]]:
        "Call the LLM API."

        if llm_api_config is None or deepseek_stream_calling is None:
            logger.warning("LLM configuration is unavailable")
            return None

        api_key_list = [llm_api_config.QWEN_KEY_API]

        for api_key in api_key_list:
            try:
                if print_thinking:
                    print(f"""[{task_name}] Model reasoning in progress""")

                result = qwen_stream_completion(model=model, api_key=api_key, content=prompt, temperature=temperature)

                if result:
                    parsed_result = result  # Normalize the appendix schema for internal callers.
                    if "selected_records" in parsed_result:
                        parsed_result["selected_documents"] = parsed_result.pop("selected_records")
                    for field in parsed_result.get("selected_fields", []):
                        if isinstance(field, dict) and "record" in field:
                            field["document"] = field.pop("record")
                    if parsed_result:
                        if print_thinking:
                            print(f"\n{'='*50}")
                            print(f"""[{task_name}] Reasoning complete""")
                            if 'reason' in parsed_result:
                                print(f"""\nRationale: {parsed_result['reason']}""")
                            if 'selected_documents' in parsed_result:
                                docs = parsed_result['selected_documents']
                                if docs:
                                    print(f"""Selected documents: {docs}""")
                            if 'selected_fields' in parsed_result:
                                fields = parsed_result['selected_fields']
                                if fields:
                                    print(f"""Selected fields: {len(fields)}""")
                                    for f in fields:  # Display selected fields.
                                        if isinstance(f, dict):
                                            print(f"   - {f.get('document', '')}: {f.get('field', '')}")
                            print(f"{'='*50}\n")
                        return parsed_result
                    else:
                        logger.info("Response parsing failed; trying another API key")
                        continue
                else:
                    logger.info("Call failed; trying another API key")
                    continue

            except Exception as e:
                logger.error(f"""API call exception: {str(e)}""")
                continue

        logger.error("All API keys failed")
        return None

    def _parse_api_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        "Parse an API response."
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

    # Batch retrieval methods

    def batch_retrieve_by_field(
        self,
        center: str,
        patients: List[str],
        document_name: str,
        field_names: Union[str, List[str]]
    ) -> Dict[str, RetrievalResult]:
        "Retrieve specified fields for multiple patients."
        results = {}
        for patient in patients:
            logger.info(f"""Retrieving data for patient {patient}""")
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
        "Retrieve condition-matched data for multiple patients."
        results = {}
        for patient in patients:
            logger.info(f"""Running semantic retrieval for patient {patient}""")
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
        "Retrieve specified documents for multiple patients."
        results = {}
        for patient in patients:
            logger.info(f"""Retrieving selected documents for patient {patient}""")
            results[patient] = self.retrieve_by_documents(
                center=center,
                patient=patient,
                document_names=document_names,
                task_goal=task_goal
            )
        return results


# Usage examples

def example_usage():
    "Demonstrate retrieval with synthetic patient identifiers."

    # Initialize the retriever with configurable token limit and default extension.
    retriever = MedicalRetriever(
        data_dir="../data/raw_processed",
        output_dir="../data/interim",
        print_thinking=True,
        max_input_tokens=8096,
        default_extension=".csv"  # Default extension
    )

    # Example 1: Exact retrieval; .csv is added when the extension is absent.
    print("\n" + "="*60)
    print("Example 1: Exact retrieval (default .csv extension)")
    print("="*60)
    result = retriever.retrieve_by_field(
        center="Center-A",
        patient="Patient-001",
        document_name="Admission note",  # Automatically completed to Admission note.csv.
        field_names=["Sex", "Age", "Department"]
    )
    print(f"""Retrieval succeeded: {result.success}""")
    print(f"""Source documents: {result.source_documents}""")
    print(f"""Retrieved data (records format):""")
    for record in result.data:
        print(f"  {record}")

    # Example 2: Retrieve by search condition with automatic batching.
    # print("\n" + "="*60)
    # print("Example 2: Search-condition retrieval with automatic batching")
    # print("="*60)
    # result = retriever.retrieve_by_condition(
    #     center="Center-A",
    #     patient="Patient-001",
    #     search_condition="Preoperative records",
    #     task_goal=retrieval_goal(6)
    # )
    # print(f"Retrieval succeeded: {result.success}")
    # print(f"Source documents: {result.source_documents}")
    # print(f"Document selection rationale: {result.document_selection_reason}")
    # print(f"Field selection rationale: {result.field_extraction_reason}")
    # print(f"Retrieved records: {len(result.data)}")

    # Example 3: Retrieve several specified documents, with optional extensions.
    print("\n" + "="*60)
    print("Example 3: Retrieve selected documents (extensions completed automatically)")
    print("="*60)
    result = retriever.retrieve_by_documents(
        center="Center-A",
        patient="Patient-001",
        document_names=[
            "Admission note",      # Automatically completed to Admission note.csv.
            "Complete blood count report",    # Automatically completed to Complete blood count report.csv.
            "Biochemistry tests",      # Automatically completed to Biochemistry tests.csv.
            "Coagulation function",
            "Electrocardiogram report",
            "Echocardiogram"
        ],
        task_goal="Retrieve infection markers, cardiac function and coagulation data"
    )
    # print(f"Retrieval succeeded: {result.success}")
    # print(f"Source documents: {result.source_documents}")
    # print(f"Retrieved records: {len(result.data)}")
    #
    # Convert retrieved records directly to a DataFrame.
    # if result.data:
    #     print("\nDataFrame:")
    #     df = pd.DataFrame(result.data)
    #     print(df.head())
    #
    # Example 4: Batch retrieval
    # print("\n" + "="*60)
    # print("Example 4: Retrieve multiple patients")
    # print("="*60)
    # patients = ["Patient-001", "Patient-002", "Patient-003"]
    # results = retriever.batch_retrieve_by_condition(
    #     center="Center-A",
    #     patients=patients,
    #     search_condition="Preoperative records",
    #     task_goal="Retrieve preoperative complete blood count and coagulation indicators"
    # )
    # for patient, result in results.items():
    #     print(f"{patient}: documents={result.source_documents}, records={len(result.data)}, success={result.success}")


if __name__ == "__main__":
    example_usage()