from prompt_templates import render_prompt, system_prompt, retrieval_goal, RETRIEVAL_TASKS

import os, sys, json, base64
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Set
import pandas as pd
from openai import OpenAI
import logging
import fitz
import warnings
from logger import init_logger
from pdf_fields_config import FIELDS_DICT
from llm_api_config import QWEN_KEY_API
from retrieval_tool import DataPathRetriever

warnings.filterwarnings('ignore')
base_dir = os.path.split(os.path.realpath(__file__))[0]
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
if not logger.handlers:
    logger = init_logger('../data/log')

RAW_PROCESSED_DIR = "../data/raw_processed"
RAW_PDF_EXTRACT_DIR = "../data/raw_pdf_extract"
RAW_DIR = "../data/raw"
# Progress-note column definitions
COMMON_COLS = ["Patient name", "Sex", "Age", "Department", "Bed number", "Inpatient ID"]

PRE_OP_COLS = COMMON_COLS + [
    "Initial progress note date", "Initial progress note",
    "Preoperative day 1 ward-round note date", "Preoperative day 1 ward-round note",
    "Preoperative day 2 ward-round note date", "Preoperative day 2 ward-round note"
]

POST_OP_COLS = COMMON_COLS + [
    "Day-of-surgery postoperative progress note date", "Day-of-surgery postoperative progress note",
    "Postoperative day 1 senior physician ward-round date", "Postoperative day 1 senior physician ward-round note",
    "Postoperative day 2 ward-round note date", "Postoperative day 2 ward-round note",
    "Postoperative day 3 routine ward-round note date", "Postoperative day 3 routine ward-round note",
    "Discharge progress note date", "Discharge progress note"
]

def create_dir_if_not_exist(dir_path: str) -> str:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


class PDFExtractor:
    "Extract PDF content as single-entry or multi-entry records."

    def __init__(self, api_type: str = "qwen", api_key: str = None,
                 base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
                 model: str = "qwen3-vl-plus"):
        self.api_type = api_type
        self.api_key = api_key or QWEN_KEY_API
        self.model = model
        self.base_url = base_url

    def pdf_to_images(self, pdf_path: str, dpi: int = 200) -> List[str]:
        images_base64 = []
        pdf = fitz.open(pdf_path)
        for page in pdf:
            zoom = dpi / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")
            images_base64.append(img_base64)
        pdf.close()
        return images_base64

    def build_single_row_prompt(self, fields: List[str]) -> str:
        return render_prompt('single_entry', **{"predefined clinical field list": ", ".join(fields)})

    def build_multi_row_prompt(self, fields: List[str]) -> str:
        return render_prompt('multi_entry', **{"predefined clinical field list": ", ".join(fields)})

    def build_prompt(self, fields: List[str], multi_row: bool = False) -> str:
        if multi_row:
            return self.build_multi_row_prompt(fields)
        return self.build_single_row_prompt(fields)

    def extract_fields_from_image(self, images_base64: List[str], fields: List[str],
                                   multi_row: bool = False) -> Union[Dict[str, str], List[Dict[str, str]]]:
        prompt = self.build_prompt(fields, multi_row)
        content = [{"type": "text", "text": prompt}]
        for img_base64 in images_base64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_base64}"}
            })

        try:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            completion = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}]
            )
            result_text = completion.choices[0].message.content
            logger.info(f"""Extracted content:\n{result_text}""")

            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
            result_text = result_text.strip()

            result = json.loads(result_text)

            if multi_row:
                if isinstance(result, dict):
                    result = [result]
                return result
            else:
                if isinstance(result, list) and len(result) > 0:
                    result = result[0]
                return result

        except json.JSONDecodeError:
            logger.error(f"""JSON parsing failed: {result_text}""")
            empty = {field: "Parsing failed" for field in fields}
            return None
        except Exception as e:
            logger.error(f"""API call failed: {e}""")
            return None

    def extract_from_pdf(self, pdf_path: str, fields: List[str],
                         multi_row: bool = False) -> Union[Dict[str, str], List[Dict[str, str]]]:
        logger.info(f"""Processing PDF: {pdf_path}, multi-entry mode: {multi_row}""")
        images_base64 = self.pdf_to_images(pdf_path)
        result = self.extract_fields_from_image(images_base64, fields, multi_row)

        if result is None:
            logger.info("Extraction or JSON parsing failed. Retrying extraction.")
            result = self.extract_fields_from_image(images_base64, fields, multi_row)

            if result is not None:
                return result
            else:
                logger.info("Extraction retry failed. Please check the configuration.")
                return None
        else:
            return result



class PatientDataManager:
    "Manage patient data extraction and cleaning."

    MULTI_ROW_KEYWORDS = ["Long-term medication orders"]

    def __init__(self):
        self.raw_processed_dir = Path(RAW_PROCESSED_DIR)
        self.raw_pdf_extract_dir = Path(RAW_PDF_EXTRACT_DIR)
        self.raw_dir = Path(RAW_DIR)
        self.extractor = PDFExtractor()
        # Use DataPathRetriever to locate patient data.
        self.path_retriever = DataPathRetriever(
            raw_dir=RAW_DIR,
            raw_processed_dir=RAW_PROCESSED_DIR,
            raw_pdf_extract_dir=RAW_PDF_EXTRACT_DIR
        )

    def is_multi_row_pdf(self, pdf_name: str) -> bool:
        pdf_lower = pdf_name.lower()
        for kw in self.MULTI_ROW_KEYWORDS:
            if kw.lower() in pdf_lower:
                return True
        return False

    def match_target_doc(self, pdf_name: str, target_docs: List[str]) -> bool:
        "Check whether a PDF filename matches any requested document."
        if not target_docs:
            return True
        pdf_lower = pdf_name.lower()
        for target in target_docs:
            if target.lower() in pdf_lower:
                return True
        return False

    # Methods delegated to DataPathRetriever
    
    def get_data_centers(self, base_dir: Path) -> List[str]:
        "List data centers through DataPathRetriever."
        return self.path_retriever.get_data_centers(base_dir)

    def find_patient_in_center(self, base_dir: Path, center: str, patient_name: str) -> Optional[Tuple[str, Path]]:
        "Find a patient in the specified center through DataPathRetriever."
        return self.path_retriever.find_patient_in_center(base_dir, center, patient_name)

    def find_patient_in_dir(self, base_dir: Path, patient_name: str,
                            center: str = None) -> Optional[Tuple[str, Path]]:
        "Find a patient in a directory through DataPathRetriever."
        return self.path_retriever.find_patient_in_dir(base_dir, patient_name, center)

    def find_patient_in_raw_subdirs(self, patient_name: str,
                                     center: str = None) -> Optional[Tuple[str, Path]]:
        "Find a patient in raw subdirectories through DataPathRetriever."
        return self.path_retriever.find_patient_in_raw(patient_name, center)

    def get_doc_names_from_dir(self, dir_path: Path, extension: str) -> Set[str]:
        "List document names through DataPathRetriever."
        return self.path_retriever.get_doc_names_from_dir(dir_path, extension)

    def find_missing_docs(self, raw_path: Path, processed_path: Path) -> List[str]:
        "Identify missing documents through DataPathRetriever."
        return self.path_retriever.find_missing_docs(raw_path, processed_path)

    def extract_missing_docs(self, center: str, patient_name: str,
                              raw_path: Path, missing_docs: List[str]) -> bool:
        "Extract missing documents, reusing CSV files in raw_pdf_extract before extracting PDFs from raw. Arguments: center (center name), patient_name (patient identifier), raw_path (patient source directory), and missing_docs (document stems). Return whether extraction succeeded."
        if not missing_docs:
            return True

        logger.info(f"""Processing missing documents: {missing_docs}""")

        # Patient path in raw_pdf_extract
        extract_dir = self.raw_pdf_extract_dir / center / patient_name
        # Destination directory for newly extracted files
        target_dir = self.raw_pdf_extract_dir / center / patient_name
        create_dir_if_not_exist(str(target_dir))

        # List existing CSV files in raw_pdf_extract.
        existing_csv_names = self.get_doc_names_from_dir(extract_dir, '.csv')

        # Separate existing extractions from documents requiring PDF extraction.
        docs_in_extract = []
        docs_need_extract = []

        for doc_name in missing_docs:
            if doc_name in existing_csv_names:
                docs_in_extract.append(doc_name)
            else:
                docs_need_extract.append(doc_name)

        extracted_count = 0

        # 1. Retain existing extracted documents for subsequent cleaning.
        if docs_in_extract:
            logger.info(f"""Found {len(docs_in_extract)} extracted documents in raw_pdf_extract: {docs_in_extract}""")
            extracted_count += len(docs_in_extract)

        # 2. Extract the remaining documents from raw.
        if docs_need_extract:
            logger.info(f"""Extracting {len(docs_need_extract)} documents from raw: {docs_need_extract}""")

            for doc_name in docs_need_extract:
                # Find the corresponding PDF file.
                pdf_file = raw_path / f"{doc_name}.pdf"
                if not pdf_file.exists():
                    pdf_file = raw_path / f"{doc_name}.PDF"

                if pdf_file.exists():
                    pdf_name = pdf_file.name
                    fields = self.get_fields_for_pdf(pdf_name)
                    if not fields:
                        logger.warning(f"""No field configuration for {pdf_name}; skipping.""")
                        continue

                    try:
                        multi_row = self.is_multi_row_pdf(pdf_name)
                        result = self.extractor.extract_from_pdf(str(pdf_file), fields, multi_row)

                        output_filename = f"{doc_name}.csv"
                        output_path = target_dir / output_filename

                        if isinstance(result, list):
                            df = pd.DataFrame(result)
                            logger.info(f"""Extracted {len(result)} records""")
                        else:
                            df = pd.DataFrame([result])

                        df = df.reindex(columns=fields)
                        df.to_csv(output_path, index=False, encoding="utf-8-sig")
                        logger.info(f"""Extraction complete: {pdf_name} -> {output_path}""")
                        extracted_count += 1
                    except Exception as e:
                        logger.error(f"""PDF extraction failed for {pdf_name}: {e}""")
                else:
                    csv_file = raw_path / f"{doc_name}.csv"
                    if not csv_file.exists():
                        logger.warning(f"""File not found: {doc_name}""")
                        continue
                    else:
                        extracted_count += 1
                        output_filename = f"{doc_name}.csv"
                        output_path = target_dir / output_filename
                        df = pd.read_csv(csv_file)
                        df.to_csv(output_path, index=False, encoding="utf-8-sig")
                        logger.info(f"""Extraction complete: {doc_name} -> {output_path}""")


        logger.info(f"""Missing-document processing complete: {extracted_count} files processed""")
        return extracted_count > 0

    def is_progress_note(self, file_name: str) -> bool:
        "Check whether a file contains progress notes."
        return "progress notes" in file_name.lower()

    def split_progress_note(self, df: pd.DataFrame, output_path: Path) -> Tuple[
        Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        "Split a progress-note DataFrame into preoperative and postoperative files derived from output_path. Return the two DataFrames."
        output_dir = output_path.parent
        base_name = output_path.stem  # Filename without the extension

        # Select preoperative progress-note columns.
        pre_op_cols_exist = [col for col in PRE_OP_COLS if col in df.columns]
        df_pre_op = df[pre_op_cols_exist].copy() if pre_op_cols_exist else None

        # Select postoperative progress-note columns.
        post_op_cols_exist = [col for col in POST_OP_COLS if col in df.columns]
        df_post_op = df[post_op_cols_exist].copy() if post_op_cols_exist else None

        # Clean and save preoperative progress notes.
        if df_pre_op is not None and len(pre_op_cols_exist) > len(COMMON_COLS):
            df_pre_op = df_pre_op.dropna(axis=1, how='all')
            df_pre_op = df_pre_op.loc[:, ~(df_pre_op == '').all()]
            df_pre_op = df_pre_op.loc[:, ~(df_pre_op == 'np.nan').all()]

            pre_op_path = output_dir / f"""Preoperative_{base_name}.csv"""
            df_pre_op.to_csv(pre_op_path, index=False, encoding="utf-8-sig")
            logger.info(f"""Preoperative progress notes saved to: {pre_op_path}""")
        else:
            df_pre_op = None
            logger.warning(f"""No preoperative progress-note columns found""")

        # Clean and save postoperative progress notes.
        if df_post_op is not None and len(post_op_cols_exist) > len(COMMON_COLS):
            df_post_op = df_post_op.dropna(axis=1, how='all')
            df_post_op = df_post_op.loc[:, ~(df_post_op == '').all()]
            df_post_op = df_post_op.loc[:, ~(df_post_op == 'np.nan').all()]

            post_op_path = output_dir / f"""Postoperative_{base_name}.csv"""
            df_post_op.to_csv(post_op_path, index=False, encoding="utf-8-sig")
            logger.info(f"""Postoperative progress notes saved to: {post_op_path}""")
        else:
            df_post_op = None
            logger.warning(f"""No postoperative progress-note columns found""")

        return df_pre_op, df_post_op

    def clean_csv(self, input_path: Path, output_path: Path):
        try:
            df = pd.read_csv(input_path, encoding="utf-8-sig")
            df_cleaned = df.dropna(axis=1, how='all')
            df_cleaned = df_cleaned.loc[:, ~(df_cleaned == '').all()]
            df_cleaned = df_cleaned.loc[:, ~(df_cleaned == 'np.nan').all()]
            create_dir_if_not_exist(str(output_path.parent))

            # Split progress notes into preoperative and postoperative records.
            if self.is_progress_note(input_path.name):
                logger.info(f"""Splitting progress notes into preoperative and postoperative records: {input_path.name}""")
                df_pre, df_post = self.split_progress_note(df_cleaned, output_path)
                # Return the cleaned source DataFrame to preserve the interface.
                return df_cleaned
            else:
                df_cleaned.to_csv(output_path, index=False, encoding="utf-8-sig")
                logger.info(f"""Cleaned data saved to: {output_path}""")
                return df_cleaned
        except Exception as e:
            logger.error(f"""CSV cleaning failed for {input_path}: {e}""")
            return None

    def process_extracted_to_processed(self, center: str, patient_name: str,
                                        target_docs: List[str] = None):
        "Clean extracted CSV files and save them to processed data, optionally limiting processing to specified documents."
        source_dir = self.raw_pdf_extract_dir / center / patient_name
        target_dir = self.raw_processed_dir / center / patient_name
        create_dir_if_not_exist(str(target_dir))
        csv_files = list(source_dir.glob("*.csv"))
        if not csv_files:
            logger.warning(f"""No CSV files found in {source_dir}""")
            return

        for csv_file in csv_files:
            if target_docs and not self.match_target_doc(csv_file.name, target_docs):
                continue
            output_path = target_dir / csv_file.name
            self.clean_csv(csv_file, output_path)
        logger.info(f"""Cleaned data for {patient_name} from raw_pdf_extract into raw_processed""")

    def process_missing_docs_to_processed(self, center: str, patient_name: str,
                                           missing_docs: List[str]):
        "Clean missing-document CSV files and save them to processed data."
        source_dir = self.raw_pdf_extract_dir / center / patient_name
        target_dir = self.raw_processed_dir / center / patient_name
        create_dir_if_not_exist(str(target_dir))

        for doc_name in missing_docs:
            csv_file = source_dir / f"{doc_name}.csv"
            if csv_file.exists():
                output_path = target_dir / f"{doc_name}.csv"
                self.clean_csv(csv_file, output_path)

        logger.info(f"""Cleaned missing documents for {patient_name} from raw_pdf_extract into raw_processed""")

    def extract_pdfs_for_patient(self, center: str, patient_name: str, source_path: Path,
                                  target_docs: List[str] = None) -> bool:
        "Extract PDF documents for a patient."
        target_dir = self.raw_pdf_extract_dir / center / patient_name
        create_dir_if_not_exist(str(target_dir))
        pdf_files = list(source_path.glob("*.pdf")) + list(source_path.glob("*.PDF"))
        if not pdf_files:
            logger.warning(f"""No PDF files found in {source_path}""")
            return False

        extracted_count = 0
        for pdf_file in pdf_files:
            pdf_name = pdf_file.name

            if target_docs and not self.match_target_doc(pdf_name, target_docs):
                continue

            fields = self.get_fields_for_pdf(pdf_name)
            if not fields:
                logger.warning(f"""No field configuration for {pdf_name}; skipping.""")
                continue

            try:
                multi_row = self.is_multi_row_pdf(pdf_name)
                result = self.extractor.extract_from_pdf(str(pdf_file), fields, multi_row)

                if result is None:
                    logger.info("File extraction failed")
                    return False

                output_filename = pdf_name.replace(".pdf", ".csv").replace(".PDF", ".csv")
                output_path = target_dir / output_filename

                if isinstance(result, list):
                    df = pd.DataFrame(result)
                    logger.info(f"""Extracted {len(result)} records""")
                else:
                    df = pd.DataFrame([result])

                df = df.reindex(columns=fields)
                df.to_csv(output_path, index=False, encoding="utf-8-sig")
                logger.info(f"""Extraction complete: {pdf_name} -> {output_path}""")
                extracted_count += 1
            except Exception as e:
                logger.error(f"""PDF extraction failed for {pdf_name}: {e}""")
                return False

        if target_docs:
            logger.info(f"""Selected-document extraction complete: {extracted_count} files extracted""")
        return extracted_count > 0

    def get_fields_for_pdf(self, pdf_name: str) -> Optional[List[str]]:
        pdf_base = pdf_name.replace(".pdf", "").replace(".PDF", "")
        pdf_base_lower = pdf_base.lower()
        sorted_keys = sorted(FIELDS_DICT.keys(), key=len, reverse=True)
        for key in sorted_keys:
            if key.lower() in pdf_base_lower:
                logger.info(f"""PDF '{pdf_name}' matched keyword '{key}'""")
                return FIELDS_DICT[key]
        return None

    def process_patient(self, patient_name: str, force_extract: bool = False,
                        target_docs: List[str] = None, center: str = None) -> bool:
        for i in range(3):
            if self.single_process_patient(patient_name=patient_name, force_extract=force_extract,
                                        target_docs=target_docs, center=center):
                return True
            else:
                logger.info("Retrying extraction")
                continue
        return False

    def single_process_patient(self, patient_name: str, force_extract: bool = False,
                        target_docs: List[str] = None, center: str = None) -> bool:
        "Process patient data. patient_name identifies the patient; force_extract re-extracts all documents; target_docs selects PDF filenames by partial matching; center limits lookup to one center, or None searches all centers. Return whether processing succeeded."
        logger.info(f"""Processing patient: {patient_name}""")
        if center:
            logger.info(f"""Selected data center: {center}""")
        if target_docs:
            logger.info(f"""Documents selected for re-extraction: {target_docs}""")

        # Re-extract target_docs directly when specified; skip the missing-document check.
        if target_docs:
            return self._reextract_target_docs(patient_name, target_docs, center)

        # Track the located center and processed-data path.
        found_center = None
        processed_path = None

        # Step 1: Check raw_processed unless re-extraction is forced.
        if not force_extract:
            result = self.find_patient_in_dir(self.raw_processed_dir, patient_name, center)
            if result:
                found_center, patient_path = result
                processed_path = patient_path
                logger.info(f"""Found processed data for {patient_name} in raw_processed/{found_center}""")

        # Step 2: Look for source data if processed data are absent or extraction is forced.
        if not processed_path:
            # Check raw_pdf_extract.
            result = self.find_patient_in_dir(self.raw_pdf_extract_dir, patient_name, center)
            if result:
                found_center, patient_path = result
                logger.info(f"""Found extracted data for {patient_name} in raw_pdf_extract/{found_center}""")
                self.process_extracted_to_processed(found_center, patient_name)
                processed_path = self.raw_processed_dir / found_center / patient_name
            else:
                # Check raw.
                result = self.find_patient_in_raw_subdirs(patient_name, center)
                if result:
                    found_center, patient_path = result
                    logger.info(f"""Found source PDFs for {patient_name} in raw/{found_center}""")
                    if self.extract_pdfs_for_patient(found_center, patient_name, patient_path):
                        self.process_extracted_to_processed(found_center, patient_name)
                        processed_path = self.raw_processed_dir / found_center / patient_name
                    else:
                        return False

        # Step 3: Check and process missing documents unless target_docs were specified.
        if processed_path and found_center:
            # 3.1 Check for missing PDF-derived documents.
            self._check_and_fill_missing_docs(found_center, patient_name, processed_path)

            # 3.2 Add the perioperative registration record and follow-up data.
            # self._extract_registry_data_from_master(found_center, patient_name, processed_path)
            self.display_patient_data(processed_path)
            return True

        # No patient data found.
        if center:
            logger.warning(f"""No data found for {patient_name} in center {center}""")
        else:
            logger.warning(f"""No data found for patient {patient_name}""")
        return False

    def _check_and_fill_missing_docs(self, center: str, patient_name: str,
                                     processed_path: Path) -> None:
        "Check and fill missing documents for patient_name in center and processed_path. Reuse raw_pdf_extract files before extracting source PDFs from raw."
        raw_result = self.find_patient_in_raw_subdirs(patient_name, center)
        if not raw_result:
            logger.warning(f"""No source data for {patient_name} in raw; cannot check for missing documents""")
            return

        _, raw_path = raw_result
        missing_docs = self.find_missing_docs(raw_path, processed_path)

        if missing_docs:
            logger.info(f"""Found {len(missing_docs)} missing documents; processing: {missing_docs}""")
            if self.extract_missing_docs(center, patient_name, raw_path, missing_docs):
                self.process_missing_docs_to_processed(center, patient_name, missing_docs)
                logger.info(f"""Missing documents processed successfully""")
            else:
                logger.warning(f"""Failed to process missing documents""")
        else:
            logger.info(f"""All documents are present; no additional processing needed""")

    def _reextract_target_docs(self, patient_name: str, target_docs: List[str],
                                center: str = None) -> bool:
        "Re-extract specified documents and update processed data."
        result = self.find_patient_in_raw_subdirs(patient_name, center)
        if not result:
            if center:
                logger.warning(f"""No source data for {patient_name} in the raw directory of center {center}""")
            else:
                logger.warning(f"""No source data for {patient_name} in raw""")
            return False

        found_center, source_path = result
        logger.info(f"""Found source data for {patient_name} in raw/{found_center}""")

        if self.extract_pdfs_for_patient(found_center, patient_name, source_path, target_docs):
            self.process_extracted_to_processed(found_center, patient_name, target_docs)
            processed_path = self.raw_processed_dir / found_center / patient_name
            logger.info(f"""Re-extracted and updated selected documents in {processed_path}""")
            self.display_patient_data(processed_path)
            return True
        else:
            logger.warning(f"""Unable to extract any selected documents: {target_docs}""")
            return False

    def display_patient_data(self, patient_path: Path):
        logger.info(f"\n{'=' * 50}")
        logger.info(f"""Patient data location: {patient_path}""")
        logger.info(f"{'=' * 50}")
        csv_files = list(patient_path.glob("*.csv"))
        if csv_files:
            logger.info(f"""Found {len(csv_files)} data files:""")
            for csv_file in csv_files:
                logger.info(f"  - {csv_file.name}")
                try:
                    df = pd.read_csv(csv_file, encoding="utf-8-sig")
                    logger.info(f"""    Columns: {len(df.columns)}, rows: {len(df)}""")
                except Exception as e:
                    logger.info(f"""    Read failed: {e}""")

    def list_centers(self) -> Dict[str, List[str]]:
        "List all data centers and their directories."
        result = {
            "raw": self.get_data_centers(self.raw_dir),
            "raw_pdf_extract": self.get_data_centers(self.raw_pdf_extract_dir),
            "raw_processed": self.get_data_centers(self.raw_processed_dir)
        }
        return result