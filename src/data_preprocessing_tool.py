
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
# 病程记录列名定义
COMMON_COLS = ["姓名", "性别", "年龄", "科室", "床号", "住院号"]

PRE_OP_COLS = COMMON_COLS + [
    "首次病程记录日期", "首次病程记录",
    "术前入院第一天查房记录日期", "术前入院第一天查房记录",
    "术前入院第二天查房记录日期", "术前入院第二天查房记录"
]

POST_OP_COLS = COMMON_COLS + [
    "术后当天手术病程记录日期", "术后当天手术病程记录",
    "术后第一天上级医生查房日期", "术后第一天上级医生查房记录",
    "术后第二天查房记录日期", "术后第二天查房记录",
    "术后第三天日常查房记录日期", "术后第三天日常查房记录",
    "出院病程记录日期", "出院病程记录"
]

def create_dir_if_not_exist(dir_path: str) -> str:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


class PDFExtractor:
    """PDF内容提取器，支持单行和多行数据"""

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
        fields_str = "、".join(fields)
        return f"""请仔细查看这些医疗文档图片，提取以下字段的信息：

需要提取的字段：{fields_str}

请以JSON格式返回结果，格式如下：
{{
    "字段1": "值1",
    "字段2": "值2",
    ...
}}

1、如果某个字段在文档中找不到，请填写""。
2、首次病程记录里面，不要遗漏了"诊断依据及鉴别诊断"及其中的"辅助检查"，不要遗漏"诊疗经过"的内容。
3、请确保提取准确，没有遗漏，也没有错误。
4、如果提取内容是入院记录，要确保每个字段提取正确，特别是 "肛门"和"外生殖器"字段。
5、请只返回JSON，不要有其他内容。"""

    def build_multi_row_prompt(self, fields: List[str]) -> str:
        fields_str = "、".join(fields)
        return f"""请仔细查看这些医疗文档图片，提取以下字段的信息：

需要提取的字段：{fields_str}

【重要】这些文档可能包含多条记录（如多次检查结果、多个药物处方、多次随访记录等）。
请提取所有记录，以JSON数组格式返回，格式如下：
[
    {{"字段1": "值1", "字段2": "值2", ...}},
    {{"字段1": "值1", "字段2": "值2", ...}},
    ...
]

注意事项：
1. 如果文档中只有一条记录，也请返回包含一个对象的数组。
2. 如果某个字段在某条记录中找不到，请填写""。
3. 每条独立的记录（如不同日期的检查、不同的处方等）应作为数组中的一个单独对象。
4. 请确保提取准确完整，不要遗漏任何记录。
5. 请只返回JSON数组，不要有其他内容。"""

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
            logger.info(f"读取内容如下：\n{result_text}")

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
            logger.error(f"JSON解析失败: {result_text}")
            empty = {field: "解析失败" for field in fields}
            return None
        except Exception as e:
            logger.error(f"API调用失败: {e}")
            return None

    def extract_from_pdf(self, pdf_path: str, fields: List[str],
                         multi_row: bool = False) -> Union[Dict[str, str], List[Dict[str, str]]]:
        logger.info(f"处理PDF: {pdf_path}, 多行模式: {multi_row}")
        images_base64 = self.pdf_to_images(pdf_path)
        result = self.extract_fields_from_image(images_base64, fields, multi_row)

        if result is None:
            logger.info("出现异常！！！提取失败或JSON解析失败！！！尝试重新提取")
            result = self.extract_fields_from_image(images_base64, fields, multi_row)

            if result is not None:
                return result
            else:
                logger.info("尝试提取失败！！！请人工检查配置")
                return None
        else:
            return result



class PatientDataManager:
    """患者数据管理器"""

    MULTI_ROW_KEYWORDS = ["长期医嘱"]

    def __init__(self):
        self.raw_processed_dir = Path(RAW_PROCESSED_DIR)
        self.raw_pdf_extract_dir = Path(RAW_PDF_EXTRACT_DIR)
        self.raw_dir = Path(RAW_DIR)
        self.extractor = PDFExtractor()
        # 使用 DataPathRetriever 进行数据路径检索
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
        """检查PDF文件名是否匹配目标文档列表中的任一项"""
        if not target_docs:
            return True
        pdf_lower = pdf_name.lower()
        for target in target_docs:
            if target.lower() in pdf_lower:
                return True
        return False

    # ==================== 委托给 DataPathRetriever 的方法 ====================
    
    def get_data_centers(self, base_dir: Path) -> List[str]:
        """获取数据中心列表（委托给 DataPathRetriever）"""
        return self.path_retriever.get_data_centers(base_dir)

    def find_patient_in_center(self, base_dir: Path, center: str, patient_name: str) -> Optional[Tuple[str, Path]]:
        """在指定的数据中心中查找患者（委托给 DataPathRetriever）"""
        return self.path_retriever.find_patient_in_center(base_dir, center, patient_name)

    def find_patient_in_dir(self, base_dir: Path, patient_name: str,
                            center: str = None) -> Optional[Tuple[str, Path]]:
        """在目录中查找患者（委托给 DataPathRetriever）"""
        return self.path_retriever.find_patient_in_dir(base_dir, patient_name, center)

    def find_patient_in_raw_subdirs(self, patient_name: str,
                                     center: str = None) -> Optional[Tuple[str, Path]]:
        """在 raw 子目录中查找患者（委托给 DataPathRetriever）"""
        return self.path_retriever.find_patient_in_raw(patient_name, center)

    def get_doc_names_from_dir(self, dir_path: Path, extension: str) -> Set[str]:
        """获取目录中的文档名（委托给 DataPathRetriever）"""
        return self.path_retriever.get_doc_names_from_dir(dir_path, extension)

    def find_missing_docs(self, raw_path: Path, processed_path: Path) -> List[str]:
        """查找缺失的文档（委托给 DataPathRetriever）"""
        return self.path_retriever.find_missing_docs(raw_path, processed_path)

    def extract_missing_docs(self, center: str, patient_name: str,
                              raw_path: Path, missing_docs: List[str]) -> bool:
        """
        提取缺失的文档
        优先从 raw_pdf_extract 目录复制已有的 CSV，没有的话再从 raw 目录提取 PDF

        Args:
            center: 数据中心名称
            patient_name: 患者姓名
            raw_path: raw目录中患者的路径
            missing_docs: 缺失的文档基础名称列表

        Returns:
            是否成功提取
        """
        if not missing_docs:
            return True

        logger.info(f"开始处理缺失的文档: {missing_docs}")

        # raw_pdf_extract 中患者的路径
        extract_dir = self.raw_pdf_extract_dir / center / patient_name
        # 目标目录（用于存放新提取的文件）
        target_dir = self.raw_pdf_extract_dir / center / patient_name
        create_dir_if_not_exist(str(target_dir))

        # 获取 raw_pdf_extract 中已有的 CSV 文件
        existing_csv_names = self.get_doc_names_from_dir(extract_dir, '.csv')

        # 分类：哪些在 raw_pdf_extract 中已有，哪些需要从 raw 提取
        docs_in_extract = []
        docs_need_extract = []

        for doc_name in missing_docs:
            if doc_name in existing_csv_names:
                docs_in_extract.append(doc_name)
            else:
                docs_need_extract.append(doc_name)

        extracted_count = 0

        # 1. 处理 raw_pdf_extract 中已有的文档（直接记录，后续会被清洗到 processed）
        if docs_in_extract:
            logger.info(f"在 raw_pdf_extract 中找到 {len(docs_in_extract)} 个已提取的文档: {docs_in_extract}")
            extracted_count += len(docs_in_extract)

        # 2. 从 raw 目录提取剩余的文档
        if docs_need_extract:
            logger.info(f"需要从 raw 目录提取 {len(docs_need_extract)} 个文档: {docs_need_extract}")

            for doc_name in docs_need_extract:
                # 尝试找到对应的PDF文件
                pdf_file = raw_path / f"{doc_name}.pdf"
                if not pdf_file.exists():
                    pdf_file = raw_path / f"{doc_name}.PDF"

                if pdf_file.exists():
                    pdf_name = pdf_file.name
                    fields = self.get_fields_for_pdf(pdf_name)
                    if not fields:
                        logger.warning(f"未找到 {pdf_name} 的字段配置，跳过")
                        continue

                    try:
                        multi_row = self.is_multi_row_pdf(pdf_name)
                        result = self.extractor.extract_from_pdf(str(pdf_file), fields, multi_row)

                        output_filename = f"{doc_name}.csv"
                        output_path = target_dir / output_filename

                        if isinstance(result, list):
                            df = pd.DataFrame(result)
                            logger.info(f"提取到 {len(result)} 条记录")
                        else:
                            df = pd.DataFrame([result])

                        df = df.reindex(columns=fields)
                        df.to_csv(output_path, index=False, encoding="utf-8-sig")
                        logger.info(f"提取完成: {pdf_name} -> {output_path}")
                        extracted_count += 1
                    except Exception as e:
                        logger.error(f"提取PDF失败 {pdf_name}: {e}")
                else:
                    csv_file = raw_path / f"{doc_name}.csv"
                    if not csv_file.exists():
                        logger.warning(f"未找到文件: {doc_name}")
                        continue
                    else:
                        extracted_count += 1
                        output_filename = f"{doc_name}.csv"
                        output_path = target_dir / output_filename
                        df = pd.read_csv(csv_file)
                        df.to_csv(output_path, index=False, encoding="utf-8-sig")
                        logger.info(f"提取完成: {doc_name} -> {output_path}")


        logger.info(f"缺失文档处理完成，共处理 {extracted_count} 个文件")
        return extracted_count > 0

    def is_progress_note(self, file_name: str) -> bool:
        """判断是否为病程记录文件"""
        return "病程记录" in file_name

    def split_progress_note(self, df: pd.DataFrame, output_path: Path) -> Tuple[
        Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        将病程记录拆分为术前和术后两个文件

        Args:
            df: 原始病程记录DataFrame
            output_path: 原始输出路径

        Returns:
            (术前DataFrame, 术后DataFrame)
        """
        output_dir = output_path.parent
        base_name = output_path.stem  # 不含扩展名的文件名

        # 提取术前病程记录列
        pre_op_cols_exist = [col for col in PRE_OP_COLS if col in df.columns]
        df_pre_op = df[pre_op_cols_exist].copy() if pre_op_cols_exist else None

        # 提取术后病程记录列
        post_op_cols_exist = [col for col in POST_OP_COLS if col in df.columns]
        df_post_op = df[post_op_cols_exist].copy() if post_op_cols_exist else None

        # 清洗并保存术前病程记录
        if df_pre_op is not None and len(pre_op_cols_exist) > len(COMMON_COLS):
            df_pre_op = df_pre_op.dropna(axis=1, how='all')
            df_pre_op = df_pre_op.loc[:, ~(df_pre_op == '').all()]
            df_pre_op = df_pre_op.loc[:, ~(df_pre_op == 'np.nan').all()]

            pre_op_path = output_dir / f"术前{base_name}.csv"
            df_pre_op.to_csv(pre_op_path, index=False, encoding="utf-8-sig")
            logger.info(f"术前病程记录保存到: {pre_op_path}")
        else:
            df_pre_op = None
            logger.warning(f"未找到术前病程记录相关列")

        # 清洗并保存术后病程记录
        if df_post_op is not None and len(post_op_cols_exist) > len(COMMON_COLS):
            df_post_op = df_post_op.dropna(axis=1, how='all')
            df_post_op = df_post_op.loc[:, ~(df_post_op == '').all()]
            df_post_op = df_post_op.loc[:, ~(df_post_op == 'np.nan').all()]

            post_op_path = output_dir / f"术后{base_name}.csv"
            df_post_op.to_csv(post_op_path, index=False, encoding="utf-8-sig")
            logger.info(f"术后病程记录保存到: {post_op_path}")
        else:
            df_post_op = None
            logger.warning(f"未找到术后病程记录相关列")

        return df_pre_op, df_post_op

    def clean_csv(self, input_path: Path, output_path: Path):
        try:
            df = pd.read_csv(input_path, encoding="utf-8-sig")
            df_cleaned = df.dropna(axis=1, how='all')
            df_cleaned = df_cleaned.loc[:, ~(df_cleaned == '').all()]
            df_cleaned = df_cleaned.loc[:, ~(df_cleaned == 'np.nan').all()]
            create_dir_if_not_exist(str(output_path.parent))

            # 判断是否为病程记录，如果是则拆分为术前和术后
            if self.is_progress_note(input_path.name):
                logger.info(f"检测到病程记录文件，进行术前/术后拆分: {input_path.name}")
                df_pre, df_post = self.split_progress_note(df_cleaned, output_path)
                # 返回原始清洗后的DataFrame（保持接口一致）
                return df_cleaned
            else:
                df_cleaned.to_csv(output_path, index=False, encoding="utf-8-sig")
                logger.info(f"清洗后保存到: {output_path}")
                return df_cleaned
        except Exception as e:
            logger.error(f"清洗CSV失败 {input_path}: {e}")
            return None

    def process_extracted_to_processed(self, center: str, patient_name: str,
                                        target_docs: List[str] = None):
        """将提取的CSV清洗后保存到processed目录，支持指定文档"""
        source_dir = self.raw_pdf_extract_dir / center / patient_name
        target_dir = self.raw_processed_dir / center / patient_name
        create_dir_if_not_exist(str(target_dir))
        csv_files = list(source_dir.glob("*.csv"))
        if not csv_files:
            logger.warning(f"在 {source_dir} 未找到CSV文件")
            return

        for csv_file in csv_files:
            if target_docs and not self.match_target_doc(csv_file.name, target_docs):
                continue
            output_path = target_dir / csv_file.name
            self.clean_csv(csv_file, output_path)
        logger.info(f"已将 {patient_name} 的数据从 raw_pdf_extract 清洗到 raw_processed")

    def process_missing_docs_to_processed(self, center: str, patient_name: str,
                                           missing_docs: List[str]):
        """将缺失文档的CSV清洗后保存到processed目录"""
        source_dir = self.raw_pdf_extract_dir / center / patient_name
        target_dir = self.raw_processed_dir / center / patient_name
        create_dir_if_not_exist(str(target_dir))

        for doc_name in missing_docs:
            csv_file = source_dir / f"{doc_name}.csv"
            if csv_file.exists():
                output_path = target_dir / f"{doc_name}.csv"
                self.clean_csv(csv_file, output_path)

        logger.info(f"已将 {patient_name} 的缺失文档从 raw_pdf_extract 清洗到 raw_processed")

    def extract_pdfs_for_patient(self, center: str, patient_name: str, source_path: Path,
                                  target_docs: List[str] = None) -> bool:
        """为患者提取PDF文件"""
        target_dir = self.raw_pdf_extract_dir / center / patient_name
        create_dir_if_not_exist(str(target_dir))
        pdf_files = list(source_path.glob("*.pdf")) + list(source_path.glob("*.PDF"))
        if not pdf_files:
            logger.warning(f"在 {source_path} 未找到PDF文件")
            return False

        extracted_count = 0
        for pdf_file in pdf_files:
            pdf_name = pdf_file.name

            if target_docs and not self.match_target_doc(pdf_name, target_docs):
                continue

            fields = self.get_fields_for_pdf(pdf_name)
            if not fields:
                logger.warning(f"未找到 {pdf_name} 的字段配置，跳过")
                continue

            try:
                multi_row = self.is_multi_row_pdf(pdf_name)
                result = self.extractor.extract_from_pdf(str(pdf_file), fields, multi_row)

                if result is None:
                    logger.info("文件提取失败！！！")
                    return False

                output_filename = pdf_name.replace(".pdf", ".csv").replace(".PDF", ".csv")
                output_path = target_dir / output_filename

                if isinstance(result, list):
                    df = pd.DataFrame(result)
                    logger.info(f"提取到 {len(result)} 条记录")
                else:
                    df = pd.DataFrame([result])

                df = df.reindex(columns=fields)
                df.to_csv(output_path, index=False, encoding="utf-8-sig")
                logger.info(f"提取完成: {pdf_name} -> {output_path}")
                extracted_count += 1
            except Exception as e:
                logger.error(f"提取PDF失败 {pdf_name}: {e}")
                return False

        if target_docs:
            logger.info(f"指定文档提取完成，共提取 {extracted_count} 个文件")
        return extracted_count > 0

    def get_fields_for_pdf(self, pdf_name: str) -> Optional[List[str]]:
        pdf_base = pdf_name.replace(".pdf", "").replace(".PDF", "")
        pdf_base_lower = pdf_base.lower()
        sorted_keys = sorted(FIELDS_DICT.keys(), key=len, reverse=True)
        for key in sorted_keys:
            if key.lower() in pdf_base_lower:
                logger.info(f"PDF '{pdf_name}' 匹配到关键词 '{key}'")
                return FIELDS_DICT[key]
        return None

    def process_patient(self, patient_name: str, force_extract: bool = False,
                        target_docs: List[str] = None, center: str = None) -> bool:
        for i in range(3):
            if self.single_process_patient(patient_name=patient_name, force_extract=force_extract,
                                        target_docs=target_docs, center=center):
                return True
            else:
                logger.info("尝试重新提取！！！")
                continue
        return False

    def single_process_patient(self, patient_name: str, force_extract: bool = False,
                        target_docs: List[str] = None, center: str = None) -> bool:
        """
        处理患者数据

        Args:
            patient_name: 患者姓名
            force_extract: 强制重新提取所有文档
            target_docs: 指定要重新提取的文档列表（模糊匹配PDF文件名）
            center: 指定数据中心名称，None则自动搜索所有中心

        Returns:
            处理是否成功
        """
        logger.info(f"开始处理患者: {patient_name}")
        if center:
            logger.info(f"指定数据中心: {center}")
        if target_docs:
            logger.info(f"指定重新提取的文档: {target_docs}")

        # 如果指定了target_docs，直接进行指定文档的重新提取（不检查缺失）
        if target_docs:
            return self._reextract_target_docs(patient_name, target_docs, center)

        # 用于跟踪找到的中心和处理后的路径
        found_center = None
        processed_path = None

        # 步骤1：检查 raw_processed 目录（非强制提取时）
        if not force_extract:
            result = self.find_patient_in_dir(self.raw_processed_dir, patient_name, center)
            if result:
                found_center, patient_path = result
                processed_path = patient_path
                logger.info(f"✓ 在 raw_processed/{found_center} 中找到患者 {patient_name} 的已处理数据")

        # 步骤2：如果 raw_processed 中没有找到或需要强制提取
        if not processed_path:
            # 检查 raw_pdf_extract 目录
            result = self.find_patient_in_dir(self.raw_pdf_extract_dir, patient_name, center)
            if result:
                found_center, patient_path = result
                logger.info(f"✓ 在 raw_pdf_extract/{found_center} 中找到患者 {patient_name} 的提取数据")
                self.process_extracted_to_processed(found_center, patient_name)
                processed_path = self.raw_processed_dir / found_center / patient_name
            else:
                # 检查 raw 目录
                result = self.find_patient_in_raw_subdirs(patient_name, center)
                if result:
                    found_center, patient_path = result
                    logger.info(f"✓ 在 raw/{found_center} 中找到患者 {patient_name} 的原始PDF")
                    if self.extract_pdfs_for_patient(found_center, patient_name, patient_path):
                        self.process_extracted_to_processed(found_center, patient_name)
                        processed_path = self.raw_processed_dir / found_center / patient_name
                    else:
                        return False

        # 步骤3：统一检查缺失文档并补全（除了target_docs情况外都执行）
        if processed_path and found_center:
            # 3.1 检查PDF类文档缺失
            self._check_and_fill_missing_docs(found_center, patient_name, processed_path)

            # 3.2 步骤4: 加入ACL损伤患者围术期信息登记表和随访数据
            # self._extract_registry_data_from_master(found_center, patient_name, processed_path)
            self.display_patient_data(processed_path)
            return True

        # 未找到任何数据
        if center:
            logger.warning(f"✗ 未在数据中心 {center} 中找到患者 {patient_name} 的任何信息")
        else:
            logger.warning(f"✗ 未找到患者 {patient_name} 的任何信息")
        return False

    def _check_and_fill_missing_docs(self, center: str, patient_name: str,
                                     processed_path: Path) -> None:
        """
        检查并补全缺失的文档
        优先从 raw_pdf_extract 获取，没有的话再从 raw 提取

        Args:
            center: 数据中心名称
            patient_name: 患者姓名
            processed_path: raw_processed中患者数据的路径
        """
        raw_result = self.find_patient_in_raw_subdirs(patient_name, center)
        if not raw_result:
            logger.warning(f"在raw目录中未找到患者 {patient_name} 的原始数据，无法检查缺失")
            return

        _, raw_path = raw_result
        missing_docs = self.find_missing_docs(raw_path, processed_path)

        if missing_docs:
            logger.info(f"发现 {len(missing_docs)} 个缺失文档，开始补充处理: {missing_docs}")
            if self.extract_missing_docs(center, patient_name, raw_path, missing_docs):
                self.process_missing_docs_to_processed(center, patient_name, missing_docs)
                logger.info(f"✓ 已补充处理缺失的文档")
            else:
                logger.warning(f"✗ 补充处理缺失文档失败")
        else:
            logger.info(f"✓ 文档完整，无需补充")

    def _reextract_target_docs(self, patient_name: str, target_docs: List[str],
                                center: str = None) -> bool:
        """重新提取指定的文档并更新到processed目录"""
        result = self.find_patient_in_raw_subdirs(patient_name, center)
        if not result:
            if center:
                logger.warning(f"✗ 在数据中心 {center} 的raw目录中未找到患者 {patient_name} 的数据库")
            else:
                logger.warning(f"✗ 在raw目录中未找到患者 {patient_name} 的数据库")
            return False

        found_center, source_path = result
        logger.info(f"✓ 在 raw/{found_center} 中找到患者 {patient_name} 的数据库")

        if self.extract_pdfs_for_patient(found_center, patient_name, source_path, target_docs):
            self.process_extracted_to_processed(found_center, patient_name, target_docs)
            processed_path = self.raw_processed_dir / found_center / patient_name
            logger.info(f"✓ 已重新提取并更新指定文档到 {processed_path}")
            self.display_patient_data(processed_path)
            return True
        else:
            logger.warning(f"✗ 未能提取任何指定的文档: {target_docs}")
            return False

    def display_patient_data(self, patient_path: Path):
        logger.info(f"\n{'=' * 50}")
        logger.info(f"患者数据位置: {patient_path}")
        logger.info(f"{'=' * 50}")
        csv_files = list(patient_path.glob("*.csv"))
        if csv_files:
            logger.info(f"找到 {len(csv_files)} 个数据文件:")
            for csv_file in csv_files:
                logger.info(f"  - {csv_file.name}")
                try:
                    df = pd.read_csv(csv_file, encoding="utf-8-sig")
                    logger.info(f"    列数: {len(df.columns)}, 行数: {len(df)}")
                except Exception as e:
                    logger.info(f"    读取失败: {e}")

    def list_centers(self) -> Dict[str, List[str]]:
        """列出所有数据中心及其目录"""
        result = {
            "raw": self.get_data_centers(self.raw_dir),
            "raw_pdf_extract": self.get_data_centers(self.raw_pdf_extract_dir),
            "raw_processed": self.get_data_centers(self.raw_processed_dir)
        }
        return result