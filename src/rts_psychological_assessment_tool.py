"""
_____________________________
  Author: Claude Assistant
   Time : 2026/01/01
   File : acl_rsi_prediction.py
   Desc : ACL-RSI评分预测 (v2.2 零样本过滤版 - 无数据不入参)
______________________________
"""
import os
import pickle
import sys
import json
import logging
import re
from typing import Dict, List, Any, Optional

import pandas as pd

from llm_api_config import QWEN_KEY_API
from streaming_llm_handler import qwen_stream_completion
from retrieval_tool import MedicalRetriever, RetrievalResult
from logger import init_logger

# 基础配置
base_dir = os.path.split(os.path.realpath(__file__))[0]
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
if not logger.handlers:
    logger = init_logger('../data/log')



class QwenAPIClient:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or QWEN_KEY_API

    def chat(self, prompt: str, system_prompt: str = "你是一位资深的运动医学专家。", temperature: float = 0.6):
        try:
            return qwen_stream_completion(system_prompt=system_prompt, content=prompt,
                                          model='qwen3-max', api_key=self.api_key, temperature=temperature)
        except Exception as e:
            logger.error(f"Qwen API Error: {e}")
            return None


class ACLRSIPredictor:
    def __init__(self, data_dir: str = "../data/raw_processed", output_dir: str = "../data/interim", qwen_api_key: str = None, raw_dir: str = "../data/raw"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.qwen_client = QwenAPIClient(api_key=qwen_api_key)
        self.retrieve_results = {}
        # 定义统一的输出文件名
        self.retrieve_results_filename = "acl_rsi_retrieval_result.json"
        self.predict_report_filename = "acl_rsi_predict_report.json"
        self.raw_dir =raw_dir
        self.complete_follow_up_filename = "完整的随访数据(包含ACL-RSI+术前Tegner+术后Tegner).csv"

        self.api_attempt_times = 3


        if MedicalRetriever:
            self.retriever = MedicalRetriever(
                data_dir=data_dir, output_dir=output_dir, print_thinking=True, max_input_tokens=8096, default_extension=".csv"
            )
        else:
            sys.exit(1)

    # --- 核心工具：JSON结构化更新 ---
    def _update_retrieve_json_file(self, center: str, patient: str, step_key: str, inner_key: str, data: Any):
        """
        功能：更新 JSON 文件中的特定嵌套字段
        逻辑：json_content[step_key][inner_key] = serialized_data

        Args:
            patient: 患者姓名 (用于确定文件夹)
            step_key: 一级键 (如 'step1_patient_base_profile')
            inner_key: 二级键 (如 '登记表_retrieval_result')
            data: 要保存的数据 (RetrievalResult对象, 字符串, 或字典)
        """
        # 1. 路径准备
        center_dir = os.path.join(self.output_dir, center)
        if not os.path.exists(center_dir):
            os.makedirs(center_dir)

        patient_dir = os.path.join(center_dir, patient)
        if not os.path.exists(patient_dir):
            os.makedirs(patient_dir)

        file_path = os.path.join(patient_dir, self.retrieve_results_filename)

        # 2. 数据序列化 (对象 -> 字典)
        save_data = data
        if isinstance(data, RetrievalResult): # 处理 RetrievalResult 对象
            save_data = {
                "success": data.success,
                "field_empty": data.field_empty,
                "data_count": len(data.data) if data.data else 0,
                "source_documents": data.source_documents,
                "data": data.data if data.data else [],
                "document_selection_reason": data.document_selection_reason,
                "field_extraction_reason": data.field_extraction_reason,
                "error_message": data.error_message
            }

        # 3. 读取现有文件
        full_content = {}
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    full_content = json.load(f)
            except json.JSONDecodeError:
                full_content = {}

        # 4. 确保一级键存在
        if step_key not in full_content or not isinstance(full_content[step_key], dict):
            full_content[step_key] = {}

        # 5. 更新二级键
        full_content[step_key][inner_key] = save_data

        # 6. 回写文件
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(full_content, f, ensure_ascii=False, indent=2)
            print(f"  💾 [JSON更新] {step_key} -> {inner_key} 已保存")
        except Exception as e:
            logger.error(f"写入文件失败: {e}")

    # --- 校验工具 ---
    def _check_success_retrieval(self, result: RetrievalResult, step_name: str) -> bool:
        """校验检索是否有效"""
        if not result.success:
            print(f"  ❌ [{step_name}] 检索失败: {result.error_message}")
            return False

        return True

    def _ensure_output_dir(self, center: str, patient: str) -> str:
        center_path = os.path.join(self.output_dir, center)
        if not os.path.exists(center_path): os.makedirs(center_path)

        path = os.path.join(center_path, patient)
        if not os.path.exists(path): os.makedirs(path)
        return path

    def _save_result(self, center: str, patient: str, filename: str, data: Any):
        path = self._ensure_output_dir(center=center, patient=patient)
        file_path = os.path.join(path, filename)

        # 加入 真实ACL-RSI评分
        raw_follow_up_data = pd.read_csv(f'{self.raw_dir}/{center}/{self.complete_follow_up_filename}')

        ground_truth_score = raw_follow_up_data[raw_follow_up_data['姓名'] == patient]['ACL-RSI'].values[0].item()

        final_report = {'患者': patient, '院区': center, 'ground_truth_ACL-RSI': ground_truth_score}

        for k, v in data.items():
            if k == 'score':
                k = 'predicted_ACL-RSI'
            final_report[k] = v

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, ensure_ascii=False, indent=4)  # 确保中文不乱码，格式化输出

    # =========================================================================
    # Step 1: 核心基线构建
    # =========================================================================
    def step1_patient_base_profile(self, center: str, patient: str):
        print("\n" + "="*50)
        print("🏗️ Step 1: 核心基线构建")
        print("="*50)

        valid_docs_content = [] # 只存储有内容的文档文本

        # 1.1 检索登记表
        print("  🔍 1.1 检索围术期登记表...")
        res_reg = self.retriever.retrieve_by_documents(
            center=center, patient=patient,
            document_names=["ACL损伤患者围术期信息登记表"],
            task_goal=f"""请检索与ACL损伤术后ACL-RSI评分的相关字段，特别地，重点关注以下关键基线信息：人口学特征： 年龄、性别、BMI、职业/运动水平（竞技/非竞技）。
            手术核心参数： 手术日期（关键Time Anchor，用于分割术前/术后）、移植物类型（自体/异体/人工）、半月板处理（缝合/切除/未处理）。
            损伤机制： 受伤时间、受伤场景（接触性/非接触性）。以上只是参考，具体情况请具体处理。"""
        )


        if self._check_success_retrieval(res_reg, "登记表"):
            self._update_retrieve_json_file(
                center=center,
                patient=patient,
                step_key="step1_patient_base_profile",
                inner_key="perioperative_registration_retrieval",
                data=res_reg)
            valid_docs_content.append(self._format_single_result("登记表", res_reg))

        # 1.2 检索病历文书
        print("  🔍 1.2 检索手术与入院记录...")
        res_docs = self.retriever.retrieve_by_documents(
            center=center, patient=patient,
            document_names=["手术记录", "入院记录"],
            task_goal="""请检索与ACL损伤术后ACL-RSI评分的相关字段，重点关注以下关键基线信息：
                        1. 手术核心参数：手术名称、移植物类型(自体/异体/人工)、半月板处理方式(缝合/切除/成形)。
                        2. 损伤机制：受伤时间、受伤具体场景(接触性/非接触性)、受伤原因。
                        3. 基础信息：性别、年龄、主诉、现病史、既往史、个人史。以上只是参考，具体情况请具体处理。
                        """
        )
        if self._check_success_retrieval(res_docs, "手术与入院记录"):
            self._update_retrieve_json_file(
                center=center,
                patient=patient,
                step_key="step1_patient_base_profile",
                inner_key="hospital_admission_and_surgery_record_retrieval",
                data=res_docs)
            valid_docs_content.append(self._format_single_result("手术与入院记录", res_docs))

        # 只有当确实检索到了内容时，才调用 LLM 生成摘要
        if valid_docs_content:
            print("  🤖 1.3 生成基线摘要...")
            combined_text = "\n".join(valid_docs_content)
            summary_prompt = f"""请从运动医学专家的角度，总结以下的内容，为后续ACL-RSI评估和可解释提供指导。特别地，可以提取以下内容：手术日期、移植物、半月板、损伤机制等。"
                       \n{combined_text}\n
                        请以JSON格式输出结果：\n
                                {{
                                    'summary': 总结的内容。
                                    'reasoning': 总结的思路。
                                }}
                                \n"""

            summary = self.qwen_client.chat(
                    prompt=summary_prompt)

            # 总结失败时，使用原始数据
            if not summary:
                logger.info("总结失败，使用原始检索数据")
                summary = valid_docs_content

            self._update_retrieve_json_file(
                center=center,
                patient=patient,
                step_key="step1_patient_base_profile",
                inner_key="patient_base_profile_summary",
                data=summary)

            self.retrieve_results["patient_base_profile_summary"] = summary
        else:
            print("  ⚠️ Step 1 未检索到任何有效文档，跳过摘要生成。")
            self.retrieve_results["patient_base_profile_summary"] = None # 显式置空

    # =========================================================================
    # Step 2: 术前状态回溯
    # =========================================================================
    def step2_pre_op_retrospection(self, center: str, patient: str):
        print("\n" + "="*50)
        print("⏪ Step 2: 术前状态回溯")
        print("="*50)

        baseline = self.retrieve_results.get("patient_base_profile_summary")

        if baseline is not None:
            print("已获取患者基本画像。")
        # 即使没有基线，也尝试检索，但在Prompt里说明情况
        baseline_context = f"{baseline}" if baseline else "基础画像信息未知"

        res = self.retriever.retrieve_by_condition(
            center=center, patient=patient,
            search_condition="手术前的临床文本、检查报告等",
            task_goal=f"""请从运动医学专家的角度，根据给定的患者基础画像总结，筛选与术后ACL-RSI评估和归因相关的术前文档和字段。\n
            患者基础画像：{baseline_context}\n
            特别地，可考虑以下角度：
            术前功能： 术前 ROM (活动度)、术前 KT-1000/2000 侧侧差、术前肌萎缩情况。\n
            术前主观评分： 术前 IKDC、Lysholm 评分。\n
            心理痕迹： 是否提及“不敢发力”、“不稳感”、“恐惧再次受伤”、受伤到手术的等待时间（等待越久，心理代偿越重）。\n
            检索术前ROM、IKDC评分、不稳感描述\n"""
        )

        if self._check_success_retrieval(res, "术前数据"):
            # self.retrieve_results["step2_pre_op"] = res
            self._update_retrieve_json_file(
                center=center,
                patient=patient,
                step_key="step2_pre_op_retrospection",
                inner_key="preoperative_retrieval",
                data=res)
        # else:
        #     self.retrieve_results["step2_pre_op"] = None # 显式置空

    # =========================================================================
    # Step 3: 术后动态追踪
    # =========================================================================
    def step3_post_op_tracking(self, center: str, patient: str):
        print("\n" + "="*50)
        print("⏩ Step 3: 术后动态追踪")
        print("="*50)

        baseline = self.retrieve_results.get("patient_base_profile_summary")
        if baseline is not None:
            print("已获取患者基本画像。")
        baseline_context = f"{baseline}" if baseline else "基础画像信息未知"

        res = self.retriever.retrieve_by_condition(
            center=center, patient=patient,
            search_condition="术后, 随访, 复查",
            task_goal=f"""请从运动医学专家的角度，根据给定的患者基础画像总结，筛选与术后ACL-RSI评估和归因相关的术后文档和字段。\n
            患者基础画像：{baseline_context}\n
            特别地，寻找符合以下类别的字段或描述：\n
            **1. 疼痛与创伤体验 (Pain & Trauma Experience)**
            * *逻辑：早期疼痛越剧烈，远期运动恐惧(Kinesiophobia)越强。*
            * **寻找字段：** VAS评分、疼痛描述（剧烈、难以忍受）、夜间痛、镇痛药物使用情况。\n
            
            **2. 早期功能受限与僵硬 (Functional Setbacks)**
            * *逻辑：术后伸直受限是远期 ACL-RSI 低下的强预测因子。*
            * **寻找字段：** 膝关节活动度 (ROM)、**伸直受限 (Extension Deficit)**、被动屈曲角度、关节僵硬、是否需要手法松解。\n
            
            **3. 并发症与不良事件 (Complications - "Confidence Killers")**
            * *逻辑：任何非预期并发症都会严重打击患者康复信心。*
            * **寻找字段：** 感染 (Infection)、发热、深静脉血栓 (DVT)、伤口愈合不良、关节积液/浮髌试验 (Effusion)、肿胀程度 (周径差)。\n
            
            **4. 影像学结构隐患 (Structural Integrity)**
            * *逻辑：骨隧道位置不佳或内固定问题会导致关节力学异常，患者会主观感觉“不稳”或“卡顿”，从而降低信心。*
            * **寻找字段：** - **CT/X线：** 骨隧道位置 (Tunnel Position)、是否等长 (Isometric)、内固定物位置 (Screw/Button)、髁间窝成形 (Notchplasty)、骨隧道扩大 (Tunnel Enlargement)。
                - **半月板/软骨：** 是否合并半月板缝合（影响负重信心）、软骨损伤等级。"""
        )

        if self._check_success_retrieval(res, "术后数据"):
            # self.retrieve_results["step3_post_op"] = res
            self._update_retrieve_json_file(
                center=center,
                patient=patient,
                step_key="step3_post_op_tracking",
                inner_key="postoperative_retrieval",
                data=res)
        # else:
        #     self.retrieve_results["step3_post_op"] = None # 显式置空

    def get_predict_prompt(self, evidence):

        prediction_prompt = f"""您是一位拥有丰富临床经验的运动医学专家。请基于提供的【住院期间病历数据】，对患者**未来康复后（术后6-12个月）**的重返运动心理准备度（ACL-RSI）进行预后预测。

【核心预测逻辑】
虽然证据数据仅限于住院期间（入院-手术-出院），但你需要利用这些早期临床特征作为"预后因子"，推断患者在完成康复并面临重返运动决策时的心理状态。
**核心公式：未来预测分 = (基于个人素质的理想基准分) - (手术复杂性带来的长期心理阴影折扣) - (早期功能缺失带来的信心磨损)**

有效证据数据
{evidence}

一、 第一步：设定理想基准分 (Base Score)
根据患者的手术史、人口学特征和术前运动水平，设定无并发症情况下的理想分数（天花板）：

* **特殊熔断规则（翻修手术）**：
    * 若病历显示为**翻修手术（Revision）**或**再次断裂**：无论职业水平如何，基准分直接锁定为 **60-70分**。再次受伤的经历是心理信心的永久性打击。

* **常规基准规则（针对初次手术）**：
    * **高基准 (85-100分)**：年轻(<30岁) + 职业/竞技水平(Tegner高) + 强烈的重返赛场意愿。
    * **中基准 (75-85分)**：中年(30-45岁) + 规律业余运动习惯 + 运动意愿中等。
    * **低基准 (<75分)**：大龄(>45岁) + 低运动需求/久坐人群 + BMI偏高。

二、 第二步：应用临床"折扣"因子 (Deductions)
请仔细审查住院/出院记录，若发现以下风险因素，在基准分上进行**累加扣除**：

1.  **手术复杂性带来的"起跑线滞后" (权重最高)**
    * **半月板/软骨的高危处理**：
        * **普通缝合 (1-2针)**：影响较小，扣 **0-5分**。
        * **复杂缝合 (≥3针) 或 涉及体部**：需限制屈膝角度，扣 **10分**。
        * **根部/后角缝合 (Root/Posterior Horn Repair) 或 软骨微骨折**：这是极高风险项。意味着术后需**非负重(NWB) 4-6周**。这将导致严重的肌肉萎缩和本体感觉丧失，扣 **20分**。
    * **合并韧带损伤**：
        * 伴有MCL/LCL/PCL修补：增加关节僵硬风险，扣 **15分**。

2.  **早期功能与症状的"信心磨损"**
    * **出院功能不达标 (关键指标)**：
        * 出院小结明确记载**"无法完成直腿抬高(SLR)"**、**"股四头肌收缩弱"**或**"需他人辅助搬腿"**：意味着早期康复受阻，患者极度缺乏安全感，扣 **10分**。
    * **疼痛/肿胀控制差**：
        * 出院时仍需强效止痛药，或关节积液明显(浮髌阳性)、VAS评分持续>4分，扣 **5分**。

三、 任务要求
1.  **明确预测性质**：输出的 `score` 是预测患者在**术后6-12个月**的分数，反映的是"预后潜力"。
2.  **置信度判断**：由于缺乏"康复过程"这一关键中间数据，**置信度最高通常不超过"中"**，除非患者是典型的单纯ACL断裂且恢复极快。
3.  **严格遵循数据**：若病历中未提及某项并发症，视为无该风险，不进行扣分。

四、 输出格式
请严格按照以下JSON格式输出结果：
{{
        "score": 预测评分（数字）,
        "confidence": "高/中/低",
        "reasoning": "详细的归因分析（字符串）",
        "key_factors": [
            {{"factor": "因素名称", "direction": "正面/负面", "weight": "高/中/低", "explanation": "解释"}}
        ]
    }}
"""
        return prediction_prompt

    # =========================================================================
    # Step 4: 归因预测 (严格过滤版)
    # =========================================================================
    def step4_prediction_and_attribution(self, center: str, patient: str):
        print("\n" + "="*50)
        print("🧠 Step 4: ACL-RSI评分预测与归因")
        print("="*50)

        # 4.1 聚合证据 (核心修改点：只聚合非空数据)
        evidence = self._aggregate_evidence_strictly(center=center, patient=patient)

        # 4.2 熔断检查
        if not evidence:
            print("  ⛔ [终止] Step 1-3 均未产生有效数据，停止预测。")
            return {
                "score": 0, "confidence": "无数据",
                "reasoning": "检索结果为空，无法进行预测。", "key_factors": []
            }

        print(f"  📊 最终入参的证据模块: {list(evidence.keys())}")

        # 4.3 执行预测
        prompt = self.get_predict_prompt(evidence=evidence)

        print("  🤖 调用模型推理...")
        predict_report = self.qwen_client.chat(prompt, temperature=0.2)

        if predict_report:
            self._save_result(center=center,
                              patient=patient,
                              filename=self.predict_report_filename,
                              data=predict_report)
            return True
        else:
            print("  ❌ 模型返回无内容")
            return False

    # --- 核心修改：严格聚合函数 ---
    def _aggregate_evidence_strictly(self, center, patient) -> Dict[str, Any]:
        """
        从 acl_rsi_retrieval_result.json 文件中读取并聚合证据
        只收集非空数据，用于 Prompt 输入
        """
        evidence = {}

        # 1. 构建文件路径
        file_path = os.path.join(self.output_dir, center, patient, self.retrieve_results_filename)

        # 2. 读取文件
        if not os.path.exists(file_path):
            logger.warning(f"文件不存在: {file_path}")
            return {}

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                full_data = json.load(f)
        except Exception as e:
            logger.error(f"读取JSON失败: {e}")
            return {}

        # 3. 提取 Step 1 Summary
        # 路径: step1_patient_base_profile -> patient_base_profile_summary
        s1 = full_data.get("step1_patient_base_profile", {})
        summary = s1.get("patient_base_profile_summary")
        evidence["Step1_Base_Profile_Summary"] = summary

        # 4. 提取 Step 2 Data
        # 路径: step2_pre_op_retrospection -> 术前retrieval_result -> data
        s2 = full_data.get("step2_pre_op_retrospection", {})
        res2 = s2.get("preoperative_retrieval", {})
        data2 = res2.get("data")
        if data2 and isinstance(data2, list):
            # 过滤空项
            clean_data = [d for d in data2 if d]
            if clean_data:
                evidence["Step2_PreOp_Records"] = clean_data

        # 5. 提取 Step 3 Data
        # 路径: step3_post_op_tracking -> 术后retrieval_result -> data
        s3 = full_data.get("step3_post_op_tracking", {})
        res3 = s3.get("postoperative_retrieval", {})
        data3 = res3.get("data")
        if data3 and isinstance(data3, list):
            # 过滤空项
            clean_data = [d for d in data3 if d]
            if clean_data:
                evidence["Step3_PostOp_Records"] = clean_data

        return evidence

    def _format_single_result(self, source_name, result):
        lines = [f"【来源: {source_name}】"]
        if result.success and result.data:
            for item in result.data:
                lines.append(str(item))
        return "\n".join(lines)


    def run(self, center: str, patient: str):
        print(f"\n🚀 开始处理: {patient}")
        self.step1_patient_base_profile(center, patient)
        self.step2_pre_op_retrospection(center, patient)
        self.step3_post_op_tracking(center, patient)
        success = self.step4_prediction_and_attribution(center, patient)
        if success:
            print("\n✅ ACL_RSI评估成功")
            return True
        else:
            print("评估失败")
            return False
