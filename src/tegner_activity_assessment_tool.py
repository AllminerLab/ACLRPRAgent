
import os
import sys
import json
import logging
from typing import Dict, List, Any

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


class TegnerPredictor:
    def __init__(self, data_dir: str = "../data/raw_processed", output_dir: str = "../data/interim",
                 qwen_api_key: str = None, raw_dir: str = "../data/raw"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.qwen_client = QwenAPIClient(api_key=qwen_api_key)
        self.retrieve_results = {}
        # 定义统一的输出文件名 (修改为 Tegner 相关)
        self.retrieve_results_filename = "tegner_retrieval_result.json"
        self.predict_report_filename = "tegner_predict_report.json"
        self.raw_dir = raw_dir
        self.complete_follow_up_filename = "完整的随访数据(包含ACL-RSI+术前Tegner+术后Tegner).csv"

        if MedicalRetriever:
            self.retriever = MedicalRetriever(
                data_dir=data_dir, output_dir=output_dir, print_thinking=True, max_input_tokens=8096,
                default_extension=".csv"
            )
        else:
            sys.exit(1)

    # --- 核心工具：JSON结构化更新 ---
    def _update_retrieve_json_file(self, center: str, patient: str, step_key: str, inner_key: str, data: Any):
        """更新 JSON 文件中的特定嵌套字段"""
        center_dir = os.path.join(self.output_dir, center)
        if not os.path.exists(center_dir):
            os.makedirs(center_dir)

        patient_dir = os.path.join(center_dir, patient)
        if not os.path.exists(patient_dir):
            os.makedirs(patient_dir)
        file_path = os.path.join(patient_dir, self.retrieve_results_filename)

        save_data = data
        if isinstance(data, RetrievalResult):
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

        full_content = {}
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    full_content = json.load(f)
            except json.JSONDecodeError:
                full_content = {}

        if step_key not in full_content or not isinstance(full_content[step_key], dict):
            full_content[step_key] = {}

        full_content[step_key][inner_key] = save_data

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(full_content, f, ensure_ascii=False, indent=2)
            print(f"  💾 [JSON更新] {step_key} -> {inner_key} 已保存")
        except Exception as e:
            logger.error(f"写入文件失败: {e}")

    # --- 校验工具 ---
    def _check_success_retrieval(self, result: RetrievalResult, step_name: str) -> bool:
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
    def _save_result(self, patient: str, filename: str, data: Any, center: str):
        path = self._ensure_output_dir(center=center, patient=patient)
        file_path = os.path.join(path, filename)

        # 加入 真实 Tegner 评分
        raw_follow_up_data = pd.read_csv(f'{self.raw_dir}/{center}/{self.complete_follow_up_filename}')

        # 修改：读取 '术后Tegner' 列 (请确保CSV列名匹配，若为 'Tegner' 请自行调整)

        ground_truth_score = raw_follow_up_data[raw_follow_up_data['姓名'] == patient]['术后Tegner'].values[0].item()
        preoperative_score = raw_follow_up_data[raw_follow_up_data['姓名'] == patient]['术前Tegner'].values[0].item()

        final_report = {'患者': patient, '院区': center, 'preoperative_Tegner': preoperative_score,
                        'ground_truth_Tegner': ground_truth_score}

        for k, v in data.items():
            if k == 'score':
                k = 'predicted_Tegner'
            final_report[k] = v

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, ensure_ascii=False, indent=4)

    # =========================================================================
    # Step 1: 核心基线构建 (侧重运动能力)
    # =========================================================================
    def step1_patient_base_profile(self, center: str, patient: str):
        print("\n" + "=" * 50)
        print("🏗️ Step 1: 核心基线构建 (运动背景)")
        print("=" * 50)

        valid_docs_content = []

        # 1.1 检索登记表
        print("  🔍 1.1 检索围术期登记表...")
        res_reg = self.retriever.retrieve_by_documents(
            center=center, patient=patient,
            document_names=["ACL损伤患者围术期信息登记表"],
            task_goal=f"""请检索与ACL损伤术后Tegner运动评分相关的字段。
            Tegner评分高度依赖于患者的【运动项目类型】和【竞技水平】。
            重点关注：
            1. 人口学特征：年龄、性别、BMI、职业（区分重体力/轻体力/久坐）。
            2. 运动习惯：受伤前从事的运动项目（如足球、篮球、滑雪 vs 跑步、游泳）、运动频率、运动水平（国家级/竞技/娱乐/久坐）。
            3. 损伤与手术：手术日期、半月板处理情况（切除/缝合）、软骨损伤情况（影响长期运动能力）。
            """
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
            task_goal="""请检索与术后运动能力(Tegner)预测相关的关键基线信息：
                        1. 运动需求描述：现病史或主诉中关于受伤场景的描述（例如"打篮球落地时"、"踢足球变向时"），这暗示了患者的运动偏好。
                        2. 关节结构完整性：半月板是切除还是缝合（切除可能影响长期高强度运动），软骨是否有损伤，韧带重建方式。
                        3. 职业信息：是否为运动员、军人或重体力劳动者。
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

        if valid_docs_content:
            print("  🤖 1.3 生成基线摘要...")
            combined_text = "\n".join(valid_docs_content)
            summary_prompt =f"""请从运动医学专家的角度，总结以下内容，重点在于评估患者的【潜在运动需求】和【身体基础】。
                       请明确提取：受伤前的Tegner级别暗示（如足球竞技属于高分段，散步属于低分段）、职业体力要求、半月板及软骨状态。
                       \n{combined_text}\n
                        请以JSON格式输出结果：\n
                                {{
                                    'summary': 总结的内容。
                                    'reasoning': 总结的思路。
                                }}"""


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
            self.retrieve_results["patient_base_profile_summary"] = None

    # =========================================================================
    # Step 2: 术前状态回溯 (锚定原始水平)
    # =========================================================================
    def step2_pre_op_retrospection(self, center: str, patient: str):
        print("\n" + "=" * 50)
        print("⏪ Step 2: 术前状态回溯 (锚定原始水平)")
        print("=" * 50)

        baseline = self.retrieve_results.get("patient_base_profile_summary")
        baseline_context = f"{baseline}" if baseline else "基础画像信息未知"

        res = self.retriever.retrieve_by_condition(
            center=center, patient=patient,
            search_condition="手术前的临床文本、检查报告，术前Tenger评分等",
            task_goal=f"""请从运动医学专家的角度，根据给定的患者基础画像总结，筛选与术后Tegner评估和归因相关的术前文档和字段。
            Tegner评分不仅看现状，更看【受伤前】的能力。
            患者基础画像：{baseline_context}\n
            请检索：
            1. 受伤前运动能力：是否有明确记录"术前Tegner评分"或具体的运动水平描述。
            2. 术前功能状态：术前 ROM、关节不稳程度（Pivot Shift 试验）。
            3. 运动意愿：是否表达了强烈的重返赛场意愿（这通常与较高的术后Tegner相关）。
            """
        )

        if self._check_success_retrieval(res, "术前数据"):
            self._update_retrieve_json_file(
                center=center,
                patient=patient,
                step_key="step2_pre_op_retrospection",
                inner_key="preoperative_retrieval",
                data=res)

    # =========================================================================
    # Step 3: 术后动态追踪 (实际活动等级)
    # =========================================================================
    def step3_post_op_tracking(self, center: str, patient: str):
        print("\n" + "=" * 50)
        print("⏩ Step 3: 术后动态追踪 (实际活动等级)")
        print("=" * 50)

        baseline = self.retrieve_results.get("patient_base_profile_summary")
        baseline_context = f"{baseline}" if baseline else "基础画像信息未知"

        res = self.retriever.retrieve_by_condition(
            center=center, patient=patient,
            search_condition="手术后的检查报告, 随访, 复查，临床文本，出院记录等",
            task_goal=f"""请从运动医学专家的角度，根据给定的患者基础画像总结，筛选与术后Tegner评估和归因相关的术后文档和字段。
            Tegner 评分衡量的是【实际上能做什么】，而不是【心理上敢不敢做】。
            患者基础画像：{baseline_context}\n
            重点检索：
            1. 当前运动项目：患者是否已经恢复跑步？是否恢复球类运动？是竞技还是娱乐性质？
            2. 工作恢复情况：是否恢复重体力劳动？
            3. 膝关节表现：在剧烈运动（如急停、变向、跳跃）时是否有肿胀、疼痛或不稳。
            4. 客观查体：Lachman、轴移试验结果（松弛度直接影响高水平运动重返）。
            """
        )

        if self._check_success_retrieval(res, "术后数据"):
            self._update_retrieve_json_file(
                center=center,
                patient=patient,
                step_key="step3_post_op_tracking",
                inner_key="postoperative_retrieval",
                data=res)

    def get_predict_prompt(self, evidence):
        # 注意：Tegner 是 0-10 分，不同于 ACL-RSI 的 0-100 分
        predict_prompt = f"""
        基于以下【仅有的】有效医疗记录预测 Tegner 运动水平评分。
        注意：JSON中不存在的步骤代表该阶段数据缺失，请勿编造。

        ### 有效证据数据
        {json.dumps(evidence, ensure_ascii=False, indent=2)}

        ### 任务
        1. 预测 Tegner 评分 (0-10 整数)。
        2. 进行归因 (引用上述 JSON 中存在的具体证据)。
        3. 输出JSON: {{score, confidence, reasoning, key_factors}}
        """

        prediction_prompt = f"""你是一位资深的运动医学专家，专门研究ACL重建术后患者的运动重返能力评估。\n

        基于以下【仅有的】有效医疗记录预测 **Tegner Activity Scale (Tegner 运动评级)** 评分，并进行归因分析。\n

        ### 有效证据数据\n
        {evidence}

        ## Tegner 评分参考标准 (0-10分)\n
        - **10分**: 国际/国家级竞技体育（足球、橄榄球等对抗性运动）。
        - **9分**: 竞技体育（足球、冰球、摔跤等）。
        - **7-8分**: 接触性运动（篮球、足球）的娱乐/竞技水平；或从事重体力劳动。
        - **6分**: 娱乐性运动（网球、羽毛球、滑雪、慢跑）；每周至少5次。
        - **5分**: 娱乐性运动（每周1-5次）；重体力劳动。
        - **3-4分**: 轻体力劳动；散步、游泳。
        - **0-2分**: 因膝关节问题严重受限，病退或日常生活受限。

        ## 影响 Tegner 评分的主要因素\n
        1. **受伤前的运动水平**（上限通常不超过伤前水平）。
        2. **手术效果**（半月板完整性、韧带稳定性如轴移试验）。
        3. **术后恢复现状**（是否已恢复跑步、跳跃、对抗）。
        4. **职业需求**（是否为重体力劳动者）。
        5. **半月板/软骨**： **普通缝合 (1-3针)**：影响较小，康复较快。**复杂缝合 (>3针) 或 涉及体部**：需限制屈膝角度。

        ## 请完成以下任务\n

        1. **评分预测**：给出 Tegner 预测评分（0-10之间的整数）。如果证据表明恢复良好但未提及具体对抗运动，通常在 5-7 分之间。\n
        2. **置信度评估**：评估预测的置信度（高/中/低）。\n
        3. **归因分析**：详细分析影响该患者评分的正面和负面因素。\n
        4. **关键因素**：列出最重要的3-5个影响因素。\n

        请以JSON格式输出结果：\n
        {{
            "score": 预测评分（0-10的整数）,
            "confidence": "高/中/低",
            "reasoning": "详细的归因分析（字符串）",
            "key_factors": [
                {{"factor": "因素名称", "direction": "正面/负面", "weight": "高/中/低", "explanation": "解释"}}
            ]
        }}
        \n
        ## 示例\n
        性别男，年龄25岁；主诉：左膝关节扭伤后疼痛8天。术前Tegner评分为8分。
        现病史相关：患者8天前从高处(约I米高)摔倒。
        既往史相关：平素身体健康，多进行篮球运动，否认糖尿病史，否认高血压史，否认冠心病史，否认疾病外伤史，否认输血史，否认手术史，无过敏史，预防接种史不详。
        个人史相关：饮酒史3年，每周2-3次，每次约2两。
        手术记录相关：1、内侧半月板后角纵裂，360全内半月板缝合器4针缝合固定；外侧半月板体部水平裂，用360全内半月板缝合器3针缝合固定；
        2、两条肌腱分别对折一次编织缝合成长8.0cm肌腱束一条，股骨端、胫骨端直径分别为8mm、8mm，用φ8mm扩髓器扩隧道，8mm股骨扩髓器扩髓约3.5cm，拧人9*30可吸收挤压螺钉。
        \n
        术后Tegner预测评分：6分\n
        """
        return prediction_prompt

    # =========================================================================
    # Step 4: 归因预测 (严格过滤版)
    # =========================================================================
    def step4_prediction_and_attribution(self, center: str, patient: str):
        print("\n" + "=" * 50)
        print("🧠 Step 4: Tegner 评分预测与归因")
        print("=" * 50)

        # 4.1 聚合证据
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
            self._save_result(patient, self.predict_report_filename, predict_report, center)
            return True
        else:
            print("  ❌ 模型返回无内容")
            return None

    def _aggregate_evidence_strictly(self, center, patient) -> Dict[str, Any]:
        """从 tegner_retrieval_result.json 文件中读取并聚合证据"""
        evidence = {}
        file_path = os.path.join(self.output_dir, center, patient, self.retrieve_results_filename)

        if not os.path.exists(file_path):
            logger.warning(f"文件不存在: {file_path}")
            return {}

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                full_data = json.load(f)
        except Exception as e:
            logger.error(f"读取JSON失败: {e}")
            return {}

        s1 = full_data.get("step1_patient_base_profile", {})
        summary = s1.get("patient_base_profile_summary")
        evidence["Step1_Base_Profile_Summary"] = summary

        s2 = full_data.get("step2_pre_op_retrospection", {})
        res2 = s2.get("preoperative_retrieval", {})
        data2 = res2.get("data")
        if data2 and isinstance(data2, list):
            clean_data = [d for d in data2 if d]
            if clean_data:
                evidence["Step2_PreOp_Records"] = clean_data

        s3 = full_data.get("step3_post_op_tracking", {})
        res3 = s3.get("postoperative_retrieval", {})
        data3 = res3.get("data")
        if data3 and isinstance(data3, list):
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
            print("\n✅ Tegner 评估成功")
            return True
        else:
            print("评估失败")
            return False
