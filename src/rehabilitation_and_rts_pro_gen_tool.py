
import os
import sys
import json
import logging
from typing import Dict, Any, Optional

from retrieval_tool import MedicalRetriever, RetrievalResult
from llm_api_config import QWEN_KEY_API
from streaming_llm_handler import qwen_stream_completion
from logger import init_logger

# 基础配置
base_dir = os.path.split(os.path.realpath(__file__))[0]
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
if not logger.handlers:
    logger = init_logger('../data/log')


class QwenAPIClient:
    """简易Qwen客户端封装"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or QWEN_KEY_API

    def chat(self, prompt: str, system_prompt: str, temperature: float = 0.4):
        try:
            return qwen_stream_completion(
                system_prompt=system_prompt,
                content=prompt,
                model='qwen3-max',
                api_key=self.api_key,
                temperature=temperature
            )
        except Exception as e:
            logger.error(f"Qwen API Error: {e}")
            return None


class ACLRehabPlanner:
    def __init__(self,
                 center: str,
                 patient: str,
                 data_dir: str = "../data/raw_processed",
                 interim_dir: str = "../data/interim",
                 retrieve_results_filename: str = "rehabilitation_rts_retrieve_results.json",
                 rehabilitation_rts_program_filename: str = "rehabilitation_rts_program.json",
                 acl_rsi_predict_report: str = "acl_rsi_predict_report.json",
                 tegner_predict_report: str = "tegner_predict_report.json",


                 ):

        self.center = center
        self.patient = patient
        self.interim_dir = interim_dir + '/' + center
        self.retrieve_results_filename = retrieve_results_filename
        self.rehabilitation_rts_program_filename = rehabilitation_rts_program_filename
        self.acl_rsi_predict_report = acl_rsi_predict_report
        self.tegner_predict_report = tegner_predict_report

        # 初始化检索器
        self.retriever = MedicalRetriever(
            data_dir=data_dir,
            output_dir=interim_dir,
            print_thinking=True,  # 开启思考过程打印
            max_input_tokens=6000
        )

        # 初始化LLM
        self.llm = QwenAPIClient()


    # --- 核心工具：JSON结构化更新 ---
    def _update_retrieve_json_file(self, patient: str, step_key: str, data: Any, inner_key: str = None):
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
        patient_dir = os.path.join(self.interim_dir, patient)
        if not os.path.exists(patient_dir):
            os.makedirs(patient_dir)
        file_path = os.path.join(patient_dir, self.retrieve_results_filename)

        # 2. 数据序列化 (对象 -> 字典)
        save_data = data
        if isinstance(data, RetrievalResult):  # 处理 RetrievalResult 对象
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
        if inner_key is not None:
            full_content[step_key][inner_key] = save_data
        else:
            full_content[step_key] = save_data

        # 6. 回写文件
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(full_content, f, ensure_ascii=False, indent=2)
            print(f"  💾 [JSON更新] {step_key} 已保存")
        except Exception as e:
            logger.error(f"写入文件失败: {e}")

    # --- 校验工具 ---
    def _check_success_retrieval(self, result: RetrievalResult) -> bool:
        """校验检索是否有效"""
        if not result.success:
            print(f"  ❌ 检索失败: {result.error_message}")
            return False

        return True

    def _load_prediction_report(self, filename: str) -> Dict[str, Any]:
        """加载之前步骤生成的预测报告 (Tegner / ACL-RSI)"""
        path = os.path.join(self.interim_dir, self.patient, filename)
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    report = json.load(f)
                    if 'acl_rsi' in filename and "ground_truth_ACL-RSI" in report:
                        value = report.pop('ground_truth_ACL-RSI', None)  # 第二个参数是默认值，防止KeyError

                    if "tegner" in filename and "ground_truth_Tegner" in report:
                        value = report.pop('ground_truth_Tegner', None)

                    return report

            except Exception as e:
                logger.error(f"读取预测报告失败 {filename}: {e}")

        return {}

    def step1_retrieve_hard_constraints(self) -> Dict[str, str]:
        """
        检索生理硬约束数据 (The 'Hardware')
        策略：优先术后数据，兼顾术前背景
        """
        print("\n" + "=" * 50)
        print("🏥 Step 1: 检索生理硬约束 (手术/影像/化验)")
        print("=" * 50)

        evidence = {}

        # 1.1 手术核心信息 (决定负重和活动度)
        print("  🔍 检索手术记录核心参数...")
        res_surgery = self.retriever.retrieve_by_documents(
            center=self.center,
            patient=self.patient,
            document_names=["手术记录", "出院记录"],  # 优先查这两个
            task_goal="提取与ACL运动损伤康复和未来重返运动相关的指标，比如：1.移植物类型(自体腘绳肌/骨髌腱/人工/异体)；2.半月板处理方式(缝合/切除/成形)；3.软骨处理方式(微骨折/移植/无处理)。"
        )
        if self._check_success_retrieval(res_surgery):
            # evidence['surgery_summary'] = self._format_retrieval_data(res_surgery, "手术信息缺失，请按保守方案处理")
            self._update_retrieve_json_file(
                patient=self.patient,
                step_key="surgery_summary",
                data=res_surgery)

        # 1.2 术后影像学 (决定伸膝限制)
        print("  🔍 检索术后影像学风险...")
        res_imaging = self.retriever.retrieve_by_condition(
            center=self.center,
            patient=self.patient,
            search_condition="术后X线, 术后CT, 影像报告",
            task_goal="取与ACL运动损伤康复和未来重返运动相关的指标，比如：骨隧道位置评价(是否偏前/偏垂直/位置良好)、内固定物状态。"
        )
        if self._check_success_retrieval(res_imaging):
            # evidence['imaging_risk'] = self._format_retrieval_data(res_imaging, "无术后影像学描述，默认无特殊解剖风险")
            self._update_retrieve_json_file(
                patient=self.patient,
                step_key="imaging_risk",
                data=res_imaging)

        # 1.3 术后炎症指标 (决定是否暂停进阶)
        print("  🔍 检索术后炎症指标...")
        res_lab_post = self.retriever.retrieve_by_condition(
            center=self.center,
            patient=self.patient,
            search_condition="术后的复查, 血常规, 血沉",
            task_goal="提取与ACL运动损伤康复和未来重返运动相关的指标，比如最近一次复查的：白细胞计数、中性粒细胞比例、C反应蛋白(CRP)、血沉(ESR)。"
        )
        if self._check_success_retrieval(res_lab_post):
            # evidence['inflammation_status'] = self._format_retrieval_data(res_lab_post, "近期无炎症复查数据")
            self._update_retrieve_json_file(
                patient=self.patient,
                step_key="inflammation_status",
                data=res_lab_post)


        # 1.4 术前/长期风险背景 (决定血栓预防和饮食)
        print("  🔍 检索背景风险因素...")
        res_risk = self.retriever.retrieve_by_condition(
            center=self.center,
            patient=self.patient,
            search_condition="术前检验, 生化, 凝血",
            task_goal="提取：D-二聚体(判断血栓风险)、尿酸、血糖值。"
        )
        if self._check_success_retrieval(res_risk):
            # evidence['background_risk'] = self._format_retrieval_data(res_risk, "无特殊代谢或凝血风险记录")
            self._update_retrieve_json_file(
                patient=self.patient,
                step_key="background_risk",
                data=res_risk)


        return evidence

    def _format_retrieval_data(self, result, default_msg: str) -> str:
        """将RetrievalResult对象格式化为Prompt可用的字符串"""
        lines = []
        if result.get("success") and len(result.get("data"))>0:
            for item in result.get("data"):
                lines.append(str(item))

        if len(lines) > 0:
            return "\n".join(lines)
        else:
            return default_msg


    def step2_load_soft_goals(self) -> Dict[str, Any]:
        """
        加载预测模型数据 (The 'Software')
        """
        print("\n" + "=" * 50)
        print("🧠 Step 2: 加载 Tegner & ACL-RSI 预测模型")
        print("=" * 50)

        # 读取 Tegner (目标上限)
        tegner_data = self._load_prediction_report("tegner_predict_report.json")
        tegner_score = tegner_data.get('predicted_Tegner')
        tegner_reason = {
            'reasoning': tegner_data.get('reasoning'),
            'confidence': tegner_data.get('confidence'),
            "key_factors": tegner_data.get('key_factors')
        }

        # 读取 ACL-RSI (心理配速)
        aclrsi_data = self._load_prediction_report("acl_rsi_predict_report.json")
        aclrsi_score = aclrsi_data.get('predicted_ACL-RSI')
        aclrsi_reason = {
            'reasoning': aclrsi_data.get('reasoning'),
            'confidence': aclrsi_data.get('confidence'),
            "key_factors": aclrsi_data.get('key_factors')
        }

        print(f"  🎯 运动目标 (Tegner) 评估: {tegner_score} 分")
        print(f"  💗 心理状态 (ACL-RSI) 评估: {aclrsi_score} 分")

        return {
            "tegner_score": tegner_score,
            "tegner_reason": tegner_reason,
            "aclrsi_score": aclrsi_score,
            "aclrsi_reason": aclrsi_reason
        }

    def get_evidence(self):
        evidence = {}

        # 1. 构建文件路径
        file_path = os.path.join(self.interim_dir, self.patient, self.retrieve_results_filename)

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

        evidence['surgery_summary'] = self._format_retrieval_data(full_data.get('surgery_summary'), "手术信息缺失，请按保守方案处理")

        evidence['imaging_risk'] = self._format_retrieval_data(full_data.get('imaging_risk'), "无术后影像学描述，默认无特殊解剖风险")

        evidence['inflammation_status'] = self._format_retrieval_data(full_data.get('inflammation_status'), "近期无炎症复查数据")

        evidence['background_risk'] = self._format_retrieval_data(full_data.get('background_risk'), "无特殊代谢或凝血风险记录")

        return evidence


    def generate_plan_prompt(self, evidence: Dict, predictions: Dict):
        # 构建系统 Prompt
        system_prompt = """你是一位顶尖的运动医学专家与物理治疗师（PT）。
                你需要基于医疗数据（手术、影像、检验、心理预测、运动预测等），
                为患者制定一份【精准的】、【基于标准的(Criteria-Based)】、【分阶段】的康复计划。
                请遵循生物-心理-社会模式，不仅关注膝盖，更关注人。
                """

        # 构建用户 Prompt (基于 V3.0 逻辑)
        user_prompt = f"""
                请为患者【{self.patient}】编写《术后康复计划与未来运动建议书》。\n

                ### 一、 关键决策数据 (Context Data)\n

                **1. 生理硬约束 (The "Hardware"):**\n
                * **手术核心参数：** {evidence['surgery_summary']}\n
                * **影像学风险提示：** {evidence['imaging_risk']}\n
                * **当前炎症与代谢状态：** {evidence['inflammation_status']}\n
                    {evidence['background_risk']}\n

                **2. 目标与心理调节 (The "Software"):**\n
                * **运动功能目标 (Tegner预测):** {predictions['tegner_score']} 分
                    * *归因参考:* {predictions['tegner_reason']}\n
                * **心理准备度 (ACL-RSI预测):** {predictions['aclrsi_score']} 分
                    * *归因参考:* {predictions['aclrsi_reason']}\n

                ### 二、 任务要求 (Task)\n

                请严格按照以下五个模块输出计划：\n

                #### 绝对禁忌与红线 (Red Flags)\n
                * **通用禁忌：** 术后12周内严禁开链伸膝（踢腿机）0-45度训练。
                * **通用禁忌：** 术后12周内严禁开链伸膝（踢腿机）0-45度训练。
                * **医嘱强行约束 (Doctor's Orders)：** \n
                    1. **负重红线：** 前4周内严格避免任何负重 (Strict NWB)，仅允许足尖点地 (Touch-down weight bearing) 辅助平衡。\n
                    2. **负重进阶逻辑：** 第4周后的负重比例，必须依据【手术核心参数】推演：
                       - 若半月板缝合针数多或有骨折：第4周开始极慢速增加负重（如每周增加10%-20%）。
                       - 若半月板缝合少且骨愈合好：第4周可较快过渡至全负重。
                * **其他个性化禁忌：** 1. 根据【影像学风险】，若骨隧道位置不佳，警告避免膝关节过伸锁死。
                    2. 根据【代谢状态】，若D-二聚体高或尿酸高，给出相应警示。\n

                #### 分阶段康复路径 (Phased Progression)\n
                请规划 **0-6周、6-12周、3-6个月、6个月+** 四个阶段。
                **关键要求：**
                * **0-6周特殊说明**：\n
                    - **0-4周**：专注于消肿、被动角度（ROM）及股四头肌唤醒，**严禁负重训练**。
                    - **4-6周**：作为负重过渡期，根据“手术核心参数”中的半月板及骨折情况，明确写出起始负重百分比（例如：起始20%或30%）。
                * **基于标准的进阶 (Criteria-Based)**：每个阶段末尾必须列出 **"通关考试" (Exit Criteria)**（如：跑步标准=患侧单腿蹲稳+肌力>健侧70%），而不仅仅是时间到了就进阶。
                * **移植物适配**：如果是腘绳肌腱，前6周减少主动勾腿；如果是BPTB，强调髌骨松动。\n

                #### 心理-运动干预策略 (Psychological Intervention)\n
                * 基于 ACL-RSI 分数 ({predictions['aclrsi_score']})：
                    * 若 < 60 (恐惧型)：设计“渐进式暴露训练”，强调无痛成功体验。
                    * 若 > 80 且处于早期 (激进型)：加入“现实核查”，强调生物愈合周期不可逾越。\n

                #### 重返运动专项 (Return to Sport - RTS)\n
                * 基于 Tegner 目标 ({predictions['tegner_score']} 分)：
                    * **正面清单：** 推荐的具体运动项目。
                    * **专项准备：** 若分数>7（接触性运动），必须包含急停变向(Cutting)和增强式训练(Plyometrics)。若分数低，侧重生活功能。\n

                #### 生活与营养建议\n
                * 结合检验数据（尿酸/血糖/血脂）给出饮食建议。
                * 职业康复建议（久坐或重体力）。\n

                ### 三、 输出格式\n
                * 语气专业、温暖、逻辑严密。
                * 使用以下 JSON 格式输出。
                {{
                    "术后康复计划": 术后康复计划,
                    "未来运动建议": 未来运动建议,
                    "reasoning": "详细的理由，特别是解释为何根据针数/骨折情况制定了第4周后的具体负重策略""
                }}
                """
        return {
            "user_prompt": user_prompt,
            "system_prompt": system_prompt
        }



    def step3_generate_plan(self, predictions: Dict):
        """
        生成康复计划 (核心 Prompt 构建)
        """
        print("\n" + "=" * 50)
        print("📝 Step 3: 生成智能化康复计划")
        print("=" * 50)

        evidence = self.get_evidence()

        prompt_dict = self.generate_plan_prompt(evidence=evidence, predictions=predictions)

        # 调用 LLM
        response = self.llm.chat(prompt_dict.get("user_prompt"), prompt_dict.get("system_prompt"))

        if response:
            self._save_report(response, predictions)
            return True
        else:
            logger.error("生成计划失败：LLM无响应")
            return False

    def _save_report(self, content: Any, predictions: Dict):
        """保存文件"""
        patient_dir = os.path.join(self.interim_dir, self.patient)
        if not os.path.exists(patient_dir):
            os.makedirs(patient_dir)
        filepath = os.path.join(patient_dir, f'{self.rehabilitation_rts_program_filename}')

        final_report = {'患者': self.patient, '院区': self.center,
                        'predicted_ACL-RSI': predictions['aclrsi_score'],
                        'predicted_Tegner': predictions['tegner_score']
                        }

        for k, v in content.items():
            final_report[k] = v

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(final_report, f, ensure_ascii=False, indent=4)  # 确保中文不乱码，格式化输出
            print(f"\n✅ 康复计划与未来运动建议已生成并保存至: {filepath}")
        except Exception as e:
            logger.error(f"保存文件失败: {e}")

    def run(self):
        """执行全流程"""
        print(f"🚀 开始为患者 {self.patient} 生成智能康复计划...")

        # 1. 抓取硬数据
        self.step1_retrieve_hard_constraints()

        # 2. 读取软数据
        predictions = self.step2_load_soft_goals()

        # 3. 生成报告
        success = self.step3_generate_plan(predictions)

        return success
