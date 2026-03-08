import os
import sys
import logging
import time

from data_preprocessing_tool import PatientDataManager
from rts_psychological_assessment_tool import ACLRSIPredictor
from tegner_activity_assessment_tool import  TegnerPredictor
from rehabilitation_and_rts_pro_gen_tool import ACLRehabPlanner
from logger import init_logger
from time_util import TimeTracker

# 基础配置
base_dir = os.path.split(os.path.realpath(__file__))[0]
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
if not logger.handlers:
    logger = init_logger('../data/log')

class ACLAgentPipeline:

    def __init__(self, base_dir="../data"):
        """
        初始化集成系统，设定统一的数据路径
        """
        self.base_dir = base_dir
        self.interim_dir = self.base_dir + "/interim"


    def run_single_pipeline(self, center: str, patient: str):
        """
        执行单个患者的全流程处理
        """
        time_tracker = TimeTracker(file_path=f'../data/interim/{center}/{patient}', filename="time_records.json")
        start_time = time.time()
        logger.info("\n" + "#" * 60)
        logger.info(f"🚀 启动 ACL智能体评估流程")
        logger.info(f"🏥 中心: {center}")
        logger.info(f"👤 患者: {patient}")
        logger.info("#" * 60 + "\n")

        # =================================================
        # Stage 1: 数据预处理 (Data Preprocessing)
        # =================================================
        logger.info(f"🔄 [1/4] 正在执行数据清洗与结构化...")
        try:
            manager = PatientDataManager()
            success = manager.process_patient(
                patient_name=patient,
                center=center
            )

            if success:
                logger.info("\n处理完成！")

            if not success:
                logger.error("⛔ [终止] 数据预处理失败，流程停止。")
                return None
            logger.info("✅ 数据预处理完成")
        except Exception as e:
            logger.error(f"⛔ [异常] 数据预处理阶段出错: {e}")
            return None

        # =================================================
        # Stage 2: ACL-RSI 心理评估 (Psychological Assessment)
        # =================================================
        logger.info(f"\n🧠 [2/4] 正在执行 ACL-RSI 心理准备度评估...")
        try:
            acl_rsi_predictor = ACLRSIPredictor()

            success_s2= acl_rsi_predictor.run(center=center, patient=patient)

            # 注意：如果评估失败（如缺数据），是否阻断流程？
            # 策略：如果完全失败，后续计划生成会缺数据，建议在此提示但尝试继续，
            # 或者由 Planner 处理缺失值。这里我们假设必须成功才能生成完美报告。
            if not success_s2:
                logger.warning("⚠️ ACL-RSI 评估未生成有效结果，后续报告可能缺失心理建议。")
                return None
            else:
                logger.info("✅ ACL-RSI 评估完成")
        except Exception as e:
            logger.error(f"⚠️ [异常] ACL-RSI 评估阶段出错: {e}")
            return None

        # =================================================
        # Stage 3: Tegner 运动评估 (Activity Assessment)
        # =================================================
        logger.info(f"\n🏃 [3/4] 正在执行 Tegner 运动水平评估...")
        try:
            tegner_predictor = TegnerPredictor()
            success_s3 = tegner_predictor.run(center=center, patient=patient)

            if not success_s3:
                logger.warning("⚠️ Tegner 评估未生成有效结果，后续报告可能缺失运动专项建议。")
                return None
            else:
                logger.info("✅ Tegner 评估完成")
        except Exception as e:
            logger.error(f"⚠️ [异常] Tegner 评估阶段出错: {e}")
            return None

        # =================================================
        # Stage 4: 康复计划生成 (Plan Generation)
        # =================================================
        logger.info(f"\n📝 [4/4] 正在生成智能化康复计划与运动建议...")
        try:
            planner = ACLRehabPlanner(center=center, patient=patient)
            # Planner 会读取 Stage 2 和 Stage 3 的 JSON 结果
            planner.run()
            logger.info("✅ 康复计划生成完成")
        except Exception as e:
            logger.error(f"⛔ [异常] 康复计划生成阶段出错: {e}")
            return None

        elapsed = time.time() - start_time
        logger.info("\n" + "=" * 60)
        time_tracker.add_record(start_time=start_time, end_time=time.time(), record_name=f'ACL智能体流程总耗时（min）')
        logger.info("=" * 60)
        return True

    def run_pipeline(self, center: str, patient: str):
        for i in range(2):
            success = self.run_single_pipeline(center=center, patient=patient)
            if success is not None:
                logger.info(f"{patient}({center})ACL运动康复评估成功！！")
                return
            else:
                logger.info(f"出现异常，ACL运动康复重新评估中...")

        return logger.info(f"{patient}({center})ACL运动康复评估失败")


    def run_batch(self, center: str, patient_list: list):
        """批量处理多个患者"""
        logger.info(f"📦 开始批量处理 {len(patient_list)} 位患者...")
        for p in patient_list:
            self.run_pipeline(center, p)


if __name__ == "__main__":
    # 实例化系统
    system = ACLAgentPipeline(base_dir="../data")

    system.run_batch(
        center="中山二院",
        patient_list=['A001'])