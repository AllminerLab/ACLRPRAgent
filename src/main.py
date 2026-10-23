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

# Basic configuration
base_dir = os.path.split(os.path.realpath(__file__))[0]
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
if not logger.handlers:
    logger = init_logger('../data/log')

class ACLAgentPipeline:

    def __init__(self, base_dir="../data"):
        "Initialize the integrated pipeline with a shared data directory."
        self.base_dir = base_dir
        self.interim_dir = self.base_dir + "/interim"


    def run_single_pipeline(self, center: str, patient: str):
        "Run the complete pipeline for one patient."
        time_tracker = TimeTracker(file_path=f'../data/interim/{center}/{patient}', filename="time_records.json")
        start_time = time.time()
        logger.info("\n" + "#" * 60)
        logger.info(f"""Starting the ACLRPRAgent assessment pipeline""")
        logger.info(f"""Center: {center}""")
        logger.info(f"""Patient: {patient}""")
        logger.info("#" * 60 + "\n")

        # =================================================
        # Stage 1: Data preprocessing
        # =================================================
        logger.info(f"""[1/4] Cleaning and structuring clinical data""")
        try:
            manager = PatientDataManager()
            success = manager.process_patient(
                patient_name=patient,
                center=center
            )

            if success:
                logger.info("\nProcessing complete")

            if not success:
                logger.error("[Stopped] Data preprocessing failed")
                return None
            logger.info("Data preprocessing complete")
        except Exception as e:
            logger.error(f"""[Error] Data preprocessing: {e}""")
            return None

        # =================================================
        # Stage 2: ACL-RSI psychological readiness assessment
        # =================================================
        logger.info(f"""\n[2/4] Assessing ACL-RSI psychological readiness""")
        try:
            acl_rsi_predictor = ACLRSIPredictor()

            success_s2= acl_rsi_predictor.run(center=center, patient=patient)

            # Decide whether missing assessment data should stop the pipeline.
            # Without a valid assessment, subsequent planning would lack required inputs.
            # This pipeline therefore requires a successful assessment before planning.
            if not success_s2:
                logger.warning("ACL-RSI assessment produced no valid result; psychological recommendations would be incomplete.")
                return None
            else:
                logger.info("ACL-RSI assessment complete")
        except Exception as e:
            logger.error(f"""[Error] ACL-RSI assessment: {e}""")
            return None

        # =================================================
        # Stage 3: Tegner activity assessment
        # =================================================
        logger.info(f"""\n[3/4] Assessing Tegner activity level""")
        try:
            tegner_predictor = TegnerPredictor()
            success_s3 = tegner_predictor.run(center=center, patient=patient)

            if not success_s3:
                logger.warning("Tegner assessment produced no valid result; activity recommendations would be incomplete.")
                return None
            else:
                logger.info("Tegner assessment complete")
        except Exception as e:
            logger.error(f"""[Error] Tegner assessment: {e}""")
            return None

        # =================================================
        # Stage 4: Rehabilitation plan generation
        # =================================================
        logger.info(f"""\n[4/4] Generating rehabilitation and return-to-activity recommendations""")
        try:
            planner = ACLRehabPlanner(center=center, patient=patient)
            # The planner reads the JSON outputs from stages 2 and 3.
            planner.run()
            logger.info("Rehabilitation plan generation complete")
        except Exception as e:
            logger.error(f"""[Error] Rehabilitation plan generation: {e}""")
            return None

        elapsed = time.time() - start_time
        logger.info("\n" + "=" * 60)
        time_tracker.add_record(start_time=start_time, end_time=time.time(), record_name=f"""Total ACLRPRAgent pipeline duration (min)""")
        logger.info("=" * 60)
        return True

    def run_pipeline(self, center: str, patient: str):
        for i in range(2):
            success = self.run_single_pipeline(center=center, patient=patient)
            if success is not None:
                logger.info(f"""ACL rehabilitation assessment completed for {patient} ({center})""")
                return
            else:
                logger.info(f"""An error occurred; retrying ACL rehabilitation assessment""")

        return logger.info(f"""ACL rehabilitation assessment failed for {patient} ({center})""")


    def run_batch(self, center: str, patient_list: list):
        "Process a batch of patients."
        logger.info(f"""Starting batch processing for {len(patient_list)} patients""")
        for p in patient_list:
            self.run_pipeline(center, p)


if __name__ == "__main__":
    # Initialize the pipeline.
    system = ACLAgentPipeline(base_dir="../data")

    system.run_batch(
        center="Center-A",
        patient_list=['A001'])