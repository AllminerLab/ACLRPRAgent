from prompt_templates import render_prompt, system_prompt, retrieval_goal, RETRIEVAL_TASKS

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

# Basic configuration
base_dir = os.path.split(os.path.realpath(__file__))[0]
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
if not logger.handlers:
    logger = init_logger('../data/log')




class QwenAPIClient:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or QWEN_KEY_API

    def chat(self, prompt: str, system_prompt: str = "You are a senior sports medicine specialist.", temperature: float = 0.6):
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
        # Define Tegner output filenames.
        self.retrieve_results_filename = "tegner_retrieval_result.json"
        self.predict_report_filename = "tegner_predict_report.json"
        self.raw_dir = raw_dir
        self.complete_follow_up_filename = "follow_up_scores.csv"

        if MedicalRetriever:
            self.retriever = MedicalRetriever(
                data_dir=data_dir, output_dir=output_dir, print_thinking=True, max_input_tokens=8096,
                default_extension=".csv"
            )
        else:
            sys.exit(1)

    # Structured JSON updates
    def _update_retrieve_json_file(self, center: str, patient: str, step_key: str, inner_key: str, data: Any):
        "Update a nested field in a JSON file."
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
            print(f"""  [JSON update] Saved {step_key} -> {inner_key}""")
        except Exception as e:
            logger.error(f"""Failed to write file: {e}""")

    # Validation helpers
    def _check_success_retrieval(self, result: RetrievalResult, step_name: str) -> bool:
        if not result.success:
            print(f"""  [{step_name}] Retrieval failed: {result.error_message}""")
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

        # Add the observed Tegner score to the evaluation report after prediction.
        raw_follow_up_data = pd.read_csv(f'{self.raw_dir}/{center}/{self.complete_follow_up_filename}')

        # Read the Postoperative Tegner column; adapt the input header if needed.

        ground_truth_score = raw_follow_up_data[raw_follow_up_data["Patient name"] == patient]["Postoperative Tegner"].values[0].item()
        preoperative_score = raw_follow_up_data[raw_follow_up_data["Patient name"] == patient]["Preoperative Tegner"].values[0].item()

        final_report = {"patient": patient, "center": center, 'preoperative_Tegner': preoperative_score,
                        'ground_truth_Tegner': ground_truth_score}

        for k, v in data.items():
            if k == 'score':
                k = 'predicted_Tegner'
            final_report[k] = v

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, ensure_ascii=False, indent=4)

    # =========================================================================
    # Step 1: Build the activity baseline profile.
    # =========================================================================
    def step1_patient_base_profile(self, center: str, patient: str):
        print("\n" + "=" * 50)
        print("Step 1: Build the baseline profile (activity history)")
        print("=" * 50)

        valid_docs_content = []

        # 1.1 Retrieve the perioperative registration record.
        print("  1.1 Retrieve the perioperative registration record")
        res_reg = self.retriever.retrieve_by_documents(
            center=center, patient=patient,
            document_names=["Perioperative clinical data of patients with ACL Injury"],
            task_goal=retrieval_goal(0)
        )

        if self._check_success_retrieval(res_reg, "Perioperative registration record"):
            self._update_retrieve_json_file(
                center=center,
                patient=patient,
                step_key="step1_patient_base_profile",
                inner_key="perioperative_registration_retrieval",
                data=res_reg)
            valid_docs_content.append(self._format_single_result("Perioperative registration record", res_reg))

        # 1.2 Retrieve clinical records.
        print("  1.2 Retrieve operative and admission records")
        res_docs = self.retriever.retrieve_by_documents(
            center=center, patient=patient,
            document_names=["Operative report", "Admission note"],
            task_goal=retrieval_goal(1)
        )
        if self._check_success_retrieval(res_docs, "Operative and admission records"):
            self._update_retrieve_json_file(
                center=center,
                patient=patient,
                step_key="step1_patient_base_profile",
                inner_key="hospital_admission_and_surgery_record_retrieval",
                data=res_docs)
            valid_docs_content.append(self._format_single_result("Operative and admission records", res_docs))

        if valid_docs_content:
            print("  1.3 Generate the baseline summary")
            combined_text = "\n".join(valid_docs_content)
            summary_prompt =render_prompt('tegner_baseline', **{"patient baseline profile": combined_text})


            summary = self.qwen_client.chat(
                prompt=summary_prompt)

            # Use the retrieved evidence if summarization fails.
            if not summary:
                logger.info("Summary generation failed; using the retrieved data")
                summary = valid_docs_content

            self._update_retrieve_json_file(
                center=center,
                patient=patient,
                step_key="step1_patient_base_profile",
                inner_key="patient_base_profile_summary",
                data=summary)
            self.retrieve_results["patient_base_profile_summary"] = summary
        else:
            print("  Step 1 found no valid documents; skipping summary generation")
            self.retrieve_results["patient_base_profile_summary"] = None

    # =========================================================================
    # Step 2: Review preoperative status and baseline activity.
    # =========================================================================
    def step2_pre_op_retrospection(self, center: str, patient: str):
        print("\n" + "=" * 50)
        print("Step 2: Review preoperative status (baseline activity)")
        print("=" * 50)

        baseline = self.retrieve_results.get("patient_base_profile_summary")
        baseline_context = f"{baseline}" if baseline else "Patient baseline profile unavailable"

        res = self.retriever.retrieve_by_condition(
            center=center, patient=patient,
            search_condition=RETRIEVAL_TASKS[2]["condition"],
            task_goal=retrieval_goal(2, baseline_context)
        )

        if self._check_success_retrieval(res, "Preoperative data"):
            self._update_retrieve_json_file(
                center=center,
                patient=patient,
                step_key="step2_pre_op_retrospection",
                inner_key="preoperative_retrieval",
                data=res)

    # =========================================================================
    # Step 3: Review postoperative status and activity level.
    # =========================================================================
    def step3_post_op_tracking(self, center: str, patient: str):
        print("\n" + "=" * 50)
        print("Step 3: Review postoperative status (activity level)")
        print("=" * 50)

        baseline = self.retrieve_results.get("patient_base_profile_summary")
        baseline_context = f"{baseline}" if baseline else "Patient baseline profile unavailable"

        res = self.retriever.retrieve_by_condition(
            center=center, patient=patient,
            search_condition=RETRIEVAL_TASKS[3]["condition"],
            task_goal=retrieval_goal(3, baseline_context)
        )

        if self._check_success_retrieval(res, "Postoperative data"):
            self._update_retrieve_json_file(
                center=center,
                patient=patient,
                step_key="step3_post_op_tracking",
                inner_key="postoperative_retrieval",
                data=res)

    def get_predict_prompt(self, evidence):
        return render_prompt('tegner_prediction', **{"retrieved evidences": evidence})

    # =========================================================================
    # Step 4: Predict and explain the score using nonempty evidence.
    # =========================================================================
    def step4_prediction_and_attribution(self, center: str, patient: str):
        print("\n" + "=" * 50)
        print("Step 4: Predict and explain Tegner score")
        print("=" * 50)

        # 4.1 Aggregate evidence.
        evidence = self._aggregate_evidence_strictly(center=center, patient=patient)

        # 4.2 Stop if no evidence is available.
        if not evidence:
            print("  [Stopped] Steps 1-3 produced no valid data; prediction aborted")
            return {
                "score": 0, "confidence": "No data",
                "reasoning": "Cannot predict from an empty retrieval result.", "key_factors": []
            }

        print(f"""  Evidence modules supplied to the model: {list(evidence.keys())}""")

        # 4.3 Run prediction.
        prompt = self.get_predict_prompt(evidence=evidence)

        print("  Calling the prediction model")
        predict_report = self.qwen_client.chat(prompt, temperature=0.2)

        if predict_report:
            self._save_result(patient, self.predict_report_filename, predict_report, center)
            return True
        else:
            print("  Model returned no content")
            return None

    def _aggregate_evidence_strictly(self, center, patient) -> Dict[str, Any]:
        "Read and aggregate evidence from tegner_retrieval_result.json."
        evidence = {}
        file_path = os.path.join(self.output_dir, center, patient, self.retrieve_results_filename)

        if not os.path.exists(file_path):
            logger.warning(f"""File not found: {file_path}""")
            return {}

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                full_data = json.load(f)
        except Exception as e:
            logger.error(f"""Failed to read JSON: {e}""")
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
        lines = [f"""[Source: {source_name}]"""]
        if result.success and result.data:
            for item in result.data:
                lines.append(str(item))
        return "\n".join(lines)

    def run(self, center: str, patient: str):
        print(f"""\nProcessing: {patient}""")
        self.step1_patient_base_profile(center, patient)
        self.step2_pre_op_retrospection(center, patient)
        self.step3_post_op_tracking(center, patient)
        success = self.step4_prediction_and_attribution(center, patient)
        if success:
            print("\nTegner assessment completed")
            return True
        else:
            print("Assessment failed")
            return False
