from prompt_templates import render_prompt, system_prompt, retrieval_goal, RETRIEVAL_TASKS

import os
import sys
import json
import logging
from typing import Dict, Any, Optional

from retrieval_tool import MedicalRetriever, RetrievalResult
from llm_api_config import QWEN_KEY_API
from streaming_llm_handler import qwen_stream_completion
from logger import init_logger

# Basic configuration
base_dir = os.path.split(os.path.realpath(__file__))[0]
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
if not logger.handlers:
    logger = init_logger('../data/log')


class QwenAPIClient:
    "Lightweight Qwen client wrapper."

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

        # Initialize the retriever.
        self.retriever = MedicalRetriever(
            data_dir=data_dir,
            output_dir=interim_dir,
            print_thinking=True,  # Enable reasoning output.
            max_input_tokens=6000
        )

        # Initialize the LLM client.
        self.llm = QwenAPIClient()


    # Structured JSON updates
    def _update_retrieve_json_file(self, patient: str, step_key: str, data: Any, inner_key: str = None):
        "Update a nested JSON field: json_content[step_key][inner_key] = serialized_data. patient selects the directory; step_key and inner_key identify the record; data can be a RetrievalResult, string or dictionary."
        # 1. Prepare the output path.
        patient_dir = os.path.join(self.interim_dir, patient)
        if not os.path.exists(patient_dir):
            os.makedirs(patient_dir)
        file_path = os.path.join(patient_dir, self.retrieve_results_filename)

        # 2. Serialize objects to dictionaries.
        save_data = data
        if isinstance(data, RetrievalResult):  # Handle RetrievalResult objects.
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

        # 3. Read the existing file.
        full_content = {}
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    full_content = json.load(f)
            except json.JSONDecodeError:
                full_content = {}

        # 4. Ensure the top-level key exists.
        if step_key not in full_content or not isinstance(full_content[step_key], dict):
            full_content[step_key] = {}

        # 5. Update the nested key.
        if inner_key is not None:
            full_content[step_key][inner_key] = save_data
        else:
            full_content[step_key] = save_data

        # 6. Write the updated file.
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(full_content, f, ensure_ascii=False, indent=2)
            print(f"""  [JSON update] Saved {step_key}""")
        except Exception as e:
            logger.error(f"""Failed to write file: {e}""")

    # Validation helpers
    def _check_success_retrieval(self, result: RetrievalResult) -> bool:
        "Check whether retrieval produced a valid result."
        if not result.success:
            print(f"""  Retrieval failed: {result.error_message}""")
            return False

        return True

    def _load_prediction_report(self, filename: str) -> Dict[str, Any]:
        "Load an earlier Tegner or ACL-RSI prediction report."
        path = os.path.join(self.interim_dir, self.patient, filename)
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    report = json.load(f)
                    if 'acl_rsi' in filename and "ground_truth_ACL-RSI" in report:
                        value = report.pop('ground_truth_ACL-RSI', None)  # Use a default value to avoid KeyError.

                    if "tegner" in filename and "ground_truth_Tegner" in report:
                        value = report.pop('ground_truth_Tegner', None)

                    return report

            except Exception as e:
                logger.error(f"""Failed to read prediction report {filename}: {e}""")

        return {}

    def step1_retrieve_hard_constraints(self) -> Dict[str, str]:
        "Retrieve physiological constraints, prioritizing postoperative data while retaining preoperative context."
        print("\n" + "=" * 50)
        print("Step 1: Retrieve physiological constraints (surgery, imaging and laboratory tests)")
        print("=" * 50)

        evidence = {}

        # 1.1 Surgical information relevant to weight bearing and range of motion
        print("  Retrieving core surgical parameters")
        res_surgery = self.retriever.retrieve_by_documents(
            center=self.center,
            patient=self.patient,
            document_names=["Operative report", "Discharge summary"],  # Search these two record types first.
            task_goal=retrieval_goal(8)
        )
        if self._check_success_retrieval(res_surgery):
            # Surgical information is missing; use a conservative approach.
            self._update_retrieve_json_file(
                patient=self.patient,
                step_key="surgery_summary",
                data=res_surgery)

        # 1.2 Postoperative imaging relevant to extension restrictions
        print("  Retrieving postoperative imaging risks")
        res_imaging = self.retriever.retrieve_by_condition(
            center=self.center,
            patient=self.patient,
            search_condition=RETRIEVAL_TASKS[9]["condition"],
            task_goal=retrieval_goal(9)
        )
        if self._check_success_retrieval(res_imaging):
            # No postoperative imaging description; assume no specific anatomical risk.
            self._update_retrieve_json_file(
                patient=self.patient,
                step_key="imaging_risk",
                data=res_imaging)

        # 1.3 Postoperative inflammatory markers relevant to progression
        print("  Retrieving postoperative inflammatory markers")
        res_lab_post = self.retriever.retrieve_by_condition(
            center=self.center,
            patient=self.patient,
            search_condition=RETRIEVAL_TASKS[10]["condition"],
            task_goal=retrieval_goal(10)
        )
        if self._check_success_retrieval(res_lab_post):
            # No recent inflammatory-marker reassessment data.
            self._update_retrieve_json_file(
                patient=self.patient,
                step_key="inflammation_status",
                data=res_lab_post)


        # 1.4 Background risk factors relevant to thrombosis precautions and diet
        print("  Retrieving background risk factors")
        res_risk = self.retriever.retrieve_by_condition(
            center=self.center,
            patient=self.patient,
            search_condition=RETRIEVAL_TASKS[11]["condition"],
            task_goal=retrieval_goal(11)
        )
        if self._check_success_retrieval(res_risk):
            # No specific metabolic or coagulation risk documented.
            self._update_retrieve_json_file(
                patient=self.patient,
                step_key="background_risk",
                data=res_risk)


        return evidence

    def _format_retrieval_data(self, result, default_msg: str) -> str:
        "Format a retrieval result as text for a prompt."
        lines = []
        if result.get("success") and len(result.get("data"))>0:
            for item in result.get("data"):
                lines.append(str(item))

        if len(lines) > 0:
            return "\n".join(lines)
        else:
            return default_msg


    def step2_load_soft_goals(self) -> Dict[str, Any]:
        "Load predicted scores and attribution insights."
        print("\n" + "=" * 50)
        print("Step 2: Load Tegner and ACL-RSI predictions")
        print("=" * 50)

        # Load the Tegner prediction for activity planning.
        tegner_data = self._load_prediction_report("tegner_predict_report.json")
        tegner_score = tegner_data.get('predicted_Tegner')
        tegner_reason = {
            'reasoning': tegner_data.get('reasoning'),
            'confidence': tegner_data.get('confidence'),
            "key_factors": tegner_data.get('key_factors')
        }

        # Load the ACL-RSI prediction for psychological adaptation.
        aclrsi_data = self._load_prediction_report("acl_rsi_predict_report.json")
        aclrsi_score = aclrsi_data.get('predicted_ACL-RSI')
        aclrsi_reason = {
            'reasoning': aclrsi_data.get('reasoning'),
            'confidence': aclrsi_data.get('confidence'),
            "key_factors": aclrsi_data.get('key_factors')
        }

        print(f"""  Predicted activity level (Tegner): {tegner_score}""")
        print(f"""  Predicted psychological readiness (ACL-RSI): {aclrsi_score}""")

        return {
            "tegner_score": tegner_score,
            "tegner_reason": tegner_reason,
            "aclrsi_score": aclrsi_score,
            "aclrsi_reason": aclrsi_reason
        }

    def get_evidence(self):
        evidence = {}

        # 1. Build the file path.
        file_path = os.path.join(self.interim_dir, self.patient, self.retrieve_results_filename)

        # 2. Read the file.
        if not os.path.exists(file_path):
            logger.warning(f"""File not found: {file_path}""")
            return {}

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                full_data = json.load(f)
        except Exception as e:
            logger.error(f"""Failed to read JSON: {e}""")
            return {}

        evidence['surgery_summary'] = self._format_retrieval_data(full_data.get('surgery_summary'), "Surgical information is missing; use a conservative approach")

        evidence['imaging_risk'] = self._format_retrieval_data(full_data.get('imaging_risk'), "No postoperative imaging description; assume no specific anatomical risk")

        evidence['inflammation_status'] = self._format_retrieval_data(full_data.get('inflammation_status'), "No recent inflammatory-marker reassessment data")

        evidence['background_risk'] = self._format_retrieval_data(full_data.get('background_risk'), "No specific metabolic or coagulation risk documented")

        return evidence


    def generate_plan_prompt(self, evidence: Dict, predictions: Dict):
        values = {
            "patient identifier": self.patient,
            "surgical core parameters": evidence["surgery_summary"],
            "imaging risk evidence": evidence["imaging_risk"],
            "inflammatory and metabolic status": evidence["inflammation_status"],
            "background risk evidence": evidence["background_risk"],
            "predicted Tegner score": predictions["tegner_score"],
            "Tegner score attribution insights": predictions["tegner_reason"],
            "predicted ACL-RSI score": predictions["aclrsi_score"],
            "ACL-RSI score attribution insights": predictions["aclrsi_reason"],
        }
        return {"user_prompt": render_prompt("prescription", **values),
                "system_prompt": system_prompt("prescription")}



    def step3_generate_plan(self, predictions: Dict):
        "Build the prompt and generate a rehabilitation plan."
        print("\n" + "=" * 50)
        print("Step 3: Generate a rehabilitation plan")
        print("=" * 50)

        evidence = self.get_evidence()

        prompt_dict = self.generate_plan_prompt(evidence=evidence, predictions=predictions)

        # Call the LLM.
        response = self.llm.chat(prompt_dict.get("user_prompt"), prompt_dict.get("system_prompt"))

        if response:
            self._save_report(response, predictions)
            return True
        else:
            logger.error("Plan generation failed: no LLM response")
            return False

    def _save_report(self, content: Any, predictions: Dict):
        "Save the report."
        patient_dir = os.path.join(self.interim_dir, self.patient)
        if not os.path.exists(patient_dir):
            os.makedirs(patient_dir)
        filepath = os.path.join(patient_dir, f'{self.rehabilitation_rts_program_filename}')

        final_report = {"patient": self.patient, "center": self.center,
                        'predicted_ACL-RSI': predictions['aclrsi_score'],
                        'predicted_Tegner': predictions['tegner_score']
                        }

        for k, v in content.items():
            final_report[k] = v

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(final_report, f, ensure_ascii=False, indent=4)  # Preserve Unicode text and format the JSON output.
            print(f"""\nRehabilitation and return-to-activity recommendations saved to: {filepath}""")
        except Exception as e:
            logger.error(f"""Failed to save file: {e}""")

    def run(self):
        "Run the complete workflow."
        print(f"""Generating a rehabilitation plan for {self.patient}""")

        # 1. Retrieve physiological constraints.
        self.step1_retrieve_hard_constraints()

        # 2. Load prediction outputs.
        predictions = self.step2_load_soft_goals()

        # 3. Generate the report.
        success = self.step3_generate_plan(predictions)

        return success
