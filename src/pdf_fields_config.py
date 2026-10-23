FIELDS_DICT = {
    "Complete blood count": ["Patient name", "Sex", "Age", "Department", "Bed number", "Inpatient ID", "Physician", "Clinical diagnosis", "Specimen type",
                       "Specimen ID", "Accession number", "Laboratory test",  "High-sensitivity C-reactive protein (hsCRP)", "C-reactive protein (CRP)",
                       "Serum amyloid A (SAA)","White blood cell count (WBC)", "Red blood cell count (RBC)",
                       "Hemoglobin (HGB)", "Platelet count (PLT)", "Hematocrit (HCT)", "Mean corpuscular volume (MCV)",
                       "Mean corpuscular hemoglobin (MCH)", "Mean corpuscular hemoglobin concentration (MCHC)", "Red cell distribution width (CV) (RDW-CV)",
                       "Red cell distribution width (SD) (RDW-SD)", "Lymphocyte percentage (LYM%)", "Neutrophil percentage (NEUT%)",
                       "Monocyte percentage (MONO%)","Eosinophil percentage (EOS%)", "Basophil percentage (BASO%)", "Lymphocytes (LYM)",
                       "Neutrophils (NEUT)", "Monocytes (MONO)", "Eosinophils (EOS)","Basophils (BASO)", "Plateletcrit (PCT)",
                       "Mean platelet volume (MPV)", "Platelet distribution width (PDW)", "Large platelet ratio (P-LCR)",  "Reticulocyte percentage (RET%)",
                       "Reticulocytes (RET)", "Immature reticulocyte fraction (IRF)", "Low-fluorescence reticulocyte fraction (LFR)", "Medium-fluorescence reticulocyte fraction (MFR)",
                       "High-fluorescence reticulocyte fraction (HFR)"],

    "X-ray": ["Patient name", "Sex", "Age", "Imaging ID", "Inpatient ID", "Examination date", "Department", "Bed number", "Outpatient ID", "Examination method", "Imaging findings", "Imaging diagnosis",
                    "Reporting physician", "Reviewing physician", "Report date"],


    "Erythrocyte sedimentation rate": ["Patient name","Sex","Age","Department","Bed number","Inpatient ID","Physician","Clinical diagnosis","Specimen type","Accession number","Laboratory test","Specimen ID","Erythrocyte sedimentation rate (ESR)"],

    "Biochemistry": ["Patient name","Sex", "Age","Department","Bed number","Inpatient ID","Physician",
                       "Clinical diagnosis","Specimen type", "Accession number","Laboratory test", "Specimen ID", "Potassium (K)",  "Sodium (Na)", "Chloride (Cl)", "Calcium (Ca)",
                       "Urea", "Creatinine (Crea)", "Blood urea nitrogen/creatinine (BUN/Crea)", "Carbon dioxide combining power (CO2-CP)", "Glucose (GLU)", "Total protein (TP)",
                       "Albumin (ALB)", "Globulin (GLB)", "Alanine aminotransferase (ALT)", "Aspartate aminotransferase (AST)", "Gamma-glutamyl transferase (GGT)", "Total bilirubin (TBIL)",
                       "Alkaline phosphatase (ALP)", "Cholinesterase (CHE)"],

    "MR": ["Patient name","Sex","Age","Examination date","Department","Bed number","Outpatient ID","Imaging ID","Inpatient ID","Examination method","Imaging findings", "Imaging diagnosis","Reporting physician","Reviewing physician","Report date"],


    "CT": ["Patient name","Sex","Age","Examination date","Department","Bed number","Inpatient ID", "Outpatient ID", "Imaging ID","Examination method","Imaging findings","Imaging diagnosis","Reporting physician","Reviewing physician","Report date"],


    "Admission note": ["Patient name", "Sex", "Age", "Department", "Bed number", "Inpatient ID", "Occupation", "Department admission time", "Record date", "History informant",
                 "Marital status", "Chief complaint", "History of present illness", "Past medical history", "Personal history", "Marital and reproductive history", "Family history", "Temperature","Pulse","Respiratory rate","Blood pressure", "General condition",
                 "Skin and mucous membranes", "Superficial lymph nodes", "Head and organ examination", "Neck examination", "Chest examination", "Abdominal examination", "Anus", "External genitalia", "Spine and limbs",
                 "Nervous system", "Specialist examination", "Auxiliary examinations", "Admission diagnosis", "Physician signature", "Signature date"],

    "Discharge summary": ["Patient name", "Sex", "Age", "Department", "Bed number", "Inpatient ID", "Admission date", "Surgical date", "Surgical procedure", "Wound healing",
                     "Admission diagnosis", "Discharge diagnosis", "Condition on admission", "Diagnosis and treatment course", "Condition at discharge", "Discharge instructions", "Physician signature"],


    "Operative report": ["Patient name","Sex","Age","Department","Bed number","Inpatient ID","Preoperative diagnosis","Surgical procedure","Surgical date","Postoperative diagnosis",
                     "Anesthesia method","Anesthesiologist","Surgeon","First assistant","Second assistant","Operative procedure details","Record date"],

    "Preoperative summary": ["Patient name","Sex","Age","Department","Bed number","Inpatient ID","Preoperative diagnosis","Surgical indications","Discussion participants and professional titles",
                     "Discussion date","Discussion (preoperative preparation, surgical plan, potential complications and preventive measures)","Chairperson summary","Physician signature"],

    "Progress notes": ["Patient name","Sex","Age","Department","Bed number","Inpatient ID","Initial progress note date", "Initial progress note",
                     "Preoperative day 1 ward-round note date","Preoperative day 1 ward-round note",
                     "Preoperative day 2 ward-round note date", "Preoperative day 2 ward-round note",
                     "Day-of-surgery postoperative progress note date",  "Day-of-surgery postoperative progress note",
                     "Postoperative day 1 senior physician ward-round date", "Postoperative day 1 senior physician ward-round note",  "Postoperative day 2 ward-round note date",
                    "Postoperative day 2 ward-round note", "Postoperative day 3 routine ward-round note date",  "Postoperative day 3 routine ward-round note", "Discharge progress note date",
                    "Discharge progress note"
                     ],
    "Long-term medication orders": ["Patient name","Sex","Age","Department","Bed number","Inpatient ID","Start date", "Start-date physician signature", "Start-date nurse signature", "Order number", "Long-term medication orders", "Dose per administration", "Frequency",
                       "Route of administration", "Stop date", "Stop-date physician signature", "Stop-date nurse signature", "Remarks"]

}