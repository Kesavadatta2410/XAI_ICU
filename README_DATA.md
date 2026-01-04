# Data Documentation - MIMIC-IV ICU Diabetic Patients

Comprehensive documentation of the preprocessed MIMIC-IV dataset for 500 diabetic ICU patients.

---

## Dataset Summary

- **Source**: MIMIC-IV (Medical Information Mart for Intensive Care)
- **Cohort**: Diabetic patients admitted to ICU
- **Sample Size**: 500 unique patients/admissions
- **Preprocessing**: RAM-safe chunked processing, timestamp-safe, filtered to ICU time window

---

## File Descriptions

### 1. cohort_500.csv (176 KB)

Patient-level information with diagnoses and outcomes.

| Column | Type | Description |
|--------|------|-------------|
| `subject_id` | int | Unique patient identifier |
| `hadm_id` | int | Hospital admission ID |
| `icd_code` | str | ICD-10 diagnosis code |
| `icd_version` | int | ICD version (9 or 10) |
| `admittime` | datetime | Admission timestamp |
| `dischtime` | datetime | Discharge timestamp |
| `deathtime` | datetime | Death time (if applicable) |
| `hospital_expire_flag` | int | **Target variable** (0=survived, 1=died) |
| `anchor_age` | int | Patient age (de-identified) |
| `gender` | str | M/F |
| `los` | float | Length of stay (days) |
| `stay_id` | int | ICU stay identifier |
| `intime` | datetime | ICU admission time |
| `outtime` | datetime | ICU discharge time |

**Key Statistics:**
- Mortality Rate: 11.0%
- Mean Age: 65 ± 14 years
- Gender: 62% Male, 38% Female
- Unique ICD Codes: 34

**Top ICD Codes:**
| Code | Count | Description |
|------|-------|-------------|
| E119 | 192 | Type 2 diabetes without complications |
| E1122 | 92 | Type 2 diabetes with chronic kidney disease |
| E1165 | 79 | Type 2 diabetes with hyperglycemia |
| E1140 | 17 | Type 2 diabetes with diabetic neuropathy |

---

### 2. vitals_500.csv (326 MB)

Vital sign measurements during ICU stay.

| Column | Type | Description |
|--------|------|-------------|
| `subject_id` | int | Patient identifier |
| `hadm_id` | int | Admission ID |
| `stay_id` | int | ICU stay ID |
| `charttime` | datetime | Measurement timestamp |
| `itemid` | int | Vital sign type identifier |
| `valuenum` | float | Numeric value |
| `intime` | datetime | ICU start time |
| `outtime` | datetime | ICU end time |

**Statistics:**
- Total Records: 3,096,113
- Unique ItemIDs: 1,441
- Missing Values: 61.1%
- Median per Patient: 2,587 measurements

**Common ItemIDs:**
| ItemID | Count | Typical Measurement |
|--------|-------|---------------------|
| 227969 | 102K | Heart rate (monitor) |
| 220045 | 58K | Heart rate |
| 220277 | 57K | SpO2 |
| 220210 | 57K | Respiratory rate |
| 220179 | 33K | Systolic BP |
| 220180 | 33K | Diastolic BP |

---

### 3. labs_500.csv (17 MB)

Laboratory test results.

| Column | Type | Description |
|--------|------|-------------|
| `subject_id` | int | Patient identifier |
| `hadm_id` | int | Admission ID |
| `itemid` | int | Lab test type |
| `charttime` | datetime | Test timestamp |
| `valuenum` | float | Numeric result |
| `stay_id` | int | ICU stay ID |

**Statistics:**
- Total Records: 170,432
- Unique Lab Types: 491
- Missing Values: 8.3%
- Median per Patient: 176 tests

**Common Labs:**
| ItemID | Count | Test |
|--------|-------|------|
| 52033 | 4.8K | - |
| 50934 | 4.7K | - |
| 50947 | 4.7K | - |
| 50983 | 4.3K | Sodium |
| 50971 | 4.2K | Potassium |
| 50912 | 4.1K | Creatinine |

---

### 4. pharmacy_500.csv (8.5 MB)

Pharmacy orders and medications.

| Column | Type | Description |
|--------|------|-------------|
| `subject_id` | int | Patient identifier |
| `hadm_id` | int | Admission ID |
| `pharmacy_id` | int | Pharmacy order ID |
| `starttime` | datetime | Order start |
| `stoptime` | datetime | Order stop |
| `medication` | str | Medication name |
| `doses_per_24_hrs` | float | Dosing frequency |

**Statistics:**
- Total Records: 31,033
- Patients: 499
- Median per Patient: 47 orders

---

### 5. prescriptions_500.csv (8.9 MB)

Prescription information.

| Column | Type | Description |
|--------|------|-------------|
| `subject_id` | int | Patient identifier |
| `hadm_id` | int | Admission ID |
| `drug` | str | Drug name |
| `route` | str | Administration route |
| `starttime` | datetime | Prescription start |
| `stoptime` | datetime | Prescription end |

**Statistics:**
- Total Records: 39,468
- Patients: 499
- Median per Patient: 57 prescriptions

---

### 6. emar_500.csv (11 MB)

Electronic medication administration records.

| Column | Type | Description |
|--------|------|-------------|
| `subject_id` | int | Patient identifier |
| `hadm_id` | int | Admission ID |
| `charttime` | datetime | Administration time |
| `event_txt` | str | Event description |

**Statistics:**
- Total Records: 95,265
- Patients: 487
- Median per Patient: 85 records

---

### 7. inputevents_500.csv (24 MB)

IV fluids and inputs.

| Column | Type | Description |
|--------|------|-------------|
| `subject_id` | int | Patient identifier |
| `starttime` | datetime | Input start |
| `endtime` | datetime | Input end |
| `itemid` | int | Input type |
| `amount` | float | Volume administered |
| `rate` | float | Infusion rate |

**Statistics:**
- Total Records: 75,001
- Patients: 408
- Median per Patient: 77.5 events

---

### 8. outputevents_500.csv (4.8 MB)

Output measurements (urine, drainage, etc.).

| Column | Type | Description |
|--------|------|-------------|
| `subject_id` | int | Patient identifier |
| `charttime` | datetime | Measurement time |
| `itemid` | int | Output type |
| `value` | float | Volume |

**Statistics:**
- Total Records: 35,092
- Patients: 494
- Median per Patient: 39 measurements

---

### 9. procedureevents_500.csv (1.1 MB)

ICU procedures performed.

| Column | Type | Description |
|--------|------|-------------|
| `subject_id` | int | Patient identifier |
| `itemid` | int | Procedure type |
| `starttime` | datetime | Procedure start |
| `endtime` | datetime | Procedure end |

**Statistics:**
- Total Records: 4,660
- Patients: 407
- Median per Patient: 8 procedures

---

### 10. microbiology_500.csv (308 KB)

Microbiology culture results.

| Column | Type | Description |
|--------|------|-------------|
| `subject_id` | int | Patient identifier |
| `charttime` | datetime | Collection time |
| `org_name` | str | Organism identified |

**Statistics:**
- Total Records: 3,163
- Patients: 278
- Median per Patient: 5 cultures

---

### 11. ingredientevents_500.csv (22 MB)

Medication ingredient details.

| Column | Type | Description |
|--------|------|-------------|
| `subject_id` | int | Patient identifier |
| `starttime` | datetime | Start time |
| `itemid` | int | Ingredient type |
| `amount` | float | Amount administered |

**Statistics:**
- Total Records: 97,907
- Patients: 408
- Median per Patient: 97 events

---

### 12. drg_500.csv (408 KB)

Diagnosis-related groups for billing/severity.

| Column | Type | Description |
|--------|------|-------------|
| `subject_id` | int | Patient identifier |
| `hadm_id` | int | Admission ID |
| `drg_code` | str | DRG code |
| `description` | str | DRG description |
| `drg_severity` | int | Severity index |

**Statistics:**
- Total Records: 989
- Unique DRG Codes: Multiple severity levels

---

## Time Gap Analysis (Δt)

Critical for Liquid-Mamba's time-decay mechanism:

| Statistic | Value (minutes) |
|-----------|-----------------|
| Mean | 1.1 |
| Median | 0.0 |
| Std | 6.9 |
| 95th percentile | 1.0 |
| Max | 479.0 |

**Interpretation**: Most measurements are clustered together (median Δt = 0), with occasional large gaps. The Liquid-Mamba model uses these time gaps to modulate state decay.

---

## Missing Data Patterns

| Data Type | Missing Rate |
|-----------|--------------|
| Vitals | 61.1% |
| Labs | 8.3% |

High missingness in vitals is expected—not all vital signs are measured at each charting event. The model uses explicit missingness masks to handle this.

---

## Data Quality Notes

1. **Timestamps**: All timestamps are cleaned with `errors='coerce'`
2. **ICU Filtering**: All events are filtered to the ICU time window (intime to outtime)
3. **No Imputation**: Original missing values preserved for mask-based modeling
4. **Chunked Processing**: Large files (vitals) processed in memory-safe chunks

---

## Usage for Model Training

### Target Variable
```python
y = cohort_df.groupby('hadm_id')['hospital_expire_flag'].first()
```

### Feature Modalities
- **Vitals**: Continuous time-series (itemid → feature index)
- **Labs**: Sparse time-series (itemid → feature index)  
- **Medications**: Binary indicators or embedding
- **ICD Codes**: Graph nodes for GAT encoder

### Recommended Split
- Train: 70%
- Validation: 15%
- Test: 15%
- Stratified by mortality outcome

---

*Documentation generated from EDA analysis*
