# MIMIC-IV Data Filter & Extraction Script

## Overview

`datafilter.py` extracts and prepares MIMIC-IV data for the XAI ICU Mortality Prediction System. It creates a diabetic cohort with 50k-100k patients suitable for training `research.py`.

---

## Purpose

This script solves the data preparation challenge by:
1. **Filtering** adult (≥18) diabetic ICU patients
2. **Ensuring** each patient has ≥5 chart events
3. **Stratified sampling** to maintain ~16% mortality rate
4. **Building** an ICD knowledge graph for disease relationships

---

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│ 📥 RAW MIMIC-IV DATA (Google Drive)                                 │
│   hosp/admissions.csv, icu/chartevents.csv, etc.                   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 👥 STEP 1: COHORT BUILDING                                          │
│   • Age ≥18 → ICU stays → Diabetic (ICD or Rx) → ≥5 chart events   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ ⚖️ STEP 2: STRATIFIED SAMPLING                                      │
│   • Target: 100k patients with ~16% mortality rate                 │
│   • Upsample deaths, downsample survivors as needed                │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 📤 STEP 3: DATA EXTRACTION                                          │
│   • Filter all tables to sampled patients only                     │
│   • Stream directly to disk (memory-efficient)                     │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 🔗 STEP 4: ICD KNOWLEDGE GRAPH                                      │
│   • Build hierarchical + co-occurrence edges                       │
│   • Save as icd_graph.gml for research.py                          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BASE_DIR` | Google Drive path | Root MIMIC-IV directory |
| `TARGET_SIZE` | 100000 | Target patient count (50k or 100k) |
| `DESIRED_MORTALITY_RATE` | 0.16 | Target mortality rate (15-18%) |
| `CHUNK_SIZE` | 500000 | Rows per chunk for memory efficiency |

---

## Diabetic Cohort Criteria

Patients are classified as diabetic if they have:
1. **ICD Codes**: E10-E14 (ICD-10) or 250 (ICD-9)
2. **Medications**: insulin, metformin, glipizide, glyburide, glimepiride

---

## Output Files

All files are saved to `data_{N}k/` (e.g., `data_100k/`):

| File | Description |
|------|-------------|
| `admissions_100k.csv` | Hospital admissions with mortality labels |
| `icustays_100k.csv` | ICU stay records |
| `chartevents_100k.csv` | Vital signs and lab values (~32GB) |
| `inputevents_100k.csv` | Medications and fluids |
| `outputevents_100k.csv` | Urine output, drains |
| `procedureevents_100k.csv` | ICU procedures |
| `drgcodes_100k.csv` | Diagnosis-related groups |
| `prescriptions_100k.csv` | Medication orders |
| `diagnoses_icd.csv` | ICD diagnosis codes |
| `patients_100k.csv` | Patient demographics |
| `hadm_icd.csv` | ICD codes per admission |
| `icd_graph.gml` | ICD knowledge graph (NetworkX) |
| `sampled_subjects.csv` | List of sampled patient IDs |

---

## Validation Checks (Step 5)

The script validates extraction by checking:
- ✅ Patient count matches target
- ✅ Mortality rate is within 15-18%
- ✅ All mandatory itemids present (HR, BP, SpO2, Glucose, etc.)
- ✅ Minimum 5 chart events per patient

---

## Usage

```bash
# Run on Colab with Google Drive mounted
python datafilter.py

# Expected output:
# - Cohort statistics at each step
# - Extraction progress for each file
# - Validation results
# - Total data size (GB)
```

---

## Memory Efficiency

The script is designed for limited RAM environments:
- **Chunked loading**: Processes large CSVs in 500k-row chunks
- **Streaming writes**: Writes directly to disk, no in-memory accumulation
- **Counter-based aggregation**: Efficient patient counting

---

## Integration with research.py

After running `datafilter.py`, copy the output folder to your local machine:

```bash
# From Colab
cp -r /content/data_100k /path/to/IIIT_Ranchi/data100k

# Then run research.py
python research.py
```

The `research.py` script expects:
- `data100k/admissions_100k.csv`
- `data100k/chartevents_100k.csv`
- `data100k/icd_graph.gml`
- etc.

---

## Mandatory Chart Event ItemIDs

| ItemID | Description | Required For |
|--------|-------------|--------------|
| 220045 | Heart Rate | Vital signs |
| 220179 | Systolic BP | Vital signs |
| 220180 | Diastolic BP | Vital signs |
| 220210 | Respiratory Rate | Vital signs |
| 220277 | SpO2 | Vital signs |
| 223761 | Temperature | Vital signs |
| 220621 | Glucose | Diabetic monitoring |
| 225664 | Bicarbonate | DKA detection |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No eligible patients found" | Check file paths and data completeness |
| Memory errors | Reduce `CHUNK_SIZE` to 250000 |
| Missing mandatory itemids | Verify chartevents extraction |
| Low mortality rate | Increase `DESIRED_MORTALITY_RATE` |
