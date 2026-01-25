# MIMIC-IV Exploratory Data Analysis (EDA) Documentation

## Overview

This document describes the exploratory data analysis performed on the MIMIC-IV dataset for the XAI ICU Mortality Prediction System. The analysis is implemented in `eda.py` and produces comprehensive visualizations and statistics.

---

## Dataset Summary

| Metric | Value |
|--------|-------|
| **Total Patients** | 42,951 |
| **Total Admissions** | 166,062 |
| **In-Hospital Mortality Rate** | 4.3% |
| **Median Length of Stay** | 4.2 days (101.8 hours) |

---

## Data Files Analyzed

The EDA analyzes the following MIMIC-IV files from `data100k/`:

| File | Description | Size |
|------|-------------|------|
| `admissions_100k.csv` | Hospital admissions with mortality labels | 29 MB |
| `icustays_100k.csv` | ICU stay records | 10 MB |
| `chartevents_100k.csv` | Vital signs and lab values | ~32 GB |
| `prescriptions_100k.csv` | Medication orders | 1.8 GB |
| `patients_100k.csv` | Patient demographics | 1.5 MB |

---

## Analysis Components

### 1. Cohort Analysis

The cohort analysis examines patient demographics and outcomes:

- **Age distribution** of ICU patients
- **Gender distribution** and mortality differences
- **Hospital mortality** breakdown by ICU type
- **Length of stay** distributions

**Key Finding:** The 4.3% mortality rate reflects in-hospital deaths only, which is appropriate for the prediction task.

### 2. Vital Signs Analysis

Analyzes the distribution and patterns of key physiological measurements:

| ItemID | Measurement | Physiological Range |
|--------|-------------|---------------------|
| 220045 | Heart Rate | 20-300 bpm |
| 220179 | Systolic BP | 40-300 mmHg |
| 220180 | Diastolic BP | 20-200 mmHg |
| 220210 | Respiratory Rate | 4-60 /min |
| 220277 | SpO2 | 50-100% |
| 223761 | Temperature | 32-42°C |
| 220621 | Glucose | 40-500 mg/dL |
| 225664 | Bicarbonate | 10-40 mEq/L |

**Statistics:**
- Total chart records: 795,956
- Records per patient: ~3,618
- Missing rate: 0%

### 3. Time Gap Analysis (Critical for Liquid Mamba)

The irregular time intervals between clinical measurements are essential for the Liquid Neural Network architecture:

| Statistic | Value |
|-----------|-------|
| Mean Δt | 2.9 minutes |
| Median Δt | 0.0 minutes |
| Std Δt | 16.5 minutes |

**Clinical Insight:** High variance in time gaps confirms the need for adaptive time constant modeling (τ adaptation in Liquid cells). Frequent vitals (every few minutes) vs infrequent labs (every several hours) require different temporal dynamics.

### 4. Medication Analysis

| Metric | Value |
|--------|-------|
| Total Prescriptions | 10,538,716 |
| Patients with Rx | 42,923 |
| Unique Drugs | 6,817 |
| Vasopressor Patients | 24,613 |

**Key Finding:** 24,613 patients received vasopressors, which is useful for outcome definition and severity stratification.

### 5. Missingness Analysis

Examines patterns of missing data across clinical measurements:

- Identifies which features have highest missingness
- Informs imputation strategies
- Highlights data quality issues

---

## Output Artifacts

### Visualizations (`eda_results/`)

| File | Description |
|------|-------------|
| `cohort_analysis.png` | Patient demographics and mortality |
| `vitals_analysis.png` | Vital signs distributions |
| `time_gap_analysis.png` | Temporal gap patterns |
| `medications_analysis.png` | Prescription patterns |
| `missingness_analysis.png` | Missing data patterns |
| `summary_dashboard.png` | Comprehensive overview |

### Data Files (`eda_results/`)

| File | Description |
|------|-------------|
| `eda_stats.json` | Summary statistics in JSON format |
| `eda_report.md` | Markdown report of findings |

---

## Implications for Model Design

### For Phase 1 (Digital Twin - research.py)

1. **Irregular Sampling**: High variance in time gaps confirms need for Liquid Mamba's adaptive time constant
2. **Rich Feature Space**: Dense vital signs support multi-trajectory uncertainty quantification
3. **Patient Heterogeneity**: Wide variation in measurements per patient requires robust sequence padding

### For Phase 2 (Safety Layer - patent.py)

1. **Medication Data**: 6,817 unique drugs available for contraindication rule mining
2. **Vasopressor Usage**: High vasopressor rate enables outcome stratification
3. **Missing Data**: Missingness patterns inform feature engineering

---

## Usage

```bash
# Run EDA analysis
python eda.py

# Expected outputs:
# - Visualization plots in eda_results/
# - Statistics in eda_results/eda_stats.json
# - Report in eda_results/eda_report.md
```

---

## Dependencies

```
eda.py
├── pandas
├── numpy
├── matplotlib
├── seaborn
└── json
```

---

*Documentation for EDA Analysis Pipeline - Clinical AI System v2.0*
