<<<<<<< HEAD
# ICU Mortality Prediction in Diabetic Patients

A comprehensive deep learning pipeline for predicting ICU mortality in diabetic patients using the MIMIC-IV dataset, featuring:

- **Liquid-Mamba** temporal encoder with dynamic time-decay
- **ICD Knowledge Graph** with Graph Attention Networks (GAT)
- **Diffusion-based counterfactual XAI** explanations
- **Uncertainty-aware predictions**

## 📊 Dataset Overview

This project uses preprocessed MIMIC-IV data from 500 diabetic ICU patients:

| Metric | Value |
|--------|-------|
| Total Patients | 500 |
| Mortality Rate | 11.0% |
| Mean Age | 65 ± 14 years |
| Gender (M/F) | 62% / 38% |
| Unique ICD Codes | 34 |
| Total Vital Records | 3,096,113 |
| Total Lab Records | 170,432 |

### Data Files (in `data/` folder)

| File | Records | Description |
|------|---------|-------------|
| `cohort_500.csv` | 500 | Patient demographics, ICD codes, mortality labels |
| `vitals_500.csv` | 3.1M | Vital signs (HR, BP, SpO2, etc.) |
| `labs_500.csv` | 170K | Laboratory results |
| `pharmacy_500.csv` | 31K | Pharmacy orders |
| `prescriptions_500.csv` | 39K | Prescription data |
| `emar_500.csv` | 95K | Electronic medication administration |
| `inputevents_500.csv` | 75K | IV input events |
| `outputevents_500.csv` | 35K | Output measurements |
| `procedureevents_500.csv` | 4.7K | ICU procedures |
| `microbiology_500.csv` | 3.2K | Microbiology cultures |
| `ingredientevents_500.csv` | 98K | Medication ingredients |
| `drg_500.csv` | 989 | Diagnosis-related groups |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Patient Timeline                         │
│  (Vitals, Labs, Meds, Procedures) + Δt + Missingness Mask  │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│               Liquid-Mamba Temporal Encoder                 │
│         (Dynamic decay driven by time gaps Δt)             │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      │ Temporal Embedding (256-dim)
                      │
┌─────────────────────┴───────────────────────────────────────┐
│                    Fusion Layer                             │
│       Temporal (256) + Graph Comorbidity (64) → 320        │
└─────────────────────┬───────────────────────────────────────┘
                      │                        ▲
                      │                        │ Graph Embedding
                      │            ┌───────────┴───────────────┐
                      │            │    ICD Knowledge Graph    │
                      │            │  (Hierarchical GAT with   │
                      │            │   time-activated nodes)   │
                      │            └───────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   Prediction Head                           │
│        Mortality Probability + Uncertainty Estimate         │
└─────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              Diffusion-based XAI Module                     │
│      Counterfactual survival trajectory generation          │
│         (Conditioned on patient latent state)              │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
IIIT Ranchi/
├── data/                    # Preprocessed MIMIC-IV CSVs
│   ├── cohort_500.csv
│   ├── vitals_500.csv
│   └── ...
├── eda.py                   # Exploratory data analysis
├── eda_results/             # EDA visualizations
│   ├── summary_dashboard.png
│   ├── cohort_analysis.png
│   ├── vitals_analysis.png
│   ├── labs_analysis.png
│   ├── time_gap_analysis.png
│   └── eda_report.md
├── research.py              # Main pipeline implementation
├── README.md                # This file
├── README_DATA.md           # Detailed data documentation
└── requirements.txt         # Python dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run EDA Analysis

```bash
python eda.py
```

This generates visualizations in `eda_results/` and a summary report.

### 3. Train Models

```bash
python research.py --train
```

## 📈 Key Findings from EDA

1. **Class Imbalance**: 11% mortality rate requires AUPRC as primary metric
2. **Irregular Time Series**: Median time gap (Δt) = 0 minutes with high variance → confirms need for Liquid-Mamba
3. **High Missingness**: 61% missing rate in vitals values → mask-based modeling critical
4. **Rich Event Data**: 3M+ vitals, 170K labs across 10+ event types
5. **ICD Hierarchy**: 34 unique ICD-10 diabetes codes (E10xx, E11xx series)

## 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **AUPRC** | Primary metric (handles class imbalance) |
| AUROC | Discrimination ability |
| Calibration | Probability reliability |
| XAI Realism | Distance between generated and real survivors |

## 🔬 Model Comparison

| Model | Description |
|-------|-------------|
| LSTM Baseline | 2-layer LSTM, fixed timesteps |
| Standard Mamba | SSM without Δt modulation |
| **Liquid-Mamba + Graph + Diffusion** | Full system with all components |

## 📚 References

- MIMIC-IV Dataset: https://physionet.org/content/mimiciv/
- Mamba: Selective State Spaces for Sequence Modeling
- Graph Attention Networks (GAT)
- Diffusion Models for XAI

## 📝 License

This project is for research purposes. MIMIC-IV data requires PhysioNet credentialing.

---

*Built for ICU mortality prediction research at IIIT Ranchi*
=======
# XAI_ICU
>>>>>>> d9b2c4a37385b605d2d2019c5f8ba7996eeca741
