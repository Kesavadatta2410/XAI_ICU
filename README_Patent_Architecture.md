# Patent.py Architecture Documentation - DEPLOYMENT MODE

## Overview

This document describes the technical architecture of `patent.py`, which implements a **Clinical AI System Deployment Engine** for patient risk assessment with uncertainty quantification and safety guardrails.

> [!IMPORTANT]
> **This is a DEPLOYMENT-ONLY script.** It does NOT train any models.
> It loads pre-trained models from `research.py` via the `deployment_package.pth` artifact.

The system implements all six phases from the clinical AI patent:
- **Phase 1-2**: Digital Twin Sandbox with Uncertainty + Safety Layer
- **Phase 3**: Human-in-the-Loop Feedback Collection
- **Phase 4**: Dynamic Knowledge Base Management
- **Phase 5**: Multi-Site Validation
- **Phase 6**: Real-Time Alert Deployment

---

## Unified Train-Deploy Pipeline

**Pipeline Overview:**

```
┌─────────────────────────────────────────────────────────────────────┐
│               🔬 research.py (TRAINING - Producer)                  │
│   Train Model → Save deployment_package.pth                        │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    results/deployment_package.pth
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│               🏥 patent.py (DEPLOYMENT - Consumer)                  │
│   load_digital_twin → run_simulation → apply_safety_layer          │
└─────────────────────────────────────────────────────────────────────┘
```

**How the Code Works (`patent.py`):**

The `patent.py` script is a **deployment-only** engine that loads a pre-trained model from `research.py`. It begins by calling `load_digital_twin()` which reads the `deployment_package.pth` file, reconstructs the `ICUMortalityPredictor` model using saved hyperparameters, and loads the trained weights along with preprocessing artifacts (feature stats, vocabulary, ICD graph). The `run_simulation()` function performs **Monte Carlo Dropout** by running the model 50 times with dropout enabled during inference, producing a mean risk prediction and variance for uncertainty quantification. Finally, `apply_safety_layer()` checks the prediction against 6 clinical safety rules (hyperkalemia, hypoxia, shock, lactate, bradycardia, unstable tachycardia) and can override low-risk predictions when critical vital signs are detected. All results are saved to the `pat_res/` directory with visualizations and JSON audit logs.

### Deployment Package Contents

The `results/deployment_package.pth` file from `research.py` contains:

| Component | Description |
|-----------|-------------|
| `model_state_dict` | Trained Liquid Mamba model weights |
| `config_dict` | Model hyperparameters for re-instantiation |
| `vocab_size` | Vocabulary size for item embeddings |
| `n_icd_nodes` | Number of ICD graph nodes |
| `feature_stats` | Normalization statistics (mean, std, min, max) |
| `itemid_to_idx` | Feature vocabulary mapping |
| `icd_adj_matrix` | ICD code adjacency matrix |
| `icd_code_to_idx` | ICD code to index mapping |

> [!NOTE]
> **PyTorch 2.6+ Compatibility**: Checkpoints are loaded with `weights_only=False` because `feature_stats` contains numpy arrays.

---

## System Architecture

**Execution Flow:**

```
Step 1: LOAD DEPLOYMENT PACKAGE
┌─────────────────────────────────────────────────────────────────────┐
│  deployment_package.pth                                             │
│         │                                                           │
│         └──→ load_digital_twin()                                    │
│                    ├──→ ICUMortalityPredictor (model)              │
│                    ├──→ feature_stats + itemid_to_idx              │
│                    └──→ icd_adj_matrix + icd_code_to_idx           │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
Step 2: DIGITAL TWIN SIMULATION
┌─────────────────────────────────────────────────────────────────────┐
│  Patient Data + Model + ICD Graph                                   │
│         │                                                           │
│         └──→ run_simulation(n_runs=50)                             │
│                    ├──→ MC Dropout (50 forward passes)             │
│                    └──→ Mean Risk + Variance + 95% CI              │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
Step 3: SAFETY LAYER
┌─────────────────────────────────────────────────────────────────────┐
│  Predicted Risk + Patient Vitals                                    │
│         │                                                           │
│         └──→ apply_safety_layer()                                  │
│                    ├──→ Check 6 clinical rules                     │
│                    └──→ Override if critical values detected       │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
Step 4: RESULTS
┌─────────────────────────────────────────────────────────────────────┐
│  Final Risk + Override Status                                       │
│         ├──→ Visualizations (PNG)                                  │
│         └──→ JSON Reports (deployment_results, safety_audit_log)   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Configuration (`Config` class)

**Location**: Lines 45-78

| Category | Parameters |
|----------|------------|
| **Deployment** | `deployment_package_path="results/deployment_package.pth"` |
| **Simulation** | `mc_samples=50`, `uncertainty_threshold=0.4`, `confidence_level=0.9` |
| **Model** (loaded from package) | `embed_dim`, `hidden_dim`, `graph_dim`, etc. |

---

### 2. Model Classes (Duplicated from research.py)

**Location**: Lines 85-395

For standalone deployment, the following model classes are included:

| Class | Description |
|-------|-------------|
| `ODELiquidCell` | ODE-based liquid neural cell with adaptive time constants |
| `LiquidMambaEncoder` | Full temporal encoder for irregular time-series |
| `GraphAttentionNetwork` | Multi-head GAT for ICD embeddings |
| `CrossAttentionFusion` | Temporal + Graph fusion module |
| `UncertaintyMortalityHead` | Aleatoric uncertainty prediction |
| `CounterfactualDiffusion` | Counterfactual XAI module |
| `ICUMortalityPredictor` | Complete model combining all components |

---

### 3. `load_digital_twin()` Function

**Location**: Lines 400-460

```python
def load_digital_twin(checkpoint_path: str, device: str = "cpu") -> Tuple:
    """
    Load trained model and preprocessing artifacts from deployment package.
    
    Returns:
        model: Loaded ICUMortalityPredictor
        scaler: StandardScaler for input preprocessing
        config_dict: Model configuration
        icd_adj: ICD adjacency matrix
        itemid_to_idx: Vocabulary mapping
    """
```

**Features**:
- Graceful error handling if deployment package not found
- Automatic config restoration from checkpoint
- Device-aware loading (CPU/CUDA)

---

### 4. `run_simulation()` Function

**Location**: Lines 465-520

Implements Digital Twin simulation with Monte Carlo Dropout:

```python
def run_simulation(model, patient_data, icd_adj, n_runs=50):
    """
    Run n_runs MC Dropout simulations for uncertainty estimation.
    
    Returns:
        mean_risk: Average predicted mortality probability
        variance: Prediction variance
        std: Standard deviation
        lower_bound: 95% CI lower bound
        upper_bound: 95% CI upper bound
    """
    model.train()  # Enable dropout for MC sampling
    
    for _ in range(n_runs):
        # Forward pass with dropout active
        predictions.append(model(patient_data))
    
    model.eval()  # Reset to evaluation mode
    return {
        'mean_risk': predictions.mean(),
        'variance': predictions.var(),
        ...
    }
```

---

### 5. `apply_safety_layer()` Function

**Location**: Lines 525-620

Rule-based safety checks that can override model predictions:

| Rule ID | Trigger Condition | Override Action |
|---------|-------------------|-----------------|
| `HYPERKALEMIA_OVERRIDE` | K+ > 6.0 mEq/L | Risk → max(pred, 0.7) |
| `HYPOXIA_OVERRIDE` | SpO2 < 85% | Risk → max(pred, 0.75) |
| `SHOCK_OVERRIDE` | SBP < 70 mmHg | Risk → max(pred, 0.8) |
| `LACTATE_OVERRIDE` | Lactate > 4.0 mmol/L | Risk → max(pred, 0.65) |
| `BRADYCARDIA_OVERRIDE` | HR < 40 bpm | Risk → max(pred, 0.6) |
| `UNSTABLE_TACHY_OVERRIDE` | HR > 150 + SBP < 90 | Risk → max(pred, 0.7) |

```python
def apply_safety_layer(predicted_risk, patient_state):
    """
    Override low-risk predictions when critical values present.
    
    Example:
        if predicted_risk < 0.5 and potassium > 6.0:
            final_risk = "High (Safety Trigger: Hyperkalemia)"
    """
```

---

### 6. Medical Knowledge Base

**Location**: Lines 700-800

Pre-loaded clinical safety rules with guideline sources:

| Rule ID | Name | Source |
|---------|------|--------|
| `STROKE_BP` | BP Management in Stroke | AHA Stroke Guidelines 2019 |
| `RENAL_NSAID` | NSAID in Renal Impairment | KDIGO CKD Guidelines |
| `HYPERKALEMIA_K` | K+ Supplementation in Hyperkalemia | KDIGO AKI Guidelines |
| `BRADY_BETABLOCK` | Beta-Blockers in Bradycardia | AHA Arrhythmia Guidelines |

---

## Execution Flow

### CLI Modes

```bash
# Mode 1: Full deployment run (2000 patients, 200 MC samples)
python patent.py
# → Runs all phases 1-6

# Mode 2: Quick demo (50 patients, 10 MC samples)
python patent.py --quick-demo
# → Fast validation mode

# Mode 3: Resume from cached data (Phase 6 only)
python patent.py --resume
# → Loads pat_res/deployment_results.json and safety_audit_log.json
# → Skips phases 1-5, runs Phase 6 alert system only
```

### Full Pipeline Execution

```bash
# Step 1: Train model with research.py
python research.py
# → Produces: results/deployment_package.pth

# Step 2: Deploy with patent.py
python patent.py
# → Loads deployment_package.pth
# → Runs Digital Twin simulations (1000+ patients)
# → Applies Safety Layer
# → Runs Phases 3-6 demos
# → Generates all results to pat_res/
```

---

## Diabetic Digital Twin (DiabeticDigitalTwin class)

**Location**: Lines 450-670

The `DiabeticDigitalTwin` class is a standalone deployment wrapper for diabetic ICU patients with specialized safety rules.

### Data Flow: Input → Output

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        INPUT: MIMIC-IV DATA                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  1. admissions_10k.csv    → Hospital admission records                      │
│  2. icustays_10k.csv      → ICU stay information                           │
│  3. chartevents_10k.csv   → Vital signs time-series (HR, BP, SpO2, etc.)   │
│  4. prescriptions_10k.csv → Diabetic medication filtering (Insulin, etc.)  │
│  5. drgcodes_10k.csv      → DRG codes for ICD graph                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING (research.py)                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  1. Diabetic Cohort Filtering                                               │
│     → Filter patients on Insulin/Metformin/Glipizide                       │
│  2. Time-Series Tensor Creation                                             │
│     → Values, timestamps, masks, modality indicators                        │
│  3. Feature Engineering                                                     │
│     → Glucose (mandatory), Bicarbonate, vital signs, labs                  │
│  4. ICD Knowledge Graph                                                     │
│     → Adjacency matrix, patient activation vectors                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MODEL (Liquid Mamba + GAT + Cross-Attention)            │
├─────────────────────────────────────────────────────────────────────────────┤
│  Inputs:                                                                    │
│    - values: (batch, seq_len) normalized vital/lab values                  │
│    - delta_t: (batch, seq_len) time gaps in hours                          │
│    - mask: (batch, seq_len) observation mask                               │
│    - modality: (batch, seq_len) vitals/labs/medications indicator          │
│    - item_idx: (batch, seq_len) vocabulary indices                         │
│    - icd_activation: (batch, n_icd_nodes) patient diagnosis mask           │
│                                                                             │
│  Processing:                                                                │
│    LiquidMamba(values, delta_t, mask) → temporal embedding                 │
│    GAT(icd_activation, icd_adj) → disease embedding                        │
│    CrossAttention(temporal, disease) → fused embedding                     │
│    UncertaintyHead(fused) → mortality_prob + uncertainty                   │
│                                                                             │
│  Outputs:                                                                   │
│    - prob: (batch,) mortality probability                                  │
│    - uncertainty: (batch,) aleatoric uncertainty                           │
│    - logit: (batch,) raw logit for loss computation                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DIGITAL TWIN SIMULATION (patent.py)                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  MC Dropout (50 runs with dropout=True):                                    │
│    → mean_risk: Average predicted mortality                                 │
│    → std: Epistemic uncertainty                                            │
│    → lower_bound: 2.5% percentile (95% CI)                                 │
│    → upper_bound: 97.5% percentile (95% CI)                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DIABETIC SAFETY LAYER                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  Rule 1: HYPOGLYCEMIA_OVERRIDE                                              │
│    Trigger: Glucose < 70 AND model_risk < 0.2                              │
│    Action: Override to High Risk (0.7)                                      │
│    Guideline: ADA Diabetes Care Standards                                   │
│                                                                             │
│  Rule 2: DKA_DETECTION                                                      │
│    Trigger: Glucose > 250 AND Bicarbonate < 18                             │
│    Action: Override to High Risk (0.8), flag "Diabetic Ketoacidosis Risk"  │
│    Guideline: ADA DKA Management Protocol                                   │
│                                                                             │
│  Rule 3: SEVERE_HYPERGLYCEMIA                                               │
│    Trigger: Glucose > 400                                                   │
│    Action: Override to Medium-High Risk (0.6), evaluate for HHS            │
│    Guideline: ADA Hyperglycemic Crisis Guidelines                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             OUTPUT FILES                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  VISUALIZATIONS (pat_res/):                                                 │
│    • digital_twin_simulation.png    - Risk & uncertainty distribution      │
│    • safety_layer_analysis.png      - Override statistics                  │
│    • diabetic_xai_analysis.png      - Glucose thresholds & safety flags   │
│    • uncertainty_quantification.png - MC dropout analysis                  │
│    • xai_dashboard.png              - Comprehensive XAI summary            │
│                                                                             │
│  JSON REPORTS (pat_res/):                                                   │
│    • deployment_results.json        - Complete simulation summary          │
│    • safety_audit_log.json          - Diabetic safety records per patient  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### DiabeticDigitalTwin Methods

| Method | Description |
|--------|-------------|
| `__init__(deployment_path)` | Load model, config, scaler, feature_names from deployment package |
| `simulate_patient(patient_data, n_simulations=50)` | Run 50 MC dropout simulations, return mean risk + 95% CI |
| `check_safety(risk_score, patient_vitals)` | Apply hypoglycemia, DKA, hyperglycemia safety rules |
| `generate_report(patient_id, simulation, safety, vitals)` | Generate clinical report string |

---

## Output Files

| File | Description |
|------|-------------|
| `pat_res/digital_twin_simulation.png` | Risk distribution and uncertainty analysis |
| `pat_res/safety_layer_analysis.png` | Override statistics and rule triggers |
| `pat_res/diabetic_xai_analysis.png` | Glucose distribution, risk vs glucose, safety overrides |
| `pat_res/uncertainty_quantification.png` | MC dropout uncertainty, risk vs uncertainty, 95% CIs |
| `pat_res/xai_dashboard.png` | Comprehensive 3x3 XAI dashboard |
| `pat_res/deployment_results.json` | Complete simulation summary |
| `pat_res/safety_audit_log.json` | 1000 patient safety records with flags |
| `pat_res/results_summary.json` | Model performance metrics (AUROC 0.964) |
| `pat_res/phases_3_6_demo_results.json` | Phase 3-6 demo outputs |
| `pat_res/feedback_log.json` | Phase 3: Clinician feedback records |
| `pat_res/medical_rules.json` | Phase 4: Dynamic knowledge base |
| `pat_res/multisite_report.json` | Phase 5: Multi-site comparison |
| `pat_res/multisite_comparison.png` | Phase 5: Site comparison charts |
| `pat_res/roc_curve.png` | ROC curve visualization |

---

## Key Differences from Training Mode

| Aspect | Old `patent.py` (Training) | New `patent.py` (Deployment) |
|--------|---------------------------|------------------------------|
| **Model** | DigitalTwinModel (LSTM) | ICUMortalityPredictor (Liquid Mamba) |
| **Training** | 30 epochs training | No training |
| **Model Source** | Trained in-script | Loaded from `deployment_package.pth` |
| **Uncertainty** | MC Dropout (in-house) | MC Dropout (from research.py model) |
| **Checkpoint** | Saves `best_model.pt` | Loads `deployment_package.pth` |

---

## Error Handling

If `deployment_package.pth` is not found:

```
============================================================
ERROR: Deployment package not found!
============================================================
Path: results/deployment_package.pth

This script requires a pre-trained model from research.py.
Please run research.py first to train and generate the model:

    python research.py

This will create: results/deployment_package.pth
============================================================
```

---

## Dependencies

```
patent.py (Deployment Mode)
├── numpy
├── pandas
├── torch
│   ├── nn
│   └── nn.functional
├── sklearn
│   ├── preprocessing (StandardScaler)
│   └── metrics
├── matplotlib
└── seaborn
```

---

*Documentation updated for Clinical AI System v2.0 - Deployment Mode*
