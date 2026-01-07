# ICU Mortality Prediction System - Architecture

## Overview

This document describes the technical architecture of the **ICU Mortality Prediction System** implemented in `research.py`. The system combines state-of-the-art deep learning techniques to predict in-hospital mortality for ICU patients using irregular time-series data and disease knowledge graphs.

> [!IMPORTANT]
> **This is the TRAINING ENGINE.** After training, it produces `results/deployment_package.pth` which is consumed by `patent.py` for deployment.

---

## Unified Train-Deploy Pipeline

**Pipeline Overview:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                    🔬 research.py (TRAINING)                        │
│  MIMIC-IV Data → Train Model → Evaluate → XAI → deployment_package │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    results/deployment_package.pth
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    🏥 patent.py (DEPLOYMENT)                        │
│  Load Model → Run Simulations → Apply Safety Layer → Reports       │
└─────────────────────────────────────────────────────────────────────┘
```

**How the Code Works (`research.py`):**

The `research.py` script implements an end-to-end ICU mortality prediction system. It begins by loading MIMIC-IV clinical data (chartevents, admissions, icustays, diagnosis codes) using the `ICUDataProcessor` class, which builds patient timelines with gap-awareness and physiological normalization. The `ICDHierarchicalGraph` constructs a disease knowledge graph from DRG codes, capturing comorbidity relationships. The core model (`ICUMortalityPredictor`) combines a **Liquid Mamba Encoder** for irregular time-series processing with an ODE-based adaptive time constant, a **Graph Attention Network** for extracting disease embeddings, and a **Cross-Attention Fusion** module that merges temporal and graph features. The `UncertaintyMortalityHead` produces mortality probabilities with aleatoric uncertainty, while the `CounterfactualDiffusion` module generates XAI explanations. After training with focal loss and class balancing, the script saves a comprehensive deployment package containing model weights, preprocessing statistics, vocabulary mappings, and ICD graph data for use by `patent.py`.

### Deployment Package Contents

The `results/deployment_package.pth` file contains:

| Component | Description |
|-----------|-------------|
| `model_state_dict` | Trained Liquid Mamba model weights |
| `config_dict` | All hyperparameters for model re-instantiation |
| `vocab_size` | Vocabulary size for clinical item embeddings |
| `n_icd_nodes` | Number of ICD graph nodes |
| `feature_stats` | Normalization statistics (mean, std, min, max) |
| `itemid_to_idx` | Feature vocabulary mapping |
| `icd_adj_matrix` | ICD code adjacency matrix |
| `icd_code_to_idx` | ICD code to index mapping |

> [!NOTE]
> **PyTorch 2.6+ Compatibility**: Checkpoints are loaded with `weights_only=False` because `feature_stats` contains numpy arrays.

---

## Data Source & Patient Selection

### Data Files Loaded

`research.py` loads the following **7 MIMIC-IV files** from the `data_10k/` directory:

| File | Purpose | Key Columns | Memory Strategy |
|------|---------|-------------|-----------------|
| `admissions_10k.csv` | Mortality labels | `hadm_id`, `hospital_expire_flag` | Full load |
| `icustays_10k.csv` | ICU stay periods | `stay_id`, `hadm_id`, `intime`, `outtime` | Full load |
| `drgcodes_10k.csv` | **Diagnosis codes** | `hadm_id`, `drg_code`, `description` | Full load |
| `chartevents_10k.csv` | Vitals & labs | `charttime`, `itemid`, `valuenum` | **Chunked (~2M rows)** |
| `inputevents_10k.csv` | Medications/fluids | `starttime`, `itemid`, `amount` | Limited (500K rows) |
| `outputevents_10k.csv` | Outputs (urine, etc.) | `charttime`, `itemid`, `value` | Full load |
| `procedureevents_10k.csv` | Procedures | `starttime`, `itemid` | Full load |

> [!NOTE]
> **DRG codes** (Diagnosis-Related Groups) from `drgcodes_10k.csv` are used to build the disease knowledge graph. These provide real diagnosis information for each hospital admission.

### Patient Selection Criteria

The system applies **strict criteria** to ensure high-quality timelines:

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| **Data Source** | ICU stays only | Focus on critical care |
| **Minimum Events** | ≥ 5 per patient | Ensures sufficient temporal data |
| **Valid Values** | Non-null `valuenum` | Only measured observations |

```python
# Critical filter: Skip patients with sparse data
if len(patient_events) < 5:
    continue  # Excluded from cohort
```

### Resulting Cohort Statistics

| Metric | Value |
|--------|-------|
| **Total ICU Stays** | ~2,292 |
| **Mortality Rate** | ~11.6% |
| **Train Samples** | 1,604 (11.7% mortality) |
| **Validation Samples** | 344 (11.6% mortality) |
| **Test Samples** | 344 (11.6% mortality) |

### Outcome Definition

Mortality is defined as:
- `hospital_expire_flag = 1` (patient died during hospital stay)

> [!NOTE]
> The **11.6% mortality rate** reflects in-hospital death only, which is lower than the 21.2% deterioration rate in `patent.py` because it excludes non-fatal adverse events.

### Why Fewer Samples Than `patent.py`?

| Factor | `patent.py` (11,550) | `research.py` (2,292) |
|--------|----------------------|----------------------|
| **Unit of Analysis** | All hospital admissions | ICU stays only |
| **Minimum Events** | No minimum | ≥ 5 events required |
| **Data Completeness** | 6-hour aggregated windows | Raw event timelines |
| **Memory Limits** | Full data | Chunked/sampled |

The Liquid Mamba model in `research.py` requires **dense, high-quality temporal sequences**, hence the stricter filtering.

### Diabetic Cohort Filtering

The system specifically filters for **diabetic ICU patients** using prescription medications as a proxy:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                      DIABETIC COHORT FILTERING                             │
├────────────────────────────────────────────────────────────────────────────┤
│  Step 1: Try diagnoses_icd.csv (ICD-10: E10-E14, ICD-9: 250)              │
│      ↓  If not found:                                                      │
│  Step 2: Use prescriptions_10k.csv as proxy                               │
│      ↓  Filter for medications:                                            │
│         • Insulin (all formulations)                                       │
│         • Metformin                                                        │
│         • Glipizide                                                        │
│         • Glyburide                                                        │
│         • Glimepiride                                                      │
│      ↓                                                                     │
│  Diabetic cohort: ~1,516 ICU stays (from ~2,578 total)                    │
└────────────────────────────────────────────────────────────────────────────┘
```

**Mandatory Features for Diabetic Patients:**

| Feature | ItemID | Physiological Range | Purpose |
|---------|--------|---------------------|---------|
| **Glucose** | 220621 | 40-500 mg/dL | Hypoglycemia/hyperglycemia detection |
| **Bicarbonate** | 225664 | 10-40 mEq/L | DKA detection (Bicarb < 18) |

---

## System Architecture

**Data Flow Overview:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 📥 INPUT DATA                                                               │
│   MIMIC-IV (chartevents, admissions, icustays) + ICD/DRG Diagnosis Codes   │
└─────────────────────────────────────────────────────────────────────────────┘
                    │                                    │
                    ▼                                    ▼
┌─────────────────────────────────┐    ┌─────────────────────────────────────┐
│ 🔧 PHASE 1: Data Preprocessing  │    │ 📊 PHASE 2: ICD Knowledge Graph    │
│   ICUDataProcessor              │    │   ICDHierarchicalGraph             │
│   - Patient Timelines           │    │   - Build adjacency matrix         │
│   - Gap-awareness (delta_t)     │    │   - GraphAttentionNetwork (GAT)    │
│   - Missingness masks           │    │   - Disease Embeddings             │
│   - Fixed-length Tensors        │    └─────────────────────────────────────┘
└─────────────────────────────────┘                      │
                    │                                    │
                    ▼                                    │
┌─────────────────────────────────┐                      │
│ ⏱️ PHASE 3: Liquid Mamba        │                      │
│   ODELiquidCell (adaptive τ)    │                      │
│   LiquidMambaEncoder            │                      │
│   → Temporal Embeddings         │                      │
└─────────────────────────────────┘                      │
                    │                                    │
                    └───────────────┬────────────────────┘
                                    ▼
              ┌─────────────────────────────────────────────┐
              │ 🔗 PHASE 4: Cross-Attention Fusion          │
              │   Temporal Emb + Graph Emb → Fused Emb     │
              └─────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
┌─────────────────────────────────┐   ┌─────────────────────────────────────┐
│ 🎯 PHASE 5: Prediction          │   │ 🔍 PHASE 6: Explainability (XAI)   │
│   UncertaintyMortalityHead      │   │   CounterfactualDiffusion          │
│   - Mortality Probability       │   │   - "What-if" scenarios            │
│   - Aleatoric Uncertainty       │   │   - Minimal changes for survival   │
└─────────────────────────────────┘   └─────────────────────────────────────┘
```

**Architecture Summary:** The system processes clinical time-series through six phases. Raw MIMIC-IV events are preprocessed into gap-aware patient timelines with missingness indicators. Simultaneously, diagnosis codes build a knowledge graph processed by a Graph Attention Network. The Liquid Mamba encoder uses ODE-based dynamics to handle irregular sampling. Cross-attention fuses temporal and graph embeddings before the uncertainty-aware mortality head makes predictions. Optionally, the diffusion XAI module generates counterfactual explanations.

---

## Component Details

### Phase 1: Data Preprocessing (`ICUDataProcessor`)

| Component | Description |
|-----------|-------------|
| **Location** | Lines 88-363 |
| **Input** | MIMIC-IV CSV files (`admissions`, `icustays`, `chartevents`, `inputevents`, `outputevents`) |
| **Output** | Fixed-length tensors with patient timelines |

**Key Features:**
- **Physiological Range Clipping**: Enforces clinical validity (e.g., HR: 20-300 bpm)
- **Log Transform**: Applied to skewed labs (Creatinine, Lactate, BUN, Bilirubin)
- **Gap-Awareness**: Computes `delta_t` (time since last observation) in hours
- **Modality Tags**: Distinguishes chartevents (0), inputevents (1), outputevents (2)
- **Missingness Masks**: Binary indicator for observed vs missing values

```python
# Output tensor structure per patient:
{
    'values': torch.Tensor,      # Normalized feature values
    'delta_t': torch.Tensor,     # Time gaps (hours)
    'mask': torch.Tensor,        # Observation mask
    'modality': torch.Tensor,    # Event type (0/1/2)
    'item_idx': torch.Tensor,    # Feature vocabulary index
    'length': int                # Actual sequence length
}
```

---

### Phase 2: ICD Knowledge Graph (`ICDHierarchicalGraph` + `GraphAttentionNetwork`)

| Component | Description |
|-----------|-------------|
| **Graph Construction** | Lines 370-472 |
| **GAT Network** | Lines 475-555 |
| **Input** | ICD diagnosis codes per patient |
| **Output** | Patient comorbidity embedding `(batch, graph_dim)` |

**Graph Structure:**
- **Nodes**: ICD-10 codes (all unique codes in dataset)
- **Edges**: 
  1. **Hierarchical** - Based on ICD prefix similarity
  2. **Co-occurrence** - Diseases appearing together in patients

**Graph Attention Network:**
```
Node Embeddings (n_nodes, embed_dim)
        ↓
Multi-Head Attention (4 heads)
        ↓
Adjacency-Masked Attention Scores
        ↓
Weighted Aggregation
        ↓
Patient Comorbidity Embedding
```

---

### Phase 3: Liquid Mamba Encoder (`ODELiquidCell` + `LiquidMambaEncoder`)

| Component | Description |
|-----------|-------------|
| **ODE Cell** | Lines 565-631 |
| **Encoder** | Lines 634-709 |
| **Input** | Patient timeline tensors |
| **Output** | Temporal embedding `(batch, hidden_dim)` |

**Mathematical Foundation:**

The Liquid Neural Network uses an ODE-based formulation:

$$\frac{dh}{dt} = \frac{1}{\tau(\Delta t)} \cdot (\sigma(W_x \cdot x + W_h \cdot h + b) - h)$$

Where:
- **τ(Δt)** = Adaptive time constant based on observation gap
- **Small Δt** → Large τ → Slow dynamics (frequent vitals)
- **Large Δt** → Small τ → Fast adaptation (sparse labs)

**Discretization (Euler):**
$$h_{t+1} = h_t + \Delta t \cdot \frac{dh}{dt}$$

---

### Phase 4: Cross-Attention Fusion (`CrossAttentionFusion`)

| Component | Description |
|-----------|-------------|
| **Location** | Lines 716-759 |
| **Input** | Temporal embedding + Graph embedding |
| **Output** | Fused embedding `(batch, temporal_dim)` |

**Architecture:**
```
[Temporal Emb] + [Graph Emb]
         ↓ Concatenate
    Fusion MLP (2 layers)
         ↓
    Layer Norm + Residual
         ↓
    Fused Embedding
```

---

### Phase 5: Uncertainty-Aware Mortality Head (`UncertaintyMortalityHead`)

| Component | Description |
|-----------|-------------|
| **Location** | Lines 766-821 |
| **Input** | Fused embedding |
| **Output** | Probability, Uncertainty, Logit |

**Uncertainty Types:**

| Type | Method | Interpretation |
|------|--------|----------------|
| **Aleatoric** | Learned log-variance head | Data/noise uncertainty |
| **Epistemic** | MC Dropout (10 samples) | Model uncertainty |

---

### Phase 6: Counterfactual Diffusion (`CounterfactualDiffusion`)

| Component | Description |
|-----------|-------------|
| **Location** | Lines 828-937 |
| **Input** | Patient embedding + target outcome |
| **Output** | Counterfactual trajectory |

**Purpose:** Generates "what-if" scenarios showing minimal changes needed for survival, providing actionable clinical insights.

---

## Complete Model (`ICUMortalityPredictor`)

| Location | Lines 944-1031 |
|----------|----------------|

**Forward Pass:**
```python
def forward(values, delta_t, mask, modality, item_idx, icd_activation, icd_adj):
    # 1. Temporal encoding
    temporal_emb, hidden_states = temporal_encoder(values, delta_t, mask, modality, item_idx)
    
    # 2. Graph encoding
    graph_emb, node_attention = graph_encoder(icd_activation, icd_adj)
    
    # 3. Fusion
    fused_emb = fusion(temporal_emb, graph_emb)
    
    # 4. Prediction
    prob, uncertainty, logit = mortality_head(fused_emb)
    
    return prob, uncertainty, logit
```

---

## Configuration

```python
@dataclass
class Config:
    # Data
    data_dir: str = "data_10k"
    max_seq_len: int = 128
    
    # Model Architecture
    embed_dim: int = 64          # Base embedding dimension
    hidden_dim: int = 128        # Liquid Mamba hidden dim
    graph_dim: int = 64          # ICD graph embedding dim
    n_mamba_layers: int = 2      # Number of Liquid Mamba layers
    n_attention_heads: int = 4   # Cross-attention heads
    dropout: float = 0.2
    
    # Training
    batch_size: int = 8
    epochs: int = 50
    lr: float = 1e-3
    
    # Diffusion XAI
    diffusion_steps: int = 50
    
    # Uncertainty
    mc_dropout_samples: int = 10
```

---

## Output Artifacts

### Deployment Package (Primary Output)
| File | Description |
|------|-------------|
| `results/deployment_package.pth` | **🔑 Main output: Model + preprocessing for patent.py** |

> [!TIP]
> The `deployment_package.pth` is the key artifact that bridges training and deployment. It contains everything needed for `patent.py` to run Digital Twin simulations without any retraining.

### Model & Metrics
| File | Description |
|------|-------------|
| `checkpoints/best_model.pt` | Best model weights + metadata (same as deployment_package) |
| `results/metrics.json` | Test metrics (AUROC, AUPRC, Brier, etc.) |

### Training Visualizations
| File | Description |
|------|-------------|
| `results/training_curves.png` | Loss/AUROC/AUPRC over epochs |
| `results/calibration.png` | Reliability diagram |
| `results/uncertainty_analysis.png` | Uncertainty vs prediction |
| `results/dca.png` | Decision Curve Analysis |

### Explainable AI (XAI) Outputs
| File | Description |
|------|-------------|
| `results/counterfactual_explanations.json` | Per-patient counterfactual results |
| `results/feature_importance.json` | Feature importance via gradient saliency |
| `results/feature_importance.png` | Clinical feature importance bar chart |
| `results/counterfactual_analysis.png` | Proximity vs sparsity trade-off plots |
| `results/xai_dashboard.png` | **Comprehensive 4-panel XAI dashboard** |

### Diabetic-Specific XAI Explanations

The counterfactual generation includes **Glucose trajectory modification explanations** for diabetic patients:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                COUNTERFACTUAL XAI FOR DIABETIC PATIENTS                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  For 5 high-risk diabetic patients, generate explanations like:             │
│                                                                             │
│  "Patient 1: To survive, Patient 1 (Risk: 86.5%) would need:               │
│    • Maintain glucose within target range (70-180 mg/dL)                   │
│    • Reduce simulated vitals by magnitude 0.674                            │
│    • Required feature changes: 121 modifications"                          │
│                                                                             │
│  Metrics tracked:                                                           │
│    - Validity: % counterfactuals that flip to survivor                     │
│    - Proximity: Distance from original (should be minimal)                 │
│    - Sparsity: Number of features changed (should be low)                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Usage

### Training (Producer)
```bash
# Run training pipeline
python research.py

# Expected output:
# - Train model on MIMIC-IV data
# - Generate XAI explanations
# - Save deployment_package.pth

# Runtime: ~10-30 minutes (GPU recommended)
```

### Deployment (Consumer)
```bash
# After research.py completes, run:
python patent.py

# patent.py will:
# - Load deployment_package.pth
# - Run Digital Twin simulations
# - Apply Safety Layer overrides
# - Generate deployment results
```

---

*Documentation updated for Train-Deploy Pipeline v2.0*

