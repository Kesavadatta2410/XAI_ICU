# ICU Mortality Prediction - Logic Flow

## Pipeline Overview

This document explains the logical flow from raw MIMIC-IV data to mortality predictions with explainable counterfactuals.

---

## Step 1: Data Processing

**Input:** Raw MIMIC-IV tables (chartevents, labevents, diagnoses_icd)

**Process:**
1. Filter diabetic ICU patients (ICD codes E10-E14)
2. Build patient timelines with timestamps
3. Apply physiological clipping (e.g., HR: 20-300 bpm)
4. Log-transform skewed labs (creatinine, bilirubin)
5. Z-score normalization
6. Compute time gaps (Δt) between observations

**Output:** Normalized tensors with values, masks, timestamps, modalities

---

## Step 2: Temporal Encoding (ODE-Liquid)

**Why ODE?** ICU data is irregularly sampled—labs every 6-12 hours, vitals every few minutes. Traditional RNNs assume fixed intervals.

**Logic:**
1. For each timestep t with time gap Δt:
   - Compute adaptive time constant: τ = τ_min + softplus(W × Δt)
   - Apply ODE dynamics: dh/dt = (f(x,h) - h) / τ
   - Euler step: h_new = h + Δt × dh/dt
2. If observation missing, rely purely on evolved state
3. If observed, blend with observation-driven update

**Output:** Temporal embedding (64-dim vector per patient)

---

## Step 3: Disease Context (Knowledge Graph)

**Why Graph?** Comorbidities matter—diabetes + CKD has different mortality than diabetes alone.

**Logic:**
1. Build adjacency matrix:
   - Hierarchical edges (ICD prefix similarity)
   - Co-occurrence edges (diseases appearing together)
2. Patient activates their diagnosed nodes
3. Multi-head GAT propagates information
4. Pool activated nodes → Graph embedding

**Output:** Disease embedding (32-dim vector per patient)

---

## Step 4: Fusion & Prediction

**Logic:**
1. Concatenate temporal + graph embeddings
2. Project through MLP fusion layers
3. Prediction head outputs:
   - μ (logit for mortality probability)
   - σ (aleatoric uncertainty)
4. Final probability: sigmoid(μ)

**Output:** P(mortality), uncertainty σ

---

## Step 5: Training

**Loss:** BCEWithLogitsLoss (numerically stable)

**Optimization:**
- AdamW with weight decay
- Cosine annealing LR schedule
- Gradient clipping (max norm = 1.0)
- Early stopping (patience = 10)

**Metrics Tracked:**
- Accuracy, F1, AUROC, AUPRC
- Brier score (calibration)
- Mean uncertainty

---

## Step 6: XAI - Counterfactual Generation

**Goal:** "What minimal changes would make this high-risk patient survive?"

**Logic:**
1. Take patient's fused embedding
2. Condition diffusion model on "survival" label
3. Denoise from random noise → counterfactual embedding
4. Compare original vs counterfactual to identify key differences

**Constraints:**
- Immutable variables (age, gender) cannot be modified
- Changes must be physiologically feasible

---

## Step 7: Evaluation & Visualization

**Generated Outputs:**

| Plot | Purpose |
|------|---------|
| Training Curves | Monitor convergence |
| Calibration | Is 70% prediction = 70% mortality? |
| Uncertainty Analysis | High uncertainty = less confident |
| Decision Curve | Net benefit vs treat-all/none |



---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| ODE over RNN | Handles irregular sampling naturally |
| Co-occurrence graph | Captures cohort-specific disease patterns |
| Aleatoric uncertainty | Distinguishes confident vs uncertain predictions |
| BCEWithLogitsLoss | Prevents numerical instability |
| Diffusion XAI | Generates realistic counterfactuals |

---

## Running the Pipeline

```bash
# Ensure data is in data/ folder
python research.py
```

Results saved to `results/` folder.
