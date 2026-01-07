# Patent.py Logic Documentation - DEPLOYMENT MODE

## Overview

This document explains the **clinical and algorithmic logic** behind `patent.py`, which implements a Clinical AI System **Deployment Engine** for patient risk assessment with safety guardrails.

> [!IMPORTANT]
> **DEPLOYMENT MODE**: This script does NOT train any models.
> It loads a pre-trained model from `research.py` via `results/deployment_package.pth`.

---

## Unified Train-Deploy Pipeline

**Pipeline Overview:**

```
┌─────────────────────────────────────────────────────────────────────┐
│               🔬 research.py (PRODUCER)                             │
│   Train Liquid Mamba → Generate XAI → Save deployment_package.pth │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    deployment_package.pth (artifact)
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│               🏥 patent.py (CONSUMER) - THIS SCRIPT                │
│   load_digital_twin → run_simulation → apply_safety_layer → Report│
└─────────────────────────────────────────────────────────────────────┘
```

**How the Code Works (`patent.py`):**

This script implements a **Digital Twin deployment** for clinical risk assessment. It does NOT train any models. The `load_digital_twin()` function loads a pre-trained Liquid Mamba model along with all preprocessing artifacts from the deployment package. The `run_simulation()` function uses **Monte Carlo Dropout** to estimate prediction uncertainty by running 50 stochastic forward passes and computing mean risk ± variance. The `apply_safety_layer()` applies 6 evidence-based clinical rules that can override model predictions when critical vital signs are detected (e.g., K+ > 6.0 mEq/L triggers hyperkalemia override). All decisions are logged for audit compliance.

### What Changed from Training Mode?

| Aspect | Old `patent.py` (Training) | New `patent.py` (Deployment) |
|--------|---------------------------|------------------------------|
| **Purpose** | Train + Evaluate | Load + Simulate |
| **Model** | DigitalTwinModel (LSTM) | ICUMortalityPredictor (Liquid Mamba) |
| **Training** | 30 epochs | None |
| **Model Source** | Trained in-script | Loaded from `deployment_package.pth` |
| **Output** | `best_model.pt` | `deployment_results.json` |

> [!NOTE]
> **PyTorch 2.6+ Compatibility**: Checkpoints are loaded with `weights_only=False` because `feature_stats` contains numpy arrays.

---
We select a cohort of adult patients who are sick enough to benefit from prediction but have enough data for reliable modeling.

```
Patient Selection Criteria:
├── Age ≥ 18 years (adults only)
├── Hospital stay > 24 hours (excludes trivial cases)
└── Complete admission record
```

**Time Windowing**:
Clinical observations are aggregated into 6-hour windows to:
- Smooth out noise from individual measurements
- Capture clinically meaningful trends
- Match typical nursing assessment intervals

```
Admission ──┬──────┬──────┬──────┬──────┬──► Discharge
            │  6h  │  6h  │  6h  │  6h  │
            └──────┴──────┴──────┴──────┘
            Window Window Window Window
               1      2      3      4
```

**Outcome Definition** (Clinical Deterioration):
A patient is labeled as "deteriorated" if ANY of these occur within 48 hours:
- Transfer to ICU
- Initiation of vasopressors (indicating shock)
- Start of mechanical ventilation
- Death

This composite outcome captures major clinical deterioration events.

---

### Step 1.2: Digital Twin Model

**Why LSTM?**
- Clinical data is **sequential** (measurements over time)
- Past values influence future predictions
- LSTMs handle **variable-length** sequences naturally

**Why Monte Carlo Dropout?**
Instead of just predicting "this patient has 60% risk", we want to know:
- "This patient has 60% ± 15% risk" (confident prediction)
- "This patient has 60% ± 35% risk" (uncertain - need more information)

```
Standard Prediction:          MC Dropout Prediction:
     ┌─────┐                       ┌─────┐────────┐
     │ 60% │                       │ 60% │ ±15%   │
     └─────┘                       └─────┴────────┘
                                   Mean   Uncertainty
```

**Algorithm**:
```
1. During training: Dropout randomly zeros neurons (regularization)
2. During inference: Keep dropout ACTIVE (unusual!)
3. Run N=1000 forward passes with different dropout masks
4. Each pass gives slightly different prediction
5. Mean = Best estimate, Std Dev = Uncertainty
```

---

### Step 1.3: Uncertainty Quantification

**Prediction Intervals**:
We compute 90% confidence intervals using the MC samples:

```python
# From 1000 MC samples, sort and take percentiles
lower_bound = np.percentile(samples, 5)   # 5th percentile
upper_bound = np.percentile(samples, 95)  # 95th percentile
```

**High-Variance Flagging**:
Patients with uncertainty > 40% are flagged for special attention:

| Uncertainty Level | Interpretation | Action |
|-------------------|----------------|--------|
| < 20% | High confidence | Trust prediction |
| 20-40% | Moderate uncertainty | Normal monitoring |
| > 40% | High variance | Flag for review |

**Clinical Rationale**:
High uncertainty often indicates:
- Unusual patient presentation
- Missing critical data
- Patient at decision boundary (could go either way)

---

## Phase 2: Safety Layer Construction

### Step 2.1: Medical Knowledge Base

**Why Safety Rules?**
Even the best AI can make dangerous recommendations. We need guardrails based on established medical knowledge.

**Rule Structure**:
```
IF [Patient Condition] AND [AI Suggests Action]
THEN [Block/Warn] + [Explain Why]
```

**Example Rules**:

#### Rule 1: BP Management in Stroke
```
IF:
  - Patient has ischemic stroke (ICD: I63.*)
  - Current BP > 185/110 mmHg
  - AI suggests aggressive BP lowering

THEN:
  - BLOCK the recommendation
  - EXPLAIN: "Permissive hypertension needed for cerebral perfusion"
  - SOURCE: AHA/ASA Stroke Guidelines
```

#### Rule 2: Nephrotoxic Drugs in Renal Impairment
```
IF:
  - Creatinine > 2.0 mg/dL (renal impairment)
  - AI suggests: NSAIDs, aminoglycosides, or contrast

THEN:
  - BLOCK the recommendation
  - EXPLAIN: "High risk of acute kidney injury"
  - SOURCE: KDIGO Guidelines
```

#### Rule 3: Anticoagulation in Active Bleeding
```
IF:
  - Active bleeding documented
  - AI suggests: heparin, warfarin, or DOACs

THEN:
  - BLOCK the recommendation
  - EXPLAIN: "Contraindicated with active hemorrhage"
  - SOURCE: ACCP Antithrombotic Guidelines
```

---

### Step 2.2: Rule Engine Implementation

**Screening Algorithm**:

```
For each AI recommendation:
    1. Extract patient's current state
       - Vitals, labs, diagnoses, medications
    
    2. Find all applicable rules
       - Match patient conditions to rule triggers
    
    3. Check each rule for violations
       - Does recommendation conflict with rule action?
    
    4. If violation found:
       - Block recommendation
       - Log to audit trail
       - Return explanation
    
    5. If no violations:
       - Allow recommendation
       - Log for monitoring
```

**Severity Levels**:

| Level | Action | Use Case |
|-------|--------|----------|
| CRITICAL | Hard block | Life-threatening violations |
| WARNING | Soft block + confirm | Significant risk |
| INFO | Allow + notify | Minor concerns |

---

### Step 2.3: Retrospective Validation

**Purpose**: Measure how often the safety layer would have prevented real harm.

**Metrics**:
- **True Prevention Rate**: Cases where blocking prevented actual harm
- **False Positive Rate**: Cases where blocking was overly conservative

**Ideal Balance**:
- High true prevention (catch dangerous recommendations)
- Low false positives (don't block too many safe recommendations)

---

## Key Algorithms

### 1. Time Window Aggregation

```python
def aggregate_to_windows(events, window_hours=6):
    """
    Convert irregular events to fixed time windows.
    
    Logic:
    - Group events by time window
    - Take mean of numeric values within window
    - Forward-fill missing windows from previous
    """
    for each window in admission_windows:
        events_in_window = filter(events, time in window)
        if events_in_window:
            values[window] = mean(events_in_window)
        else:
            values[window] = values[window-1]  # Forward fill
    return values
```

### 2. Outcome Labeling

```python
def label_deterioration(patient, horizon_hours=48):
    """
    Label patient as 1 (deteriorated) or 0 (stable).
    
    Deterioration = ANY of:
    - ICU transfer within horizon
    - Vasopressor started within horizon
    - Ventilation started within horizon
    - Death within horizon
    """
    events_in_horizon = get_events(patient, hours=horizon_hours)
    
    icu_transfer = any(event.type == 'ICU_TRANSFER')
    vasopressor = any(event.drug in VASOPRESSORS)
    ventilation = any(event.type == 'VENTILATION')
    death = patient.death_time <= horizon
    
    return int(icu_transfer or vasopressor or ventilation or death)
```

### 3. Monte Carlo Uncertainty

```python
def mc_uncertainty(model, x, n_samples=1000):
    """
    Compute prediction mean and uncertainty via MC Dropout.
    
    Key insight: Dropout during inference creates an ensemble
    of models, whose disagreement indicates uncertainty.
    """
    model.train()  # Enable dropout
    
    predictions = []
    for _ in range(n_samples):
        pred = model(x)  # Different dropout mask each time
        predictions.append(pred)
    
    mean_pred = np.mean(predictions)
    uncertainty = np.std(predictions)
    
    # Interpretation:
    # Low std = high confidence = model always predicts similar value
    # High std = low confidence = dropout significantly changes prediction
    
    return mean_pred, uncertainty
```

### 4. Safety Rule Matching

```python
def check_rule(rule, patient_state, recommendation):
    """
    Check if a recommendation violates a safety rule.
    
    Logic:
    1. Check if patient matches rule conditions
    2. Check if recommendation matches blocked actions
    3. Return violation if both match
    """
    # Check conditions
    for condition_key, condition_value in rule.conditions:
        patient_value = patient_state.get(condition_key)
        if not matches(patient_value, condition_value):
            return None  # Rule doesn't apply
    
    # Check if recommendation is blocked
    for blocked_action in rule.blocked_actions:
        if recommendation matches blocked_action:
            return Violation(
                rule=rule,
                severity=rule.severity,
                explanation=rule.explanation
            )
    
    return None  # Recommendation is safe
```

---

## Evaluation Metrics

### For Phase 1 (Prediction)

| Metric | Purpose | Good Value |
|--------|---------|------------|
| **AUC-ROC** | Overall discrimination | > 0.80 |
| **AUC-PR** | Performance on rare events | > 0.50 |
| **F1 Score** | Balance of precision/recall | > 0.50 |
| **Calibration** | Predicted probs match reality | ECE < 0.10 |
| **Coverage** | % of intervals containing truth | ≈ 90% |

### For Phase 2 (Safety)

| Metric | Purpose | Good Value |
|--------|---------|------------|
| **Harm Prevention Rate** | % dangerous recs blocked | > 95% |
| **False Positive Rate** | % safe recs blocked | < 10% |
| **Audit Completeness** | All decisions logged | 100% |

---

## Clinical Workflow Integration

```
Patient Data
     │
     ▼
┌─────────────┐
│   Phase 1   │ ─── Prediction: "70% risk of deterioration"
│ Digital Twin│ ─── Uncertainty: "±12% (confident)"
└─────────────┘
     │
     ▼
┌─────────────┐
│  AI System  │ ─── Recommendation: "Start vasopressor"
│  (External) │
└─────────────┘
     │
     ▼
┌─────────────┐
│   Phase 2   │ ─── Check against rules
│Safety Layer │ ─── No violations found
└─────────────┘
     │
     ▼
┌─────────────┐
│  Clinician  │ ─── Reviews AI suggestion + uncertainty
│  Decision   │ ─── Makes final decision
└─────────────┘
```

---

## Key Design Decisions

### Why 6-Hour Windows?
- Matches typical nursing assessment frequency
- Balances granularity vs noise reduction
- Clinically interpretable ("morning shift data")

### Why 48-Hour Prediction Horizon?
- Long enough to enable intervention
- Short enough for actionable predictions
- Matches typical ICU transfer decision window

### Why 40% Uncertainty Threshold?
- Empirically derived from calibration studies
- High enough to avoid over-flagging
- Low enough to catch genuinely uncertain cases

### Why Hard-Coded Rules (vs. Learned)?
- Medical knowledge should be explicit and auditable
- Rules can be traced to clinical guidelines
- Easier to update when guidelines change
- Avoids "black box" safety decisions

---

## Limitations and Future Work

### Current Limitations
1. Rules are static (don't adapt to new evidence)
2. Limited to MIMIC-IV data patterns
3. No real-time streaming support
4. Single-site validation only

### Planned Improvements
- **Phase 3**: Human-in-the-loop learning
- **Phase 4**: Continual knowledge base updating
- **Phase 5**: Multi-site validation
- **Phase 6**: Real-time deployment architecture

---

*Logic Documentation for Clinical AI System v1.0*
