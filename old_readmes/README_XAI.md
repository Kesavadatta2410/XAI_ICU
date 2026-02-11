# XAI Analysis for ICU Mortality Prediction

Comprehensive explainability analysis for the Liquid Mamba, Transformer, and GRU-D models.

## Features

### 1. Global Feature Importance
- **ICD Code Importance**: Top-20 diagnosis codes by attention weight
- **Statistical Analysis**: Mean ± std across all test patients
- **Model Comparison**: Side-by-side importance across models

### 2. Per-Patient Explanations  
- **Timeline Visualization**: Vital signs over time with predictions
- **Risk Assessment**: Color-coded mortality probability (green/orange/red)
- **Uncertainty Quantification**: Model confidence for each prediction

### 3. Mamba-Specific Visualizations
- **ODE Dynamics**: Liquid state evolution during patient stay
- **Phase Portraits**: Trajectory visualization in hidden space  
- **State Transitions**: Critical moments where states change rapidly

### 4. Counterfactual Analysis *(Coming Soon)*
- **What-If Scenarios**: "If glucose was 20% lower, risk would be X%"
- **Intervention Recommendations**: Which changes most reduce mortality risk
- **Sensitivity Analysis**: How robust are predictions to input changes

## Usage

### Basic Analysis
```bash
# Analyze Liquid Mamba model with 20 sample patients
python xai_analysis.py --model LiquidMamba --n-patients 20

# Analyze all models
python xai_analysis.py --model all --n-patients 50

# Generate HTML dashboard
python xai_analysis.py --model all --export-dashboard
```

### Requirements
- Trained model checkpoints in `checkpoints/` directory
- MIMIC-IV data in `data100k/` directory  
- All dependencies from `research.py`

## Output Directory Structure

```
results/xai/
├── feature_importance/
│   ├── LiquidMamba_importance.csv
│   ├── LiquidMamba_icd_importance.png
│   ├── Transformer_importance.png
│   └── GRUD_importance.png
│
├── mamba_dynamics/
│   ├── patient_12345_ode_dynamics.png
│   ├── patient_67890_ode_dynamics.png
│   └── ... (more patients)
│
├── patients/
│   ├── patient_12345_timeline.png
│   ├── patient_67890_timeline.png
│   └── ... (more patients)
│
└── counterfactuals/  (coming soon)
    ├── intervention_glucose.png
    ├── intervention_bp.png
    └── summary.json
```

## Outputs Explained

### Feature Importance Plot
![Example](docs/feature_importance_example.png)

Shows the top-20 ICD diagnosis codes that the model pays most attention to when making predictions. Higher bars = more important for mortality prediction.

### Patient Timeline
![Example](docs/patient_timeline_example.png)

**Top Panel**: Vital sign measurements over time during ICU stay  
**Bottom Panel**: Predicted mortality risk with color-coded thresholds
- Green (< 30%): Low risk
- Orange (30-70%): Medium risk  
- Red (> 70%): High risk

### Mamba ODE Dynamics
![Example](docs/ode_dynamics_example.png)

**Left Panel**: Evolution of liquid state dimensions over time  
**Right Panel**: Phase portrait showing state trajectory

Unique to Liquid Mamba! Shows how the ODE-based liquid computation evolves throughout the patient's ICU stay.

## CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | `LiquidMamba` | Which model to analyze: `LiquidMamba`, `Transformer`, `GRUD`, or `all` |
| `--n-patients` | int | `20` | Number of patients to analyze in detail |
| `--export-dashboard` | flag | False | Generate HTML dashboard (coming soon) |

## Expected Runtime

- **LiquidMamba**: ~5-10 minutes for 20 patients
- **All models**: ~15-30 minutes for 20 patients  
- **Full analysis (100 patients)**: ~1-2 hours

Runtime depends on GPU availability and batch size.

## Notes

⚠️ **Prerequisites**:
1. Models must be trained first using `research.py`
2. Checkpoints must exist in `checkpoints/` directory:
   - `LiquidMamba_best.pth`
   - `Transformer_best.pth`  
   - `GRUD_best.pth`

🔬 **Current Status**: 
- ✅ Feature importance
- ✅ Per-patient timelines
- ✅ Mamba ODE dynamics
- ⏳ Counterfactual analysis (in development)
- ⏳ HTML dashboard (in development)

## Citation

If you use these XAI techniques in your research, please cite our paper:

```bibtex
@article{yourlastname2024liquid,
  title={Liquid Mamba: ODE-Driven Selective State Spaces for ICU Mortality Prediction},
  author={Your Name},
  journal={TBD},
  year={2024}
}
```
