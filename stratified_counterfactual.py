"""
Stratified Counterfactual Validity Analysis
============================================
Adam-optimizer counterfactual generation, stratified by risk tier.
Designed for GPU (RTX 3050+). On GPU: 150 patients × 80 steps ≈ 5 min.

Usage:
    python stratified_counterfactual.py                          # defaults: 150 pts, 80 steps
    python stratified_counterfactual.py --n_patients 300 --steps 80

Output:
    results/xai/counterfactuals/stratified_validity_results.json
    results/xai/counterfactuals/stratified_validity_figure.png
    results/xai/counterfactuals/counterfactual_examples.csv
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from research import (
    ICUMortalityPredictor, ICUDataset, collate_fn,
    Config, set_seed, load_mimic_data, build_icd_graph, prepare_sequences
)
from xai_analysis import XAIConfig, load_trained_model

# ─────────────────────────────────────────────────────────────────────────────
RISK_TIERS = {
    "Extreme (>80%)":    (0.80, 1.01),
    "High (60-80%)":     (0.60, 0.80),
    "Moderate (40-60%)": (0.40, 0.60),
    "Low (<40%)":        (0.00, 0.40),
}
DECISION_THRESHOLD = 0.50


# ─────────────────────────────────────────────────────────────────────────────
# COUNTERFACTUAL — Adam optimizer (accurate, GPU-optimised)
# ─────────────────────────────────────────────────────────────────────────────

def generate_counterfactual_gradient(
    model,
    batch: dict,
    icd_adj: torch.Tensor,
    device: str,
    target_risk: float = 0.49,
    n_steps: int = 80,
    step_size: float = 0.02,
    perturbation_budget: float = 2.0,
    feature_clip: float = 3.0,
) -> Tuple[float, float, float, np.ndarray]:
    """
    Adam-based counterfactual generation (GPU-optimised).

    Iteratively perturbs input features toward lower mortality risk
    using Adam gradient descent with an L2 budget constraint.
    On RTX 3050: ~2ms/step → 80 steps × 150 patients ≈ 5 min.

    Returns
    -------
    original_risk, final_risk, proximity_score, delta_features
    """
    model.eval()

    values_orig = batch['values'].to(device).float()
    delta_t     = batch['delta_t'].to(device).float()
    mask        = batch['mask'].to(device).float()
    modality    = batch['modality'].to(device)
    item_idx    = batch['item_idx'].to(device)
    icd_act     = batch['icd_activation'].to(device).float()

    with torch.no_grad():
        prob_orig, _, _, _ = model(
            values_orig, delta_t, mask, modality,
            item_idx, icd_act, icd_adj, return_internals=True
        )
    original_risk = prob_orig.item()

    delta = torch.zeros_like(values_orig, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([delta], lr=step_size)

    best_risk  = original_risk
    best_delta = delta.detach().clone()

    for _ in range(n_steps):
        optimizer.zero_grad()

        values_perturbed = (values_orig + delta).clamp(-feature_clip, feature_clip)
        prob, _, _, _ = model(
            values_perturbed, delta_t, mask, modality,
            item_idx, icd_act, icd_adj, return_internals=True
        )

        loss = F.mse_loss(prob, torch.tensor([[target_risk]], device=device))
        l2   = delta.norm(p=2)
        if l2 > perturbation_budget:
            loss = loss + 0.1 * (l2 - perturbation_budget)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            l2 = delta.norm(p=2)
            if l2 > perturbation_budget:
                delta.data = delta.data * (perturbation_budget / l2)

        cur_risk = prob.item()
        if cur_risk < best_risk:
            best_risk  = cur_risk
            best_delta = delta.detach().clone()

        if cur_risk < DECISION_THRESHOLD:
            break   # early stop

    delta_np  = best_delta.squeeze(0).cpu().numpy()
    proximity = float(np.linalg.norm(delta_np))
    return original_risk, best_risk, proximity, delta_np


# ─────────────────────────────────────────────────────────────────────────────
# STRATIFIED VALIDITY
# ─────────────────────────────────────────────────────────────────────────────

def _get_tier(risk: float) -> str:
    for name, (lo, hi) in RISK_TIERS.items():
        if lo <= risk < hi:
            return name
    return "Low (<40%)"


def compute_stratified_validity(
    model,
    test_loader: DataLoader,
    icd_adj: torch.Tensor,
    device: str,
    n_patients: int = 50,
    n_steps: int = 20,
    perturbation_budget: float = 2.0,
) -> Tuple[Dict, pd.DataFrame]:

    print("\n" + "=" * 70)
    print("  STRATIFIED COUNTERFACTUAL VALIDITY ANALYSIS")
    print(f"  Patients: {n_patients} | Steps/pt: {n_steps} | Budget: {perturbation_budget}")
    est_sec = n_patients * n_steps * 0.002  # ~2ms/step on GPU
    print(f"  Estimated time: ~{max(est_sec/60, 1):.1f} min (GPU) / ~{est_sec*60:.0f} min (CPU)")
    print("=" * 70)

    model.eval()
    records = []
    n_done  = 0

    for batch_idx, batch in enumerate(tqdm(test_loader,
                                           desc="Generating CFs",
                                           total=n_patients)):
        if n_done >= n_patients:
            break

        try:
            orig, final, prox, delta = generate_counterfactual_gradient(
                model, batch, icd_adj, device,
                n_steps=n_steps, perturbation_budget=perturbation_budget,
            )
        except Exception as e:
            tqdm.write(f"  Patient {batch_idx} error: {e}")
            n_done += 1
            continue

        tier  = _get_tier(orig)
        valid = (orig >= DECISION_THRESHOLD) and (final < DECISION_THRESHOLD)

        records.append({
            "patient_idx":    batch_idx,
            "original_risk":  round(orig, 4),
            "final_risk":     round(final, 4),
            "risk_reduction": round(orig - final, 4),
            "proximity":      round(prox, 4),
            "tier":           tier,
            "valid":          valid,
        })
        n_done += 1

    df = pd.DataFrame(records)

    # Aggregate
    stratified = {}
    for tname in RISK_TIERS:
        sub = df[df['tier'] == tname]
        if len(sub) == 0:
            stratified[tname] = {"n_patients": 0, "validity_rate": None,
                                  "mean_proximity": None, "std_proximity": None,
                                  "mean_risk_reduction": None,
                                  "original_risks": [], "final_risks": []}
            continue

        flippable = sub[sub['original_risk'] >= DECISION_THRESHOLD]
        vr = (float(flippable['valid'].sum() / len(flippable))
              if len(flippable) > 0 else None)

        stratified[tname] = {
            "n_patients":          int(len(sub)),
            "n_flippable":         int(len(flippable)),
            "validity_rate":       round(vr, 4) if vr is not None else None,
            "mean_proximity":      round(float(sub['proximity'].mean()), 4),
            "std_proximity":       round(float(sub['proximity'].std(skipna=True)), 4),
            "mean_risk_reduction": round(float(sub['risk_reduction'].mean()), 4),
            "original_risks":      sub['original_risk'].tolist(),
            "final_risks":         sub['final_risk'].tolist(),
        }

    flip_all = df[df['original_risk'] >= DECISION_THRESHOLD]
    overall  = (float(flip_all['valid'].sum() / len(flip_all))
                if len(flip_all) > 0 else 0.0)

    results = {
        "per_patient":             records,
        "stratified":              stratified,
        "overall_validity_rate":   round(overall, 4),
        "total_patients_analyzed": n_done,
        "config": {"n_steps": n_steps, "perturbation_budget": perturbation_budget,
                   "decision_threshold": DECISION_THRESHOLD},
    }

    # Print table
    print("\n  ┌─────────────────────────┬──────────┬──────────────┬────────────────────┐")
    print("  │ Risk Tier               │ Patients │ Validity (%) │ Mean Risk Reduction│")
    print("  ├─────────────────────────┼──────────┼──────────────┼────────────────────┤")
    for t, s in stratified.items():
        if s['n_patients'] == 0:
            continue
        vr_s = f"{s['validity_rate']*100:.1f}%" if s['validity_rate'] is not None else "N/A"
        rr_s = f"{s['mean_risk_reduction']*100:.1f}%"
        print(f"  │ {t:<23} │ {s['n_patients']:>8} │ {vr_s:>12} │ {rr_s:>18} │")
    print("  └─────────────────────────┴──────────┴──────────────┴────────────────────┘")
    print(f"\n  Overall validity (flippable patients only): {overall*100:.1f}%")

    return results, df


# ─────────────────────────────────────────────────────────────────────────────
# 4-PANEL FIGURE
# ─────────────────────────────────────────────────────────────────────────────

def plot_stratified_validity(results: Dict, df: pd.DataFrame,
                             save_path: Path, dpi: int = 300):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Stratified Counterfactual Validity — LiquidMamba ICU",
                 fontsize=14, fontweight='bold', y=1.02)

    COLORS = {
        "Extreme (>80%)":    '#d62728',
        "High (60-80%)":     '#ff7f0e',
        "Moderate (40-60%)": '#2ca02c',
        "Low (<40%)":        '#1f77b4',
    }
    tnames     = list(RISK_TIERS.keys())
    stratified = results['stratified']

    # A: Validity bar
    ax = axes[0, 0]
    valid_t = [t for t in tnames
               if stratified[t]['n_patients'] > 0
               and stratified[t]['validity_rate'] is not None]
    rates   = [stratified[t]['validity_rate'] * 100 for t in valid_t]
    bars    = ax.bar(range(len(valid_t)), rates,
                     color=[COLORS[t] for t in valid_t],
                     alpha=0.85, edgecolor='black', linewidth=0.8)
    for bar, rate, t in zip(bars, rates, valid_t):
        nf = stratified[t].get('n_flippable', 0)
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5, f"{rate:.1f}%\n(n={nf})",
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.axhline(50, color='black', linestyle='--', lw=1.2, alpha=0.6)
    ax.set_xticks(range(len(valid_t)))
    ax.set_xticklabels(valid_t, fontsize=8, rotation=15, ha='right')
    ax.set_ylabel('Validity Rate (%)', fontsize=10)
    ax.set_title('(A) Validity Rate by Risk Tier', fontsize=11, fontweight='bold')
    ax.set_ylim(0, max(rates + [20]) + 20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # B: Proximity box
    ax = axes[0, 1]
    bdata, blabels, bcolors = [], [], []
    for t in tnames:
        sub = df[df['tier'] == t]
        if len(sub) > 0:
            bdata.append(sub['proximity'].values)
            blabels.append(t)
            bcolors.append(COLORS[t])
    if bdata:
        bp = ax.boxplot(bdata, labels=blabels, patch_artist=True,
                        medianprops=dict(color='black', linewidth=2),
                        flierprops=dict(marker='o', markersize=3, alpha=0.5))
        for patch, c in zip(bp['boxes'], bcolors):
            patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_ylabel('Proximity (L₂ ‖Δx‖)', fontsize=10)
    ax.set_title('(B) Counterfactual Difficulty\n(higher = harder to flip)',
                 fontsize=11, fontweight='bold')
    ax.set_xticklabels(blabels, fontsize=8, rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # C: Risk reduction violin
    ax = axes[1, 0]
    vdata, vlabels, vcolors = [], [], []
    for t in tnames:
        sub = df[df['tier'] == t]
        if len(sub) > 1:
            vdata.append(sub['risk_reduction'].values * 100)
            vlabels.append(t); vcolors.append(COLORS[t])
    if vdata:
        parts = ax.violinplot(vdata, positions=range(len(vdata)),
                              showmedians=True, showextrema=True)
        for pc, c in zip(parts['bodies'], vcolors):
            pc.set_facecolor(c); pc.set_alpha(0.7)
        parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(2)
    ax.axhline(0, color='gray', linestyle='--', lw=1, alpha=0.7)
    ax.set_xticks(range(len(vlabels)))
    ax.set_xticklabels(vlabels, fontsize=8, rotation=15, ha='right')
    ax.set_ylabel('Risk Reduction (%)', fontsize=10)
    ax.set_title('(C) Risk Reduction Achieved\n(positive = toward survival)',
                 fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # D: Scatter original vs final
    ax = axes[1, 1]
    for t in tnames:
        sub = df[df['tier'] == t]
        if len(sub) > 0:
            ax.scatter(sub['original_risk'], sub['final_risk'],
                       c=COLORS[t], alpha=0.7, s=40, label=t,
                       edgecolors='white', linewidths=0.4)
    ax.plot([0,1],[0,1],'k--',lw=1.2,alpha=0.6,label='No change')
    ax.axhline(DECISION_THRESHOLD, color='red', linestyle=':', lw=1.5,
               alpha=0.8, label=f'Boundary ({DECISION_THRESHOLD})')
    ax.axvline(DECISION_THRESHOLD, color='red', linestyle=':', lw=1.5, alpha=0.8)
    ax.fill_betweenx([0, DECISION_THRESHOLD],
                     [DECISION_THRESHOLD]*2, [1.0]*2,
                     alpha=0.08, color='green', label='Valid CF region')
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_xlabel('Original Risk', fontsize=10)
    ax.set_ylabel('Final Risk (after perturbation)', fontsize=10)
    ax.set_title('(D) Risk Trajectory After Counterfactual',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved figure → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN (callable from xai_analysis.py OR standalone)
# ─────────────────────────────────────────────────────────────────────────────

def run_stratified_analysis(
    model, test_loader, icd_adj, device,
    output_dir: Path,
    n_patients: int = 50,
    n_steps: int = 20,
    perturbation_budget: float = 2.0,
    dpi: int = 300,
) -> Dict:
    cf_dir = output_dir / "counterfactuals"
    cf_dir.mkdir(parents=True, exist_ok=True)

    results, df = compute_stratified_validity(
        model, test_loader, icd_adj, device,
        n_patients=n_patients, n_steps=n_steps,
        perturbation_budget=perturbation_budget,
    )

    # Save JSON
    jpath = cf_dir / "stratified_validity_results.json"
    jout  = {k: v for k, v in results.items() if k != "per_patient"}
    jout["stratified"] = {
        t: {k2: v2 for k2, v2 in s.items()
            if k2 not in ("original_risks", "final_risks")}
        for t, s in results["stratified"].items()
    }
    jout["per_patient"] = results["per_patient"]
    with open(jpath, "w") as f:
        json.dump(jout, f, indent=2)
    print(f"  ✅ JSON → {jpath}")

    # Save CSV
    cpath = cf_dir / "counterfactual_examples.csv"
    df.to_csv(cpath, index=False)
    print(f"  ✅ CSV  → {cpath}")

    # Save figure
    plot_stratified_validity(results, df,
                             cf_dir / "stratified_validity_figure.png", dpi=dpi)

    print("\n  CLINICAL INTERPRETATION")
    print("  " + "-"*60)
    interps = {
        "Extreme (>80%)":    "Multi-organ failure — beyond intervention reversal. Escalate.",
        "High (60-80%)":     "Critically ill but potentially reversible with aggressive Rx.",
        "Moderate (40-60%)": "Decision boundary reachable — CFs directly actionable at bedside.",
        "Low (<40%)":        "Already below threshold — focus on maintaining current status.",
    }
    for t, msg in interps.items():
        s  = results["stratified"].get(t, {})
        vr = s.get("validity_rate")
        vr_s = f"{vr*100:.1f}%" if vr is not None else "N/A"
        print(f"  ► {t} (n={s.get('n_patients',0)}, validity={vr_s})")
        print(f"    {msg}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Stratified Counterfactual Validity")
    parser.add_argument('--n_patients', type=int, default=150,
                        help='Patients to analyze (default 150, ~5 min on GPU)')
    parser.add_argument('--steps',      type=int,   default=80,
                        help='Adam steps per patient (default 80)')
    parser.add_argument('--budget',     type=float, default=2.0)
    parser.add_argument('--output_dir', type=str,   default='results/xai')
    args = parser.parse_args()

    xai_cfg = XAIConfig()
    config  = Config()
    set_seed(xai_cfg.seed)
    device  = xai_cfg.device

    est_gpu = args.n_patients * args.steps * 0.002 / 60
    est_cpu = args.n_patients * args.steps * 0.05 / 60
    print(f"Device: {device}  |  GPU est: ~{est_gpu:.1f} min  |  CPU est: ~{est_cpu:.0f} min")

    data = load_mimic_data(config.data_dir)
    icd_graph, _ = build_icd_graph(data)
    tensors, labels, vocab_size = prepare_sequences(data, icd_graph, config)

    all_ids  = labels.index.tolist()
    test_ids = all_ids[int(0.9 * len(labels)):]
    loader   = DataLoader(
        ICUDataset(tensors, labels, icd_graph, test_ids),
        batch_size=1, collate_fn=collate_fn, shuffle=False
    )
    icd_adj = icd_graph.adj_matrix.to(device)

    model = load_trained_model(
        "LiquidMamba", ICUMortalityPredictor,
        {"vocab_size": vocab_size, "n_icd_nodes": icd_graph.n_nodes, "config": config},
        Path(xai_cfg.checkpoint_dir) / "LiquidMamba_best.pth", device,
    )

    run_stratified_analysis(
        model, loader, icd_adj, device,
        output_dir=Path(args.output_dir),
        n_patients=args.n_patients,
        n_steps=args.steps,
        perturbation_budget=args.budget,
    )

if __name__ == "__main__":
    main()
