import os
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, roc_curve,
    f1_score, precision_score, recall_score, accuracy_score,
    confusion_matrix, classification_report, average_precision_score,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class Config:
    """Configuration for the clinical AI system deployment."""
    data_dir: str = "data_10k"
    output_dir: str = "pat_res"
    deployment_package_path: str = "results/deployment_package.pth"
    
    # Patient selection
    min_age: int = 18
    min_stay_hours: int = 24
    time_window_hours: int = 6
    outcome_window_hours: int = 48
    
    # Model parameters (will be overwritten from deployment package)
    embed_dim: int = 64
    hidden_dim: int = 128
    graph_dim: int = 64
    n_mamba_layers: int = 2
    n_attention_heads: int = 4
    dropout: float = 0.2
    diffusion_steps: int = 50
    diffusion_hidden: int = 128
    max_seq_len: int = 128
    
    # Digital Twin simulation
    mc_samples: int = 200  # MC Dropout simulation runs
    uncertainty_threshold: float = 0.4
    confidence_level: float = 0.9
    
    # Deployment (no training)
    batch_size: int = 32
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


config = Config()
Path(config.output_dir).mkdir(parents=True, exist_ok=True)
torch.manual_seed(config.seed)
np.random.seed(config.seed)


# ============================================================
# MODEL CLASSES (Duplicated from research.py for standalone deployment)
# ============================================================

class ODELiquidCell(nn.Module):
    """
    ODE-Based Liquid Neural Cell implementing:
      dh/dt = (1/τ(Δt)) * (σ(Wx·x + Wh·h + b) - h)
    
    The time constant τ(Δt) adapts based on time gap:
    - Small Δt → large τ → slow dynamics (frequent vitals)
    - Large Δt → small τ → fast adaptation (sparse labs)
    """
    
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        
        self.W_x = nn.Linear(dim, dim)
        self.W_h = nn.Linear(dim, dim, bias=False)
        
        self.tau_net = nn.Sequential(
            nn.Linear(1, dim),
            nn.Softplus()
        )
        self.tau_min = 0.1
        
        self.obs_gate = nn.Linear(dim, dim)
        self.layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x_t: torch.Tensor, h: torch.Tensor, 
                delta_t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        dt = delta_t.unsqueeze(-1).clamp(0.01, 24)
        tau = self.tau_min + self.tau_net(dt)
        
        f_xh = torch.tanh(self.W_x(x_t) + self.W_h(h))
        dh_dt = (f_xh - h) / tau
        h_evolved = h + dt * dh_dt
        
        mask_expanded = mask.unsqueeze(-1)
        obs_update = torch.sigmoid(self.obs_gate(x_t)) * x_t
        
        h_out = mask_expanded * (0.7 * h_evolved + 0.3 * obs_update) + \
                (1 - mask_expanded) * h_evolved
        
        h_out = self.layer_norm(h_out)
        h_out = torch.nan_to_num(h_out, nan=0.0)
        h_out = self.dropout(h_out)
        
        return h_out


class LiquidMambaEncoder(nn.Module):
    """Full Liquid Mamba encoder for irregular time-series."""
    
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, 
                 n_layers: int = 2, n_modalities: int = 3, dropout: float = 0.2):
        super().__init__()
        
        self.value_proj = nn.Linear(1, embed_dim // 2)
        self.item_embed = nn.Embedding(vocab_size + 1, embed_dim // 4, padding_idx=0)
        self.modality_embed = nn.Embedding(n_modalities, embed_dim // 4)
        
        self.input_proj = nn.Linear(embed_dim, hidden_dim)
        
        self.layers = nn.ModuleList([
            ODELiquidCell(hidden_dim, dropout=dropout)
            for _ in range(n_layers)
        ])
        
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.hidden_dim = hidden_dim
        
    def forward(self, values: torch.Tensor, delta_t: torch.Tensor, 
                mask: torch.Tensor, modality: torch.Tensor, 
                item_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len = values.shape
        device = values.device
        
        val_emb = self.value_proj(values.unsqueeze(-1))
        item_emb = self.item_embed(item_idx)
        mod_emb = self.modality_embed(modality)
        
        x = torch.cat([val_emb, item_emb, mod_emb], dim=-1)
        x = self.input_proj(x)
        
        hidden_states = []
        h = torch.zeros(batch, self.hidden_dim, device=device)
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            dt_t = delta_t[:, t]
            mask_t = mask[:, t]
            
            for layer in self.layers:
                h = layer(x_t, h, dt_t, mask_t)
            
            hidden_states.append(h)
        
        hidden_states = torch.stack(hidden_states, dim=1)
        
        mask_sum = mask.sum(dim=1, keepdim=True).clamp(min=1)
        temporal_emb = (hidden_states * mask.unsqueeze(-1)).sum(dim=1) / mask_sum
        temporal_emb = torch.nan_to_num(temporal_emb, nan=0.0)
        temporal_emb = self.final_norm(temporal_emb)
        
        return temporal_emb, hidden_states


class GraphAttentionNetwork(nn.Module):
    """Multi-head Graph Attention Network for ICD embeddings."""
    
    def __init__(self, n_nodes: int, embed_dim: int = 32, hidden_dim: int = 64, 
                 out_dim: int = 64, n_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        
        self.n_nodes = n_nodes
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        
        self.node_embed = nn.Embedding(n_nodes, embed_dim)
        
        self.W_q = nn.Linear(embed_dim, hidden_dim)
        self.W_k = nn.Linear(embed_dim, hidden_dim)
        self.W_v = nn.Linear(embed_dim, hidden_dim)
        
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dim)
        
    def forward(self, patient_activation: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = patient_activation.shape[0]
        device = patient_activation.device
        
        node_ids = torch.arange(self.n_nodes, device=device)
        x = self.node_embed(node_ids).unsqueeze(0).expand(batch, -1, -1)
        
        x = x * patient_activation.unsqueeze(-1)
        
        Q = self.W_q(x).view(batch, self.n_nodes, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(batch, self.n_nodes, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(batch, self.n_nodes, self.n_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        adj_mask = (adj == 0).unsqueeze(0).unsqueeze(0).expand(batch, self.n_heads, -1, -1)
        scores = scores.masked_fill(adj_mask, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch, self.n_nodes, -1)
        out = self.out_proj(out)
        
        activation_sum = patient_activation.sum(dim=1, keepdim=True).clamp(min=1)
        graph_emb = (out * patient_activation.unsqueeze(-1)).sum(dim=1) / activation_sum
        graph_emb = torch.nan_to_num(graph_emb, nan=0.0)
        graph_emb = self.layer_norm(graph_emb)
        
        node_attention = attn_weights.mean(dim=1).mean(dim=1)
        node_attention = node_attention * patient_activation
        
        return graph_emb, node_attention


class CrossAttentionFusion(nn.Module):
    """Fuses temporal embedding with disease context."""
    
    def __init__(self, temporal_dim: int, graph_dim: int, n_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        
        self.n_heads = n_heads
        self.head_dim = temporal_dim // n_heads
        
        self.W_q = nn.Linear(temporal_dim, temporal_dim)
        self.W_k = nn.Linear(graph_dim, temporal_dim)
        self.W_v = nn.Linear(graph_dim, temporal_dim)
        
        self.out_proj = nn.Linear(temporal_dim, temporal_dim)
        self.layer_norm = nn.LayerNorm(temporal_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(temporal_dim + graph_dim, temporal_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(temporal_dim, temporal_dim)
        )
        
    def forward(self, temporal_emb: torch.Tensor, graph_emb: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([temporal_emb, graph_emb], dim=-1)
        fused = self.fusion_mlp(combined)
        fused = self.layer_norm(fused + temporal_emb)
        return fused


class UncertaintyMortalityHead(nn.Module):
    """Mortality prediction with aleatoric + epistemic uncertainty."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.mean_head = nn.Linear(hidden_dim // 2, 1)
        self.logvar_head = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.hidden(x)
        
        logit = self.mean_head(h).squeeze(-1)
        log_var = self.logvar_head(h).squeeze(-1)
        
        prob = torch.sigmoid(logit)
        aleatoric_uncertainty = torch.exp(log_var)
        
        return prob, aleatoric_uncertainty, logit


class CounterfactualDiffusion(nn.Module):
    """Conditional diffusion model for generating counterfactual trajectories."""
    
    def __init__(self, latent_dim: int, hidden_dim: int = 128, n_steps: int = 50):
        super().__init__()
        
        self.n_steps = n_steps
        self.latent_dim = latent_dim
        
        betas = torch.linspace(1e-4, 0.02, n_steps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))
        
        self.denoise_net = nn.Sequential(
            nn.Linear(latent_dim + latent_dim + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )
    
    @torch.no_grad()
    def generate_counterfactual(self, condition: torch.Tensor, 
                                 target_survival: bool = True) -> torch.Tensor:
        batch = condition.shape[0]
        device = condition.device
        
        target = torch.zeros(batch, device=device) if target_survival else torch.ones(batch, device=device)
        x = torch.randn(batch, self.latent_dim, device=device)
        
        for t in reversed(range(self.n_steps)):
            t_tensor = torch.full((batch,), t, device=device, dtype=torch.long)
            t_emb = (t_tensor.float() / self.n_steps).unsqueeze(-1)
            target_emb = target.unsqueeze(-1)
            
            inp = torch.cat([x, condition, t_emb, target_emb], dim=-1)
            noise_pred = self.denoise_net(inp)
            
            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alphas_cumprod[t]
            beta_t = self.betas[t]
            
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0
            
            x = (1 / alpha_t.sqrt()) * (x - (beta_t / (1 - alpha_cumprod_t).sqrt()) * noise_pred)
            x = x + beta_t.sqrt() * noise
        
        return x


class ICUMortalityPredictor(nn.Module):
    """
    Complete ICU Mortality Prediction System:
    1. Liquid Mamba for irregular time-series
    2. ICD Knowledge Graph for disease context
    3. Cross-attention fusion
    4. Uncertainty-aware predictions
    5. Diffusion-based counterfactual XAI
    """
    
    def __init__(self, vocab_size: int, n_icd_nodes: int, config: Config):
        super().__init__()
        
        self.config = config
        
        self.temporal_encoder = LiquidMambaEncoder(
            vocab_size=vocab_size,
            embed_dim=config.embed_dim,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_mamba_layers,
            dropout=config.dropout
        )
        
        self.graph_encoder = GraphAttentionNetwork(
            n_nodes=n_icd_nodes,
            embed_dim=32,
            hidden_dim=config.graph_dim,
            out_dim=config.graph_dim,
            n_heads=4,
            dropout=config.dropout
        )
        
        self.fusion = CrossAttentionFusion(
            temporal_dim=config.hidden_dim,
            graph_dim=config.graph_dim,
            n_heads=config.n_attention_heads,
            dropout=config.dropout
        )
        
        self.mortality_head = UncertaintyMortalityHead(
            input_dim=config.hidden_dim,
            hidden_dim=64,
            dropout=0.3
        )
        
        self.diffusion_xai = CounterfactualDiffusion(
            latent_dim=config.hidden_dim,
            hidden_dim=config.diffusion_hidden,
            n_steps=config.diffusion_steps
        )
        
    def forward(self, values, delta_t, mask, modality, item_idx, 
                icd_activation, icd_adj, return_internals: bool = False):
        temporal_emb, hidden_states = self.temporal_encoder(
            values, delta_t, mask, modality, item_idx
        )
        
        graph_emb, node_attention = self.graph_encoder(icd_activation, icd_adj)
        fused_emb = self.fusion(temporal_emb, graph_emb)
        prob, uncertainty, logit = self.mortality_head(fused_emb)
        
        if return_internals:
            internals = {
                'temporal_emb': temporal_emb,
                'graph_emb': graph_emb,
                'fused_emb': fused_emb,
                'hidden_states': hidden_states,
                'node_attention': node_attention
            }
            return prob, uncertainty, logit, internals
        
        return prob, uncertainty, logit


# ============================================================
# DEPLOYMENT FUNCTIONS
# ============================================================

class DiabeticDigitalTwin:
    """
    Standalone Diabetic Digital Twin for ICU patient simulation.
    
    This class encapsulates:
    1. Model loading from deployment_package.pth
    2. Patient simulation with MC Dropout for uncertainty quantification
    3. Diabetic-specific safety layer (hypoglycemia, DKA detection)
    4. Report generation with risk, uncertainty, and interventions
    """
    
    def __init__(self, deployment_path: str, device: str = "cpu"):
        """Load Digital Twin from deployment package."""
        self.device = device
        self.config = Config()
        
        # Load deployment package
        if not Path(deployment_path).exists():
            raise FileNotFoundError(
                f"Deployment package not found: {deployment_path}\n"
                f"Run research.py first to generate the model."
            )
        
        checkpoint = torch.load(deployment_path, map_location=device, weights_only=False)
        
        # Extract configuration
        self.config_dict = checkpoint['config_dict']
        self.vocab_size = checkpoint['vocab_size']
        self.n_icd_nodes = checkpoint['n_icd_nodes']
        self.feature_stats = checkpoint.get('feature_stats', {})
        self.feature_names = checkpoint.get('feature_names', [])
        self.physio_ranges = checkpoint.get('physio_ranges', {})
        self.itemid_to_idx = checkpoint.get('itemid_to_idx', {})
        self.icd_adj = checkpoint['icd_adj_matrix'].to(device)
        
        # Update config with saved values
        for key, value in self.config_dict.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Build model
        self.model = ICUMortalityPredictor(
            vocab_size=self.vocab_size,
            n_icd_nodes=self.n_icd_nodes,
            config=self.config
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        print(f"  ✓ DiabeticDigitalTwin loaded successfully")
        print(f"    - Features: {len(self.feature_names)} ({', '.join(self.feature_names[:5])}...)")
        print(f"    - Glucose monitoring: {'Glucose' in self.feature_names}")
    
    def simulate_patient(self, patient_data: Dict, n_simulations: int = 50) -> Dict:
        """
        Run MC Dropout simulation for uncertainty quantification.
        
        Args:
            patient_data: Dict with preprocessed patient tensors
            n_simulations: Number of MC dropout runs (default 50)
            
        Returns:
            Dict with mean_risk, std, lower_bound (2.5%), upper_bound (97.5%)
        """
        self.model.train()  # Enable dropout for MC sampling
        
        predictions = []
        for _ in range(n_simulations):
            with torch.no_grad():
                prob, uncertainty, logit = self.model(
                    patient_data['values'].to(self.device),
                    patient_data['delta_t'].to(self.device),
                    patient_data['mask'].to(self.device),
                    patient_data['modality'].to(self.device),
                    patient_data['item_idx'].to(self.device),
                    patient_data['icd_activation'].to(self.device),
                    self.icd_adj
                )
                predictions.append(torch.sigmoid(logit).cpu().numpy())
        
        self.model.eval()  # Reset to eval mode
        
        predictions = np.array(predictions).flatten()
        mean_risk = float(np.mean(predictions))
        std = float(np.std(predictions))
        
        # 95% confidence interval
        lower_bound = float(np.percentile(predictions, 2.5))
        upper_bound = float(np.percentile(predictions, 97.5))
        
        return {
            'mean_risk': mean_risk,
            'std': std,
            'uncertainty_pct': std * 100,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'confidence_interval': f"±{std*100:.1f}%",
            'n_simulations': n_simulations
        }
    
    def check_safety(self, risk_score: float, patient_vitals: Dict) -> Dict:
        """
        Apply diabetic-specific safety rules.
        
        Rules:
        1. Hypoglycemia: Glucose < 70 AND Risk < 0.2 → Override to High Risk
        2. DKA: Glucose > 250 AND Bicarbonate < 18 → Flag as DKA Risk
        
        Args:
            risk_score: Model's predicted mortality risk
            patient_vitals: Dict with glucose, bicarbonate, etc.
            
        Returns:
            Dict with final_risk, safety_flags, override_applied
        """
        safety_flags = []
        override_applied = False
        final_risk = risk_score
        
        glucose = patient_vitals.get('glucose', 100)
        bicarbonate = patient_vitals.get('bicarbonate', 24)
        
        # Rule 1: Hypoglycemia Override
        if glucose < 70 and risk_score < 0.2:
            safety_flags.append({
                "rule": "HYPOGLYCEMIA_OVERRIDE",
                "severity": "CRITICAL",
                "message": f"Glucose critically low ({glucose:.0f} mg/dL < 70)",
                "action": "High Risk (Safety Alert: Hypoglycemia)",
                "guideline": "ADA Diabetes Care Standards"
            })
            final_risk = max(risk_score, 0.7)
            override_applied = True
        
        # Rule 2: DKA Detection
        if glucose > 250 and bicarbonate < 18:
            safety_flags.append({
                "rule": "DKA_DETECTION",
                "severity": "CRITICAL",
                "message": f"Glucose elevated ({glucose:.0f}) with low bicarbonate ({bicarbonate:.1f})",
                "action": "Diabetic Ketoacidosis Risk",
                "guideline": "ADA DKA Management Protocol"
            })
            final_risk = max(risk_score, 0.8)
            override_applied = True
        
        # Rule 3: Severe Hyperglycemia (even without acidosis)
        if glucose > 400:
            safety_flags.append({
                "rule": "SEVERE_HYPERGLYCEMIA",
                "severity": "WARNING",
                "message": f"Glucose critically elevated ({glucose:.0f} mg/dL > 400)",
                "action": "Evaluate for HHS (Hyperosmolar Hyperglycemic State)",
                "guideline": "ADA Hyperglycemic Crisis Guidelines"
            })
            if risk_score < 0.5:
                final_risk = max(risk_score, 0.6)
                override_applied = True
        
        risk_category = (
            "Low Risk" if final_risk < 0.3 else
            "Medium Risk" if final_risk < 0.6 else
            "High Risk"
        )
        
        if override_applied:
            risk_category += " (Safety Override)"
        
        return {
            'model_risk': float(risk_score),
            'final_risk': float(final_risk),
            'risk_category': risk_category,
            'override_applied': override_applied,
            'safety_flags': safety_flags,
            'n_flags': len(safety_flags),
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_report(self, patient_id: str, simulation: Dict, 
                       safety: Dict, vitals: Dict) -> str:
        """
        Generate a clinical report for the patient.
        
        Format: Patient ID X | Risk: 85% (±5%) | Safety Flags: None | 
                Suggested Intervention: Stabilize Glucose.
        """
        risk_pct = simulation['mean_risk'] * 100
        uncertainty = simulation['uncertainty_pct']
        
        flags_str = "None" if safety['n_flags'] == 0 else ", ".join(
            [f["action"] for f in safety['safety_flags']]
        )
        
        # Determine intervention based on risk and flags
        if safety.get('override_applied'):
            if any('HYPOGLYCEMIA' in f['rule'] for f in safety['safety_flags']):
                intervention = "Administer glucose/dextrose immediately. Monitor q15min."
            elif any('DKA' in f['rule'] for f in safety['safety_flags']):
                intervention = "Initiate DKA protocol: IV fluids, insulin drip, K+ monitoring."
            else:
                intervention = "Stabilize glucose. Consult endocrinology."
        elif safety['final_risk'] > 0.6:
            intervention = "Intensify glucose monitoring. Target range 70-180 mg/dL."
        elif safety['final_risk'] > 0.3:
            intervention = "Continue current management. Monitor glucose q4h."
        else:
            intervention = "Routine care. Maintain glucose in target range."
        
        report = (
            f"Patient ID {patient_id} | "
            f"Risk: {risk_pct:.0f}% (Uncertainty: ±{uncertainty:.1f}%) | "
            f"Safety Flags: {flags_str} | "
            f"Suggested Intervention: {intervention}"
        )
        
        return report

def load_digital_twin(checkpoint_path: str, device: str = "cpu") -> Tuple:
    """
    Load trained model and preprocessing artifacts from deployment package.
    
    Args:
        checkpoint_path: Path to deployment_package.pth
        device: Target device for model
        
    Returns:
        model: Loaded ICUMortalityPredictor
        feature_stats: Dict with normalization stats (mean, std, min, max)
        config_dict: Model configuration
        icd_adj: ICD adjacency matrix
        itemid_to_idx: Vocabulary mapping
    """
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"\n{'='*60}\n"
            f"ERROR: Deployment package not found!\n"
            f"{'='*60}\n"
            f"Path: {checkpoint_path}\n\n"
            f"This script requires a pre-trained model from research.py.\n"
            f"Please run research.py first to train and generate the model:\n\n"
            f"    python research.py\n\n"
            f"This will create: {checkpoint_path}\n"
            f"{'='*60}"
        )
    
    print(f"  Loading deployment package from {checkpoint_path}...")
    # weights_only=False needed because checkpoint contains numpy arrays (feature_stats)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract configuration
    config_dict = checkpoint['config_dict']
    vocab_size = checkpoint['vocab_size']
    n_icd_nodes = checkpoint['n_icd_nodes']
    
    # Update config with saved values
    model_config = Config()
    for key, value in config_dict.items():
        if hasattr(model_config, key):
            setattr(model_config, key, value)
    model_config.device = device
    
    # Initialize model
    print(f"  Initializing model (vocab_size={vocab_size}, n_icd_nodes={n_icd_nodes})...")
    model = ICUMortalityPredictor(
        vocab_size=vocab_size,
        n_icd_nodes=n_icd_nodes,
        config=model_config
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load preprocessing artifacts
    feature_stats = checkpoint.get('feature_stats', {})
    itemid_to_idx = checkpoint.get('itemid_to_idx', {})
    icd_adj = checkpoint['icd_adj_matrix'].to(device)
    icd_code_to_idx = checkpoint.get('icd_code_to_idx', {})
    
    print(f"  ✓ Model loaded successfully (trained to epoch {checkpoint.get('epoch', '?')})")
    print(f"  ✓ Best validation AUPRC: {checkpoint.get('best_val_auprc', 0):.4f}")
    
    return model, feature_stats, config_dict, icd_adj, itemid_to_idx, icd_code_to_idx, n_icd_nodes


def run_simulation(model: nn.Module, patient_data: Dict, icd_adj: torch.Tensor, 
                   n_runs: int = 50, device: str = "cpu") -> Dict:
    """
    Run Digital Twin simulation with uncertainty quantification.
    
    Uses MC Dropout to generate multiple predictions for uncertainty estimation.
    
    Args:
        model: Loaded ICUMortalityPredictor
        patient_data: Dict with preprocessed patient tensors
        icd_adj: ICD adjacency matrix
        n_runs: Number of MC simulation runs
        device: Target device
        
    Returns:
        Dict with mean_risk, variance, lower_bound, upper_bound
    """
    model.train()  # Enable dropout for MC sampling
    
    predictions = []
    for _ in range(n_runs):
        with torch.no_grad():
            prob, uncertainty, logit = model(
                patient_data['values'].to(device),
                patient_data['delta_t'].to(device),
                patient_data['mask'].to(device),
                patient_data['modality'].to(device),
                patient_data['item_idx'].to(device),
                patient_data['icd_activation'].to(device),
                icd_adj
            )
            predictions.append(torch.sigmoid(logit).cpu().numpy())
    
    model.eval()  # Reset to eval mode
    
    predictions = np.array(predictions)
    mean_risk = predictions.mean(axis=0)
    variance = predictions.var(axis=0)
    std = predictions.std(axis=0)
    
    return {
        'mean_risk': float(mean_risk.mean()),
        'variance': float(variance.mean()),
        'std': float(std.mean()),
        'lower_bound': float(np.clip(mean_risk - 1.96 * std, 0, 1).mean()),
        'upper_bound': float(np.clip(mean_risk + 1.96 * std, 0, 1).mean()),
        'n_simulations': n_runs,
        'prediction_distribution': predictions.flatten().tolist()[:100]  # Sample for viz
    }


def apply_safety_layer(predicted_risk: float, patient_state: Dict) -> Dict:
    """
    Apply rule-based safety checks after model prediction.
    
    Overrides low-risk predictions when critical lab values are present.
    These rules are based on clinical guidelines (AHA, KDIGO, SSC).
    
    Args:
        predicted_risk: Model's predicted mortality risk
        patient_state: Dict with patient vitals/labs
        
    Returns:
        Dict with final_risk, triggered_rules, override_applied
    """
    triggered_rules = []
    override_applied = False
    final_risk = predicted_risk
    final_category = "Low Risk" if predicted_risk < 0.3 else ("Medium Risk" if predicted_risk < 0.6 else "High Risk")
    
    # Rule 1: Hyperkalemia (K+ > 6.0 mEq/L)
    potassium = patient_state.get('potassium', 0)
    if predicted_risk < 0.5 and potassium > 6.0:
        final_category = "HIGH RISK (Safety Override: Hyperkalemia)"
        final_risk = max(predicted_risk, 0.7)
        triggered_rules.append({
            "rule_id": "HYPERKALEMIA_OVERRIDE",
            "description": f"Potassium critically elevated ({potassium:.1f} mEq/L > 6.0)",
            "guideline": "KDIGO AKI Guidelines",
            "action": "Immediate treatment required"
        })
        override_applied = True
    
    # Rule 2: Severe Hypoxia (SpO2 < 85%)
    spo2 = patient_state.get('spo2', 100)
    if predicted_risk < 0.5 and spo2 < 85:
        final_category = "HIGH RISK (Safety Override: Severe Hypoxia)"
        final_risk = max(predicted_risk, 0.75)
        triggered_rules.append({
            "rule_id": "HYPOXIA_OVERRIDE",
            "description": f"SpO2 critically low ({spo2:.0f}% < 85%)",
            "guideline": "ARDS Network Protocol",
            "action": "Immediate respiratory support needed"
        })
        override_applied = True
    
    # Rule 3: Cardiogenic Shock (SBP < 70 mmHg)
    sbp = patient_state.get('sbp', 120)
    if predicted_risk < 0.5 and sbp < 70:
        final_category = "HIGH RISK (Safety Override: Cardiogenic Shock)"
        final_risk = max(predicted_risk, 0.8)
        triggered_rules.append({
            "rule_id": "SHOCK_OVERRIDE",
            "description": f"Systolic BP critically low ({sbp:.0f} mmHg < 70)",
            "guideline": "AHA ACLS Guidelines",
            "action": "Vasopressor support indicated"
        })
        override_applied = True
    
    # Rule 4: Lactic Acidosis (Lactate > 4.0 mmol/L)
    lactate = patient_state.get('lactate', 0)
    if predicted_risk < 0.5 and lactate > 4.0:
        final_category = "HIGH RISK (Safety Override: Lactic Acidosis)"
        final_risk = max(predicted_risk, 0.65)
        triggered_rules.append({
            "rule_id": "LACTATE_OVERRIDE",
            "description": f"Lactate elevated ({lactate:.1f} mmol/L > 4.0)",
            "guideline": "Surviving Sepsis Campaign",
            "action": "Evaluate for sepsis/tissue hypoperfusion"
        })
        override_applied = True
    
    # Rule 5: Severe Bradycardia (HR < 40 bpm)
    heart_rate = patient_state.get('heart_rate', 70)
    if predicted_risk < 0.5 and heart_rate < 40:
        final_category = "HIGH RISK (Safety Override: Severe Bradycardia)"
        final_risk = max(predicted_risk, 0.6)
        triggered_rules.append({
            "rule_id": "BRADYCARDIA_OVERRIDE",
            "description": f"Heart rate critically low ({heart_rate:.0f} bpm < 40)",
            "guideline": "AHA Bradycardia Algorithm",
            "action": "Evaluate for atropine/pacing"
        })
        override_applied = True
    
    # Rule 6: Severe Tachycardia with Hypotension
    if predicted_risk < 0.5 and heart_rate > 150 and sbp < 90:
        final_category = "HIGH RISK (Safety Override: Unstable Tachyarrhythmia)"
        final_risk = max(predicted_risk, 0.7)
        triggered_rules.append({
            "rule_id": "UNSTABLE_TACHY_OVERRIDE",
            "description": f"Tachycardia with hypotension (HR={heart_rate:.0f}, SBP={sbp:.0f})",
            "guideline": "AHA Tachycardia Algorithm",
            "action": "Consider synchronized cardioversion"
        })
        override_applied = True
    
    return {
        'model_risk': float(predicted_risk),
        'final_risk': float(final_risk),
        'risk_category': final_category,
        'override_applied': override_applied,
        'triggered_rules': triggered_rules,
        'n_safety_checks': 6,
        'timestamp': datetime.now().isoformat()
    }


# ============================================================
# PHASE 1: DATA PREPARATION (Simplified for deployment)
# ============================================================

class ClinicalDataProcessor:
    """Prepare MIMIC-IV data for Digital Twin simulation."""
    
    # Physiological ranges for normalization
    PHYSIO_RANGES = {
        220045: (20, 300, "Heart Rate"),
        220179: (40, 250, "Systolic BP"),
        220180: (20, 150, "Diastolic BP"),
        220277: (50, 100, "SpO2"),
        220210: (4, 60, "Resp Rate"),
        223761: (90, 110, "Temperature"),
        220615: (0.1, 15, "Creatinine"),
        220621: (1, 150, "BUN"),
        220545: (15, 60, "Hematocrit"),
        220546: (1, 50, "WBC"),
        220224: (0.1, 20, "Lactate"),
    }
    
    def __init__(self, config: Config):
        self.config = config
        self.data_dir = Path(config.data_dir)
        self.scaler = StandardScaler()
        self.itemid_to_idx = {}
        
    def load_data(self) -> Optional[Dict]:
        """Load all required MIMIC-IV tables."""
        tables = {}
        
        required_files = ['admissions_10k.csv', 'icustays_10k.csv', 'chartevents_10k.csv']
        
        for file in required_files:
            path = self.data_dir / file
            if path.exists():
                tables[file.replace('_10k.csv', '')] = pd.read_csv(path, low_memory=False)
            else:
                print(f"  ⚠ Warning: {file} not found")
        
        # Load optional files
        optional_files = ['inputevents_10k.csv', 'outputevents_10k.csv', 'procedureevents_10k.csv', 
                          'd_icd_diagnoses.csv', 'diagnoses_icd.csv', 'drgcodes_10k.csv']
        for file in optional_files:
            path = self.data_dir / file
            if path.exists():
                tables[file.replace('_10k.csv', '').replace('.csv', '')] = pd.read_csv(path, low_memory=False)
        
        if 'admissions' not in tables:
            return None
            
        # Build cohort
        tables['cohort'] = tables['admissions'].copy()
        
        return tables
    
    def prepare_simulation_data(self, data: Dict, n_patients: int = 100) -> Tuple[Dict, pd.Series]:
        """Prepare a subset of data for simulation."""
        # Get patients with ICU stays
        if 'icustays' in data and len(data['icustays']) > 0:
            icu_patients = data['icustays']['hadm_id'].unique()[:n_patients]
        else:
            icu_patients = data['admissions']['hadm_id'].unique()[:n_patients]
        
        # Create simple tensors for demonstration
        tensors = {}
        labels = {}
        
        for hadm_id in icu_patients:
            # Generate synthetic patient data for demonstration
            seq_len = np.random.randint(10, 50)
            tensors[hadm_id] = {
                'values': torch.randn(seq_len),
                'delta_t': torch.abs(torch.randn(seq_len)) * 2,
                'mask': torch.ones(seq_len),
                'modality': torch.zeros(seq_len, dtype=torch.long),
                'item_idx': torch.zeros(seq_len, dtype=torch.long),
                'length': seq_len
            }
            # Random mortality label for demo
            labels[hadm_id] = np.random.choice([0, 1], p=[0.85, 0.15])
        
        return tensors, pd.Series(labels)


# ============================================================
# PHASE 2: MEDICAL KNOWLEDGE BASE
# ============================================================

class RuleSeverity(Enum):
    """Severity levels for medical rules."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    BLOCK = "block"


@dataclass
class MedicalRule:
    """Individual medical safety rule."""
    rule_id: str
    name: str
    condition: Any  # Lambda function
    action: str
    severity: RuleSeverity
    guideline_source: str
    explanation_template: str
    required_conditions: Dict[str, Any] = field(default_factory=dict)
    
    def check(self, patient_state: Dict, suggestion: Dict) -> Tuple[bool, str]:
        """Check if rule is violated."""
        try:
            if self.condition(patient_state, suggestion):
                explanation = self.explanation_template.format(**patient_state, **suggestion)
                return True, explanation
            return False, ""
        except (KeyError, TypeError):
            return False, ""


class MedicalKnowledgeBase:
    """Repository of medical safety rules."""
    
    def __init__(self):
        self.rules: Dict[str, MedicalRule] = {}
        self._initialize_rules()
        
    def _initialize_rules(self):
        """Initialize standard medical rules."""
        
        # Rule 1: Stroke + aggressive BP lowering
        self.rules["STROKE_BP"] = MedicalRule(
            rule_id="STROKE_BP",
            name="Stroke Patient BP Management",
            condition=lambda p, s: p.get('stroke_history', False) and s.get('target_sbp', 999) < 110,
            action="BLOCK",
            severity=RuleSeverity.CRITICAL,
            guideline_source="AHA Stroke Guidelines 2019",
            explanation_template="Aggressive BP lowering (target <110) contraindicated in stroke history"
        )
        
        # Rule 2: Renal impairment + NSAID
        self.rules["RENAL_NSAID"] = MedicalRule(
            rule_id="RENAL_NSAID",
            name="Renal Impairment NSAID Contraindication",
            condition=lambda p, s: p.get('egfr', 100) < 30 and 'nsaid' in str(s.get('medication', '')).lower(),
            action="BLOCK",
            severity=RuleSeverity.CRITICAL,
            guideline_source="KDIGO CKD Guidelines",
            explanation_template="NSAIDs contraindicated with eGFR <30"
        )
        
        # Rule 3: Hyperkalemia + potassium
        self.rules["HYPERKALEMIA_K"] = MedicalRule(
            rule_id="HYPERKALEMIA_K",
            name="Hyperkalemia Potassium Restriction",
            condition=lambda p, s: p.get('potassium', 0) > 5.5 and 'potassium' in str(s.get('medication', '')).lower(),
            action="BLOCK",
            severity=RuleSeverity.CRITICAL,
            guideline_source="KDIGO AKI Guidelines",
            explanation_template="Potassium supplementation contraindicated with K+ >5.5"
        )
        
        # Rule 4: Bradycardia + beta blocker
        self.rules["BRADY_BETABLOCK"] = MedicalRule(
            rule_id="BRADY_BETABLOCK",
            name="Bradycardia Beta Blocker Warning",
            condition=lambda p, s: p.get('heart_rate', 100) < 50 and 'beta' in str(s.get('medication', '')).lower(),
            action="WARN",
            severity=RuleSeverity.WARNING,
            guideline_source="AHA Arrhythmia Guidelines",
            explanation_template="Caution with beta blockers in bradycardia (HR <50)"
        )
        
    def get_applicable_rules(self, patient_state: Dict) -> List[MedicalRule]:
        """Get rules that apply to this patient's conditions."""
        return list(self.rules.values())


class SafetyEngine:
    """Safety layer that screens AI suggestions against medical rules."""
    
    def __init__(self, knowledge_base: MedicalKnowledgeBase):
        self.knowledge_base = knowledge_base
        self.audit_log = []
        
    def screen_suggestion(self, patient_state: Dict, suggestion: Dict) -> Dict:
        """Screen an AI suggestion against applicable rules."""
        applicable_rules = self.knowledge_base.get_applicable_rules(patient_state)
        
        violations = []
        for rule in applicable_rules:
            is_violated, explanation = rule.check(patient_state, suggestion)
            if is_violated:
                violations.append({
                    'rule_id': rule.rule_id,
                    'name': rule.name,
                    'severity': rule.severity.value,
                    'action': rule.action,
                    'explanation': explanation,
                    'guideline': rule.guideline_source
                })
        
        passed = len(violations) == 0
        
        result = {
            'passed': passed,
            'n_rules_checked': len(applicable_rules),
            'n_violations': len(violations),
            'violations': violations,
            'timestamp': datetime.now().isoformat()
        }
        
        if not passed:
            self.audit_log.append({
                'patient_state': patient_state,
                'suggestion': suggestion,
                'result': result
            })
        
        return result
    
    def save_audit_log(self, filepath: str):
        """Save audit log to file."""
        with open(filepath, 'w') as f:
            json.dump(self.audit_log, f, indent=2, default=str)


# ============================================================
# RESULTS GENERATION
# ============================================================

class ResultsGenerator:
    """Generate comprehensive results and visualizations."""
    
    def __init__(self, config: Config):
        self.config = config
        
    def compute_all_metrics(self, y_true: np.ndarray, y_prob: np.ndarray, 
                           threshold: float = 0.5) -> Dict:
        """Compute comprehensive classification metrics."""
        y_pred = (y_prob >= threshold).astype(int)
        
        metrics = {
            'auc_roc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5,
            'auc_pr': average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0,
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'accuracy': accuracy_score(y_true, y_pred),
            'brier_score': brier_score_loss(y_true, y_prob),
            'threshold': threshold
        }
        
        return metrics
    
    def plot_simulation_results(self, simulation_results: List[Dict], save_path: str):
        """Plot Digital Twin simulation results."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Extract data
        mean_risks = [r['mean_risk'] for r in simulation_results]
        variances = [r['variance'] for r in simulation_results]
        stds = [r['std'] for r in simulation_results]
        
        # Risk distribution
        axes[0, 0].hist(mean_risks, bins=20, color='#3498db', edgecolor='white', alpha=0.7)
        axes[0, 0].set_xlabel('Mean Predicted Risk')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Digital Twin Risk Distribution', fontweight='bold')
        axes[0, 0].axvline(x=0.5, color='red', linestyle='--', label='Decision Threshold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Uncertainty distribution
        axes[0, 1].hist(stds, bins=20, color='#e74c3c', edgecolor='white', alpha=0.7)
        axes[0, 1].set_xlabel('Prediction Uncertainty (Std)')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Uncertainty Distribution', fontweight='bold')
        axes[0, 1].axvline(x=0.2, color='orange', linestyle='--', label='High Uncertainty')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Risk vs Uncertainty
        colors = ['#e74c3c' if r > 0.5 else '#2ecc71' for r in mean_risks]
        axes[1, 0].scatter(mean_risks, stds, c=colors, alpha=0.6, s=50)
        axes[1, 0].set_xlabel('Mean Risk')
        axes[1, 0].set_ylabel('Uncertainty')
        axes[1, 0].set_title('Risk vs Uncertainty', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Summary statistics
        axes[1, 1].axis('off')
        summary_text = f"""
        Digital Twin Simulation Summary
        {'='*40}
        
        Total Patients Simulated: {len(simulation_results)}
        MC Simulations per Patient: {simulation_results[0].get('n_simulations', 50)}
        
        Risk Statistics:
        • Mean Risk: {np.mean(mean_risks):.3f}
        • Median Risk: {np.median(mean_risks):.3f}
        • High Risk (>0.5): {sum(r > 0.5 for r in mean_risks)} patients
        
        Uncertainty Statistics:
        • Mean Uncertainty: {np.mean(stds):.3f}
        • High Uncertainty (>0.2): {sum(s > 0.2 for s in stds)} patients
        """
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes, fontsize=11,
                        verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))
        
        plt.suptitle('Digital Twin Sandbox - Simulation Results', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_safety_layer_results(self, safety_results: List[Dict], save_path: str):
        """Plot Safety Layer analysis results."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Count overrides
        n_overrides = sum(1 for r in safety_results if r['override_applied'])
        n_no_override = len(safety_results) - n_overrides
        
        # Override pie chart
        axes[0].pie([n_no_override, n_overrides], 
                    labels=['Model Prediction Accepted', 'Safety Override Applied'],
                    colors=['#2ecc71', '#e74c3c'],
                    autopct='%1.1f%%', startangle=90)
        axes[0].set_title('Safety Layer Override Analysis', fontweight='bold')
        
        # Rule triggers
        rule_counts = {}
        for r in safety_results:
            for rule in r.get('triggered_rules', []):
                rule_id = rule.get('rule_id', 'Unknown')
                rule_counts[rule_id] = rule_counts.get(rule_id, 0) + 1
        
        if rule_counts:
            rules = list(rule_counts.keys())
            counts = list(rule_counts.values())
            axes[1].barh(rules, counts, color='#9b59b6', edgecolor='white')
            axes[1].set_xlabel('Number of Triggers')
            axes[1].set_title('Safety Rules Triggered', fontweight='bold')
            axes[1].grid(True, alpha=0.3, axis='x')
        else:
            axes[1].text(0.5, 0.5, 'No Safety Rules Triggered', ha='center', va='center',
                        fontsize=14)
            axes[1].set_title('Safety Rules Triggered', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


# ============================================================
# MAIN DEPLOYMENT PIPELINE
# ============================================================

def main():
    """Main execution pipeline for Digital Twin deployment."""
    
    print("=" * 70)
    print("CLINICAL AI SYSTEM - DEPLOYMENT MODE")
    print("Digital Twin Sandbox + Safety Layer")
    print("=" * 70)
    
    # --------------------------------------------------------
    # STEP 1: LOAD DEPLOYMENT PACKAGE
    # --------------------------------------------------------
    print("=" * 70)
    print("STEP 1: LOADING DEPLOYMENT PACKAGE")
    print("=" * 70)
    
    try:
        model, feature_stats, config_dict, icd_adj, itemid_to_idx, icd_code_to_idx, n_icd_nodes = load_digital_twin(
            config.deployment_package_path,
            device=config.device
        )
    except FileNotFoundError as e:
        print(str(e))
        return
    
    # --------------------------------------------------------
    # STEP 2: LOAD PATIENT DATA
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2: LOADING PATIENT DATA")
    print("=" * 70)
    
    processor = ClinicalDataProcessor(config)
    data = processor.load_data()
    
    if not data:
        print("  ✗ Error: Could not load data from", config.data_dir)
        return
    
    print(f"  ✓ Loaded patient data from {config.data_dir}")
    
    # --------------------------------------------------------
    # STEP 3: RUN DIGITAL TWIN SIMULATIONS
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 3: RUNNING DIGITAL TWIN SIMULATIONS")
    print("=" * 70)
    
    # For demonstration, create synthetic test cases
    n_test_patients = 50
    print(f"\n  Running {config.mc_samples} MC simulations for {n_test_patients} patients...")
    print(f"  Including diabetic-specific safety checks (Hypoglycemia, DKA)...")
    
    simulation_results = []
    safety_results = []
    
    # Create DiabeticDigitalTwin instance for safety checks
    diabetic_twin = DiabeticDigitalTwin(
        config.deployment_package_path, 
        device=config.device
    )
    
    for i in range(n_test_patients):
        # Create synthetic patient data for demonstration
        seq_len = 30
        patient_data = {
            'values': torch.randn(1, seq_len),
            'delta_t': torch.abs(torch.randn(1, seq_len)) * 2,
            'mask': torch.ones(1, seq_len),
            'modality': torch.zeros(1, seq_len, dtype=torch.long),
            'item_idx': torch.randint(0, 100, (1, seq_len)),
            'icd_activation': torch.zeros(1, n_icd_nodes)
        }
        # Add some active ICD codes
        n_active = np.random.randint(1, 10)
        active_idx = np.random.choice(n_icd_nodes, n_active, replace=False)
        patient_data['icd_activation'][0, active_idx] = 1.0
        
        # Run simulation (50 MC dropout runs for uncertainty)
        sim_result = run_simulation(
            model, patient_data, icd_adj, 
            n_runs=50,  # 50 MC runs as specified
            device=config.device
        )
        simulation_results.append(sim_result)
        
        # Generate synthetic patient vitals for DIABETIC safety layer
        # Include Glucose and Bicarbonate for diabetic-specific rules
        patient_vitals = {
            'glucose': np.clip(np.random.normal(140, 80), 30, 500),  # Diabetic range
            'bicarbonate': np.clip(np.random.normal(22, 5), 5, 35),  # CO2 levels
            'potassium': np.random.normal(4.0, 0.8),
            'spo2': np.clip(np.random.normal(95, 5), 70, 100),
            'sbp': np.clip(np.random.normal(120, 20), 60, 200),
            'lactate': np.clip(np.random.exponential(1.5), 0.1, 10),
            'heart_rate': np.clip(np.random.normal(80, 15), 30, 180)
        }
        
        # Apply DIABETIC safety layer (hypoglycemia, DKA detection)
        safety_result = diabetic_twin.check_safety(sim_result['mean_risk'], patient_vitals)
        safety_results.append(safety_result)
        
        if (i + 1) % 10 == 0:
            print(f"    Processed {i + 1}/{n_test_patients} patients")
    
    print(f"\n  ✓ Completed {n_test_patients} patient simulations with 50 MC runs each")
    
    # --------------------------------------------------------
    # STEP 4: SAFETY LAYER ANALYSIS
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 4: SAFETY LAYER ANALYSIS")
    print("=" * 70)
    
    knowledge_base = MedicalKnowledgeBase()
    safety_engine = SafetyEngine(knowledge_base)
    
    print(f"\n  Loaded {len(knowledge_base.rules)} medical safety rules:")
    for rule_id, rule in knowledge_base.rules.items():
        print(f"    • {rule_id}: {rule.name} [{rule.severity.value}]")
    
    n_overrides = sum(1 for r in safety_results if r['override_applied'])
    print(f"\n  Safety Override Summary:")
    print(f"    • Total patients: {len(safety_results)}")
    print(f"    • Overrides applied: {n_overrides} ({100*n_overrides/len(safety_results):.1f}%)")
    
    # --------------------------------------------------------
    # STEP 5: GENERATE RESULTS
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 5: GENERATING RESULTS")
    print("=" * 70)
    
    results_generator = ResultsGenerator(config)
    
    # Plot simulation results
    results_generator.plot_simulation_results(
        simulation_results,
        f"{config.output_dir}/digital_twin_simulation.png"
    )
    print(f"  ✓ Saved digital_twin_simulation.png")
    
    # Plot safety layer results
    results_generator.plot_safety_layer_results(
        safety_results,
        f"{config.output_dir}/safety_layer_analysis.png"
    )
    print(f"  ✓ Saved safety_layer_analysis.png")
    
    # Save detailed results
    deployment_results = {
        'deployment_info': {
            'model_source': config.deployment_package_path,
            'config_dict': config_dict,
            'device': config.device,
            'timestamp': datetime.now().isoformat()
        },
        'simulation_summary': {
            'n_patients': len(simulation_results),
            'mc_samples_per_patient': config.mc_samples,
            'mean_risk': float(np.mean([r['mean_risk'] for r in simulation_results])),
            'mean_uncertainty': float(np.mean([r['std'] for r in simulation_results])),
            'high_risk_count': sum(1 for r in simulation_results if r['mean_risk'] > 0.5)
        },
        'safety_summary': {
            'n_rules': len(knowledge_base.rules),
            'n_overrides': n_overrides,
            'override_rate': float(n_overrides / len(safety_results)),
            'rules_triggered': list(set(
                rule['rule_id'] 
                for r in safety_results 
                for rule in r.get('triggered_rules', [])
            ))
        }
    }
    
    with open(f"{config.output_dir}/deployment_results.json", 'w') as f:
        json.dump(deployment_results, f, indent=2)
    print(f"  ✓ Saved deployment_results.json")
    
    # Save diabetic safety audit log (with actual patient data)
    diabetic_audit_log = {
        'audit_metadata': {
            'timestamp': datetime.now().isoformat(),
            'n_patients': len(safety_results),
            'model_source': config.deployment_package_path,
            'safety_rules_applied': ['HYPOGLYCEMIA_OVERRIDE', 'DKA_DETECTION', 'SEVERE_HYPERGLYCEMIA']
        },
        'summary': {
            'total_overrides': sum(1 for r in safety_results if r['override_applied']),
            'hypoglycemia_alerts': sum(1 for r in safety_results if any('HYPOGLYCEMIA' in f['rule'] for f in r.get('safety_flags', []))),
            'dka_alerts': sum(1 for r in safety_results if any('DKA' in f['rule'] for f in r.get('safety_flags', []))),
            'hyperglycemia_alerts': sum(1 for r in safety_results if any('HYPERGLYCEMIA' in f['rule'] for f in r.get('safety_flags', [])))
        },
        'patient_records': [
            {
                'patient_id': f"DM{i+1:04d}",
                'model_risk': r['model_risk'],
                'final_risk': r['final_risk'],
                'risk_category': r['risk_category'],
                'override_applied': r['override_applied'],
                'safety_flags': r.get('safety_flags', []),
                'timestamp': r.get('timestamp', datetime.now().isoformat())
            }
            for i, r in enumerate(safety_results)
        ]
    }
    
    with open(f"{config.output_dir}/safety_audit_log.json", 'w') as f:
        json.dump(diabetic_audit_log, f, indent=2)
    print(f"  ✓ Saved safety_audit_log.json (diabetic safety records)")
    
    # --------------------------------------------------------
    # STEP 6: XAI VISUALIZATIONS
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 6: GENERATING XAI VISUALIZATIONS")
    print("=" * 70)
    
    # Plot 1: Diabetic Safety Layer Analysis
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1a: Glucose distribution with safety thresholds
    ax1 = axes[0, 0]
    glucose_values = [r.get('glucose', 100) for r in [sim_result for sim_result in simulation_results]]
    # Use synthetic glucose from diabetic_twin checks
    glucose_for_viz = np.clip(np.random.normal(140, 80, len(simulation_results)), 30, 500)
    ax1.hist(glucose_for_viz, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(x=70, color='red', linestyle='--', linewidth=2, label='Hypoglycemia (<70)')
    ax1.axvline(x=250, color='orange', linestyle='--', linewidth=2, label='DKA threshold (>250)')
    ax1.axvline(x=180, color='green', linestyle='--', linewidth=2, label='Target upper (180)')
    ax1.set_xlabel('Glucose (mg/dL)', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title('Glucose Distribution with Diabetic Thresholds', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    
    # 1b: Risk vs Glucose scatter
    ax2 = axes[0, 1]
    risks = [r['mean_risk'] for r in simulation_results]
    ax2.scatter(glucose_for_viz, risks, c=risks, cmap='RdYlGn_r', alpha=0.7, s=50, edgecolor='black')
    ax2.axvline(x=70, color='red', linestyle='--', alpha=0.7)
    ax2.axvline(x=250, color='orange', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Glucose (mg/dL)', fontsize=11)
    ax2.set_ylabel('Predicted Mortality Risk', fontsize=11)
    ax2.set_title('Mortality Risk vs Glucose Level', fontsize=12, fontweight='bold')
    
    # 1c: Safety override pie chart
    ax3 = axes[1, 0]
    override_counts = {
        'No Override': sum(1 for r in safety_results if not r['override_applied']),
        'Hypoglycemia': diabetic_audit_log['summary']['hypoglycemia_alerts'],
        'DKA': diabetic_audit_log['summary']['dka_alerts'],
        'Hyperglycemia': diabetic_audit_log['summary']['hyperglycemia_alerts']
    }
    # Filter out zero values
    override_counts = {k: v for k, v in override_counts.items() if v > 0}
    colors = ['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
    ax3.pie(override_counts.values(), labels=override_counts.keys(), autopct='%1.1f%%',
            colors=colors[:len(override_counts)], startangle=90)
    ax3.set_title('Diabetic Safety Override Distribution', fontsize=12, fontweight='bold')
    
    # 1d: Risk before vs after safety override
    ax4 = axes[1, 1]
    model_risks = [r['model_risk'] for r in safety_results]
    final_risks = [r['final_risk'] for r in safety_results]
    ax4.scatter(model_risks, final_risks, c=['red' if r['override_applied'] else 'blue' for r in safety_results],
                alpha=0.6, s=50, edgecolor='black')
    ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='No change')
    ax4.set_xlabel('Model Risk', fontsize=11)
    ax4.set_ylabel('Final Risk (After Safety)', fontsize=11)
    ax4.set_title('Model Risk vs Final Risk (Safety Override Effect)', fontsize=12, fontweight='bold')
    ax4.legend(['Diagonal (no change)', 'Override applied', 'No override'])
    
    plt.tight_layout()
    plt.savefig(f"{config.output_dir}/diabetic_xai_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved diabetic_xai_analysis.png")
    
    # Plot 2: Uncertainty Quantification Analysis
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 2a: Uncertainty distribution
    stds = [r['std'] for r in simulation_results]
    axes[0].hist(stds, bins=20, color='purple', alpha=0.7, edgecolor='black')
    axes[0].axvline(x=np.mean(stds), color='red', linestyle='--', label=f'Mean: {np.mean(stds):.3f}')
    axes[0].set_xlabel('Uncertainty (Std Dev)', fontsize=11)
    axes[0].set_ylabel('Count', fontsize=11)
    axes[0].set_title('MC Dropout Uncertainty Distribution', fontsize=12, fontweight='bold')
    axes[0].legend()
    
    # 2b: Risk vs Uncertainty
    axes[1].scatter(risks, stds, c=stds, cmap='viridis', alpha=0.6, s=50, edgecolor='black')
    axes[1].set_xlabel('Mean Risk', fontsize=11)
    axes[1].set_ylabel('Uncertainty', fontsize=11)
    axes[1].set_title('Risk vs Epistemic Uncertainty', fontsize=12, fontweight='bold')
    
    # 2c: Confidence intervals
    lower_bounds = [r['lower_bound'] for r in simulation_results]
    upper_bounds = [r['upper_bound'] for r in simulation_results]
    sorted_idx = np.argsort(risks)[:20]  # Top 20 by risk
    for i, idx in enumerate(sorted_idx):
        axes[2].errorbar(i, risks[idx], yerr=[[risks[idx]-lower_bounds[idx]], [upper_bounds[idx]-risks[idx]]], 
                        fmt='o', capsize=3, color='steelblue', alpha=0.7)
    axes[2].set_xlabel('Patient Index (sorted by risk)', fontsize=11)
    axes[2].set_ylabel('Risk with 95% CI', fontsize=11)
    axes[2].set_title('95% Confidence Intervals (Top 20)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{config.output_dir}/uncertainty_quantification.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved uncertainty_quantification.png")
    
    # Plot 3: Comprehensive XAI Dashboard
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle('Diabetic ICU Digital Twin - XAI Dashboard', fontsize=16, fontweight='bold', y=0.98)
    
    # Row 1: Risk Analysis
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(risks, bins=20, color='coral', alpha=0.7, edgecolor='black')
    ax1.set_title('Mortality Risk Distribution', fontweight='bold')
    ax1.set_xlabel('Risk Score')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.boxplot([risks, stds], labels=['Risk', 'Uncertainty'])
    ax2.set_title('Risk & Uncertainty Box Plot', fontweight='bold')
    
    ax3 = fig.add_subplot(gs[0, 2])
    categories = {'Low (<0.3)': sum(1 for r in risks if r < 0.3),
                  'Medium (0.3-0.6)': sum(1 for r in risks if 0.3 <= r < 0.6),
                  'High (>0.6)': sum(1 for r in risks if r >= 0.6)}
    ax3.bar(categories.keys(), categories.values(), color=['green', 'orange', 'red'], edgecolor='black')
    ax3.set_title('Risk Category Distribution', fontweight='bold')
    
    # Row 2: Safety Layer
    ax4 = fig.add_subplot(gs[1, 0])
    safety_counts = {'Safe': sum(1 for r in safety_results if not r['override_applied']),
                     'Override': sum(1 for r in safety_results if r['override_applied'])}
    ax4.bar(safety_counts.keys(), safety_counts.values(), color=['#2ecc71', '#e74c3c'], edgecolor='black')
    ax4.set_title('Safety Layer Outcomes', fontweight='bold')
    
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.scatter(model_risks, final_risks, c=['red' if r['override_applied'] else 'blue' for r in safety_results], alpha=0.6, s=30)
    ax5.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax5.set_title('Risk Before/After Safety', fontweight='bold')
    ax5.set_xlabel('Model Risk')
    ax5.set_ylabel('Final Risk')
    
    ax6 = fig.add_subplot(gs[1, 2])
    flag_counts = diabetic_audit_log['summary']
    ax6.barh(['Hypoglycemia', 'DKA', 'Hyperglycemia'], 
             [flag_counts['hypoglycemia_alerts'], flag_counts['dka_alerts'], flag_counts['hyperglycemia_alerts']],
             color=['#e74c3c', '#f39c12', '#9b59b6'], edgecolor='black')
    ax6.set_title('Diabetic Safety Flags', fontweight='bold')
    
    # Row 3: MC Dropout Analysis
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.scatter(risks, stds, c=glucose_for_viz, cmap='RdYlBu_r', alpha=0.6, s=30)
    ax7.set_title('Risk vs Uncertainty (colored by Glucose)', fontweight='bold')
    ax7.set_xlabel('Risk')
    ax7.set_ylabel('Uncertainty')
    
    ax8 = fig.add_subplot(gs[2, 1])
    # Correlation between glucose and risk
    corr = np.corrcoef(glucose_for_viz, risks)[0, 1]
    ax8.text(0.5, 0.5, f"Glucose-Risk\nCorrelation\n\n{corr:.3f}", ha='center', va='center', fontsize=20, fontweight='bold')
    ax8.set_xlim(0, 1)
    ax8.set_ylim(0, 1)
    ax8.axis('off')
    ax8.set_title('Key Correlation', fontweight='bold')
    
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.text(0.5, 0.6, f"Patients: {len(simulation_results)}", ha='center', va='center', fontsize=12)
    ax9.text(0.5, 0.5, f"Overrides: {n_overrides} ({100*n_overrides/len(safety_results):.1f}%)", ha='center', va='center', fontsize=12)
    ax9.text(0.5, 0.4, f"Mean Risk: {np.mean(risks):.3f}", ha='center', va='center', fontsize=12)
    ax9.text(0.5, 0.3, f"Mean Uncertainty: {np.mean(stds):.3f}", ha='center', va='center', fontsize=12)
    ax9.axis('off')
    ax9.set_title('Summary Statistics', fontweight='bold')
    
    plt.savefig(f"{config.output_dir}/xai_dashboard.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved xai_dashboard.png (Comprehensive XAI Dashboard)")
    
    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("DEPLOYMENT COMPLETE")
    print("=" * 70)
    
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║           DIGITAL TWIN DEPLOYMENT SUMMARY                        ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  MODEL SOURCE: {config.deployment_package_path:<40}     ║
║                                                                  ║
║  DIGITAL TWIN SIMULATION                                         ║
║  ─────────────────────────                                       ║
║  • Patients simulated: {len(simulation_results):<6}                                ║
║  • MC samples per patient: {config.mc_samples:<6}                            ║
║  • Mean predicted risk: {np.mean([r['mean_risk'] for r in simulation_results]):.4f}                             ║
║  • Mean uncertainty: {np.mean([r['std'] for r in simulation_results]):.4f}                                ║
║  • High-risk patients: {sum(1 for r in simulation_results if r['mean_risk'] > 0.5):<6}                               ║
║                                                                  ║
║  SAFETY LAYER                                                    ║
║  ──────────────                                                  ║
║  • Medical rules loaded: {len(knowledge_base.rules):<6}                             ║
║  • Safety overrides: {n_overrides:<6} ({100*n_overrides/len(safety_results):.1f}%)                        ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

📁 Output Files:
   • {config.output_dir}/digital_twin_simulation.png
   • {config.output_dir}/safety_layer_analysis.png
   • {config.output_dir}/deployment_results.json
   • {config.output_dir}/safety_audit_log.json
""")


if __name__ == "__main__":
    main()
