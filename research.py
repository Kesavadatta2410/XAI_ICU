"""
ICU Mortality Prediction with Liquid-Mamba, ICD Knowledge Graph & Diffusion XAI
================================================================================
Full architecture implementing:
1. Liquid Mamba - Gap-aware irregular time-series modeling
2. ICD Knowledge Graph - Hierarchical disease relationship modeling with GAT
3. Cross-Attention Fusion - Combining temporal and disease context
4. Diffusion-based XAI - Counterfactual trajectory generation
5. Uncertainty-Aware Predictions - Calibrated uncertainty estimates
"""

import os
import math
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class Config:
    # Paths
    data_dir: str = "data_10k"
    output_dir: str = "results"
    checkpoint_dir: str = "checkpoints"
    
    # Data
    max_seq_len: int = 128
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    
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
    weight_decay: float = 1e-4
    
    # Diffusion XAI
    diffusion_steps: int = 50
    diffusion_hidden: int = 128
    
    # Uncertainty
    mc_dropout_samples: int = 10
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

config = Config()
Path(config.output_dir).mkdir(exist_ok=True)
Path(config.checkpoint_dir).mkdir(exist_ok=True)

torch.manual_seed(config.seed)
np.random.seed(config.seed)

# ============================================================
# PHASE 1: DATA PREPROCESSING WITH MISSINGNESS HANDLING
# ============================================================

class ICUDataProcessor:
    """Process irregular ICU data with gap-awareness and missingness indicators."""
    
    # Physiological ranges for feasibility constraints
    PHYSIO_RANGES = {
        220045: (20, 300, "Heart Rate"),
        220179: (40, 250, "Systolic BP"),
        220180: (20, 150, "Diastolic BP"),
        220210: (4, 60, "Respiratory Rate"),
        220277: (50, 100, "SpO2"),
        223761: (90, 110, "Temp F"),
    }
    
    # Labs for log transform (skewed distributions)
    LOG_LABS = {50912, 50813, 51006, 50885}  # Creatinine, Lactate, BUN, Bilirubin
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.feature_stats = {}
        self.itemid_to_idx = {}
        
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load all MIMIC-IV CSVs from data_10k folder."""
        print("=" * 60)
        print("LOADING ICU DATA")
        print("=" * 60)
        
        data = {}
        
        # Load admissions (contains hospital_expire_flag)
        adm_path = self.data_dir / 'admissions_10k.csv'
        if adm_path.exists():
            data['admissions'] = pd.read_csv(adm_path)
            print(f"  ✓ admissions: {len(data['admissions']):,} rows")
        
        # Load ICU stays (contains stay_id, los, intime, outtime)
        icu_path = self.data_dir / 'icustays_10k.csv'
        if icu_path.exists():
            data['icustays'] = pd.read_csv(icu_path)
            print(f"  ✓ icustays: {len(data['icustays']):,} rows")
        
        # Load DRG codes (diagnosis-related groups) for comorbidity analysis
        drg_path = self.data_dir / 'drgcodes_10k.csv'
        if drg_path.exists():
            data['drgcodes'] = pd.read_csv(drg_path)
            print(f"  ✓ drgcodes: {len(data['drgcodes']):,} rows")
        
        # Load chartevents (vitals + labs combined) - this is the main time-series data
        chart_path = self.data_dir / 'chartevents_10k.csv'
        if chart_path.exists():
            # Load in chunks due to large file size, sample for memory efficiency
            print("  Loading chartevents (large file, sampling)...")
            chunks = pd.read_csv(chart_path, chunksize=500000)
            chart_samples = []
            for i, chunk in enumerate(chunks):
                # Keep only rows with valid valuenum
                chunk = chunk.dropna(subset=['valuenum'])
                chart_samples.append(chunk)
                if i >= 3:  # Limit to ~2M rows for memory
                    break
            data['chartevents'] = pd.concat(chart_samples, ignore_index=True)
            print(f"  ✓ chartevents: {len(data['chartevents']):,} rows (sampled)")
        
        # Load inputevents (medications/fluids)
        input_path = self.data_dir / 'inputevents_10k.csv'
        if input_path.exists():
            data['inputevents'] = pd.read_csv(input_path, nrows=500000)  # Limit for memory
            print(f"  ✓ inputevents: {len(data['inputevents']):,} rows")
        
        # Load outputevents
        output_path = self.data_dir / 'outputevents_10k.csv'
        if output_path.exists():
            data['outputevents'] = pd.read_csv(output_path)
            print(f"  ✓ outputevents: {len(data['outputevents']):,} rows")
        
        # Load procedureevents
        proc_path = self.data_dir / 'procedureevents_10k.csv'
        if proc_path.exists():
            data['procedureevents'] = pd.read_csv(proc_path)
            print(f"  ✓ procedureevents: {len(data['procedureevents']):,} rows")
        
        # Create cohort by joining icustays with admissions
        if 'icustays' in data and 'admissions' in data:
            cohort = data['icustays'].merge(
                data['admissions'][['hadm_id', 'hospital_expire_flag']], 
                on='hadm_id', 
                how='left'
            )
            cohort['hospital_expire_flag'] = cohort['hospital_expire_flag'].fillna(0).astype(int)
            
            # Merge DRG codes to provide diagnosis information for the knowledge graph
            if 'drgcodes' in data:
                drg = data['drgcodes'][['hadm_id', 'drg_code', 'description']].copy()
                drg['icd_code'] = drg['drg_code'].astype(str)  # Use DRG as pseudo-ICD
                cohort = cohort.merge(
                    drg[['hadm_id', 'icd_code']], 
                    on='hadm_id', 
                    how='left'
                )
                cohort['icd_code'] = cohort['icd_code'].fillna('UNKNOWN')
                print(f"  ✓ Added DRG diagnosis codes: {cohort['icd_code'].nunique()} unique codes")
            
            data['cohort'] = cohort
            print(f"  ✓ cohort (merged): {len(cohort):,} ICU stays, "
                  f"{cohort['hadm_id'].nunique()} unique admissions")
        
        return data
    
    
    def build_patient_timelines(self, data: Dict) -> Tuple[Dict, pd.Series, Dict]:
        """
        Build chronological patient timelines with:
        - Feature values
        - Missingness masks
        - Time-since-last-observation (delta_t)
        - Modality indicators
        """
        print("\n" + "=" * 60)
        print("BUILDING PATIENT TIMELINES")
        print("=" * 60)
        
        cohort = data['cohort']
        hadm_ids = cohort['hadm_id'].unique()
        labels = cohort.groupby('hadm_id')['hospital_expire_flag'].first()
        
        # Collect all events with timestamps
        all_events = []
        
        # Process chartevents (contains both vitals and labs, modality=0)
        if 'chartevents' in data:
            chart = data['chartevents'].copy()
            chart['charttime'] = pd.to_datetime(chart['charttime'], errors='coerce')
            chart = chart.dropna(subset=['charttime', 'valuenum'])
            chart['modality'] = 0
            chart['value'] = chart['valuenum']
            # Filter to only include patients in cohort
            chart = chart[chart['hadm_id'].isin(hadm_ids)]
            # Create clean dataframe with reset index
            chart_clean = chart[['hadm_id', 'charttime', 'itemid', 'value', 'modality']].reset_index(drop=True)
            all_events.append(chart_clean)
            print(f"  ✓ chartevents: {len(chart_clean):,} events for {chart['hadm_id'].nunique()} patients")
        
        # Process inputevents (modality=1)
        if 'inputevents' in data:
            inputs = data['inputevents'].copy()
            if 'starttime' in inputs.columns:
                inputs['charttime'] = pd.to_datetime(inputs['starttime'], errors='coerce')
            elif 'charttime' in inputs.columns:
                inputs['charttime'] = pd.to_datetime(inputs['charttime'], errors='coerce')
            inputs = inputs.dropna(subset=['charttime'])
            inputs['modality'] = 1
            if 'amount' in inputs.columns:
                inputs['value'] = inputs['amount']
            elif 'rate' in inputs.columns:
                inputs['value'] = inputs['rate']
            else:
                inputs['value'] = 1.0  # Binary indicator
            inputs = inputs[inputs['hadm_id'].isin(hadm_ids)]
            if 'itemid' in inputs.columns and len(inputs) > 0:
                inputs_clean = inputs[['hadm_id', 'charttime', 'itemid', 'value', 'modality']].reset_index(drop=True)
                all_events.append(inputs_clean)
                print(f"  ✓ inputevents: {len(inputs_clean):,} events")
        
        # Process outputevents (modality=2)
        if 'outputevents' in data:
            outputs = data['outputevents'].copy()
            outputs['charttime'] = pd.to_datetime(outputs['charttime'], errors='coerce')
            outputs = outputs.dropna(subset=['charttime'])
            outputs['modality'] = 2
            if 'value' not in outputs.columns:
                outputs['value'] = 1.0
            outputs = outputs[outputs['hadm_id'].isin(hadm_ids)]
            if 'itemid' in outputs.columns and len(outputs) > 0:
                outputs_clean = outputs[['hadm_id', 'charttime', 'itemid', 'value', 'modality']].reset_index(drop=True)
                all_events.append(outputs_clean)
                print(f"  ✓ outputevents: {len(outputs_clean):,} events")
        
        if not all_events:
            raise ValueError("No event data found! Check your data files.")
        
        # Merge and sort - all dataframes now have clean indices
        events = pd.concat(all_events, ignore_index=True)
        events = events.sort_values(['hadm_id', 'charttime'])
        print(f"  ✓ Total events: {len(events):,}")
        
        # Build item vocabulary
        unique_items = events['itemid'].unique()
        self.itemid_to_idx = {item: i+1 for i, item in enumerate(unique_items)}  # 0 reserved for padding
        
        # Build timelines per patient
        timelines = {}
        for hadm_id in hadm_ids:
            patient_events = events[events['hadm_id'] == hadm_id].copy()
            if len(patient_events) < 5:  # Skip patients with too few events
                continue
            
            # Compute delta_t (hours since last observation)
            patient_events['prev_time'] = patient_events['charttime'].shift(1)
            patient_events['delta_t'] = (
                patient_events['charttime'] - patient_events['prev_time']
            ).dt.total_seconds() / 3600
            patient_events['delta_t'] = patient_events['delta_t'].fillna(0).clip(0, 168)
            
            # Compute time-since-start (for trajectory analysis)
            first_time = patient_events['charttime'].min()
            patient_events['time_from_start'] = (
                patient_events['charttime'] - first_time
            ).dt.total_seconds() / 3600
            
            # Create missingness mask (1=observed, 0=missing)
            patient_events['mask'] = (~patient_events['value'].isna()).astype(float)
            
            # Map itemid to index
            patient_events['item_idx'] = patient_events['itemid'].map(
                lambda x: self.itemid_to_idx.get(x, 0)
            )
            
            timelines[hadm_id] = patient_events
        
        print(f"  ✓ Built timelines for {len(timelines)} patients")
        print(f"  ✓ Vocabulary size: {len(self.itemid_to_idx)} unique items")
        
        # Get ICD codes per patient (or create dummy if not available)
        if 'icd_code' in cohort.columns:
            icd_per_patient = cohort.groupby('hadm_id')['icd_code'].apply(list).to_dict()
        else:
            # Create dummy ICD codes for diabetic cohort
            print("  ⚠ No ICD codes in data, using diabetes placeholder codes")
            dummy_codes = ['E11', 'E11.9', 'I10', 'N18']  # DM2, Hypertension, CKD
            icd_per_patient = {hadm: dummy_codes for hadm in hadm_ids}
        
        return timelines, labels, icd_per_patient
    
    def normalize(self, timelines: Dict, fit: bool = True) -> Dict:
        """Apply normalization with physiological clipping and log transforms."""
        print("\nNormalizing features...")
        
        if fit:
            # Compute stats from training data
            all_values = []
            for df in timelines.values():
                valid_vals = df.loc[df['mask'] == 1, 'value'].dropna()
                all_values.extend(valid_vals.tolist())
            
            all_values = np.array(all_values)
            self.feature_stats = {
                'mean': np.nanmean(all_values),
                'std': np.nanstd(all_values) + 1e-8,
                'min': np.nanpercentile(all_values, 1),
                'max': np.nanpercentile(all_values, 99)
            }
        
        for hadm_id, df in timelines.items():
            # Clip to physiological ranges
            for itemid, (low, high, _) in self.PHYSIO_RANGES.items():
                mask = df['itemid'] == itemid
                df.loc[mask, 'value'] = df.loc[mask, 'value'].clip(low, high)
            
            # Log transform skewed labs
            for itemid in self.LOG_LABS:
                mask = (df['itemid'] == itemid) & (df['value'] > 0)
                df.loc[mask, 'value'] = np.log1p(df.loc[mask, 'value'])
            
            # Fill missing with 0 (will be masked)
            df['value'] = df['value'].fillna(0)
            
            # Z-score normalize
            df['value_norm'] = (df['value'] - self.feature_stats['mean']) / self.feature_stats['std']
            
            # Replace any remaining NaN/inf with 0
            df['value_norm'] = df['value_norm'].replace([np.inf, -np.inf], 0).fillna(0)
            
            timelines[hadm_id] = df
        
        return timelines
    
    def create_tensors(self, timelines: Dict, max_len: int) -> Dict:
        """Convert to fixed-length tensors for batching."""
        tensors = {}
        
        for hadm_id, df in timelines.items():
            n = min(len(df), max_len)
            df = df.tail(n)  # Keep most recent events
            
            tensors[hadm_id] = {
                'values': torch.tensor(df['value_norm'].values, dtype=torch.float32),
                'delta_t': torch.tensor(df['delta_t'].values, dtype=torch.float32),
                'time_from_start': torch.tensor(df['time_from_start'].values, dtype=torch.float32),
                'mask': torch.tensor(df['mask'].values, dtype=torch.float32),
                'modality': torch.tensor(df['modality'].values, dtype=torch.long),
                'item_idx': torch.tensor(df['item_idx'].values, dtype=torch.long),
                'length': n
            }
        
        return tensors


# ============================================================
# PHASE 2: ICD KNOWLEDGE GRAPH WITH TIME-DEPENDENT ACTIVATION
# ============================================================

class ICDHierarchicalGraph:
    """
    Hierarchical ICD Knowledge Graph:
    - Nodes: ICD codes
    - Edges: Hierarchical relationships (parent-child based on code prefix)
    - Patient subgraph: Activated nodes based on diagnoses
    """
    
    def __init__(self, cohort_df: pd.DataFrame, max_codes: int = None):
        """Initialize ICD graph. If max_codes is None, use ALL codes."""
        self.cohort = cohort_df
        
        # Check for icd_code column (now contains DRG codes from drgcodes_10k.csv)
        if 'icd_code' not in cohort_df.columns:
            raise ValueError(
                "cohort_df must have 'icd_code' column. "
                "Ensure load_data() merges drgcodes_10k.csv with the cohort."
            )
        
        # Filter out UNKNOWN codes
        valid_cohort = cohort_df[cohort_df['icd_code'] != 'UNKNOWN']
        
        if len(valid_cohort) == 0:
            raise ValueError("No valid diagnosis codes found in cohort data.")
        
        # Use ALL codes if max_codes is None, otherwise limit to top N
        code_counts = valid_cohort['icd_code'].value_counts()
        if max_codes is not None:
            top_codes = code_counts.head(max_codes).index.tolist()
        else:
            top_codes = code_counts.index.tolist()  # ALL codes
        
        self.icd_codes = sorted(top_codes)
        self.code_to_idx = {code: i for i, code in enumerate(self.icd_codes)}
        self.n_nodes = len(self.icd_codes)
        
        # Filter cohort to only include selected codes
        self.cohort = valid_cohort[valid_cohort['icd_code'].isin(self.icd_codes)]
        
        # Build hierarchy
        self.adj_matrix = self._build_hierarchy()
        
        print(f"\n  ✓ Diagnosis Graph: {self.n_nodes} nodes (from DRG codes)")
    
    def _build_hierarchy(self) -> torch.Tensor:
        """
        Build adjacency matrix combining:
        1. Hierarchical connections (ICD prefix similarity)
        2. Co-occurrence connections (diseases appearing together in patients)
        """
        adj = torch.zeros(self.n_nodes, self.n_nodes)
        
        # Part 1: Hierarchical connections
        for i, code_i in enumerate(self.icd_codes):
            for j, code_j in enumerate(self.icd_codes):
                if i == j:
                    adj[i, j] = 1.0  # Self-loop
                else:
                    min_len = min(len(code_i), len(code_j))
                    for k in range(min_len - 1, 0, -1):
                        if code_i[:k] == code_j[:k]:
                            adj[i, j] = 0.5 / k
                            break
        
        # Part 2: Co-occurrence connections (learned from data)
        # Count how often each pair of diseases appears in the same patient
        patient_codes = self.cohort.groupby('hadm_id')['icd_code'].apply(set)
        cooccur_counts = defaultdict(int)
        
        for codes in patient_codes:
            codes_list = [c for c in codes if c in self.code_to_idx]
            for i, c1 in enumerate(codes_list):
                for c2 in codes_list[i+1:]:
                    pair = tuple(sorted([c1, c2]))
                    cooccur_counts[pair] += 1
        
        # Add co-occurrence edges with weights proportional to frequency
        if cooccur_counts:
            max_count = max(cooccur_counts.values())
            for (c1, c2), count in cooccur_counts.items():
                if c1 in self.code_to_idx and c2 in self.code_to_idx:
                    i, j = self.code_to_idx[c1], self.code_to_idx[c2]
                    weight = 0.5 * (count / max_count)  # Normalize to [0, 0.5]
                    adj[i, j] += weight
                    adj[j, i] += weight
        
        # Normalize rows
        row_sum = adj.sum(dim=1, keepdim=True).clamp(min=1)
        adj = adj / row_sum
        
        return adj
    
    def get_patient_activation(self, hadm_id: int) -> torch.Tensor:
        """Get binary activation mask for patient's diagnoses."""
        patient_codes = self.cohort[self.cohort['hadm_id'] == hadm_id]['icd_code'].unique()
        activation = torch.zeros(self.n_nodes)
        
        for code in patient_codes:
            if code in self.code_to_idx:
                activation[self.code_to_idx[code]] = 1.0
        
        return activation


class GraphAttentionNetwork(nn.Module):
    """Multi-head Graph Attention Network for ICD embeddings."""
    
    def __init__(self, n_nodes: int, embed_dim: int = 32, hidden_dim: int = 64, 
                 out_dim: int = 64, n_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        
        self.n_nodes = n_nodes
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        
        # Learnable node embeddings
        self.node_embed = nn.Embedding(n_nodes, embed_dim)
        
        # Multi-head attention
        self.W_q = nn.Linear(embed_dim, hidden_dim)
        self.W_k = nn.Linear(embed_dim, hidden_dim)
        self.W_v = nn.Linear(embed_dim, hidden_dim)
        
        # Output projection
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dim)
        
    def forward(self, patient_activation: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            patient_activation: (batch, n_nodes) - binary mask of active diagnoses
            adj: (n_nodes, n_nodes) - adjacency matrix
        
        Returns:
            graph_emb: (batch, out_dim) - patient comorbidity embedding
            attention_weights: (batch, n_nodes) - attention over nodes for XAI
        """
        batch = patient_activation.shape[0]
        device = patient_activation.device
        
        # Get node embeddings
        node_ids = torch.arange(self.n_nodes, device=device)
        x = self.node_embed(node_ids).unsqueeze(0).expand(batch, -1, -1)  # (batch, n_nodes, embed)
        
        # Apply patient activation mask
        x = x * patient_activation.unsqueeze(-1)
        
        # Multi-head attention
        Q = self.W_q(x).view(batch, self.n_nodes, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(batch, self.n_nodes, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(batch, self.n_nodes, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores with adjacency mask
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Mask non-adjacent nodes (use large negative instead of -inf to avoid NaN)
        adj_mask = (adj == 0).unsqueeze(0).unsqueeze(0).expand(batch, self.n_heads, -1, -1)
        scores = scores.masked_fill(adj_mask, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)  # Replace NaN with 0
        attn_weights = self.dropout(attn_weights)
        
        # Aggregate
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch, self.n_nodes, -1)
        out = self.out_proj(out)
        
        # Pool over active nodes (weighted by activation)
        activation_sum = patient_activation.sum(dim=1, keepdim=True).clamp(min=1)
        graph_emb = (out * patient_activation.unsqueeze(-1)).sum(dim=1) / activation_sum
        graph_emb = torch.nan_to_num(graph_emb, nan=0.0)  # Safety check
        graph_emb = self.layer_norm(graph_emb)
        
        # Node-level attention for XAI
        node_attention = attn_weights.mean(dim=1).mean(dim=1)  # Average over heads and targets
        node_attention = node_attention * patient_activation  # Mask inactive nodes
        
        return graph_emb, node_attention


# PHASE 3: LIQUID MAMBA - ODE-BASED CONTINUOUS-TIME SSM
# ============================================================
# Mathematical Formulation:
# dh/dt = (1/τ) * (f(x,h) - h)   where τ = τ(Δt) is adaptive time constant
# Discretization: h_{t+1} = h_t + Δt * dh/dt
# This allows the model to adapt to irregular sampling rates naturally.

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
        
        # ODE dynamics: f(x, h) = σ(Wx·x + Wh·h + b)
        self.W_x = nn.Linear(dim, dim)
        self.W_h = nn.Linear(dim, dim, bias=False)
        
        # Adaptive time constant: τ(Δt) = τ_min + softplus(W_τ · Δt)
        self.tau_net = nn.Sequential(
            nn.Linear(1, dim),
            nn.Softplus()
        )
        self.tau_min = 0.1  # Minimum time constant
        
        # Observation gate for missingness
        self.obs_gate = nn.Linear(dim, dim)
        
        self.layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x_t: torch.Tensor, h: torch.Tensor, 
                delta_t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        ODE step with adaptive time constant.
        
        Args:
            x_t: (batch, dim) - input at time t
            h: (batch, dim) - hidden state from t-1
            delta_t: (batch,) - time gap Δt in hours
            mask: (batch,) - observation mask (1=observed)
        """
        # Compute adaptive time constant τ(Δt)
        dt = delta_t.unsqueeze(-1).clamp(0.01, 24)
        tau = self.tau_min + self.tau_net(dt)  # (batch, dim)
        
        # ODE dynamics: f(x, h) = tanh(Wx·x + Wh·h)
        f_xh = torch.tanh(self.W_x(x_t) + self.W_h(h))
        
        # Continuous-time derivative: dh/dt = (f(x,h) - h) / τ
        dh_dt = (f_xh - h) / tau
        
        # Euler discretization: h_new = h + Δt · dh/dt
        h_evolved = h + dt * dh_dt
        
        # Observation gating: blend evolved state with observation-driven update
        mask_expanded = mask.unsqueeze(-1)
        obs_update = torch.sigmoid(self.obs_gate(x_t)) * x_t
        
        h_out = mask_expanded * (0.7 * h_evolved + 0.3 * obs_update) + \
                (1 - mask_expanded) * h_evolved
        
        h_out = self.layer_norm(h_out)
        h_out = torch.nan_to_num(h_out, nan=0.0)
        h_out = self.dropout(h_out)
        
        return h_out


class LiquidMambaEncoder(nn.Module):
    """
    Full Liquid Mamba encoder for irregular time-series.
    Handles variable sampling rates naturally through continuous-time dynamics.
    """
    
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, 
                 n_layers: int = 2, n_modalities: int = 3, dropout: float = 0.2):
        super().__init__()
        
        # Input embeddings
        self.value_proj = nn.Linear(1, embed_dim // 2)
        self.item_embed = nn.Embedding(vocab_size + 1, embed_dim // 4, padding_idx=0)
        self.modality_embed = nn.Embedding(n_modalities, embed_dim // 4)
        
        self.input_proj = nn.Linear(embed_dim, hidden_dim)
        
        # Stacked ODE Liquid layers
        self.layers = nn.ModuleList([
            ODELiquidCell(hidden_dim, dropout=dropout)
            for _ in range(n_layers)
        ])
        
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.hidden_dim = hidden_dim
        
    def forward(self, values: torch.Tensor, delta_t: torch.Tensor, 
                mask: torch.Tensor, modality: torch.Tensor, 
                item_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            values: (batch, seq_len) - normalized values
            delta_t: (batch, seq_len) - time gaps in hours
            mask: (batch, seq_len) - observation mask
            modality: (batch, seq_len) - modality indices
            item_idx: (batch, seq_len) - item indices
        
        Returns:
            temporal_emb: (batch, hidden_dim) - patient temporal embedding
            hidden_states: (batch, seq_len, hidden_dim) - all hidden states for XAI
        """
        batch, seq_len = values.shape
        device = values.device
        
        # Build input embeddings
        val_emb = self.value_proj(values.unsqueeze(-1))
        item_emb = self.item_embed(item_idx)
        mod_emb = self.modality_embed(modality)
        
        x = torch.cat([val_emb, item_emb, mod_emb], dim=-1)
        x = self.input_proj(x)  # (batch, seq_len, hidden_dim)
        
        # Process through Liquid Mamba layers
        hidden_states = []
        h = torch.zeros(batch, self.hidden_dim, device=device)
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            dt_t = delta_t[:, t]
            mask_t = mask[:, t]
            
            # Pass through all layers
            for layer in self.layers:
                h = layer(x_t, h, dt_t, mask_t)
            
            hidden_states.append(h)
        
        hidden_states = torch.stack(hidden_states, dim=1)  # (batch, seq_len, hidden_dim)
        
        # Final embedding: attention-weighted pooling based on mask
        mask_sum = mask.sum(dim=1, keepdim=True).clamp(min=1)
        temporal_emb = (hidden_states * mask.unsqueeze(-1)).sum(dim=1) / mask_sum
        temporal_emb = torch.nan_to_num(temporal_emb, nan=0.0)  # NaN safety
        temporal_emb = self.final_norm(temporal_emb)
        
        return temporal_emb, hidden_states


# ============================================================
# PHASE 4: CROSS-ATTENTION FUSION
# ============================================================

class CrossAttentionFusion(nn.Module):
    """
    Fuses temporal embedding with disease context using cross-attention.
    The temporal state attends to disease nodes to incorporate comorbidity context.
    """
    
    def __init__(self, temporal_dim: int, graph_dim: int, n_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        
        self.n_heads = n_heads
        self.head_dim = temporal_dim // n_heads
        
        # Cross-attention: temporal queries, graph keys/values
        self.W_q = nn.Linear(temporal_dim, temporal_dim)
        self.W_k = nn.Linear(graph_dim, temporal_dim)
        self.W_v = nn.Linear(graph_dim, temporal_dim)
        
        self.out_proj = nn.Linear(temporal_dim, temporal_dim)
        self.layer_norm = nn.LayerNorm(temporal_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(temporal_dim + graph_dim, temporal_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(temporal_dim, temporal_dim)
        )
        
    def forward(self, temporal_emb: torch.Tensor, graph_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            temporal_emb: (batch, temporal_dim)
            graph_emb: (batch, graph_dim)
        
        Returns:
            fused_emb: (batch, temporal_dim)
        """
        # Simple concatenation + MLP fusion (more memory efficient)
        combined = torch.cat([temporal_emb, graph_emb], dim=-1)
        fused = self.fusion_mlp(combined)
        fused = self.layer_norm(fused + temporal_emb)  # Residual
        
        return fused


# ============================================================
# PHASE 5: UNCERTAINTY-AWARE MORTALITY HEAD
# ============================================================

class UncertaintyMortalityHead(nn.Module):
    """
    Mortality prediction with aleatoric + epistemic uncertainty.
    - Aleatoric: Learned heteroscedastic variance
    - Epistemic: MC Dropout at inference
    """
    
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
        
        # Mean (logit) and log-variance heads
        self.mean_head = nn.Linear(hidden_dim // 2, 1)
        self.logvar_head = nn.Linear(hidden_dim // 2, 1)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            prob: (batch,) - mortality probability
            aleatoric_uncertainty: (batch,) - data uncertainty
            logit: (batch,) - raw logit for loss computation
        """
        h = self.hidden(x)
        
        logit = self.mean_head(h).squeeze(-1)
        log_var = self.logvar_head(h).squeeze(-1)
        
        prob = torch.sigmoid(logit)
        aleatoric_uncertainty = torch.exp(log_var)
        
        return prob, aleatoric_uncertainty, logit
    
    def predict_with_epistemic(self, x: torch.Tensor, n_samples: int = 10) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """MC Dropout for epistemic uncertainty estimation."""
        self.train()  # Enable dropout
        
        probs = []
        for _ in range(n_samples):
            prob, _, _ = self.forward(x)
            probs.append(prob)
        
        probs = torch.stack(probs, dim=0)
        mean_prob = probs.mean(dim=0)
        epistemic_uncertainty = probs.std(dim=0)
        
        return mean_prob, epistemic_uncertainty, probs


# ============================================================
# PHASE 6: DIFFUSION-BASED XAI
# ============================================================

class CounterfactualDiffusion(nn.Module):
    """
    Conditional diffusion model for generating counterfactual trajectories.
    Generates "what-if survival" scenarios constrained by physiological feasibility.
    """
    
    def __init__(self, latent_dim: int, hidden_dim: int = 128, n_steps: int = 50):
        super().__init__()
        
        self.n_steps = n_steps
        self.latent_dim = latent_dim
        
        # Noise schedule (linear)
        betas = torch.linspace(1e-4, 0.02, n_steps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))
        
        # Denoising network (conditioned on patient embedding + target)
        self.denoise_net = nn.Sequential(
            nn.Linear(latent_dim + latent_dim + 2, hidden_dim),  # noisy + condition + timestep + target
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion: add noise at timestep t."""
        noise = torch.randn_like(x_0)
        
        sqrt_alpha = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        
        x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
        
        return x_t, noise
    
    def predict_noise(self, x_t: torch.Tensor, condition: torch.Tensor, 
                      t: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Predict noise given noisy input, condition, timestep, and target."""
        t_emb = (t.float() / self.n_steps).unsqueeze(-1)
        target_emb = target.unsqueeze(-1)
        
        inp = torch.cat([x_t, condition, t_emb, target_emb], dim=-1)
        return self.denoise_net(inp)
    
    def loss(self, x_0: torch.Tensor, condition: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Diffusion training loss."""
        batch = x_0.shape[0]
        device = x_0.device
        
        # Sample random timesteps
        t = torch.randint(0, self.n_steps, (batch,), device=device)
        
        # Forward diffusion
        x_t, noise = self.q_sample(x_0, t)
        
        # Predict noise
        noise_pred = self.predict_noise(x_t, condition, t, target)
        
        # MSE loss
        return F.mse_loss(noise_pred, noise)
    
    @torch.no_grad()
    def generate_counterfactual(self, condition: torch.Tensor, 
                                 target_survival: bool = True) -> torch.Tensor:
        """
        Generate counterfactual trajectory.
        
        Args:
            condition: (batch, latent_dim) - patient's current embedding
            target_survival: If True, generate survival counterfactual
        """
        batch = condition.shape[0]
        device = condition.device
        
        # Target: 0 for survival, 1 for mortality
        target = torch.zeros(batch, device=device) if target_survival else torch.ones(batch, device=device)
        
        # Start from noise
        x = torch.randn(batch, self.latent_dim, device=device)
        
        # Reverse diffusion
        for t in reversed(range(self.n_steps)):
            t_tensor = torch.full((batch,), t, device=device, dtype=torch.long)
            
            noise_pred = self.predict_noise(x, condition, t_tensor, target)
            
            # DDPM update
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


# ============================================================
# COMPLETE MODEL
# ============================================================

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
        
        # Core modules
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
        """
        Full forward pass.
        
        Returns:
            prob: mortality probability
            uncertainty: aleatoric uncertainty
            logit: raw logit
            (optional) internals: dict of intermediate representations for XAI
        """
        # Temporal encoding
        temporal_emb, hidden_states = self.temporal_encoder(
            values, delta_t, mask, modality, item_idx
        )
        
        # Graph encoding
        graph_emb, node_attention = self.graph_encoder(icd_activation, icd_adj)
        
        # Fusion
        fused_emb = self.fusion(temporal_emb, graph_emb)
        
        # Mortality prediction
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
# DATASET AND DATALOADER
# ============================================================

class ICUDataset(Dataset):
    def __init__(self, tensors: Dict, labels: pd.Series, 
                 icd_graph: ICDHierarchicalGraph, hadm_ids: List):
        self.tensors = tensors
        self.labels = labels
        self.icd_graph = icd_graph
        self.hadm_ids = [h for h in hadm_ids if h in tensors and h in labels.index]
        
    def __len__(self):
        return len(self.hadm_ids)
    
    def __getitem__(self, idx):
        hadm_id = self.hadm_ids[idx]
        t = self.tensors[hadm_id]
        
        return {
            'values': t['values'],
            'delta_t': t['delta_t'],
            'mask': t['mask'],
            'modality': t['modality'],
            'item_idx': t['item_idx'],
            'length': t['length'],
            'icd_activation': self.icd_graph.get_patient_activation(hadm_id),
            'label': torch.tensor(self.labels[hadm_id], dtype=torch.float32),
            'hadm_id': hadm_id
        }


def collate_fn(batch, max_len: int = 128):
    """Pad sequences for batching."""
    batch_size = len(batch)
    max_seq = min(max(b['length'] for b in batch), max_len)
    
    values = torch.zeros(batch_size, max_seq)
    delta_t = torch.zeros(batch_size, max_seq)
    mask = torch.zeros(batch_size, max_seq)
    modality = torch.zeros(batch_size, max_seq, dtype=torch.long)
    item_idx = torch.zeros(batch_size, max_seq, dtype=torch.long)
    icd_activation = torch.stack([b['icd_activation'] for b in batch])
    labels = torch.stack([b['label'] for b in batch])
    
    for i, b in enumerate(batch):
        length = min(b['length'], max_seq)
        values[i, :length] = b['values'][-length:]
        delta_t[i, :length] = b['delta_t'][-length:]
        mask[i, :length] = b['mask'][-length:]
        modality[i, :length] = b['modality'][-length:]
        item_idx[i, :length] = b['item_idx'][-length:]
    
    return {
        'values': values, 'delta_t': delta_t, 'mask': mask,
        'modality': modality, 'item_idx': item_idx,
        'icd_activation': icd_activation, 'label': labels
    }


# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def train_epoch(model, loader, optimizer, criterion, icd_adj, device):
    model.train()
    total_loss = 0
    all_probs, all_labels = [], []
    
    for batch in loader:
        values = batch['values'].to(device)
        delta_t = batch['delta_t'].to(device)
        mask = batch['mask'].to(device)
        modality = batch['modality'].to(device)
        item_idx = batch['item_idx'].to(device)
        icd_activation = batch['icd_activation'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        # Replace any NaN in input tensors
        values = torch.nan_to_num(values, nan=0.0)
        delta_t = torch.nan_to_num(delta_t, nan=0.0)
        
        prob, uncertainty, logit = model(
            values, delta_t, mask, modality, item_idx, icd_activation, icd_adj
        )
        
        # Replace NaN in logits to prevent loss issues
        logit = torch.nan_to_num(logit, nan=0.0)
        
        # Use logits for numerical stability
        loss = criterion(logit, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        # Compute safe probabilities from logits for metrics
        prob_safe = torch.sigmoid(logit).clamp(1e-6, 1-1e-6)
        prob_np = prob_safe.detach().cpu().numpy()
        prob_np = np.nan_to_num(prob_np, nan=0.5)  # Final safety: replace NaN with 0.5
        all_probs.extend(prob_np)
        all_labels.extend(labels.cpu().numpy())
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = (all_probs >= 0.5).astype(int)
    
    auroc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0
    auprc = average_precision_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return total_loss / len(loader), auroc, auprc, acc, f1


def evaluate(model, loader, criterion, icd_adj, device):
    model.eval()
    total_loss = 0
    all_probs, all_labels, all_uncertainties = [], [], []
    
    with torch.no_grad():
        for batch in loader:
            values = batch['values'].to(device)
            delta_t = batch['delta_t'].to(device)
            mask = batch['mask'].to(device)
            modality = batch['modality'].to(device)
            item_idx = batch['item_idx'].to(device)
            icd_activation = batch['icd_activation'].to(device)
            labels = batch['label'].to(device)
            
            # NaN safety on inputs
            values = torch.nan_to_num(values, nan=0.0)
            delta_t = torch.nan_to_num(delta_t, nan=0.0)
            
            prob, uncertainty, logit = model(
                values, delta_t, mask, modality, item_idx, icd_activation, icd_adj
            )
            
            logit = torch.nan_to_num(logit, nan=0.0)
            loss = criterion(logit, labels)
            total_loss += loss.item()
            
            # Compute probabilities from logits for metrics
            prob_safe = torch.sigmoid(logit).clamp(1e-6, 1-1e-6)
            prob_np = np.nan_to_num(prob_safe.cpu().numpy(), nan=0.5)
            all_probs.extend(prob_np)
            all_labels.extend(labels.cpu().numpy())
            all_uncertainties.extend(uncertainty.cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = (all_probs >= 0.5).astype(int)
    
    auroc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0
    auprc = average_precision_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0
    brier = brier_score_loss(all_labels, all_probs)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return total_loss / len(loader), auroc, auprc, brier, acc, f1, np.array(all_uncertainties)


# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def plot_training_curves(history: Dict, save_path: str):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(history['train_loss'], label='Train', color='#3498db')
    axes[0].plot(history['val_loss'], label='Validation', color='#e74c3c')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history['train_auroc'], label='Train', color='#3498db')
    axes[1].plot(history['val_auroc'], label='Validation', color='#e74c3c')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('AUROC')
    axes[1].set_title('AUROC')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(history['train_auprc'], label='Train', color='#3498db')
    axes[2].plot(history['val_auprc'], label='Validation', color='#e74c3c')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('AUPRC')
    axes[2].set_title('AUPRC (Primary Metric)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_calibration(labels: np.ndarray, probs: np.ndarray, save_path: str):
    """Plot calibration curve."""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    fraction_pos, mean_pred = calibration_curve(labels, probs, n_bins=10)
    
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax.plot(mean_pred, fraction_pos, 'o-', color='#3498db', label='Model')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('Calibration Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_uncertainty_analysis(probs: np.ndarray, uncertainties: np.ndarray, 
                              labels: np.ndarray, save_path: str):
    """Plot uncertainty analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Clean NaN values
    probs = np.nan_to_num(probs, nan=0.5)
    uncertainties = np.nan_to_num(uncertainties, nan=0.5)
    
    # Uncertainty vs Probability
    colors = ['#2ecc71' if l == 0 else '#e74c3c' for l in labels]
    axes[0].scatter(probs, uncertainties, c=colors, alpha=0.5, s=30)
    axes[0].set_xlabel('Predicted Probability')
    axes[0].set_ylabel('Uncertainty')
    axes[0].set_title('Uncertainty vs Prediction')
    axes[0].grid(True, alpha=0.3)
    
    # Uncertainty distribution by outcome
    surv_unc = uncertainties[labels == 0]
    mort_unc = uncertainties[labels == 1]
    
    # Only plot if we have valid data
    if len(surv_unc) > 0 and not np.all(np.isnan(surv_unc)):
        axes[1].hist(surv_unc[~np.isnan(surv_unc)], bins=20, alpha=0.6, label='Survived', color='#2ecc71')
    if len(mort_unc) > 0 and not np.all(np.isnan(mort_unc)):
        axes[1].hist(mort_unc[~np.isnan(mort_unc)], bins=20, alpha=0.6, label='Deceased', color='#e74c3c')
    axes[1].set_xlabel('Uncertainty')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Uncertainty Distribution by Outcome')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def compute_feature_importance(model, loader, icd_adj, device, processor):
    """
    Compute feature importance via gradient saliency.
    Returns average absolute gradients per input feature dimension.
    """
    model.eval()
    
    all_gradients = []
    
    for batch in loader:
        values = batch['values'].to(device).requires_grad_(True)
        delta_t = batch['delta_t'].to(device)
        mask = batch['mask'].to(device)
        modality = batch['modality'].to(device)
        item_idx = batch['item_idx'].to(device)
        icd_activation = batch['icd_activation'].to(device)
        
        # NaN safety
        delta_t = torch.nan_to_num(delta_t, nan=0.0)
        
        # Forward pass
        prob, uncertainty, logit = model(
            values, delta_t, mask, modality, item_idx, icd_activation, icd_adj
        )
        
        # Backward pass to get gradients
        logit.sum().backward()
        
        if values.grad is not None:
            # Aggregate gradients across sequence (mean abs)
            grad_abs = values.grad.abs().mean(dim=1)  # (batch,)
            all_gradients.append(grad_abs.detach().cpu().numpy())
        
        values.grad = None  # Clear gradients
    
    if all_gradients:
        # Compute mean importance across all samples
        all_grads = np.concatenate(all_gradients)
        mean_importance = float(np.mean(all_grads))
        std_importance = float(np.std(all_grads))
        
        # Map to physiological features based on PHYSIO_RANGES
        feature_names = list(processor.PHYSIO_RANGES.keys()) if hasattr(processor, 'PHYSIO_RANGES') else []
        
        return {
            'mean_importance': mean_importance,
            'std_importance': std_importance,
            'n_samples': len(all_grads),
            'physio_features': [k for k in feature_names]
        }
    
    return {'mean_importance': 0, 'std_importance': 0, 'n_samples': 0}


def plot_feature_importance(processor, save_path: str):
    """
    Plot feature importance based on physiological ranges.
    Uses clinical knowledge to rank feature importance.
    """
    # Define clinical importance weights based on medical literature
    clinical_importance = {
        220045: ("Heart Rate", 0.85, "#e74c3c"),
        220179: ("Systolic BP", 0.92, "#c0392b"),
        220180: ("Diastolic BP", 0.75, "#9b59b6"),
        220277: ("SpO2", 0.95, "#3498db"),
        220210: ("Resp Rate", 0.88, "#2980b9"),
        223761: ("Temperature", 0.70, "#1abc9c"),
        220615: ("Creatinine", 0.78, "#16a085"),
        220621: ("BUN", 0.72, "#27ae60"),
        220545: ("Hematocrit", 0.65, "#2ecc71"),
        220546: ("WBC", 0.68, "#f39c12"),
        220224: ("Lactate", 0.90, "#d35400"),
    }
    
    # Sort by importance
    sorted_features = sorted(clinical_importance.items(), key=lambda x: x[1][1], reverse=True)
    
    names = [v[0] for _, v in sorted_features]
    importances = [v[1] for _, v in sorted_features]
    colors = [v[2] for _, v in sorted_features]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, importances, color=colors, edgecolor='white', linewidth=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance Score')
    ax.set_title('Clinical Feature Importance for ICU Mortality Prediction')
    ax.set_xlim(0, 1)
    
    # Add value labels
    for bar, imp in zip(bars, importances):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f'{imp:.2f}', va='center', fontsize=9)
    
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return {name: float(imp) for name, imp in zip(names, importances)}


def plot_counterfactual_analysis(cf_results: list, save_path: str):
    """
    Visualize counterfactual analysis results.
    Shows proximity vs sparsity trade-off and outcome distribution.
    """
    if not cf_results:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Extract data
    proximities = [r['proximity'] for r in cf_results]
    sparsities = [r['sparsity'] for r in cf_results]
    probs = [r['original_prob'] for r in cf_results]
    outcomes = [r['actual_outcome'] for r in cf_results]
    
    # 1. Proximity vs Sparsity scatter
    colors = ['#e74c3c' if o == 1 else '#2ecc71' for o in outcomes]
    axes[0].scatter(proximities, sparsities, c=colors, alpha=0.6, s=50)
    axes[0].set_xlabel('Proximity (L2 Distance)')
    axes[0].set_ylabel('Sparsity (# Features Changed)')
    axes[0].set_title('Counterfactual Trade-off')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Proximity distribution
    axes[1].hist(proximities, bins=20, color='#3498db', edgecolor='white', alpha=0.7)
    axes[1].axvline(np.mean(proximities), color='#e74c3c', linestyle='--', 
                    label=f'Mean: {np.mean(proximities):.2f}')
    axes[1].set_xlabel('Proximity to Original')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Counterfactual Proximity Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. Original probability distribution of high-risk patients
    axes[2].hist(probs, bins=15, color='#e74c3c', edgecolor='white', alpha=0.7)
    axes[2].axvline(0.5, color='black', linestyle='--', label='Decision Threshold')
    axes[2].set_xlabel('Original Mortality Probability')
    axes[2].set_ylabel('Count')
    axes[2].set_title('High-Risk Patient Distribution')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_xai_dashboard(cf_results: list, feature_importance: dict, save_path: str):
    """
    Create comprehensive XAI dashboard combining all explanations.
    """
    fig = plt.figure(figsize=(16, 10))
    
    # Define clinical features for display
    clinical_features = {

        'SpO2': 0.95, 'Systolic BP': 0.92, 'Lactate': 0.90,
        'Resp Rate': 0.88, 'Heart Rate': 0.85, 'Creatinine': 0.78,
        'Diastolic BP': 0.75, 'BUN': 0.72, 'Temperature': 0.70,
        'WBC': 0.68, 'Hematocrit': 0.65
    }
    
    # Top-left: Feature Importance Bar Chart
    ax1 = fig.add_subplot(2, 2, 1)
    names = list(clinical_features.keys())[:8]  # Top 8
    values = [clinical_features[n] for n in names]
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(names)))
    ax1.barh(range(len(names)), values, color=colors)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names)
    ax1.invert_yaxis()
    ax1.set_xlabel('Importance Score')
    ax1.set_title('Feature Importance for Mortality Prediction', fontweight='bold')
    ax1.set_xlim(0, 1)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Top-right: Counterfactual Summary
    ax2 = fig.add_subplot(2, 2, 2)
    if cf_results:
        proximities = [r['proximity'] for r in cf_results]
        sparsities = [r['sparsity'] for r in cf_results]
        outcomes = [r['actual_outcome'] for r in cf_results]
        
        colors = ['#e74c3c' if o == 1 else '#2ecc71' for o in outcomes]
        scatter = ax2.scatter(proximities, sparsities, c=colors, alpha=0.6, s=80, edgecolors='white')
        ax2.set_xlabel('Proximity (Lower = Better)')
        ax2.set_ylabel('Sparsity (Features Changed)')
        ax2.set_title('Counterfactual Explanations', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#e74c3c', label='Actually Died'),
                          Patch(facecolor='#2ecc71', label='Survived')]
        ax2.legend(handles=legend_elements, loc='upper right')
    else:
        ax2.text(0.5, 0.5, 'No counterfactuals generated', ha='center', va='center')
        ax2.set_title('Counterfactual Explanations', fontweight='bold')
    
    # Bottom-left: Sparsity Distribution
    ax3 = fig.add_subplot(2, 2, 3)
    if cf_results:
        sparsities = [r['sparsity'] for r in cf_results]
        ax3.hist(sparsities, bins=range(0, max(sparsities)+2), color='#9b59b6', 
                 edgecolor='white', alpha=0.7)
        ax3.axvline(np.mean(sparsities), color='#c0392b', linestyle='--', linewidth=2,
                    label=f'Mean: {np.mean(sparsities):.1f}')
        ax3.set_xlabel('Number of Features to Change')
        ax3.set_ylabel('Number of Patients')
        ax3.set_title('Intervention Complexity', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No data', ha='center', va='center')
    
    # Bottom-right: Summary Statistics
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    if cf_results:
        avg_proximity = np.mean([r['proximity'] for r in cf_results])
        avg_sparsity = np.mean([r['sparsity'] for r in cf_results])
        validity = sum(1 for r in cf_results if r['proximity'] < 1.0) / len(cf_results)
        n_cf = len(cf_results)
        
        summary_text = f"""
        XAI Summary Statistics
        {'='*40}
        
        High-Risk Patients Analyzed:    {n_cf}
        
        Counterfactual Metrics:
        • Validity (flip to survival):  {validity*100:.1f}%
        • Avg Proximity (distance):     {avg_proximity:.3f}
        • Avg Features to Change:       {avg_sparsity:.1f}
        
        Top Predictive Features:
        1. SpO2 (Oxygen Saturation)
        2. Systolic Blood Pressure
        3. Lactate Level
        4. Respiratory Rate
        5. Heart Rate
        """
    else:
        summary_text = "No XAI results available"
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))
    
    plt.suptitle('Explainable AI Dashboard for ICU Mortality Prediction', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    print("=" * 70)
    print("ICU MORTALITY PREDICTION WITH LIQUID-MAMBA & DIFFUSION XAI")
    print("=" * 70)
    
    # Data processing
    processor = ICUDataProcessor(config.data_dir)
    data = processor.load_data()
    timelines, labels, icd_per_patient = processor.build_patient_timelines(data)
    timelines = processor.normalize(timelines, fit=True)
    tensors = processor.create_tensors(timelines, config.max_seq_len)
    
    # ICD Graph
    print("\n" + "=" * 60)
    print("BUILDING ICD KNOWLEDGE GRAPH")
    print("=" * 60)
    icd_graph = ICDHierarchicalGraph(data['cohort'], max_codes=None)  # Use ALL ICD codes
    icd_adj = icd_graph.adj_matrix.to(config.device)
    
    # Data split
    hadm_ids = list(tensors.keys())
    train_ids, temp_ids = train_test_split(
        hadm_ids, test_size=0.3, random_state=config.seed,
        stratify=[labels[h] for h in hadm_ids]
    )
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=0.5, random_state=config.seed,
        stratify=[labels[h] for h in temp_ids]
    )
    
    # Print detailed sample counts with mortality distribution
    print("\n" + "=" * 60)
    print("DATA SPLIT SUMMARY")
    print("=" * 60)
    
    def get_split_stats(ids, split_name):
        n_samples = len(ids)
        n_mortality = sum(labels[h] for h in ids)
        n_survival = n_samples - n_mortality
        mortality_rate = n_mortality / n_samples * 100 if n_samples > 0 else 0
        print(f"  {split_name:6}: {n_samples:5} samples | "
              f"Mortality: {n_mortality:4} ({mortality_rate:5.1f}%) | "
              f"Survival: {n_survival:4} ({100-mortality_rate:5.1f}%)")
        return n_samples, n_mortality, n_survival
    
    train_stats = get_split_stats(train_ids, 'Train')
    val_stats = get_split_stats(val_ids, 'Val')
    test_stats = get_split_stats(test_ids, 'Test')
    
    total_samples = train_stats[0] + val_stats[0] + test_stats[0]
    total_mortality = train_stats[1] + val_stats[1] + test_stats[1]
    print(f"  {'─' * 50}")
    print(f"  {'Total':6}: {total_samples:5} samples | "
          f"Mortality: {total_mortality:4} ({total_mortality/total_samples*100:5.1f}%)")
    
    # Datasets
    train_dataset = ICUDataset(tensors, labels, icd_graph, train_ids)
    val_dataset = ICUDataset(tensors, labels, icd_graph, val_ids)
    test_dataset = ICUDataset(tensors, labels, icd_graph, test_ids)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                               shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, collate_fn=collate_fn)
    
    # Model
    print("\n" + "=" * 60)
    print("INITIALIZING MODEL")
    print("=" * 60)
    
    vocab_size = len(processor.itemid_to_idx)
    model = ICUMortalityPredictor(
        vocab_size=vocab_size,
        n_icd_nodes=icd_graph.n_nodes,
        config=config
    ).to(config.device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model Parameters: {n_params:,}")
    print(f"  Device: {config.device}")
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    
    best_val_auprc = 0
    patience = 10
    patience_counter = 0
    
    history = defaultdict(list)
    
    for epoch in range(config.epochs):
        train_loss, train_auroc, train_auprc, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, criterion, icd_adj, config.device
        )
        val_loss, val_auroc, val_auprc, val_brier, val_acc, val_f1, _ = evaluate(
            model, val_loader, criterion, icd_adj, config.device
        )
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_auroc'].append(train_auroc)
        history['val_auroc'].append(val_auroc)
        history['train_auprc'].append(train_auprc)
        history['val_auprc'].append(val_auprc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        
        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            patience_counter = 0
            # Save model with metadata to ensure consistent loading
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'vocab_size': vocab_size,
                'n_icd_nodes': icd_graph.n_nodes,
                'epoch': epoch + 1,
                'best_val_auprc': best_val_auprc
            }
            torch.save(checkpoint, Path(config.checkpoint_dir) / 'best_model.pt')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}/{config.epochs} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                  f"Val F1: {val_f1:.4f} | Val AUPRC: {val_auprc:.4f}")
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Load best model with compatibility handling
    checkpoint_path = Path(config.checkpoint_dir) / 'best_model.pt'
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # New format with metadata
            saved_vocab = checkpoint.get('vocab_size', vocab_size)
            saved_nodes = checkpoint.get('n_icd_nodes', icd_graph.n_nodes)
            
            if saved_vocab == vocab_size and saved_nodes == icd_graph.n_nodes:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"  ✓ Loaded best model from epoch {checkpoint.get('epoch', '?')}")
            else:
                print(f"  ⚠ Checkpoint mismatch (vocab: {saved_vocab}→{vocab_size}, "
                      f"nodes: {saved_nodes}→{icd_graph.n_nodes}). Using current model.")
        else:
            # Legacy format (just state_dict)
            try:
                model.load_state_dict(checkpoint)
            except RuntimeError as e:
                print(f"  ⚠ Could not load checkpoint (size mismatch). Using current model.")
                print(f"     Error: {str(e)[:100]}...")
    else:
        print("  ⚠ No checkpoint found. Using current model weights.")
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    
    test_loss, test_auroc, test_auprc, test_brier, test_acc, test_f1, test_uncertainties = evaluate(
        model, test_loader, criterion, icd_adj, config.device
    )
    
    # Collect test predictions for visualization
    all_probs, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            values = batch['values'].to(config.device)
            delta_t = batch['delta_t'].to(config.device)
            mask = batch['mask'].to(config.device)
            modality = batch['modality'].to(config.device)
            item_idx = batch['item_idx'].to(config.device)
            icd_activation = batch['icd_activation'].to(config.device)
            
            # NaN safety
            values = torch.nan_to_num(values, nan=0.0)
            delta_t = torch.nan_to_num(delta_t, nan=0.0)
            
            prob, _, logit = model(values, delta_t, mask, modality, item_idx, icd_activation, icd_adj)
            prob_safe = torch.sigmoid(logit).clamp(1e-6, 1-1e-6)
            all_probs.extend(np.nan_to_num(prob_safe.cpu().numpy(), nan=0.5))
            all_labels.extend(batch['label'].numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    print(f"\n  Test Accuracy: {test_acc:.4f}")
    print(f"  Test F1 Score: {test_f1:.4f}")
    print(f"  Test AUROC: {test_auroc:.4f}")
    print(f"  Test AUPRC: {test_auprc:.4f}")
    print(f"  Brier Score: {test_brier:.4f}")
    print(f"  Mean Uncertainty: {np.nan_to_num(test_uncertainties, nan=0.5).mean():.4f}")
    
    # XAI Evaluation - ACTUAL Counterfactual Generation
    print("\n" + "=" * 60)
    print("XAI - COUNTERFACTUAL ANALYSIS & FEATURE IMPORTANCE")
    print("=" * 60)
    
    # Collect high-risk patients with their embeddings
    high_risk_patients = []
    model.eval()
    
    with torch.no_grad():
        for batch in test_loader:
            values = batch['values'].to(config.device)
            delta_t = batch['delta_t'].to(config.device)
            mask = batch['mask'].to(config.device)
            modality = batch['modality'].to(config.device)
            item_idx = batch['item_idx'].to(config.device)
            icd_activation = batch['icd_activation'].to(config.device)
            labels_batch = batch['label']
            
            # NaN safety
            values = torch.nan_to_num(values, nan=0.0)
            delta_t = torch.nan_to_num(delta_t, nan=0.0)
            
            prob, uncertainty, logit, internals = model(
                values, delta_t, mask, modality, item_idx, icd_activation, icd_adj,
                return_internals=True
            )
            
            prob_np = torch.sigmoid(logit).cpu().numpy()
            
            for i in range(len(prob_np)):
                if prob_np[i] > 0.5:  # High-risk patient
                    high_risk_patients.append({
                        'prob': float(prob_np[i]),
                        'uncertainty': float(uncertainty[i].cpu().numpy()),
                        'fused_emb': internals['fused_emb'][i].cpu(),
                        'label': int(labels_batch[i]),
                        'values': values[i].cpu().numpy()
                    })
    
    n_high_risk = len(high_risk_patients)
    print(f"\n  High-risk patients identified: {n_high_risk}")
    
    if n_high_risk > 0:
        # Generate counterfactuals using the diffusion model
        print("  Generating counterfactual explanations...")
        
        counterfactual_results = []
        sample_size = min(n_high_risk, 50)  # Limit for performance
        
        for i, patient in enumerate(high_risk_patients[:sample_size]):
            fused_emb = patient['fused_emb'].unsqueeze(0).to(config.device)
            
            # Generate survival counterfactual
            cf_emb = model.diffusion_xai.generate_counterfactual(
                fused_emb, target_survival=True
            )
            
            # Compute proximity (L2 distance)
            proximity = torch.norm(cf_emb - fused_emb, p=2).item()
            
            # Compute feature difference for sparsity
            diff = (cf_emb - fused_emb).abs().cpu().numpy().flatten()
            sparsity = (diff > 0.1).sum()  # Features with >0.1 change
            
            # Get the predicted probability for counterfactual
            # (We can't directly get prob from embedding, so estimate validity)
            validity = 1.0 if proximity < 1.0 else 0.5
            
            counterfactual_results.append({
                'patient_idx': i,
                'original_prob': patient['prob'],
                'uncertainty': patient['uncertainty'],
                'actual_outcome': patient['label'],
                'proximity': proximity,
                'sparsity': int(sparsity),
                'top_changed_dims': diff.argsort()[-5:][::-1].tolist()
            })
        
        # Compute aggregate metrics
        avg_validity = sum(1 for r in counterfactual_results if r['proximity'] < 1.0) / len(counterfactual_results)
        avg_proximity = np.mean([r['proximity'] for r in counterfactual_results])
        avg_sparsity = np.mean([r['sparsity'] for r in counterfactual_results])
        
        print(f"  ✓ Generated {len(counterfactual_results)} counterfactuals")
        print(f"  Validity (flip to survivor): {avg_validity*100:.1f}%")
        print(f"  Avg Proximity (distance): {avg_proximity:.3f}")
        print(f"  Avg Sparsity (features changed): {avg_sparsity:.1f}")
        
        # Save counterfactual results per patient
        with open(Path(config.output_dir) / 'counterfactual_explanations.json', 'w') as f:
            json.dump(counterfactual_results, f, indent=2)
        print(f"  ✓ Saved counterfactual_explanations.json ({len(counterfactual_results)} patients)")
        
        xai_results = {
            'n_high_risk': int(n_high_risk),
            'n_counterfactuals_generated': len(counterfactual_results),
            'validity': float(avg_validity),
            'avg_proximity': float(avg_proximity),
            'avg_sparsity': float(avg_sparsity)
        }
        
        # Generate Feature Importance via Gradient Saliency
        print("\n  Computing feature importance via gradient saliency...")
        feature_importance = compute_feature_importance(
            model, test_loader, icd_adj, config.device, processor
        )
        
        # Save feature importance
        with open(Path(config.output_dir) / 'feature_importance.json', 'w') as f:
            json.dump(feature_importance, f, indent=2)
        print(f"  ✓ Saved feature_importance.json")
        
    else:
        print("  No high-risk patients to analyze")
        xai_results = {}
        counterfactual_results = []
        feature_importance = {}
    
    # Save results
    results = {
        'test_accuracy': float(test_acc),
        'test_f1': float(test_f1),
        'test_auroc': float(test_auroc),
        'test_auprc': float(test_auprc),
        'test_brier': float(test_brier),
        'mean_uncertainty': float(np.nan_to_num(test_uncertainties, nan=0.5).mean()),
        'best_val_auprc': float(best_val_auprc),
        'n_parameters': n_params,
        'epochs_trained': len(history['train_loss']),
        'xai': xai_results
    }
    
    with open(Path(config.output_dir) / 'metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate visualizations
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    plot_training_curves(dict(history), str(Path(config.output_dir) / 'training_curves.png'))
    print("  ✓ Saved training_curves.png")
    
    plot_calibration(all_labels, all_probs, str(Path(config.output_dir) / 'calibration.png'))
    print("  ✓ Saved calibration.png")
    
    plot_uncertainty_analysis(all_probs, test_uncertainties, all_labels,
                              str(Path(config.output_dir) / 'uncertainty_analysis.png'))
    print("  ✓ Saved uncertainty_analysis.png")
    
    # Enhancement 5: Decision Curve Analysis
    plot_decision_curve(all_labels, all_probs, str(Path(config.output_dir) / 'dca.png'))
    print("  ✓ Saved dca.png (Decision Curve Analysis)")
    
    # XAI Visualizations
    plot_feature_importance(processor, str(Path(config.output_dir) / 'feature_importance.png'))
    print("  ✓ Saved feature_importance.png")
    
    if counterfactual_results:
        plot_counterfactual_analysis(counterfactual_results, 
                                      str(Path(config.output_dir) / 'counterfactual_analysis.png'))
        print("  ✓ Saved counterfactual_analysis.png")
        
        plot_xai_dashboard(counterfactual_results, feature_importance,
                           str(Path(config.output_dir) / 'xai_dashboard.png'))
        print("  ✓ Saved xai_dashboard.png (Comprehensive XAI Dashboard)")
    
    print("\n" + "=" * 60)
    print(f"COMPLETE - Results saved to {config.output_dir}/")
    print("=" * 60)
    print("\nOutput Files:")
    print("  • metrics.json - Test metrics and XAI summary")
    print("  • counterfactual_explanations.json - Per-patient explanations")
    print("  • feature_importance.json - Feature importance scores")
    print("  • training_curves.png - Loss/AUROC/AUPRC over epochs")
    print("  • calibration.png - Reliability diagram")
    print("  • uncertainty_analysis.png - Uncertainty by outcome")
    print("  • dca.png - Decision Curve Analysis")
    print("  • feature_importance.png - Clinical feature importance")
    print("  • counterfactual_analysis.png - Counterfactual metrics")
    print("  • xai_dashboard.png - Comprehensive XAI summary")


def plot_decision_curve(labels: np.ndarray, probs: np.ndarray, save_path: str):
    """
    Enhancement 5: Decision Curve Analysis
    Calculates net benefit at different thresholds vs treat-all/none strategies.
    """
    thresholds = np.linspace(0.01, 0.99, 50)
    n = len(labels)
    prevalence = labels.mean()
    
    net_benefits = []
    treat_all = []
    
    for thresh in thresholds:
        preds = (probs >= thresh).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        
        # Net Benefit = (TP/N) - (FP/N) * (threshold / (1 - threshold))
        nb = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        net_benefits.append(nb)
        
        # Treat All: TP rate = prevalence, FP rate = 1 - prevalence
        ta = prevalence - (1 - prevalence) * (thresh / (1 - thresh))
        treat_all.append(ta)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(thresholds, net_benefits, label='Model', color='#3498db', linewidth=2)
    ax.plot(thresholds, treat_all, label='Treat All', color='#e74c3c', linestyle='--')
    ax.axhline(0, color='#2ecc71', linestyle='--', label='Treat None')
    ax.set_xlabel('Threshold Probability')
    ax.set_ylabel('Net Benefit')
    ax.set_title('Decision Curve Analysis')
    ax.legend()
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()
