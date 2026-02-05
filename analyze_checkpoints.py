"""
Checkpoint Analysis Script
Analyzes saved training checkpoints from research.py
"""

import torch
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def analyze_checkpoints():
    checkpoint_dir = Path("e:/Vscode/IIIT Ranchi/checkpoints")
    
    # Find all epoch checkpoints
    checkpoint_files = sorted(checkpoint_dir.glob("epoch_*_checkpoint.pt"))
    
    if not checkpoint_files:
        print("No checkpoint files found!")
        return
    
    print(f"Found {len(checkpoint_files)} checkpoints\n")
    print("=" * 80)
    
    # Collect metrics
    epochs = []
    train_losses = []
    val_losses = []
    val_auprcs = []
    val_aurocs = []
    val_f1s = []
    
    best_auprc = 0
    best_epoch = 0
    best_checkpoint = None
    
    for cp_path in checkpoint_files:
        try:
            cp = torch.load(cp_path, weights_only=False, map_location='cpu')
            epoch = cp.get('epoch', 0)
            epochs.append(epoch)
            
            train_loss = cp.get('train_loss', 0)
            val_loss = cp.get('val_loss', 0)
            val_auprc = cp.get('val_auprc', 0)
            val_auroc = cp.get('val_auroc', 0)
            val_f1 = cp.get('val_f1', 0)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_auprcs.append(val_auprc)
            val_aurocs.append(val_auroc)
            val_f1s.append(val_f1)
            
            if val_auprc > best_auprc:
                best_auprc = val_auprc
                best_epoch = epoch
                best_checkpoint = cp
            
            print(f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"AUROC: {val_auroc:.4f} | AUPRC: {val_auprc:.4f} | F1: {val_f1:.4f}")
                  
        except Exception as e:
            print(f"Error loading {cp_path.name}: {e}")
    
    print("=" * 80)
    print(f"\n🏆 BEST MODEL: Epoch {best_epoch}")
    print(f"   Best AUPRC: {best_auprc:.4f}")
    print(f"   Best AUROC: {best_checkpoint.get('val_auroc', 0):.4f}")
    print(f"   Best F1:    {best_checkpoint.get('val_f1', 0):.4f}")
    
    # Check training history from last checkpoint
    last_cp = torch.load(checkpoint_files[-1], weights_only=False, map_location='cpu')
    history = last_cp.get('history', {})
    
    if history:
        print(f"\n📊 Training History Summary (from epoch {len(checkpoint_files)}):")
        if 'train_loss' in history:
            print(f"   Final Train Loss: {history['train_loss'][-1]:.4f}")
        if 'val_loss' in history:
            print(f"   Final Val Loss:   {history['val_loss'][-1]:.4f}")
        if 'val_auprc' in history:
            print(f"   Max Val AUPRC:    {max(history['val_auprc']):.4f}")
        if 'val_auroc' in history:
            print(f"   Max Val AUROC:    {max(history['val_auroc']):.4f}")
    
    # Plot training curves
    if len(epochs) > 1:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Loss curves
        axes[0, 0].plot(epochs, train_losses, 'b-o', label='Train Loss', markersize=4)
        axes[0, 0].plot(epochs, val_losses, 'r-o', label='Val Loss', markersize=4)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training vs Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # AUROC
        axes[0, 1].plot(epochs, val_aurocs, 'g-o', label='Val AUROC', markersize=4)
        axes[0, 1].axhline(y=max(val_aurocs), color='g', linestyle='--', alpha=0.5, label=f'Best: {max(val_aurocs):.4f}')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('AUROC')
        axes[0, 1].set_title('Validation AUROC')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # AUPRC
        axes[1, 0].plot(epochs, val_auprcs, 'm-o', label='Val AUPRC', markersize=4)
        axes[1, 0].axhline(y=max(val_auprcs), color='m', linestyle='--', alpha=0.5, label=f'Best: {max(val_auprcs):.4f}')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUPRC')
        axes[1, 0].set_title('Validation AUPRC (Primary Metric)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # F1 Score
        axes[1, 1].plot(epochs, val_f1s, 'c-o', label='Val F1', markersize=4)
        axes[1, 1].axhline(y=max(val_f1s), color='c', linestyle='--', alpha=0.5, label=f'Best: {max(val_f1s):.4f}')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].set_title('Validation F1 Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = checkpoint_dir / 'training_analysis.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n📈 Saved training curves to: {output_path}")
        plt.show()
    
    # Summary Table
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Metric':<20} {'Best Value':<15} {'At Epoch':<10}")
    print("-" * 45)
    
    best_auroc_idx = np.argmax(val_aurocs)
    best_auprc_idx = np.argmax(val_auprcs)
    best_f1_idx = np.argmax(val_f1s)
    
    print(f"{'AUROC':<20} {val_aurocs[best_auroc_idx]:.4f}          {epochs[best_auroc_idx]}")
    print(f"{'AUPRC':<20} {val_auprcs[best_auprc_idx]:.4f}          {epochs[best_auprc_idx]}")
    print(f"{'F1 Score':<20} {val_f1s[best_f1_idx]:.4f}          {epochs[best_f1_idx]}")
    print(f"{'Min Val Loss':<20} {min(val_losses):.4f}          {epochs[np.argmin(val_losses)]}")
    
    return best_checkpoint

if __name__ == "__main__":
    analyze_checkpoints()
