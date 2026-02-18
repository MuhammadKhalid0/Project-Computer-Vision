#!/usr/bin/env python3
"""
Comprehensive model analysis with plots to assess learning, overfitting, and generalization.
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, roc_curve, 
    confusion_matrix, classification_report
)
import time

from cvproj_exc.osr_learning import (
    load_challenge_train_data, spl_training, mpl_training, UNKNOWN_LABEL
)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def compute_osr_metrics(y_true, y_pred, y_score, prefix=""):
    """Compute comprehensive OSR metrics."""
    y_true_binary = (y_true >= 0).astype(int)
    
    try:
        auc = roc_auc_score(y_true_binary, y_score)
    except:
        auc = 0.0
    
    kc_mask = y_true >= 0
    kc_acc = accuracy_score(y_true[kc_mask], y_pred[kc_mask]) if kc_mask.any() else 0.0
    
    unk_mask = y_true == UNKNOWN_LABEL
    unk_recall = (y_pred[unk_mask] == UNKNOWN_LABEL).sum() / float(unk_mask.sum()) if unk_mask.any() else 0.0
    
    overall_acc = accuracy_score(y_true, y_pred)
    
    per_class_acc = []
    unique_kc = np.unique(y_true[kc_mask]) if kc_mask.any() else []
    for cls in unique_kc:
        cls_mask = y_true == cls
        if cls_mask.any():
            per_class_acc.append((y_pred[cls_mask] == cls).sum() / float(cls_mask.sum()))
    balanced_rank1 = np.mean(per_class_acc) if per_class_acc else 0.0
    
    return {
        'auc_roc': float(auc),
        'kc_acc': float(kc_acc),
        'unk_recall': float(unk_recall),
        'overall_acc': float(overall_acc),
        'balanced_rank1': float(balanced_rank1)
    }


def plot_metrics_comparison(spl_train_metrics, spl_val_metrics, 
                            mpl_train_metrics, mpl_val_metrics, save_path='metrics_comparison.png'):
    """Plot train vs validation metrics comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training vs Validation Metrics Comparison', fontsize=16, fontweight='bold')
    
    metrics_names = ['AUC-ROC', 'KC Accuracy', 'Unknown Recall', 'Balanced Rank-1']
    metrics_keys = ['auc_roc', 'kc_acc', 'unk_recall', 'balanced_rank1']
    
    for idx, (name, key) in enumerate(zip(metrics_names, metrics_keys)):
        ax = axes[idx // 2, idx % 2]
        
        x = np.arange(2)
        width = 0.35
        
        spl_train_val = [spl_train_metrics[key], spl_val_metrics[key]]
        mpl_train_val = [mpl_train_metrics[key], mpl_val_metrics[key]]
        
        bars1 = ax.bar(x - width/2, spl_train_val, width, label='SPL', alpha=0.8)
        bars2 = ax.bar(x + width/2, mpl_train_val, width, label='MPL', alpha=0.8)
        
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['Train', 'Validation'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.1])
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()


def plot_overfitting_analysis(spl_train_metrics, spl_val_metrics,
                              mpl_train_metrics, mpl_val_metrics, save_path='overfitting_analysis.png'):
    """Plot overfitting gap analysis (train - validation)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Overfitting Analysis: Train - Validation Gap', fontsize=16, fontweight='bold')
    
    metrics_names = ['AUC-ROC', 'KC Accuracy', 'Unknown Recall', 'Balanced Rank-1']
    metrics_keys = ['auc_roc', 'kc_acc', 'unk_recall', 'balanced_rank1']
    
    # SPL overfitting
    ax1 = axes[0]
    spl_gaps = [spl_train_metrics[k] - spl_val_metrics[k] for k in metrics_keys]
    colors = ['red' if gap > 0.1 else 'orange' if gap > 0.05 else 'green' for gap in spl_gaps]
    bars1 = ax1.barh(metrics_names, spl_gaps, color=colors, alpha=0.7)
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax1.set_xlabel('Train - Validation Gap', fontsize=11)
    ax1.set_title('SPL Overfitting Gap', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    for i, (bar, gap) in enumerate(zip(bars1, spl_gaps)):
        ax1.text(gap + 0.01 if gap >= 0 else gap - 0.01, i,
                f'{gap:+.3f}',
                va='center', ha='left' if gap >= 0 else 'right', fontsize=9)
    
    # MPL overfitting
    ax2 = axes[1]
    mpl_gaps = [mpl_train_metrics[k] - mpl_val_metrics[k] for k in metrics_keys]
    colors = ['red' if gap > 0.1 else 'orange' if gap > 0.05 else 'green' for gap in mpl_gaps]
    bars2 = ax2.barh(metrics_names, mpl_gaps, color=colors, alpha=0.7)
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel('Train - Validation Gap', fontsize=11)
    ax2.set_title('MPL Overfitting Gap', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    for i, (bar, gap) in enumerate(zip(bars2, mpl_gaps)):
        ax2.text(gap + 0.01 if gap >= 0 else gap - 0.01, i,
                f'{gap:+.3f}',
                va='center', ha='left' if gap >= 0 else 'right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()


def plot_roc_curves(y_train_true, y_train_score, y_val_true, y_val_score, 
                   model_name, save_path):
    """Plot ROC curves for train and validation."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Train ROC
    y_train_binary = (y_train_true >= 0).astype(int)
    fpr_train, tpr_train, _ = roc_curve(y_train_binary, y_train_score)
    auc_train = roc_auc_score(y_train_binary, y_train_score)
    
    # Validation ROC
    y_val_binary = (y_val_true >= 0).astype(int)
    fpr_val, tpr_val, _ = roc_curve(y_val_binary, y_val_score)
    auc_val = roc_auc_score(y_val_binary, y_val_score)
    
    ax.plot(fpr_train, tpr_train, label=f'Train (AUC={auc_train:.4f})', linewidth=2)
    ax.plot(fpr_val, tpr_val, label=f'Validation (AUC={auc_val:.4f})', linewidth=2, linestyle='--')
    ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(f'{model_name} ROC Curves', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()


def plot_score_distributions(y_train_true, y_train_score, y_val_true, y_val_score,
                             model_name, save_path):
    """Plot score distributions for known vs unknown classes."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{model_name} Score Distributions', fontsize=16, fontweight='bold')
    
    # Train scores
    ax1 = axes[0]
    train_known_scores = y_train_score[y_train_true >= 0]
    train_unk_scores = y_train_score[y_train_true == UNKNOWN_LABEL]
    
    ax1.hist(train_known_scores, bins=50, alpha=0.6, label='Known Classes', density=True)
    ax1.hist(train_unk_scores, bins=50, alpha=0.6, label='Unknown Classes', density=True)
    ax1.set_xlabel('Confidence Score', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title('Training Set', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Validation scores
    ax2 = axes[1]
    val_known_scores = y_val_score[y_val_true >= 0]
    val_unk_scores = y_val_score[y_val_true == UNKNOWN_LABEL]
    
    ax2.hist(val_known_scores, bins=50, alpha=0.6, label='Known Classes', density=True)
    ax2.hist(val_unk_scores, bins=50, alpha=0.6, label='Unknown Classes', density=True)
    ax2.set_xlabel('Confidence Score', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title('Validation Set', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()


def plot_confusion_matrices(y_train_true, y_train_pred, y_val_true, y_val_pred,
                           model_name, save_path):
    """Plot confusion matrices for train and validation."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{model_name} Confusion Matrices', fontsize=16, fontweight='bold')
    
    # Binary: Known vs Unknown
    train_binary_true = (y_train_true >= 0).astype(int)
    train_binary_pred = (y_train_pred >= 0).astype(int)
    val_binary_true = (y_val_true >= 0).astype(int)
    val_binary_pred = (y_val_pred >= 0).astype(int)
    
    # Train confusion matrix
    cm_train = confusion_matrix(train_binary_true, train_binary_pred)
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Unknown', 'Known'], yticklabels=['Unknown', 'Known'])
    axes[0].set_title('Training Set', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=11)
    axes[0].set_xlabel('Predicted Label', fontsize=11)
    
    # Validation confusion matrix
    cm_val = confusion_matrix(val_binary_true, val_binary_pred)
    sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                xticklabels=['Unknown', 'Known'], yticklabels=['Unknown', 'Known'])
    axes[1].set_title('Validation Set', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=11)
    axes[1].set_xlabel('Predicted Label', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()


def print_detailed_report(y_train_true, y_train_pred, y_val_true, y_val_pred,
                         train_metrics, val_metrics, model_name):
    """Print detailed performance report."""
    print("\n" + "="*80)
    print(f"{model_name} DETAILED REPORT")
    print("="*80)
    
    print("\n📊 METRICS COMPARISON:")
    print("-" * 80)
    print(f"{'Metric':<20} {'Train':>12} {'Validation':>12} {'Gap':>12} {'Status':>12}")
    print("-" * 80)
    
    metrics_info = [
        ('AUC-ROC', 'auc_roc'),
        ('KC Accuracy', 'kc_acc'),
        ('Unknown Recall', 'unk_recall'),
        ('Balanced Rank-1', 'balanced_rank1'),
        ('Overall Accuracy', 'overall_acc')
    ]
    
    for name, key in metrics_info:
        train_val = train_metrics[key]
        val_val = val_metrics[key]
        gap = train_val - val_val
        
        if gap > 0.1:
            status = "⚠️ OVERFIT"
        elif gap > 0.05:
            status = "🟡 MARGINAL"
        elif gap < -0.05:
            status = "🟢 UNDERFIT"
        else:
            status = "✅ GOOD"
        
        print(f"{name:<20} {train_val:>12.4f} {val_val:>12.4f} {gap:>+12.4f} {status:>12}")
    
    print("-" * 80)
    
    # Overfitting assessment
    max_gap = max([train_metrics[k] - val_metrics[k] for k in ['auc_roc', 'kc_acc', 'balanced_rank1']])
    print(f"\n🔍 OVERFITTING ASSESSMENT:")
    if max_gap > 0.1:
        print("   ⚠️  SIGNIFICANT OVERFITTING DETECTED")
        print("   → Large gap between train and validation performance")
        print("   → Consider: regularization, less complex model, or more data")
    elif max_gap > 0.05:
        print("   🟡 MILD OVERFITTING")
        print("   → Small gap, but acceptable for this task")
    else:
        print("   ✅ GOOD GENERALIZATION")
        print("   → Train and validation performance are well-aligned")
    
    # Generalization assessment
    val_balanced = val_metrics['balanced_rank1']
    print(f"\n🎯 GENERALIZATION ASSESSMENT:")
    if val_balanced > 0.90:
        print("   ✅ EXCELLENT - Validation performance > 90%")
    elif val_balanced > 0.80:
        print("   ✅ VERY GOOD - Validation performance > 80%")
    elif val_balanced > 0.70:
        print("   🟡 GOOD - Validation performance > 70%")
    else:
        print("   ⚠️  NEEDS IMPROVEMENT - Validation performance < 70%")
    
    print("="*80)


def main():
    print("="*80)
    print("MODEL PERFORMANCE ANALYSIS WITH VISUALIZATIONS")
    print("="*80)
    
    # Load data
    print("\nLoading challenge training data...")
    x_full, y_full = load_challenge_train_data()
    print(f"Total samples: {len(x_full)}")
    print(f"Known classes: {(y_full >= 0).sum()}")
    print(f"Known unknowns: {(y_full == UNKNOWN_LABEL).sum()}")
    
    # Split into train and validation (75/25)
    print("\nSplitting data: 75% train, 25% validation...")
    x_train, x_val, y_train, y_val = train_test_split(
        x_full, y_full,
        test_size=0.25,
        random_state=42,
        stratify=y_full
    )
    print(f"Training set: {len(x_train)} samples")
    print(f"Validation set: {len(x_val)} samples")
    
    # Train SPL
    print("\n" + "="*80)
    print("TRAINING SPL")
    print("="*80)
    spl_start = time.time()
    spl_predict_fn = spl_training(x_train, y_train)
    spl_train_time = time.time() - spl_start
    print(f"Training time: {spl_train_time:.2f} seconds")
    
    # Evaluate SPL
    print("\nEvaluating SPL...")
    y_train_pred_spl, y_train_score_spl = spl_predict_fn(x_train)
    y_val_pred_spl, y_val_score_spl = spl_predict_fn(x_val)
    
    spl_train_metrics = compute_osr_metrics(y_train, y_train_pred_spl, y_train_score_spl)
    spl_val_metrics = compute_osr_metrics(y_val, y_val_pred_spl, y_val_score_spl)
    
    # Train MPL
    print("\n" + "="*80)
    print("TRAINING MPL")
    print("="*80)
    mpl_start = time.time()
    mpl_predict_fn = mpl_training(x_train, y_train)
    mpl_train_time = time.time() - mpl_start
    print(f"Training time: {mpl_train_time:.2f} seconds")
    
    # Evaluate MPL
    print("\nEvaluating MPL...")
    y_train_pred_mpl, y_train_score_mpl = mpl_predict_fn(x_train)
    y_val_pred_mpl, y_val_score_mpl = mpl_predict_fn(x_val)
    
    mpl_train_metrics = compute_osr_metrics(y_train, y_train_pred_mpl, y_train_score_mpl)
    mpl_val_metrics = compute_osr_metrics(y_val, y_val_pred_mpl, y_val_score_mpl)
    
    # Print reports
    print_detailed_report(y_train, y_train_pred_spl, y_val, y_val_pred_spl,
                         spl_train_metrics, spl_val_metrics, "SPL")
    
    print_detailed_report(y_train, y_train_pred_mpl, y_val, y_val_pred_mpl,
                         mpl_train_metrics, mpl_val_metrics, "MPL")
    
    # Generate plots
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)
    
    # Metrics comparison
    plot_metrics_comparison(spl_train_metrics, spl_val_metrics,
                           mpl_train_metrics, mpl_val_metrics,
                           'metrics_comparison.png')
    
    # Overfitting analysis
    plot_overfitting_analysis(spl_train_metrics, spl_val_metrics,
                              mpl_train_metrics, mpl_val_metrics,
                              'overfitting_analysis.png')
    
    # ROC curves
    plot_roc_curves(y_train, y_train_score_spl, y_val, y_val_score_spl,
                   'SPL', 'spl_roc_curves.png')
    
    plot_roc_curves(y_train, y_train_score_mpl, y_val, y_val_score_mpl,
                   'MPL', 'mpl_roc_curves.png')
    
    # Score distributions
    plot_score_distributions(y_train, y_train_score_spl, y_val, y_val_score_spl,
                            'SPL', 'spl_score_distributions.png')
    
    plot_score_distributions(y_train, y_train_score_mpl, y_val, y_val_score_mpl,
                            'MPL', 'mpl_score_distributions.png')
    
    # Confusion matrices
    plot_confusion_matrices(y_train, y_train_pred_spl, y_val, y_val_pred_spl,
                           'SPL', 'spl_confusion_matrices.png')
    
    plot_confusion_matrices(y_train, y_train_pred_mpl, y_val, y_val_pred_mpl,
                           'MPL', 'mpl_confusion_matrices.png')
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated plots:")
    print("  1. metrics_comparison.png - Train vs Validation metrics")
    print("  2. overfitting_analysis.png - Overfitting gap analysis")
    print("  3. spl_roc_curves.png - SPL ROC curves")
    print("  4. mpl_roc_curves.png - MPL ROC curves")
    print("  5. spl_score_distributions.png - SPL score distributions")
    print("  6. mpl_score_distributions.png - MPL score distributions")
    print("  7. spl_confusion_matrices.png - SPL confusion matrices")
    print("  8. mpl_confusion_matrices.png - MPL confusion matrices")
    print("\n✅ All plots saved in current directory!")


if __name__ == "__main__":
    main()