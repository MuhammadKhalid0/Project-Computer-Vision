#!/usr/bin/env python3
"""
hyperparameter tuning for OSR learning with overfitting awareness.

Usage:
    PYTHONPATH=src python3 hyperparameter_tuning.py

"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
import time
from itertools import product
from multiprocessing import Pool, cpu_count
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

from cvproj_exc.osr_learning import load_challenge_train_data, UNKNOWN_LABEL, normalize_features, mixup_augmentation


#SEARCH SPACE CONFIGURATION
# =============================================================================

SPL_SEARCH_SPACE = {
    # Preprocessing: None = no PCA, float = retain that fraction of variance
    'pca_options': [None, 0.90, 0.95, 0.99],
    # SVM hyperparameters
    'C_options': [1.0, 10.0, 50.0, 100.0],
    'gamma_options': ['scale', 'auto'],
    'class_weight_options': [None, 'balanced'],
    'calib_method_options': ['sigmoid'],
    # Data augmentation
    'mixup_ratio_options': [0.0, 0.3, 0.5],
    # Score combination (prob_weight; cosine_weight = 1 - prob_weight)
    'prob_weight_options': [0.4, 0.5, 0.6, 0.7],
    # Decision threshold
    'threshold_options': [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
}

MPL_SEARCH_SPACE = {
    'pca_options': [None, 0.90, 0.95, 0.99],
    'C_options': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
    'solver_options': ['lbfgs'],
    'class_weight_options': [None, 'balanced'],
    'max_iter_options': [500],
    'mixup_ratio_options': [0.0, 0.3, 0.5],
    'prob_weight_options': [0.4, 0.5, 0.6, 0.7],
    'threshold_options': [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
}

# How much to penalise overfitting (0 = ignore, 1 = heavy penalty)
OVERFIT_PENALTY_WEIGHT = 0.3


#METRICS AND SCORING
# =============================================================================

def compute_osr_metrics(y_true, y_pred, y_score):
    """Compute comprehensive OSR metrics."""
    y_true_binary = (y_true >= 0).astype(int)

    try:
        auc = roc_auc_score(y_true_binary, y_score)
    except Exception:
        auc = 0.0

    kc_mask = y_true >= 0
    kc_acc = accuracy_score(y_true[kc_mask], y_pred[kc_mask]) if kc_mask.any() else 0.0

    unk_mask = y_true == UNKNOWN_LABEL
    unk_recall = float((y_pred[unk_mask] == UNKNOWN_LABEL).sum()) / float(unk_mask.sum()) if unk_mask.any() else 0.0

    overall_acc = accuracy_score(y_true, y_pred)

    per_class_acc = []
    for cls in (np.unique(y_true[kc_mask]) if kc_mask.any() else []):
        cls_mask = y_true == cls
        if cls_mask.any():
            per_class_acc.append(float((y_pred[cls_mask] == cls).sum()) / float(cls_mask.sum()))
    balanced_rank1 = float(np.mean(per_class_acc)) if per_class_acc else 0.0

    return {
        'auc_roc': float(auc),
        'kc_acc': float(kc_acc),
        'unk_recall': float(unk_recall),
        'overall_acc': float(overall_acc),
        'balanced_rank1': float(balanced_rank1),
    }


def compute_generalization_score(train_metrics, val_metrics, overfit_penalty=OVERFIT_PENALTY_WEIGHT):
    """
    Score that rewards high validation performance AND penalises overfitting.

    generalization_score = val_score - penalty * avg_overfitting_gap (for overfitting awarness)
    """
    # Weighted validation score (aligns with competition criteria)
    val_score = (
        0.30 * val_metrics['auc_roc']
        + 0.25 * val_metrics['balanced_rank1']
        + 0.20 * val_metrics['kc_acc']
        + 0.25 * val_metrics['unk_recall']
    )

    # Overfitting penalty: average gap across the key discriminative metrics
    gaps = []
    for key in ['auc_roc', 'kc_acc', 'balanced_rank1']:
        gap = max(0.0, train_metrics[key] - val_metrics[key])
        gaps.append(gap)
    avg_gap = float(np.mean(gaps))

    gen_score = val_score - overfit_penalty * avg_gap
    return gen_score, val_score, avg_gap


# PREPROCESSING PIPELINE
#=============================================================================

class PreprocessPipeline:
    """StandardScaler ->> optional PCA ->> L2 normalisation."""

    def __init__(self, pca_components=None):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=pca_components, random_state=42) if pca_components is not None else None
        self.pca_components = pca_components

    def fit_transform(self, x):
        x = self.scaler.fit_transform(x)
        if self.pca is not None:
            x = self.pca.fit_transform(x)
        return normalize_features(x)

    def transform(self, x):
        x = self.scaler.transform(x)
        if self.pca is not None:
            x = self.pca.transform(x)
        return normalize_features(x)

    @property
    def name(self):
        if self.pca is None:
            return 'standard_l2'
        if isinstance(self.pca_components, float):
            return f'standard_pca{int(self.pca_components * 100)}_l2'
        return f'standard_pca{self.pca_components}_l2'

    @property
    def n_dims(self):
        if self.pca is not None and hasattr(self.pca, 'n_components_'):
            return int(self.pca.n_components_)
        return None


#SPL TUNING
# =============================================================================

def evaluate_spl_config(args):
    """Evaluate a single SPL configuration (worker function for multiprocessing)."""
    (pca_comp, mixup_ratio, C, gamma, cw, calib, x_train, y_train, x_val, y_val, ss) = args
    
    results = []
    
    try:
        # Build preprocessing pipeline
        pipe = PreprocessPipeline(pca_components=pca_comp)
        x_tr = pipe.fit_transform(x_train)
        x_va = pipe.transform(x_val)
        n_dims = pipe.n_dims or x_tr.shape[1]
        
        # --- Augmentation
        kc_mask = y_train >= 0
        kuc_mask = y_train == UNKNOWN_LABEL
        
        if mixup_ratio > 0 and kuc_mask.any():
            x_kc = x_tr[kc_mask]
            y_kc = y_train[kc_mask]
            x_kuc = x_tr[kuc_mask]
            x_syn = mixup_augmentation(x_kc, y_kc, x_kuc, ratio=mixup_ratio, alpha=0.4)
            x_train_aug = np.vstack([x_tr, x_syn])
            y_train_aug = np.concatenate([y_train, np.full(len(x_syn), UNKNOWN_LABEL)])
        else:
            x_train_aug = x_tr
            y_train_aug = y_train.copy()
        
        # --- SPL labels
        kc_aug = y_train_aug >= 0
        kuc_aug = y_train_aug == UNKNOWN_LABEL
        max_label = y_train_aug[kc_aug].max() if kc_aug.any() else -1
        pseudo_label = max_label + 1
        
        y_spl = y_train_aug.copy()
        y_spl[kuc_aug] = pseudo_label
        known_classes = np.unique(y_train_aug[kc_aug])
        
        # Class prototypes (from original, non-augmented data)
        prototypes = {}
        for cls in known_classes:
            cls_samples = x_tr[y_train == cls]
            if len(cls_samples) > 0:
                prototypes[cls] = np.mean(cls_samples, axis=0)
        
        # Train SVM
        base_svm = SVC(
            C=C, kernel='rbf', gamma=gamma,
            class_weight=cw, random_state=42, probability=True,
        )
        unique_labels, counts = np.unique(y_spl, return_counts=True)
        min_samples = int(counts.min())
        
        if min_samples >= 5:
            svm = CalibratedClassifierCV(base_svm, cv=3, method=calib)
        elif min_samples >= 3:
            svm = CalibratedClassifierCV(base_svm, cv=2, method=calib)
        else:
            svm = base_svm
        
        fit_t = time.time()
        svm.fit(x_train_aug, y_spl)
        fit_time = time.time() - fit_t
        
        classes = svm.classes_
        kc_idx = [i for i, c in enumerate(classes) if c in known_classes]
        if len(kc_idx) == 0:
            return results
        
        # Predict on both sets
        eval_data = {}
        for tag, x_e, y_e in [('train', x_tr, y_train), ('val', x_va, y_val)]:
            proba = svm.predict_proba(x_e)
            kp = proba[:, kc_idx]
            bi = np.argmax(kp, axis=1)
            y_pk = np.array([classes[kc_idx[i]] for i in bi])
            mkp = np.max(kp, axis=1)
            # Cosine similarity to predicted prototype
            cs = np.zeros(len(x_e))
            for j, xj in enumerate(x_e):
                if y_pk[j] in prototypes:
                    cs[j] = np.dot(xj, prototypes[y_pk[j]])
            cs_norm = (cs + 1.0) / 2.0
            eval_data[tag] = (y_pk, mkp, cs_norm, y_e)
        
        # Inner loop: sweep score weights × threshold
        for pw in ss['prob_weight_options']:
            cw_score = 1.0 - pw
            tr_comb = pw * eval_data['train'][1] + cw_score * eval_data['train'][2]
            va_comb = pw * eval_data['val'][1] + cw_score * eval_data['val'][2]
            
            for thr in ss['threshold_options']:
                tr_pred = np.where(tr_comb >= thr, eval_data['train'][0], UNKNOWN_LABEL)
                va_pred = np.where(va_comb >= thr, eval_data['val'][0], UNKNOWN_LABEL)
                
                tr_m = compute_osr_metrics(y_train, tr_pred, tr_comb)
                va_m = compute_osr_metrics(y_val, va_pred, va_comb)
                gs, vs, ag = compute_generalization_score(tr_m, va_m)
                
                results.append({
                    'gen_score': gs, 'val_score': vs, 'avg_gap': ag,
                    'params': {
                        'preprocess': pipe.name,
                        'pca_components': pca_comp,
                        'n_dims': n_dims,
                        'C': C, 'gamma': gamma,
                        'class_weight': cw if cw is not None else None,
                        'calib_method': calib,
                        'mixup_ratio': mixup_ratio,
                        'threshold': thr,
                        'score_weights': [pw, cw_score],
                    },
                    'train_metrics': tr_m,
                    'val_metrics': va_m,
                    'fit_time': fit_time,
                })
    except Exception:
        pass  # Return empty list on error
    
    return results


def tune_spl(x_train, y_train, x_val, y_val):
    """Tune SPL hyperparameters with overfitting awareness."""
    ss = SPL_SEARCH_SPACE

    n_classifier_cfgs = (
        len(ss['pca_options']) * len(ss['C_options']) * len(ss['gamma_options'])
        * len(ss['class_weight_options']) * len(ss['mixup_ratio_options'])
        * len(ss['calib_method_options'])
    )
    n_score_cfgs = len(ss['prob_weight_options']) * len(ss['threshold_options'])

    print('\n' + '=' * 80)
    print('TUNING SPL (SVM) — OVERFITTING-AWARE')
    print('=' * 80)
    print(f'  Classifier configs : {n_classifier_cfgs}')
    print(f'  Score configs each : {n_score_cfgs}')
    print(f'  Total evaluations  : {n_classifier_cfgs * n_score_cfgs}')
    print(f'  Overfit penalty    : {OVERFIT_PENALTY_WEIGHT}')
    
    # Generate all config combinations
    all_configs = []
    for pca_comp in ss['pca_options']:
        for mixup_ratio in ss['mixup_ratio_options']:
            for C, gamma, cw, calib in product(
                ss['C_options'], ss['gamma_options'],
                ss['class_weight_options'], ss['calib_method_options'],
            ):
                all_configs.append((pca_comp, mixup_ratio, C, gamma, cw, calib,
                                   x_train, y_train, x_val, y_val, ss))
    
    print(f'  Using {min(32, cpu_count(), len(all_configs))} parallel workers...')
    
    # Parallel evaluation
    t0 = time.time()
    n_jobs = min(32, cpu_count(), len(all_configs))
    with Pool(n_jobs) as pool:
        results_lists = pool.map(evaluate_spl_config, all_configs)
    
    # Flatten results
    results = [r for sublist in results_lists for r in sublist]
    elapsed = time.time() - t0
    
    print(f'\n  Done — {len(results)} valid configurations evaluated in {elapsed:.1f}s')
    results.sort(key=lambda r: r['gen_score'], reverse=True)
    return results


#MPL TUNING
# =============================================================================

def evaluate_mpl_config(args):
    """Evaluate a single MPL configuration (worker function for multiprocessing)."""
    (pca_comp, mixup_ratio, C, solver, cw_param, max_iter, x_train, y_train, x_val, y_val, ss) = args
    
    results = []
    
    try:
        pipe = PreprocessPipeline(pca_components=pca_comp)
        x_tr = pipe.fit_transform(x_train)
        x_va = pipe.transform(x_val)
        n_dims = pipe.n_dims or x_tr.shape[1]
        
        kc_mask = y_train >= 0
        kuc_mask = y_train == UNKNOWN_LABEL
        
        if mixup_ratio > 0 and kuc_mask.any():
            x_kc = x_tr[kc_mask]
            y_kc = y_train[kc_mask]
            x_kuc = x_tr[kuc_mask]
            x_syn = mixup_augmentation(x_kc, y_kc, x_kuc, ratio=mixup_ratio, alpha=0.4)
            x_train_aug = np.vstack([x_tr, x_syn])
            y_train_aug = np.concatenate([y_train, np.full(len(x_syn), UNKNOWN_LABEL)])
        else:
            x_train_aug = x_tr
            y_train_aug = y_train.copy()
        
        # MPL labels — each KUC gets a unique pseudo label
        kc_aug = y_train_aug >= 0
        kuc_aug = y_train_aug == UNKNOWN_LABEL
        max_label = y_train_aug[kc_aug].max() if kc_aug.any() else -1
        
        y_mpl = y_train_aug.copy()
        kuc_indices = np.where(kuc_aug)[0]
        for idx, ki in enumerate(kuc_indices):
            y_mpl[ki] = max_label + 1 + idx
        
        known_classes = np.unique(y_train_aug[kc_aug])
        
        # Class centroids (from original, non-augmented data)
        centroids = {}
        for cls in known_classes:
            cs = x_tr[y_train == cls]
            if len(cs) > 0:
                centroids[cls] = np.mean(cs, axis=0)
        
        lr = LogisticRegression(
            C=C, solver=solver, max_iter=max_iter,
            class_weight=cw_param, random_state=42,
        )
        fit_t = time.time()
        lr.fit(x_train_aug, y_mpl)
        fit_time = time.time() - fit_t
        
        classes = lr.classes_
        kc_idx = [i for i, c in enumerate(classes) if c in known_classes]
        if len(kc_idx) == 0:
            return results
        
        eval_data = {}
        for tag, x_e, y_e in [('train', x_tr, y_train), ('val', x_va, y_val)]:
            proba = lr.predict_proba(x_e)
            kp = proba[:, kc_idx]
            bi = np.argmax(kp, axis=1)
            y_pk = np.array([classes[kc_idx[i]] for i in bi])
            mkp = np.max(kp, axis=1)
            cs = np.zeros(len(x_e))
            for j, xj in enumerate(x_e):
                if y_pk[j] in centroids:
                    cs[j] = np.dot(xj, centroids[y_pk[j]])
            cs_norm = (cs + 1.0) / 2.0
            eval_data[tag] = (y_pk, mkp, cs_norm, y_e)
        
        for pw in ss['prob_weight_options']:
            cw_score = 1.0 - pw
            tr_comb = pw * eval_data['train'][1] + cw_score * eval_data['train'][2]
            va_comb = pw * eval_data['val'][1] + cw_score * eval_data['val'][2]
            
            for thr in ss['threshold_options']:
                tr_pred = np.where(tr_comb >= thr, eval_data['train'][0], UNKNOWN_LABEL)
                va_pred = np.where(va_comb >= thr, eval_data['val'][0], UNKNOWN_LABEL)
                
                tr_m = compute_osr_metrics(y_train, tr_pred, tr_comb)
                va_m = compute_osr_metrics(y_val, va_pred, va_comb)
                gs, vs, ag = compute_generalization_score(tr_m, va_m)
                
                results.append({
                    'gen_score': gs, 'val_score': vs, 'avg_gap': ag,
                    'params': {
                        'preprocess': pipe.name,
                        'pca_components': pca_comp,
                        'n_dims': n_dims,
                        'C': C, 'solver': solver,
                        'max_iter': max_iter,
                        'class_weight': cw_param if cw_param is not None else None,
                        'mixup_ratio': mixup_ratio,
                        'threshold': thr,
                        'score_weights': [pw, cw_score],
                    },
                    'train_metrics': tr_m,
                    'val_metrics': va_m,
                    'fit_time': fit_time,
                })
    except Exception:
        pass  # Return empty list on error
    
    return results


def tune_mpl(x_train, y_train, x_val, y_val):
    """Tune MPL hyperparameters with overfitting awareness."""
    ss = MPL_SEARCH_SPACE

    n_classifier_cfgs = (
        len(ss['pca_options']) * len(ss['C_options']) * len(ss['solver_options'])
        * len(ss['class_weight_options']) * len(ss['mixup_ratio_options'])
    )
    n_score_cfgs = len(ss['prob_weight_options']) * len(ss['threshold_options'])

    print('\n' + '=' * 80)
    print('TUNING MPL (Logistic Regression) — OVERFITTING-AWARE')
    print('=' * 80)
    print(f'  Classifier configs : {n_classifier_cfgs}')
    print(f'  Score configs each : {n_score_cfgs}')
    print(f'  Total evaluations  : {n_classifier_cfgs * n_score_cfgs}')
    print(f'  Overfit penalty    : {OVERFIT_PENALTY_WEIGHT}')
    
    # Generate all config combinations
    all_configs = []
    for pca_comp in ss['pca_options']:
        for mixup_ratio in ss['mixup_ratio_options']:
            for C, solver, cw_param, max_iter in product(
                ss['C_options'], ss['solver_options'],
                ss['class_weight_options'], ss['max_iter_options'],
            ):
                all_configs.append((pca_comp, mixup_ratio, C, solver, cw_param, max_iter,
                                   x_train, y_train, x_val, y_val, ss))
    
    print(f'  Using {min(32, cpu_count(), len(all_configs))} parallel workers...')
    
    # Parallel evaluation
    t0 = time.time()
    n_jobs = min(32, cpu_count(), len(all_configs))
    with Pool(n_jobs) as pool:
        results_lists = pool.map(evaluate_mpl_config, all_configs)
    
    # Flatten results
    results = [r for sublist in results_lists for r in sublist]
    elapsed = time.time() - t0
    
    print(f'\n  Done — {len(results)} valid configurations evaluated in {elapsed:.1f}s')
    results.sort(key=lambda r: r['gen_score'], reverse=True)
    return results


# REPORTING
#=============================================================================

def print_top_results(results, model_name, top_n=10):
    """Print top-N results with overfitting analysis."""
    print(f'\n{"=" * 100}')
    print(f'TOP {min(top_n, len(results))} {model_name} CONFIGURATIONS '
          f'(sorted by generalization score, penalty={OVERFIT_PENALTY_WEIGHT})')
    print(f'{"=" * 100}')

    for i, r in enumerate(results[:top_n]):
        p = r['params']
        tr = r['train_metrics']
        va = r['val_metrics']

        print(f'\n{"─" * 100}')
        print(f'  #{i+1}  Gen Score: {r["gen_score"]:.4f}  |  '
              f'Val Score: {r["val_score"]:.4f}  |  Avg Overfit Gap: {r["avg_gap"]:.4f}')
        print(f'  Preprocess : {p["preprocess"]}'
              + (f'  ({p["n_dims"]} dims)' if p.get("n_dims") else ''))
        print(f'  Classifier : C={p["C"]}, '
              + (f'gamma={p["gamma"]}, ' if 'gamma' in p else f'solver={p["solver"]}, ')
              + f'class_weight={p["class_weight"]}')
        print(f'  Augment    : mixup_ratio={p["mixup_ratio"]}')
        print(f'  Scoring    : prob_w={p["score_weights"][0]}, '
              f'cos_w={p["score_weights"][1]}, threshold={p["threshold"]}')
        print(f'  Fit time   : {r["fit_time"]:.2f}s')

        print(f'  {"Metric":<22} {"Train":>10} {"Val":>10} {"Gap":>10} {"Status":>10}')
        print(f'  {"─" * 62}')
        for name, key in [('AUC-ROC', 'auc_roc'), ('KC Accuracy', 'kc_acc'),
                          ('Unknown Recall', 'unk_recall'), ('Balanced Rank-1', 'balanced_rank1'),
                          ('Overall Accuracy', 'overall_acc')]:
            gap = tr[key] - va[key]
            flag = 'OVER' if gap > 0.10 else 'MARG' if gap > 0.05 else '  OK'
            print(f'  {name:<22} {tr[key]:>10.4f} {va[key]:>10.4f} {gap:>+10.4f} {flag:>10}')


# MAIN
#=============================================================================

def main():
    print('=' * 80)
    print('HYPERPARAMETER TUNING WITH OVERFITTING AWARENESS')
    print('=' * 80)
    print()
    print('This script finds parameters that generalise well by penalising')
    print('configurations where training performance greatly exceeds validation.')
    print(f'Overfit penalty weight: {OVERFIT_PENALTY_WEIGHT}')
    print()

    # Load data
    print('Loading challenge data...')
    x_full, y_full = load_challenge_train_data()
    n_kc = (y_full >= 0).sum()
    n_kuc = (y_full == UNKNOWN_LABEL).sum()
    n_classes = len(np.unique(y_full[y_full >= 0]))
    print(f'  Total samples : {len(x_full)}')
    print(f'  Known classes : {n_kc} ({n_classes} unique labels)')
    print(f'  Known unknowns: {n_kuc}')
    print(f'  Features      : {x_full.shape[1]}')

    # Train / validation split
    print('\nSplitting 75% train / 25% validation (stratified)...')
    x_train, x_val, y_train, y_val = train_test_split(
        x_full, y_full, test_size=0.25, random_state=42, stratify=y_full,
    )
    print(f'  Training  : {len(x_train)}')
    print(f'  Validation: {len(x_val)}')

    # Tune SPL
    spl_t0 = time.time()
    spl_results = tune_spl(x_train, y_train, x_val, y_val)
    spl_time = time.time() - spl_t0

    # Tune MPL
    mpl_t0 = time.time()
    mpl_results = tune_mpl(x_train, y_train, x_val, y_val)
    mpl_time = time.time() - mpl_t0

    # Print top results
    print_top_results(spl_results, 'SPL', top_n=10)
    print_top_results(mpl_results, 'MPL', top_n=10)

    # Summary
    print(f'\n{"=" * 80}')
    print('TIMING SUMMARY')
    print(f'{"=" * 80}')
    print(f'  SPL tuning: {spl_time:.1f}s ({len(spl_results)} evaluations)')
    print(f'  MPL tuning: {mpl_time:.1f}s ({len(mpl_results)} evaluations)')
    print(f'  Total     : {spl_time + mpl_time:.1f}s')


if __name__ == '__main__':
    main()