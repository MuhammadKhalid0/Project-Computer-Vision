"""
Task 5.2.3: Train an SVM classifier on the extracted CNN features.

This script loads the 512-dim ResNet18 features extracted in Task 5.2.2,
standardises them, and trains a Support Vector Machine to distinguish
balloon regions (label=1) from background regions (label=0).

We use class_weight='balanced' to compensate for the large neg/pos imbalance
(~21:1 on training set). The trained model + scaler are persisted with joblib
so that the inference script (Task 5.2.4) can reload them.

Usage:
    python train_svm.py

Input:
    results/features_train.npz
    results/features_valid.npz

Output:
    results/svm_model.joblib      (trained SVM)
    results/svm_scaler.joblib     (fitted StandardScaler)
"""

import os
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib


# Paths
RESULTS_DIR = os.path.join('..', 'results')


def load_features(split_name):
    """
    Load feature vectors and labels for a given split.

    Returns
    -------
    X : np.ndarray, shape (N, 512)
    y : np.ndarray, shape (N,)
    """
    path = os.path.join(RESULTS_DIR, f'features_{split_name}.npz')
    data = np.load(path)
    return data['features'], data['labels']


def main():
    # ------------------------------------------------------------------
    # 1. Load features
    # ------------------------------------------------------------------
    print("Loading features ...")
    X_train, y_train = load_features('train')
    X_valid, y_valid = load_features('valid')

    print(f"  Train: {X_train.shape[0]} samples "
          f"({np.sum(y_train == 1)} pos, {np.sum(y_train == 0)} neg)")
    print(f"  Valid: {X_valid.shape[0]} samples "
          f"({np.sum(y_valid == 1)} pos, {np.sum(y_valid == 0)} neg)")

    # ------------------------------------------------------------------
    # 2. Standardise features (zero mean, unit variance)
    # ------------------------------------------------------------------
    # SVMs are sensitive to feature scale, so we fit a scaler on training
    # data and apply it to both train and validation.
    print("\nStandardising features ...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)

    # ------------------------------------------------------------------
    # 3. Train SVM
    # ------------------------------------------------------------------
    print("\nTraining SVM (RBF kernel, balanced class weights) ...")
    svm = SVC(
        kernel='rbf',
        C=0.1,
        gamma='scale',
        class_weight='balanced', #because of the imbalance in the data
        probability=True,      #needed for confidence scores at inference
        verbose=False,
    )
    svm.fit(X_train, y_train)
    print("  Training complete.")
    print(f"  Support vectors: {svm.n_support_} "
          f"(neg={svm.n_support_[0]}, pos={svm.n_support_[1]})")

    # ------------------------------------------------------------------
    # 4. Evaluate on training set
    # ------------------------------------------------------------------
    print("\n--- Training set performance ---")
    y_pred_train = svm.predict(X_train)
    print(classification_report(
        y_train, y_pred_train, target_names=['background', 'balloon']))

    # ------------------------------------------------------------------
    # 5. Evaluate on validation set
    # ------------------------------------------------------------------
    print("--- Validation set performance ---")
    y_pred_valid = svm.predict(X_valid)
    print(classification_report(
        y_valid, y_pred_valid, target_names=['background', 'balloon']))

    cm = confusion_matrix(y_valid, y_pred_valid)
    print("Confusion matrix (valid):")
    print(f"  TN={cm[0, 0]:5d}  FP={cm[0, 1]:5d}")
    print(f"  FN={cm[1, 0]:5d}  TP={cm[1, 1]:5d}")

    # ------------------------------------------------------------------
    # 6. Save model and scaler
    # ------------------------------------------------------------------
    model_path = os.path.join(RESULTS_DIR, 'svm_model.joblib')
    scaler_path = os.path.join(RESULTS_DIR, 'svm_scaler.joblib')

    joblib.dump(svm, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"\nModel saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    print("\nDone.")


if __name__ == '__main__':
    main()