from typing import Callable, Tuple

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

from cvproj_exc.config import Config

UNKNOWN_LABEL = -1


def normalize_features(x: np.ndarray) -> np.ndarray:
    """L2 normalization for face embeddings."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norms + 1e-8)


def mixup_augmentation(x_kc: np.ndarray, y_kc: np.ndarray, x_kuc: np.ndarray, ratio: float = 0.5, alpha: float = 0.4) -> np.ndarray:
    """
    Generate synthetic KUCs using mixup augmentation (LORD paper Section 4.4).
    
    Parameters
    ----------
    x_kc : Known class samples
    y_kc : Known class labels
    x_kuc : Known unknown class samples
    ratio : Ratio of synthetic samples to KC samples
    alpha : Beta distribution parameter for mixup
    
    Returns
    -------
    x_synthetic : Synthetic KUC samples
    """
    n_synthetic = int(len(x_kc) * ratio)
    x_synthetic = []
    
    # Mix KC with KC to create out-of-distribution samples
    for _ in range(n_synthetic // 2):
        idx1, idx2 = np.random.choice(len(x_kc), 2, replace=False)
        lam = np.random.beta(alpha, alpha)
        x_mix = lam * x_kc[idx1] + (1 - lam) * x_kc[idx2]
        x_synthetic.append(x_mix)
    
    # Mix KC with KUC for more diverse synthetic unknowns
    if len(x_kuc) > 0:
        for _ in range(n_synthetic - len(x_synthetic)):
            idx_kc = np.random.choice(len(x_kc))
            idx_kuc = np.random.choice(len(x_kuc))
            lam = np.random.beta(alpha, alpha)
            x_mix = lam * x_kc[idx_kc] + (1 - lam) * x_kuc[idx_kuc]
            x_synthetic.append(x_mix)
    else:
        # If no KUC, just do more KC mixup
        while len(x_synthetic) < n_synthetic:
            idx1, idx2 = np.random.choice(len(x_kc), 2, replace=False)
            lam = np.random.beta(alpha, alpha)
            x_mix = lam * x_kc[idx1] + (1 - lam) * x_kc[idx2]
            x_synthetic.append(x_mix)
    
    return np.array(x_synthetic)


def spl_training(
    x_train: np.ndarray, y_train: np.ndarray
) -> Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Implementation of the single pseudo label (SPL) approach.
    Do NOT change the interface of this function. For benchmarking we expect the given inputs and
    return values. Introduce additional helper functions if desired.

    Parameters
    ----------
    x_train : array, shape (n_samples, n_features). The feature vectors for training.
    y_train : array, shape (n_samples,). The ground truth labels of samples x.

    Returns
    -------
    spl_predict_fn :
        Callable, a function that holds a reference to your trained estimator and uses it to
        predict class labels and scores for the incoming test data.

        Parameters
        ----------
        x_test : array, shape (n_test_samples, n_features). The feature vectors for testing.

        Returns
        -------
        y_pred :    array, shape (n_samples,). The predicted class labels.
        y_score :   array, shape (n_samples,).
                    The similarities or confidence scores of the predicted class labels. We assume
                    that the scores are confidence/similarity values, i.e., a high value indicates
                    that the class prediction is trustworthy.
                    To be more precise:
                    - Returning probabilities in the range 0 to 1 is fine if 1 means high
                      confidence.
                    - Returning distances in the range -inf to 0 (or +inf) is fine if 0 (or +inf)
                      means high confidence.

                    Please ensure that your score is formatted accordingly.
    """
    np.random.seed(42)

    # Preprocessing: StandardScaler → PCA95 → L2 normalization (SPL #1)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    pca = PCA(n_components=0.95, random_state=42)
    x_train_pca = pca.fit_transform(x_train_scaled)
    x_train_norm = normalize_features(x_train_pca)
    
    # Separate KCs and KUCs
    kc_mask = y_train >= 0
    kuc_mask = y_train == UNKNOWN_LABEL
    
    # No mixup augmentation for SPL #1 (mixup_ratio=0.0)
    x_train_aug = x_train_norm
    y_train_aug = y_train.copy()
    
    # For SPL: assign all KUCs a single pseudo label
    kc_mask_aug = y_train_aug >= 0
    kuc_mask_aug = y_train_aug == UNKNOWN_LABEL
    max_kc_label = y_train_aug[kc_mask_aug].max() if kc_mask_aug.any() else -1
    pseudo_label = max_kc_label + 1
    
    y_train_spl = y_train_aug.copy()
    y_train_spl[kuc_mask_aug] = pseudo_label
    
    # Store known class labels for rejection
    known_classes = np.unique(y_train_aug[kc_mask_aug])
    
    # Train SVM with SPL #1 hyperparameters: C=50.0, gamma='auto', class_weight=None
    base_svm = SVC(
        C=50.0,  # SPL #1
        kernel='rbf',
        gamma='auto',  # SPL #1
        class_weight=None,  # SPL #1
        random_state=42,
        probability=True
    )
    
    # Check if we have enough samples per class for calibration
    unique_labels, counts = np.unique(y_train_spl, return_counts=True)
    min_samples = counts.min()
    
    if min_samples >= 3:
        # Use calibration if we have enough samples (tuned: sigmoid)
        svm = CalibratedClassifierCV(base_svm, cv=2, method='sigmoid')
        svm.fit(x_train_aug, y_train_spl)
    else:
        # Skip calibration if classes have too few samples
        svm = base_svm
        svm.fit(x_train_aug, y_train_spl)
    
    # Compute class prototypes for additional scoring (use original training data, not augmented)
    class_prototypes = {}
    for cls in known_classes:
        cls_samples = x_train_norm[y_train == cls]
        if len(cls_samples) > 0:
            class_prototypes[cls] = np.mean(cls_samples, axis=0)

    def spl_predict_fn(x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Apply same preprocessing: StandardScaler → PCA → L2 normalization
        x_test_scaled = scaler.transform(x_test)
        x_test_pca = pca.transform(x_test_scaled)
        x_test_norm = normalize_features(x_test_pca)
        
        # Get probability predictions
        proba = svm.predict_proba(x_test_norm)
        
        # Find pseudo label index in classes
        classes = svm.classes_
        pseudo_idx = np.where(classes == pseudo_label)[0][0] if pseudo_label in classes else -1
        
        # Get predictions excluding pseudo class for known classes
        # Sum probabilities of all known classes
        known_class_proba = np.zeros(len(x_test_norm))
        for i, cls in enumerate(classes):
            if cls in known_classes:
                known_class_proba += proba[:, i]
        
        # Get best known class prediction
        known_class_idx = []
        for i, cls in enumerate(classes):
            if cls in known_classes:
                known_class_idx.append(i)
        
        if len(known_class_idx) > 0:
            known_proba = proba[:, known_class_idx]
            best_known_idx = np.argmax(known_proba, axis=1)
            y_pred_known = np.array([classes[known_class_idx[i]] for i in best_known_idx])
            max_known_proba = np.max(known_proba, axis=1)
        else:
            y_pred_known = np.full(len(x_test_norm), UNKNOWN_LABEL)
            max_known_proba = np.zeros(len(x_test_norm))
        
        # Additional cosine similarity scoring
        cos_scores = np.zeros(len(x_test_norm))
        for i, x in enumerate(x_test_norm):
            if y_pred_known[i] in class_prototypes:
                cos_scores[i] = np.dot(x, class_prototypes[y_pred_known[i]])
        
        # Normalize cosine similarity to [0, 1]
        cos_scores_norm = (cos_scores + 1) / 2
        
        # Combine probability and cosine similarity (SPL #1: 0.4/0.6 weights)
        # This is the confidence score for the known class prediction
        combined_score = 0.40 * max_known_proba + 0.6 * cos_scores_norm
        
        # Decision: use threshold on combined score (SPL #1: threshold=0.45)
        threshold = 0.465
        
        y_pred_final = np.where(combined_score >= threshold, y_pred_known, UNKNOWN_LABEL)
        
        # y_score represents confidence in the prediction (high = confident)
        y_score = combined_score
        
        return y_pred_final, y_score

    return spl_predict_fn


def mpl_training(
    x_train: np.ndarray, y_train: np.ndarray
) -> Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Implementation of the multi pseudo label (MPL) approach.
    Do NOT change the interface of this function. For benchmarking we expect the given inputs and
    return values. Introduce additional helper functions if desired.

    Parameters
    ----------
    x_train : array, shape (n_samples, n_features). The feature vectors for training.
    y_train : array, shape (n_samples,). The ground truth labels of samples x.

    Returns
    -------
    mpl_predict_fn :
        Callable, a function that holds a reference to your trained estimator and uses it to
        predict class labels and scores for the incoming test data.

        Parameters
        ----------
        x_test : array, shape (n_test_samples, n_features). The feature vectors for testing.

        Returns
        -------
        y_pred :    array, shape (n_samples,). The predicted class labels.
        y_score :   array, shape (n_samples,).
                    The similarities or confidence scores of the predicted class labels. We assume
                    that the scores are confidence/similarity values, i.e., a high value indicates
                    that the class prediction is trustworthy.
                    To be more precise:
                    - Returning probabilities in the range 0 to 1 is fine if 1 means high
                      confidence.
                    - Returning distances in the range -inf to 0 (or +inf) is fine if 0 (or +inf)
                      means high confidence.

                    Please ensure that your score is formatted accordingly.
    """
    np.random.seed(42)
    
    # StandardScaler + L2 normalization preprocessing (MPL #9: no PCA)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_train_norm = normalize_features(x_train_scaled)
    
    # Separate KCs and KUCs
    kc_mask = y_train >= 0
    kuc_mask = y_train == UNKNOWN_LABEL
    
    # Mixup augmentation for MPL #9 (mixup_ratio=0.3)
    if kuc_mask.any():
        x_kc = x_train_norm[kc_mask]
        y_kc = y_train[kc_mask]
        x_kuc = x_train_norm[kuc_mask]
        
        # Generate synthetic KUCs using mixup
        x_synthetic = mixup_augmentation(x_kc, y_kc, x_kuc, ratio=0.3, alpha=0.4)
        
        # Augment training data
        x_train_aug = np.vstack([x_train_norm, x_synthetic])
        y_train_aug = np.concatenate([y_train, np.full(len(x_synthetic), UNKNOWN_LABEL)])
    else:
        x_train_aug = x_train_norm
        y_train_aug = y_train.copy()
    
    # For MPL: assign each KUC sample a unique pseudo label
    kc_mask_aug = y_train_aug >= 0
    kuc_mask_aug = y_train_aug == UNKNOWN_LABEL
    max_kc_label = y_train_aug[kc_mask_aug].max() if kc_mask_aug.any() else -1
    
    y_train_mpl = y_train_aug.copy()
    # Assign unique labels to each KUC (including augmented ones)
    kuc_indices = np.where(kuc_mask_aug)[0]
    for idx, kuc_idx in enumerate(kuc_indices):
        y_train_mpl[kuc_idx] = max_kc_label + 1 + idx
    
    # Store known class labels and pseudo labels
    known_classes = np.unique(y_train_aug[kc_mask_aug])
    pseudo_labels = set(y_train_mpl[kuc_mask_aug])
    
    # Train Logistic Regression with MPL #9 hyperparameters: C=10.0, solver='lbfgs', class_weight=None
    lr = LogisticRegression(
        C=10.0,  # MPL #9
        max_iter=500,
        class_weight=None,  # MPL #9
        solver='lbfgs',
        random_state=42
    )
    
    lr.fit(x_train_aug, y_train_mpl)
    
    # Compute class centroids for distance-based scoring (use original training data, not augmented)
    class_centroids = {}
    for cls in known_classes:
        cls_samples = x_train_norm[y_train == cls]
        if len(cls_samples) > 0:
            class_centroids[cls] = np.mean(cls_samples, axis=0)

    def mpl_predict_fn(x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Apply same preprocessing: StandardScaler + L2 normalization
        x_test_scaled = scaler.transform(x_test)
        x_test_norm = normalize_features(x_test_scaled)
        
        # Get probability predictions
        proba = lr.predict_proba(x_test_norm)
        classes = lr.classes_
        
        # Find indices of known classes in the classifier's class list
        known_class_idx = [i for i, cls in enumerate(classes) if cls in known_classes]
        
        if len(known_class_idx) > 0:
            # Get probabilities for known classes only
            known_proba = proba[:, known_class_idx]
            best_known_idx = np.argmax(known_proba, axis=1)
            y_pred_known = np.array([classes[known_class_idx[i]] for i in best_known_idx])
            max_known_proba = np.max(known_proba, axis=1)
        else:
            y_pred_known = np.full(len(x_test_norm), UNKNOWN_LABEL)
            max_known_proba = np.zeros(len(x_test_norm))
        
        # Distance-based scoring to nearest known class centroid
        centroid_scores = np.zeros(len(x_test_norm))
        for i, x in enumerate(x_test_norm):
            if y_pred_known[i] in class_centroids:
                # Cosine similarity to predicted known class centroid
                centroid_scores[i] = np.dot(x, class_centroids[y_pred_known[i]])
        
        # Normalize cosine scores to [0, 1]
        centroid_scores_norm = (centroid_scores + 1) / 2
        
        # Combine probability and centroid similarity (MPL #9: 0.5/0.5 weights)
        combined_score = 0.5 * max_known_proba + 0.5 * centroid_scores_norm
        
        # Decision rule: threshold on combined score (MPL #9: threshold=0.4)
        threshold = 0.415
        
        y_pred_final = np.where(combined_score >= threshold, y_pred_known, UNKNOWN_LABEL)
        
        # y_score represents confidence in the prediction
        y_score = combined_score
        
        return y_pred_final, y_score

    return mpl_predict_fn


def load_challenge_train_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the challenge training data.

    Returns
    -------
    x : array, shape (n_samples, n_features). The feature vectors.
    y : array, shape (n_samples,). The corresponding labels of samples x.
    """
    df = pd.read_csv(Config.CHAL_TRAIN_DATA, header=None).values
    x = df[:, :-1]
    y = df[:, -1].astype(int)
    return x, y


def main():
    x_train, y_train = load_challenge_train_data()

    # TODO: implement
    spl_predict_fn = spl_training(x_train, y_train)

    # TODO: implement
    mpl_predict_fn = mpl_training(x_train, y_train)

    # TODO: No todo, but this is roughly how we will test your implementation (with real data). So
    #       please make sure that this call (besides the unit tests) does what it is supposed to do.
    #       This is random data, you can not achieve good results on it. Split your training set to
    #       validate your performance.
    x_test = np.random.rand(50, x_train.shape[1])
    y_test = np.random.randint(-1, 5, 50)
    for predict_fn in (spl_predict_fn, mpl_predict_fn):
        y_pred, y_score = predict_fn(x_test)
        print("Acc: {}".format(np.equal(y_test, y_pred).sum() / len(x_test)))


if __name__ == "__main__":
    main()
