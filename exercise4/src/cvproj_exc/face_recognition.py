import os
import pickle
from datetime import datetime
from collections import Counter
import cv2
import numpy as np

from cvproj_exc.config import Config


# FaceNet to extract face embeddings.
class FaceNet:

    def __init__(self):
        self.facenet = cv2.dnn.readNetFromONNX(str(Config.RESNET50))

    # Predict embedding from a given face image.
    def predict(self, face):
        # Normalize face image using mean subtraction.
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB) - (131.0912, 103.8827, 91.4953)

        # Forward pass through deep neural network. The input size should be 224 x 224.
        reshaped = np.moveaxis(face, 2, 0)
        reshaped = np.expand_dims(reshaped, axis=0)
        self.facenet.setInput(reshaped)
        embedding = np.squeeze(self.facenet.forward())
        return embedding / np.linalg.norm(embedding)

    @classmethod
    @property
    def embedding_dimensionality(cls):
        """Get dimensionality of the extracted embeddings."""
        return 128


# The FaceRecognizer model enables supervised face identification.
class FaceRecognizer:

    # Prepare FaceRecognizer; specify all parameters for face identification.
    def __init__(self, num_neighbours=1, max_distance=2.0, min_prob=0.0):
        # TODO: Prepare FaceNet and set all parameters for kNN.
        # Initialize FaceNet for feature extraction
        self.facenet = FaceNet()
        # Set k-NN parameters
        self.k = num_neighbours
        # Set open-set thresholds
        self.max_distance = max_distance  # Distance threshold (td)
        self.min_prob = min_prob  # Probability threshold (tp)

        # The underlying gallery: class labels and embeddings.
        self.labels = []
        self.embeddings = np.empty((0, FaceNet.embedding_dimensionality))

        # Load face recognizer from pickle file if available.
        if os.path.exists(Config.REC_GALLERY):
            self.load()

    # Save the trained model as a pickle file.
    def save(self):
        print("FaceRecognizer saving: {}".format(Config.REC_GALLERY))
        with open(Config.REC_GALLERY, "wb") as f:
            pickle.dump((self.labels, self.embeddings), f)

    # Load trained model from a pickle file.
    def load(self):
        print("FaceRecognizer loading: {}".format(Config.REC_GALLERY))
        with open(Config.REC_GALLERY, "rb") as f:
            (self.labels, self.embeddings) = pickle.load(f)

    # TODO: Train face identification with a new face with labeled identity.
    def partial_fit(self, face, label):
        # Extract color embedding from BGR face image
        color_embedding = self.facenet.predict(face)
        
        # Convert to grayscale (3-channel for FaceNet compatibility)
        gray_face = cv2.cvtColor(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        gray_embedding = self.facenet.predict(gray_face)
        
        # Store both embeddings as separate gallery entries with same label
        # Even indices = color embeddings, odd indices = grayscale embeddings
        if self.embeddings.size == 0:
            self.embeddings = np.vstack([color_embedding, gray_embedding])
        else:
            self.embeddings = np.vstack([self.embeddings, color_embedding, gray_embedding])
        
        self.labels.extend([label, label])

    # TODO: Predict the identity for a new face.
    def predict(self, face, debug_file=None) -> tuple[str, float, float]:
        # Check if gallery is empty
        if len(self.labels) == 0:
            return ('unknown', 0.0, np.inf)
        
        # Extract query embeddings (color and grayscale)
        color_embedding = self.facenet.predict(face)
        gray_face = cv2.cvtColor(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        gray_embedding = self.facenet.predict(gray_face)
        
        # Separate color and grayscale galleries
        # Gallery structure: even indices = color, odd indices = grayscale
        num_samples = len(self.labels) // 2
        if num_samples == 0:
            return ('unknown', 0.0, np.inf)
        
        color_embeddings = self.embeddings[::2]  # Even indices
        gray_embeddings = self.embeddings[1::2]   # Odd indices
        color_labels = self.labels[::2]  # Labels for color embeddings
        
        # Compute Euclidean distances to all gallery embeddings
        color_distances = np.linalg.norm(color_embeddings - color_embedding, axis=1)
        gray_distances = np.linalg.norm(gray_embeddings - gray_embedding, axis=1)
        
        # Find k nearest neighbors for each (using argpartition for efficiency)
        k_actual = min(self.k, num_samples)
        if k_actual == 0:
            return ('unknown', 0.0, np.inf)
        
        color_k_indices = np.argpartition(color_distances, k_actual - 1)[:k_actual]
        gray_k_indices = np.argpartition(gray_distances, k_actual - 1)[:k_actual]
        
        # Get labels and distances for k nearest neighbors
        color_k_labels = [color_labels[i] for i in color_k_indices]
        gray_k_labels = [color_labels[i] for i in gray_k_indices]
        color_k_distances = color_distances[color_k_indices]
        gray_k_distances = gray_distances[gray_k_indices]
        
        # Weighted voting (0.6 color, 0.4 grayscale)
        weight_color = 0.6
        weight_gray = 0.4
        label_votes = {}
        
        for label in color_k_labels:
            label_votes[label] = label_votes.get(label, 0) + weight_color
        for label in gray_k_labels:
            label_votes[label] = label_votes.get(label, 0) + weight_gray
        
        if not label_votes:
            return ('unknown', 0.0, np.inf)
        
        # Majority vote (weighted)
        predicted_label = max(label_votes, key=label_votes.get)
        
        # Compute posterior probability: p(Ci|x) = (weighted ki) / (weighted k)
        ki_color = sum(1 for l in color_k_labels if l == predicted_label)
        ki_gray = sum(1 for l in gray_k_labels if l == predicted_label)
        posterior_prob = (weight_color * ki_color + weight_gray * ki_gray) / (weight_color * k_actual + weight_gray * k_actual)
        
        # Compute distance to predicted class: d(Ci|x) = min distance to class neighbors
        class_color_dists = [color_k_distances[i] for i, l in enumerate(color_k_labels) if l == predicted_label]
        class_gray_dists = [gray_k_distances[i] for i, l in enumerate(gray_k_labels) if l == predicted_label]
        class_distance = min(class_color_dists + class_gray_dists) if (class_color_dists or class_gray_dists) else np.inf
        if debug_file:
            self._write_debug_info(
                debug_file, k_actual, color_k_labels, color_k_distances,
                label_votes, predicted_label, posterior_prob, class_distance,
                num_samples
            )
        # Open-set decision rule: reject if distance > threshold OR probability < threshold
        if class_distance > self.max_distance or posterior_prob < self.min_prob:
            return ('unknown', posterior_prob, class_distance)
        
        return (predicted_label, posterior_prob, class_distance)

    def _write_debug_info(self, debug_file, k_actual, k_labels, k_distances, 
                        label_votes, predicted_label, posterior_prob, 
                        class_distance, num_samples):
        """Write detailed k-NN debug information to file."""
        with open(debug_file, 'a') as f:
            f.write("=" * 80 + "\n")
            f.write(f"DEBUG INFO - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n")
            f.write(f"k (actual): {k_actual}\n")
            f.write(f"Total gallery samples: {num_samples}\n")
            f.write(f"Total gallery embeddings: {len(self.labels)}\n")
            f.write("\n")
            
            f.write("Top k Nearest Neighbors (sorted by distance):\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Rank':<6} {'Label':<25} {'Distance':<15} {'Count':<10}\n")
            f.write("-" * 80 + "\n")
            
            label_counts_so_far = {}
            for i, (label, dist) in enumerate(zip(k_labels, k_distances), 1):
                label_counts_so_far[label] = label_counts_so_far.get(label, 0) + 1
                f.write(f"{i:<6} {label:<25} {dist:<15.6f} {label_counts_so_far[label]:<10}\n")
            
            f.write("\n")
            f.write("Label Votes (for majority voting):\n")
            f.write("-" * 80 + "\n")
            for label, votes in sorted(label_votes.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {label}: {votes} votes\n")
            
            f.write("\n")
            f.write("Prediction Results:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Predicted Label: {predicted_label}\n")
            f.write(f"Posterior Probability: {posterior_prob:.6f} ({posterior_prob*100:.2f}%)\n")
            f.write(f"Distance to Predicted Class: {class_distance:.6f}\n")
            f.write(f"Max Distance Threshold: {self.max_distance}\n")
            f.write(f"Min Probability Threshold: {self.min_prob}\n")
            
            label_counts = Counter(k_labels)
            f.write("\n")
            f.write("Neighbor Distribution:\n")
            f.write("-" * 80 + "\n")
            for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / k_actual) * 100
                f.write(f"  {label}: {count}/{k_actual} ({percentage:.1f}%)\n")
            
            f.write("\n")
            f.write("Distance Statistics:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Min distance: {k_distances[0]:.6f}\n")
            f.write(f"Max distance: {k_distances[-1]:.6f}\n")
            f.write(f"Mean distance: {np.mean(k_distances):.6f}\n")
            f.write(f"Median distance: {np.median(k_distances):.6f}\n")
            f.write(f"Std deviation: {np.std(k_distances):.6f}\n")
            f.write("\n" * 2)
# The FaceClustering class enables unsupervised clustering of face images according to their
# identity and re-identification.
class FaceClustering:

    # Prepare FaceClustering; specify all parameters of clustering algorithm.
    def __init__(self, num_clusters=2, max_iter=200):
        # Initialize FaceNet for embedding extraction
        self.facenet = FaceNet()    

        # The underlying gallery: embeddings without class labels.
        self.embeddings = np.empty((0, FaceNet.embedding_dimensionality))

        # Number of cluster centers for k-means clustering.
        self.num_clusters = num_clusters
        # Cluster centers. Initialize as empty array (size 0) so fit() will use random initialization
        # Don't use np.empty() as it creates uninitialized values that pass the size check
        self.cluster_center = np.array([]).reshape(0, FaceNet.embedding_dimensionality)
        # Cluster index associated with the different samples.
        self.cluster_membership = []

        # Maximum number of iterations for k-means clustering.
        self.max_iter = max_iter

        # Load face clustering from pickle file if available.
        if os.path.exists(Config.CLUSTER_GALLERY):
            self.load()

    # Save the trained model as a pickle file.
    def save(self):
        print("FaceClustering saving: {}".format(Config.CLUSTER_GALLERY))
        with open(Config.CLUSTER_GALLERY, "wb") as f:
            pickle.dump(
                (self.embeddings, self.num_clusters, self.cluster_center, self.cluster_membership),
                f,
            )

    # Load trained model from a pickle file.
    def load(self):
        print("FaceClustering loading: {}".format(Config.CLUSTER_GALLERY))
        with open(Config.CLUSTER_GALLERY, "rb") as f:
            data = pickle.load(f)
            (self.embeddings, self.num_clusters, self.cluster_center, self.cluster_membership) = data
    
    # Train face clustering with a new face (unsupervised, no label).
    def partial_fit(self, face):
        # Extract embedding using FaceNet
        embedding = self.facenet.predict(face)  # Shape: (128,)
        
        # Append to embeddings array
        if self.embeddings.size == 0:
            self.embeddings = embedding.reshape(1, -1) #result (1, 128)
        else:
            self.embeddings = np.vstack([self.embeddings, embedding.reshape(1, -1)]) #result (n+1, 128)

    # Perform k-means clustering on stored embeddings.
    def fit(self):
        if len(self.embeddings) == 0:
            return None
        
        n_samples = len(self.embeddings) #return number of embeddings in the gallery
        n_features = self.embeddings.shape[1]  #128 (embedding dimension)
        
        # Incremental initialization: use existing cluster centers if available and valid,
        # otherwise use random initialization (first time or if num_clusters changed)
        has_existing_centers = (
            self.cluster_center.size > 0 and 
            self.cluster_center.shape[0] == self.num_clusters and
            self.cluster_center.shape[1] == n_features
        )
        
        if has_existing_centers:
            # Use existing centers as initialization (better initialization)
            # This allows the algorithm to adapt to new data while preserving previous structure
            print(f"Using existing cluster centers as initialization (incremental k-means)")
        else:
            # Random initialization: select k random embeddings as centers (first time only)
            random_indices = np.random.choice(n_samples, self.num_clusters, replace=False)
            self.cluster_center = self.embeddings[random_indices].copy()  # Shape: (k, 128)
            print(f"Random initialization: selected {self.num_clusters} random embeddings as initial centers")
        
        objective_history = [] #For convergence analysis plotting
        
        # k-Means iterations
        for iteration in range(self.max_iter):
            # Assignment step: assign each point to nearest center
            # Compute distance matrix D[i,j] = |embedding_i - center_j|
            # Shape: (n_samples, num_clusters) - each row is distances from one embedding to all centers
            distances = np.linalg.norm(
                self.embeddings[:, np.newaxis, :] - self.cluster_center[np.newaxis, :, :],
                axis=2
            )  # Shape: (n_samples, num_clusters)
            
            # For each embedding, find index of nearest center
            self.cluster_membership = np.argmin(distances, axis=1)  # Shape: (n_samples,)
            
            # Compute objective function: J = sum( |x_i - c_j|^2)            # Sum of squared distances from each point to its assigned center
            objective = np.sum(distances[np.arange(n_samples), self.cluster_membership]**2)
            objective_history.append(objective)
            
            # Update step: recalculate centers as means
            # For each cluster, compute centroid of assigned points
            new_centers = np.zeros_like(self.cluster_center)
            for k in range(self.num_clusters):
                mask = self.cluster_membership == k  # Boolean array: True for points in cluster k
                if np.any(mask):
                    new_centers[k] = np.mean(self.embeddings[mask], axis=0)  # Mean of cluster k
                else:
                    # Handle empty clusters: keep old center
                    new_centers[k] = self.cluster_center[k]
            
            # Check convergence: stop if centers don't change significantly
            if np.allclose(self.cluster_center, new_centers, atol=1e-6):
                break
            
            self.cluster_center = new_centers
        
        # Convert cluster_membership to list for consistency with save/load
        self.cluster_membership = self.cluster_membership.tolist()
        
        return objective_history  # Return for convergence analysis plotting

    # Predict the cluster for a new face.
    def predict(self, face) -> tuple[int, np.ndarray]:
        # Check if clustering has been performed
        # cluster_membership is only set by fit(), so if it's empty, clustering hasn't been done
        if len(self.cluster_membership) == 0:
            return None
        
        # Extract query embedding
        query_embedding = self.facenet.predict(face)  # Shape: (128,)
        
        # Compute distances to all cluster centers
        # cluster_center shape: (num_clusters, 128)
        # query_embedding shape: (128,)
        # Broadcasting: (k, 128) - (1, 128) compute distance for each center
        distances = np.linalg.norm(
            self.cluster_center - query_embedding[np.newaxis, :],
            axis=1
        )  # Shape: (num_clusters,) - one distance per cluster
        
        # Find best matching cluster (minimum distance)
        best_cluster = np.argmin(distances)
        
        # Return cluster index and distance distribution
        return (int(best_cluster), distances)
