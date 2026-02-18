import pickle

import numpy as np

from cvproj_exc.classifier import NearestNeighborClassifier

# Class label for unknown subjects in test and training data.
UNKNOWN_LABEL = -1


# Evaluation of open-set face identification.
class OpenSetEvaluation:

    def __init__(
        self,
        classifier=NearestNeighborClassifier(),
        false_alarm_rate_range=np.logspace(-3, 0, 1000, endpoint=True),
    ):
        # The false alarm rates.
        self.false_alarm_rate_range = false_alarm_rate_range

        # Datasets (embeddings + labels) used for training and testing.
        self.train_embeddings = []
        self.train_labels = []
        self.test_embeddings = []
        self.test_labels = []

        # The evaluated classifier (see classifier.py)
        self.classifier = classifier

    # Prepare the evaluation by reading training and test data from file.
    def prepare_input_data(self, train_data_file, test_data_file):
        with open(train_data_file, "rb") as f:
            (self.train_embeddings, self.train_labels) = pickle.load(f, encoding="bytes")
        with open(test_data_file, "rb") as f:
            (self.test_embeddings, self.test_labels) = pickle.load(f, encoding="bytes")

    # Run the evaluation and find performance measure (identification rates) at different
    # similarity thresholds.
    def run(self):
        # Fit the classifier on the training data
        self.classifier.fit(self.train_embeddings, self.train_labels)
        
        # Predict similarities on the test data
        prediction_labels, similarities = self.classifier.predict_labels_and_similarities(self.test_embeddings)
        
        # Initialize lists to store results
        similarity_thresholds = []
        identification_rates = []
        
        # For each false alarm rate, find a similarity threshold and compute identification rate
        for false_alarm_rate in self.false_alarm_rate_range:
            # Find similarity threshold that yields this false alarm rate
            similarity_threshold = self.select_similarity_threshold(similarities, false_alarm_rate)
            
            # Compute identification rate at this threshold
            identification_rate = self.calc_identification_rate(prediction_labels, similarities, similarity_threshold)
            
            # Store results
            similarity_thresholds.append(similarity_threshold)
            identification_rates.append(identification_rate)
        
        # Return all false alarm rates, identification rates, and similarity thresholds
        evaluation_results = {
            "similarity_thresholds": similarity_thresholds,
            "identification_rates": identification_rates,
        }
        
        return evaluation_results

    def select_similarity_threshold(self, similarities, false_alarm_rate):
        # Filter similarities for unknown faces only (false alarms are about unknown faces)
        unknown_mask = self.test_labels == UNKNOWN_LABEL
        unknown_similarities = similarities[unknown_mask]
        
        if len(unknown_similarities) == 0:
            raise ValueError("No unknown faces in test data.")
        
        # For false_alarm_rate = 0.01, we want 1% of unknown faces to be accepted
        # Accepted means similarity >= threshold, so 99% should be below threshold
        percentile = 100 * (1.0 - false_alarm_rate)
        threshold = np.percentile(unknown_similarities, percentile)
        
        return threshold

    def calc_identification_rate(self, prediction_labels, similarities, similarity_threshold):
        # Only consider known faces for identification rate
        known_mask = self.test_labels != UNKNOWN_LABEL
        known_test_labels = self.test_labels[known_mask]
        known_predictions = prediction_labels[known_mask]
        known_similarities = similarities[known_mask]
        
        if len(known_test_labels) == 0:
            return 0.0
        
        # Apply threshold: only count accepted predictions (similarity >= threshold)
        accepted_mask = known_similarities >= similarity_threshold
        accepted_test_labels = known_test_labels[accepted_mask]
        accepted_predictions = known_predictions[accepted_mask]
        
        if len(accepted_test_labels) == 0:
            return 0.0
        
        # At rank 1, check if predicted label matches true label (only for accepted predictions)
        total_correct = np.sum(accepted_predictions == accepted_test_labels)
        return total_correct / len(known_test_labels)