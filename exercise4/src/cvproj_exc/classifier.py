import cv2
import numpy as np


class NearestNeighborClassifier:

    def __init__(self):
        self.classifier = cv2.ml.KNearest_create()
        self.__reset()

    def __reset(self):
        self.classifier.setDefaultK(1)

    def fit(self, embeddings, labels):
        self.__reset()
        self.classifier.train(
            embeddings.astype(np.float32), cv2.ml.ROW_SAMPLE, labels.astype(np.float32)
        )

    def predict_labels_and_similarities(self, embeddings):
        _, prediction_labels, _, dists = self.classifier.findNearest(
            embeddings.astype(np.float32), k=1
        )
        similarities = -dists.flatten()
        return prediction_labels.flatten(), similarities
