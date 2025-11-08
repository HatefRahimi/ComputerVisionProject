import os
import pickle

import cv2
import numpy as np
from collections import Counter
from config import Config


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
    def get_embedding_dimensionality(cls):
        """Get dimensionality of the extracted embeddings."""
        return 128


# The FaceRecognizer model enables supervised face identification.
class FaceRecognizer:

    # Prepare FaceRecognizer; specify all parameters for face identification.
    def __init__(self, num_neighbours=7, max_distance=0.7, min_prob=0.9):
        # TODO: Prepare FaceNet and set all parameters for kNN.
        self.facenet = FaceNet()
        # store the k for k-NN, plus your reject thresholds
        self.k = num_neighbours
        self.max_distance = max_distance
        self.min_prob = min_prob

        # The underlying gallery: class labels and embeddings.
        self.labels = []
        self.embeddings = np.empty((0, FaceNet.get_embedding_dimensionality))

        # Load face recognizer from pickle file if available.
        if os.path.exists(Config.REC_GALLERY):
            self.load()

    # Save the trained model as a pickle file.
    def save(self):
        print("FaceRecognizer saving: {}".format(Config.CLUSTER_GALLERY))
        with open(Config.REC_GALLERY, "wb") as f:
            pickle.dump((self.labels, self.embeddings), f)

    # Load trained model from a pickle file.
    def load(self):
        print("FaceRecognizer loading: {}".format(Config.CLUSTER_GALLERY))
        with open(Config.REC_GALLERY, "rb") as f:
            (self.labels, self.embeddings) = pickle.load(f)

    # TODO: Train face identification with a new face with labeled identity.
    def partial_fit(self, face, label):
        # 1) Extract its 128-D embedding
        emb = self.facenet.predict(face)
        # added
        emb = emb.flatten()

        # 2) Grayscale embedding
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        emb_g = self.facenet.predict(gray3).flatten()

        # 3) Append to gallery
        # self.labels.append(label)
        self.labels.extend([label, label])
        self.embeddings = np.vstack([self.embeddings, emb[None, :], emb_g[None, :]])

    def _knn(self, probe_emb: np.ndarray):
        # 1) Compute L2 distances to all gallery embeddings
        dists = np.linalg.norm(self.embeddings - probe_emb, axis=1)

        # 2) Find the k nearest neighbors
        idx = np.argsort(dists)[: self.k]
        nearest_labels = [self.labels[i] for i in idx]
        nearest_distances = [dists[i] for i in idx]

        # 3) Majority vote for label
        vote, count = Counter(nearest_labels).most_common(1)[0]
        prob = count / self.k

        class_dists = [d for l, d in zip(nearest_labels, nearest_distances) if l == vote]
        class_dist = min(class_dists) if class_dists else float("inf")

        # 5) Otherwise return predicted label + confidence
        return vote, prob, class_dist

    # TODO: Predict the identity for a new face.
    def predict(self, face):
        if self.embeddings.shape[0] == 0:
            return None, 0.0

        # 1) Embed the color probe
        emb_c = self.facenet.predict(face).flatten()

        # 2) Gray probe
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        emb_g = self.facenet.predict(gray3).flatten()

        # 3) k-NN on each
        v_c, p_c, d_c = self._knn(emb_c)
        v_g, p_g, d_g = self._knn(emb_g)

        # 4) Fuse: if they agree, average; else take higher post_prob
        if v_c == v_g:
            v_c, (p_c + p_g) / 2.0, (d_c + d_g) / 2.0
        if p_c > p_g:
            v, p, d = v_c, p_c, d_c
        else:
            v, p, d = v_g, p_g, d_g

        # 4) OPEN-SET RULE:
        #    if too far (d>τᵈ) or too uncertain (p<τᵖ) → unknown
        if d > self.max_distance or p < self.min_prob:
            return "unknown", p, d
        # 5) otherwise return the predicted class
        return v, p, d


# The FaceClustering class enables unsupervised clustering of face images according to their
# identity and re-identification.
class FaceClustering:

    # Prepare FaceClustering; specify all parameters of clustering algorithm.
    def __init__(self, num_clusters=2, max_iter=25):
        # TODO: Prepare FaceNet.
        self.facenet = FaceNet()

        # The underlying gallery: embeddings without class labels.
        self.embeddings = np.empty((0, FaceNet.get_embedding_dimensionality))

        # Number of cluster centers for k-means clustering.
        self.num_clusters = num_clusters
        # Cluster centers.
        self.cluster_center = np.empty((num_clusters, FaceNet.get_embedding_dimensionality))
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
            (self.embeddings, self.num_clusters, self.cluster_center, self.cluster_membership) = (
                pickle.load(f)
            )

    # TODO
    def partial_fit(self, face):
        emb = self.facenet.predict(face).flatten()
        self.embeddings = np.vstack([self.embeddings, emb[None, :]])

    # TODO
    def fit(self):
        # 1) run k-means on self.embeddings
        X = self.embeddings
        n, dim = X.shape
        k = self.num_clusters

        # 2.a) Initialize centers by sampling k points
        idx0 = np.random.choice(n, k, replace=False)
        centers = X[idx0].copy()

        membership = np.zeros(n, dtype=int)
        objective = []

        for it in range(self.max_iter):
            # 2.b) Assignment: each point to nearest center
            #   compute distance matrix: shape (n,k)
            dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
            membership = np.argmin(dists, axis=1)

            # 2.c) Objective: sum of squared distances
            obj = 0.0
            for i in range(k):
                pts = X[membership == i]
                obj += np.sum((pts - centers[i]) ** 2)
            objective.append(obj)

            # 2.d) Update centers
            new_centers = np.zeros_like(centers)
            for i in range(k):
                pts = X[membership == i]
                if len(pts) > 0:
                    new_centers[i] = pts.mean(axis=0)
                else:
                    # if a cluster lost all points, reinitialize it
                    new_centers[i] = X[np.random.randint(n)]

            # check for convergence
            if np.allclose(new_centers, centers):
                break
            centers = new_centers

        # store results
        self.cluster_center = centers
        self.cluster_membership = membership.tolist()

        # return the objective history (for plotting if you like)
        return objective

    # TODO
    def predict(self, face):
        # 1) Embed the face
        emb = self.facenet.predict(face).flatten()  # 128-d vector

        # 2) Compute distances to each cluster center
        #    self.cluster_center is shape (k,128)
        #    so broadcasting gives a (k,) array
        dists = np.linalg.norm(self.cluster_center - emb[None, :], axis=1)

        # 3) Pick the argmin
        best = int(np.argmin(dists))

        return best, dists
