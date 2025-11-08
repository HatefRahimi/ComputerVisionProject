import pickle

import numpy as np

from classifier import NearestNeighborClassifier

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
        """
              1) Train the classifier on self.train_embeddings/self.train_labels
              2) Score each probe in self.test_embeddings → (pred_label, similarity)
              3) For each FAR in self.false_alarm_rate_range:
                   a) pick threshold τ = select_similarity_threshold(similarities, FAR)
                   b) assign 'unknown' if sim < τ, else use pred_label
                   c) compute IR = calc_identification_rate(final_labels)
              4) return dict with arrays of thresholds and IRs.
        """
        # --- 1) Fit on training data ----------------------------
        self.classifier.fit(self.train_embeddings, self.train_labels)

        # --- 2) Score the test set -----------------------------
        all_sims = []
        all_preds = []
        for emb in self.test_embeddings:
            label_arr, sim_arr = self.classifier.predict_labels_and_similarities(emb)
            label = int(label_arr)
            sim = float(sim_arr)
            all_preds.append(label)
            all_sims.append(sim)
        all_sims = np.array(all_sims)
        all_preds = np.array(all_preds)

        # --- 3) Sweep false alarm rates -------------------------
        thresholds = []
        identification_rates = []
        for far in self.false_alarm_rate_range:
            # a) choose τ so that fraction FAR of *true* unknowns would exceed τ
            τ = self.select_similarity_threshold(all_sims, far)
            thresholds.append(τ)

            # b) final predicted = unknown if sim < τ else NN‐label
            final_labels = np.where(all_sims < τ, UNKNOWN_LABEL, all_preds)

            # c) rank-1 ID rate among all probes
            ir = self.calc_identification_rate(final_labels)
            identification_rates.append(ir)

        return {
            "false_alarm_rates": np.array(self.false_alarm_rate_range),
            "similarity_thresholds": np.array(thresholds),
            "identification_rates": np.array(identification_rates),
        }

    def select_similarity_threshold(self, similarity, false_alarm_rate):
        sims = np.array(similarity)
        # pick out the unknown‐true probes
        mask_unknown = np.array(self.test_labels) == UNKNOWN_LABEL
        unk_sims = sims[mask_unknown]
        if len(unk_sims) == 0:
            # no unknowns → pick a threshold above all sims
            return sims.max() + 1e-6
        # percentile so that `false_alarm_rate` of unk_sims are above τ
        pct = 100.0 * (1.0 - false_alarm_rate)
        return np.percentile(unk_sims, pct)

    def calc_identification_rate(self, prediction_labels):
        pred = np.array(prediction_labels)
        true = np.array(self.test_labels)
        # only consider the truly known probes
        mask_known = true != UNKNOWN_LABEL
        # among those, how many were correctly labeled?
        return np.mean(pred[mask_known] == true[mask_known])
