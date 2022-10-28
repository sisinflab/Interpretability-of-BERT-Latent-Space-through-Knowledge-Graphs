import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, hamming_loss, accuracy_score
import json
import time
import os

from sklearn.model_selection import ParameterGrid

from typing import List, Dict

from ontobert.entity_classification import BinaryNNClassifier


class MultiBinaryClassifier(tf.keras.layers.Layer):

    """
    Generic multilabel classifier implementation based on a one-vs-rest
    strategy.

    """

    def __init__(
        self,
        labels: List[str] = [],
        decision_threshold=0.5,
        binary_classifier_cls=BinaryNNClassifier,
    ):
        super(MultiBinaryClassifier, self).__init__()

        self.labels = labels
        self.models = []
        self.decision_threshold = decision_threshold
        self.binary_classifier_cls = binary_classifier_cls

    def _initialize_models(self):
        """
        Initializes all the inner binary classifiers with default
        configuration.
        """
        for label in self.labels:
            model = self.binary_classifier_cls(label=label)
            self.models.append(model)

    def _perform_sampling(self, X, y):
        """
        Performs sampling on a given dataset in order to equalize the number of
        negative samples (y==0) and positive samples (y==1). Specifically, the
        negative samples are re-sampled randomly to fit the number of positive
        samples.
        """

        X_pos = X[y == 1]
        y_pos = y[y == 1]

        X_neg = X[y == 0]
        y_neg = y[y == 0]

        # sample an equal number of negative samples
        p = np.random.permutation(X_neg.shape[0])[: X_pos.shape[0]]

        X_neg = tf.gather(X_neg, p)
        y_neg = tf.gather(y_neg, p)

        X = tf.concat([X_pos, X_neg], 0)
        y = tf.concat([y_pos, y_neg], 0)

        # reshuffle

        p = np.random.permutation(X.shape[0])
        X = tf.gather(X, p)
        y = tf.gather(y, p)

        return X, y

    def fit_best(
        self,
        X,
        y,
        X_val,
        y_val,
        param_grid: Dict,
        max_epochs=50,
        optimizer=tf.keras.optimizers.Adam,
        equal_sampling=True,
    ):
        """
        Performs hyperparameter selection with early stopping for each of the
        binary classifiers.
        """

        combinations = list(ParameterGrid(param_grid))
        for idx, label in enumerate(self.labels):

            print(
                "Selecting best binary classification model for class: "
                f"{self.labels[idx]}"
            )

            best_loss = np.inf
            best_model = None
            best_params = {}

            for params in combinations:
                model = self.binary_classifier_cls(label=label, **params)

                X_train = X
                # only select the binary labels (y) for the current class
                y_train = y[:, idx]
                y_valid = y_val[:, idx]

                if equal_sampling:
                    X_train, y_train = self._perform_sampling(X_train, y_train)

                print(X_train.shape)
                print(y_train.shape)

                callback = tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=3, restore_best_weights=True
                )

                model.compile(
                    loss="binary_crossentropy",
                    optimizer=optimizer(learning_rate=params["learning_rate"]),
                    metrics=[tf.keras.metrics.BinaryAccuracy()],
                )

                history = model.fit(
                    X_train,
                    y_train,
                    epochs=max_epochs,
                    batch_size=params["batch_size"],
                    validation_data=(X_val, y_valid),
                    callbacks=[callback],
                )

                curr_loss = np.min(history.history["val_loss"])

                if curr_loss < best_loss:
                    best_loss = curr_loss
                    best_model = model
                    best_params = params

            print(f"Best params for class {self.labels[idx]}: {best_params}")

            self.models.append(best_model)

    def fit(
        self,
        X,
        y,
        lr=0.01,
        epochs=50,
        batch_size=32,
        optimizer=tf.keras.optimizers.Adam,
        equal_sampling=True,
    ):

        self._initialize_models()

        for idx, model in enumerate(self.models):
            # only select the binary labels (y) for the current class

            print(
                "Training binary classification model for class: " f"{self.labels[idx]}"
            )

            X_train = X
            y_train = y[:, idx]

            if equal_sampling:
                X_train, y_train = self._perform_sampling(X_train, y_train)

            print(X_train.shape)
            print(y_train.shape)

            model.compile(
                loss="binary_crossentropy",
                optimizer=optimizer(learning_rate=lr),
                metrics=[tf.keras.metrics.BinaryAccuracy()],
            )

            return model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    def evaluate(
        self,
        X,
        y,
        save=True,
        verbose=True,
    ):
        """
        Performs a classification report, for both the binary classifiers and
        the multilabel classifier as a whole.

        If save is True, a complete .txt report is saved in the current working
        directory.

        Receives in input X, a test dataset of shape (N, N_dim), and y, the
        list of ground truth labels of shape (N, N_labels).
        """

        results = {}
        for idx, model in enumerate(self.models):

            y_test = y[:, idx]
            preds = model.predict(X)

            report = classification_report(
                y_true=y_test,
                y_pred=np.where(preds > self.decision_threshold, 1, 0),
                output_dict=True,
            )

            if verbose:
                print(report)

            results.update({self.labels[idx]: report})

        y_pred = np.where(self.predict(X) > 0.5, 1, 0)

        print(y_pred.shape)
        print(y.shape)

        EMR = np.all(y_pred == y, axis=1).mean()
        hl = hamming_loss(y, y_pred)
        hs = accuracy_score(y, y_pred)

        print(f"Exact match ratio: {EMR}")
        print(f"Hamming loss {hl}")
        print(f"Accuracy score {hs}")

        results["EMR"] = EMR
        results["hamming_loss"] = hl
        results["hamming_score"] = hs

        if save:
            t = time.localtime()
            timestamp = time.strftime("%b-%d-%Y_%H%M", t)
            with open(f"binary_classification_results-{timestamp}.json", "w+") as f:
                json.dump(results, f, indent=4)

    def load(self, models_dir: str):
        """
        Loads a model based on previously saved weights. This function should
        only be called if the model was saved using "model.save()".

        Receives as argument the name of the folder in which the weights were
        originally saved.
        """
        for filename in os.listdir(models_dir):
            self.models.append(
                tf.keras.models.load_model(os.path.join(models_dir, filename))
            )
            self.labels.append(filename)

    def save(self):
        """
        Persists the model's weights to disk, in a folder under the current
        working directory.
        """
        t = time.localtime()
        timestamp = time.strftime("%b-%d-%Y_%H%M", t)
        for idx, model in enumerate(self.models):
            model.save(os.path.join(f"model-{timestamp}", self.labels[idx]))

    def predict(self, embeddings: tf.Tensor) -> tf.Tensor:
        """
        Receives as input a tensor of embeddings of shape (N, N_dim), where
        N_dim is the number of dimensions for the embeddings.

        Returns a tensor of shape (N, N_labels), where N is the number of
        input embeddings and N_labels is the number of binary classifiers.
        """
        return self(embeddings)

    def predict_top_label(self, embeddings: tf.Tensor) -> List[str]:
        """
        For each of the given input embeddings, predict the most likely label.
        """
        preds = []
        for embedding in embeddings:
            pred = tf.reshape(self.predict(tf.expand_dims(embedding, 0)), (-1))
            max_prob = tf.reduce_max(pred)
            if max_prob >= self.decision_threshold:
                label_idx = tf.squeeze(
                    tf.gather(
                        tf.random.shuffle(
                            tf.where(tf.math.equal(pred, max_prob)),
                        ),
                        0,
                    )
                )

                preds.append(self.labels[label_idx])
            else:
                preds.append("")

        return preds

    def call(self, embeddings: tf.Tensor) -> tf.Tensor:
        outputs = [model(embeddings) for model in self.models]
        return tf.squeeze(tf.stack(outputs, axis=-1))
