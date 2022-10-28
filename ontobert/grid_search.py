import tensorflow as tf
import numpy as np

from ontobert.link_prediction import LinkPredictor


class EarlyStopping(tf.keras.callbacks.Callback):
    def __init__(
        self,
        validation_set,
        patience=3,  # number of consecutive worse result before early stopping
        interval=10,  # interval between epochs for evaluation of early stopping
        delay=100,  # number of epochs to wait before starting evaluating early stopping
    ):
        super(EarlyStopping, self).__init__()
        self.validation_set = validation_set
        self.stopped_epoch = 0
        self.best_mr = np.inf
        self.patience = patience
        self.wait = 0
        self.delay = delay
        self.interval = interval

    def on_train_end(self, logs={}):
        if self.stopped_epoch > 0:
            print(
                f"Early stopping performed on epoch {self.stopped_epoch + 1} with best MR {self.best_mr}"
            )

    def on_epoch_end(self, epoch, logs={}):

        if epoch + 1 < self.delay:
            return

        if (epoch + 1 - self.delay) % self.interval != 0:
            return

        lp = LinkPredictor(model=self.model)
        current_mr = lp.mean_rank(self.validation_set)
        print(f"Mean rank: {current_mr}")
        if current_mr < self.best_mr:
            self.wait = 0
            self.best_mr = current_mr
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                self.model.set_weights(self.best_weights)
