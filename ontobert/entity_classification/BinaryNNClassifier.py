import tensorflow as tf


class BinaryNNClassifier(tf.keras.Model):
    """
    Implementation of a binary classifier based on a neural network and a ReLU
    activation function.

    The argument hidden_layers can be used to specifify the number of hidden
    layers as well as the number of activations in each of them (defaults to
    one hidden layer with 300 hidden neurons).
    """

    def __init__(self, label: str, hidden_layers=(300,), **kwargs):
        super(BinaryNNClassifier, self).__init__(**kwargs)
        self.label = label

        self.hidden_layers = []
        for h in hidden_layers:
            self.hidden_layers.append(tf.keras.layers.Dense(h, activation="relu"))

        self.layer_output = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, embeddings: tf.Tensor) -> tf.Tensor:
        x = embeddings

        for h in self.hidden_layers:
            x = h(x)

        x = self.layer_output(x)
        return x
