import tensorflow as tf
import numpy as np
import math

from typing import Dict, Tuple

from ontobert.graph_embedding import GraphEmbedding
from ontobert.graph_embedding.utils.sampling import NegativeSampler, UniformSampler


class GraphEmbeddingModel(tf.keras.Model):
    def __init__(
        self,
        graph_embedding: GraphEmbedding,
        negative_sampler=UniformSampler(),
        *args,
        **kwargs,
    ):

        super(GraphEmbeddingModel, self).__init__()

        self.graph_embedding = graph_embedding
        self.negative_sampler = negative_sampler
        self.negative_sampler.set_graph_embedding(self.graph_embedding)

    def call(self, raw_triples: tf.Tensor, training=False) -> tf.Tensor:
        # 1. inputs are first label-encoded
        # 2. then the score function is computed on the embeddings
        #    the score function should be defined in the implementor
        return self.score(self.graph_embedding.kg_encoder(raw_triples))

    def train_step(self, raw_triples: tf.Tensor) -> Dict[str, float]:
        # triples is a tensor of shape (N, 3)
        with tf.GradientTape() as tape:
            loss = self._loss(raw_triples)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return { "batch_loss": loss }


    # abstract methods

    def score(self, triples: tf.Tensor):
        """
        Computes the energy functions associated to a given tensor of label
        encoded triples of shape (N, 3, k) and dtype=tf.int64.

        TODO: refactor this to take "raw" triples
        """
        pass

    def _loss(self, raw_triples: tf.Tensor):
        """
        Computes the loss function associated to a given tensor of triples
        of shape (N, 3) and dtype=tf.string.
        """
        pass
