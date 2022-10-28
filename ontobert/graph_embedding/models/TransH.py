import tensorflow as tf
import numpy as np
import math

from ontobert.graph_embedding.models.GraphEmbeddingModel import GraphEmbeddingModel
from ontobert.graph_embedding.utils.metrics import distance_func
from tensorflow.keras.initializers import GlorotUniform


class TransH(GraphEmbeddingModel):
    def __init__(
        self,
        distance="L2",
        margin=1,
        C=0.25,
        *args,
        **kwargs,
    ):
        """
        Arguments:
            * margin: the "gamma" parameter used in the loss function;
            * C: regularization weight
        """

        super(TransH, self).__init__(*args, **kwargs)
        self.distance = distance
        self.margin = margin
        self.C = C
        self.relation_hyperplanes = tf.Variable(
            initial_value=GlorotUniform()(
                shape=(
                    self.graph_embedding.n_relations,
                    *self.graph_embedding.relation_embeddings.get_shape()[1:],
                )
            ),
            name="relation_hyperplanes",
        )

        if self.graph_embedding.relations_trainable:
            self.graph_embedding.relation_embeddings.assign(
                tf.nn.l2_normalize(self.graph_embedding.relation_embeddings, axis=-1)
            )

    def score(self, triples: tf.Tensor):
        _, rel, _ = tf.unstack(triples, axis=1, num=3)
        rel_hyper = tf.nn.embedding_lookup(self.relation_hyperplanes, rel)
        head, rel, tail = self.graph_embedding.lookup_embeddings(triples)

        head = tf.expand_dims(head, axis=-1)
        rel_hyper = tf.expand_dims(rel_hyper, axis=-1)
        tail = tf.expand_dims(tail, axis=-1)

        head_proj = tf.squeeze(
            head - tf.multiply(tf.matmul(rel_hyper, head, transpose_a=True), rel_hyper)
        )

        tail_proj = tf.squeeze(
            tail - tf.multiply(tf.matmul(rel_hyper, tail, transpose_a=True), rel_hyper)
        )

        return -distance_func(head_proj + rel, tail_proj, self.distance)

    def _loss(self, raw_triples: tf.Tensor):
        triples = self.graph_embedding.kg_encoder(raw_triples)

        corrupted_triples = self.negative_sampler(triples)

        # regularization on norm of relation hyperplanes
        self.relation_hyperplanes.assign(
            tf.nn.l2_normalize(self.relation_hyperplanes, axis=-1)
        )

        loss = (
            tf.reduce_sum(
                tf.clip_by_value(
                    self.margin - self.score(triples) + self.score(corrupted_triples),
                    0,
                    np.inf,
                ),
                name="pairwise_hinge_loss",
            )
            + self._soft_constraints()
        )

        return loss

    def _soft_constraints(self):

        # constraint on entity embeddings norm
        if self.graph_embedding.entities_trainable:
            scale = tf.reduce_sum(
                tf.clip_by_value(
                    tf.norm(self.graph_embedding.entity_embeddings, ord=2, axis=-1) - 1,
                    0,
                    np.inf,
                )
            )
        else:
            scale = 0

        orthogonal = tf.matmul(
            tf.expand_dims(self.relation_hyperplanes, -1),
            tf.expand_dims(self.graph_embedding.relation_embeddings, -1),
            transpose_a=True,
        )

        eps = 1e-5

        orthogonal = (
            tf.pow(
                tf.squeeze(orthogonal)
                / tf.norm(self.graph_embedding.relation_embeddings, ord=2, axis=-1),
                2,
            )
            - eps**2
        )

        orthogonal = tf.reduce_sum(tf.clip_by_value(orthogonal, 0, np.inf))
        return self.C * (scale + orthogonal)
