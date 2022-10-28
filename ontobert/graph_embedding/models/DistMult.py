import tensorflow as tf
import numpy as np

from ontobert.graph_embedding.models.GraphEmbeddingModel import GraphEmbeddingModel


class DistMult(GraphEmbeddingModel):

    """
    Implementation of DistMult (Nickel et al. 2012)

    DistMult is an extension of RESCAL, where the relation embeddings are
    constrained to be diagonal matrices to reduce the number of trainable
    parameters.
    """

    def __init__(
        self,
        C=0.0001,
        *args,
        **kwargs,
    ):

        """

        Arguments:
        * kg: the knowledge graph
        * entity_shape: the shape for the entity embeddings
        * C: regularization weight

        """
        super(DistMult, self).__init__(*args, **kwargs)
        self.C = C
        if self.graph_embedding.entities_trainable:
            self.graph_embedding.entity_embeddings.assign(
                tf.nn.l2_normalize(self.graph_embedding.entity_embeddings, axis=-1)
            )
        if self.graph_embedding.relations_trainable:
            self.graph_embedding.relation_embeddings.assign(
                tf.nn.l2_normalize(self.graph_embedding.relation_embeddings, axis=-1)
            )

    def score(self, triples: tf.Tensor):
        """
        The score is based on a bilinear function.
        """

        head, rel, tail = self.graph_embedding.lookup_embeddings(triples)

        return tf.reduce_sum(head * rel * tail, axis=1)

    def _loss(self, raw_triples: tf.Tensor):
        if self.graph_embedding.entities_trainable:
            self.graph_embedding.entity_embeddings.assign(
                tf.nn.l2_normalize(self.graph_embedding.entity_embeddings, axis=1)
            )

        triples = self.graph_embedding.kg_encoder(raw_triples)
        corrupted_triples = self.negative_sampler(triples)

        # regularization constraint on relation embeddings norm

        if self.graph_embedding.relations_trainable:
            constraint = tf.reduce_sum(
                tf.square(self.graph_embedding.relation_embeddings)
            )
        else:
            constraint = 0

        #        positive_loss = tf.reduce_sum(tf.square(1 - self.score(triples)))
        #        corrupted_loss = tf.reduce_sum(tf.square(0 - self.score(corrupted_triples)))

        loss = tf.reduce_sum(
            tf.clip_by_value(
                1 - self.score(triples) + self.score(corrupted_triples),
                0,
                np.inf,
            ),
            name="pairwise_hinge_loss",
            axis=0,
        )

        return loss + self.C * constraint
