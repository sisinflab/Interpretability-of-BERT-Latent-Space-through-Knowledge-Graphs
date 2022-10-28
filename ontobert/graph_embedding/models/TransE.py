import tensorflow as tf
import numpy as np
import math
from typing import List, Tuple

from ontobert.graph_embedding.models.GraphEmbeddingModel import GraphEmbeddingModel
from ontobert.graph_embedding.utils.metrics import distance_func


class TransE(GraphEmbeddingModel):
    """

    Attributes:

    * margin: margin hyperparameter used in the loss computation;
    * distance: distance function used to compute the triples energy (score);
    can be either 'l1' or 'l2';

    Methods:

    * fit(X): takes a knowledge graph in the form of an array of shape (N, 3),
    where N is the number of triples.

    """

    def __init__(
        self,
        distance="L2",
        margin=1,
        *args,
        **kwargs,
    ):
        """
        Arguments:
            * margin: the "gamma" parameter used in the loss function;
            * distance: the distance function used in the loss computation (can
            be either 'L1' or 'L2')

        """
        super(TransE, self).__init__(*args, **kwargs)
        self.margin = margin
        self.distance = distance

        if self.graph_embedding.relations_trainable:
            self.graph_embedding.relation_embeddings.assign(
                tf.nn.l2_normalize(self.graph_embedding.relation_embeddings, axis=-1)
            )

    def score(self, triples: tf.Tensor) -> tf.Tensor:
        head, rel, tail = self.graph_embedding.lookup_embeddings(triples)
        return -distance_func(head + rel, tail, self.distance, squared=False)

    def _loss(self, raw_triples: tf.Tensor):
        # TODO: consider moving this in the train_step of the parent class
        # and refactor this to take two input tensors (true and false triples)

        triples = self.graph_embedding.kg_encoder(raw_triples)

        corrupted_triples = self.negative_sampler(triples)

        # regularization constraint on entities norm
        if self.graph_embedding.entities_trainable:
            self.graph_embedding.entity_embeddings.assign(
                tf.nn.l2_normalize(self.graph_embedding.entity_embeddings, axis=-1)
            )

        loss = tf.reduce_sum(
            tf.clip_by_value(
                self.margin - self.score(triples) + self.score(corrupted_triples),
                0,
                np.inf,
            ),
            name="pairwise_hinge_loss",
            axis=0,
        )
        return loss
