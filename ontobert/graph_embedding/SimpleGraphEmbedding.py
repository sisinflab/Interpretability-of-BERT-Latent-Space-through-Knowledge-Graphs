import tensorflow as tf

from ontobert.graph_embedding.GraphEmbedding import GraphEmbedding
from typing import Tuple

class SimpleGraphEmbedding(GraphEmbedding):

    """
    Class that models a generic learnable graph embedding, with a given
    dimension for the latent spaces, and given initializers for entity and
    relation embeddings.
    """
    def __init__(
        self,
        entity_shape: Tuple[int],
        relation_shape: Tuple[int] = None,
        *args,
        **kwargs,
    ):

        super(SimpleGraphEmbedding, self).__init__(*args, **kwargs)
        self.entity_shape = entity_shape
        if relation_shape is None:
            self.relation_shape = entity_shape
        else:
            self.relation_shape = relation_shape

        self.entity_embeddings = tf.Variable(
            initial_value=self.entity_initializer((self.n_entities, *self.entity_shape)),
            name='entity_embeddings',
            trainable=self.entities_trainable
        )

        self.relation_embeddings = tf.Variable(
            initial_value=self.relation_initializer((self.n_relations, *self.relation_shape)),
            name='relation_embeddings',
            trainable=self.relations_trainable
        )
