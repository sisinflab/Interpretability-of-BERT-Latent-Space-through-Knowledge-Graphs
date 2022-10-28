import tensorflow as tf

from ontobert.graph_embedding import GraphEmbedding


class NegativeSampler(tf.keras.layers.Layer):
    def __init__(self, graph_embedding: GraphEmbedding = None):
        super(NegativeSampler, self).__init__()
        self.graph_embedding = graph_embedding

    def set_graph_embedding(self, graph_embedding: GraphEmbedding):
        self.graph_embedding = graph_embedding


class UniformSampler(NegativeSampler):
    """
    UniformSampler performs negative sampling on triples based on the
    ranking evaluation protocol proposed by Bordes et al. (2013). Specifically,
    each triple in the evaluation set is corrupted, substituting either the
    tail or the head randomly with a known entity from the knowledge graph.
    """

    def __init__(self, seed=100):
        super(UniformSampler, self).__init__()
        self.seed = seed

    def call(self, triples):

        if self.graph_embedding is None:
            raise Exception(
                'UniformSampler requires a GraphEmbedding instance to work. \
                Provide one using "set_graph_embedding" method.'
            )

        corrupt_tail = tf.random.uniform(
            (tf.shape(triples)[0],), minval=0, maxval=2, dtype=tf.int32, seed=self.seed
        )  # a tensor of 0s and 1s, where a 0 means that the head is corrupted
        # and 1 means that the tail is corrupted

        indices = tf.stack(
            (tf.range(tf.shape(triples)[0]), tf.math.scalar_mul(2, corrupt_tail)),
            axis=1,
        )  # indices represents the position of the corrupted element in each triple
        # (if 0, the head is corrupted; if 2, the tail is corrupted)

        updates = tf.random.uniform(
            (tf.shape(triples)[0],),
            minval=0,
            maxval=self.graph_embedding.n_entities,
            dtype=tf.int64,
            seed=self.seed,
        )  # the randomly sampled entities

        return tf.tensor_scatter_nd_update(triples, indices, updates)
