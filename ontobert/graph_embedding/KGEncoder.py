import numpy as np
import tensorflow as tf

from ontobert.graph_embedding import KnowledgeGraph


class KGEncoder(tf.keras.layers.Layer):
    """

    Encoder used to label-encode a K a knowledge graph in the form of a numpy
    array of shape (N, 3), where N is the number of triples in the graph, and
    dtype=string. The triples should have the form:

    (h, r, t)

    where r is the relation, while h and t are the head and tail entities
    respectively.

    The method outputs a new array of same shape, but with the entities and
    relations encoded by integers.

    Properties:
        * entities: ordered list of unique entities in the KG;
        * relations: ordered list of unique relations in the KG;

    """

    def __init__(self, kg: KnowledgeGraph):
        """
        Parameters:
            * X: a KG of shape (N, 3)

        """
        super(KGEncoder, self).__init__()

        self.entities_list = kg.entities
        self.relations_list = kg.relations

        self.relations = tf.convert_to_tensor(self.relations_list, dtype=tf.string)
        self.entities = tf.convert_to_tensor(self.entities_list, dtype=tf.string)

        self.relation_lookup = tf.keras.layers.StringLookup(
            vocabulary=self.relations, num_oov_indices=0
        )
        self.entity_lookup = tf.keras.layers.StringLookup(
            vocabulary=self.entities, num_oov_indices=0
        )

        self.relation_lookup_inverse = tf.keras.layers.StringLookup(
            vocabulary=self.relations, invert=True, num_oov_indices=0
        )

        self.entity_lookup_inverse = tf.keras.layers.StringLookup(
            vocabulary=self.entities, invert=True, num_oov_indices=0
        )

    def call(self, raw_triples: tf.Tensor) -> tf.Tensor:
        """
        Transforms a KG into a label-encoded representation

        Parameters:
            * X: a tensor of shape (N, 3) and dtype=tf.string

        Returns:
            * X_encoded: a label-encoded KG of shape (N, 3) and dtype=tf.int64
        """

        head, rel, tail = tf.unstack(raw_triples, axis=1)
        return tf.stack(
            (
                self.entity_lookup(head),
                self.relation_lookup(rel),
                self.entity_lookup(tail),
            ),
            axis=1,
        )

    def revert(self, labels: tf.Tensor) -> tf.Tensor:
        """
        Transforms a label-encoded KG into its original form

        Parameters:
            * X: a tensor of shape (N, 3) and dtype=tf.int64

        Returns:
            * X_encoded: an human-readable KG of shape (N, 3) and dtype=tf.string
        """

        head, rel, tail = tf.unstack(labels, axis=1)
        return tf.stack(
            (
                self.entity_lookup_inverse(head),
                self.relation_lookup_inverse(rel),
                self.entity_lookup_inverse(tail),
            ),
            axis=1,
        )
