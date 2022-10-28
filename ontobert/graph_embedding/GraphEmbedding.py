import tensorflow as tf
import numpy as np
from typing import Callable, Tuple
from tensorflow.keras.initializers import GlorotUniform

from ontobert.graph_embedding.KGEncoder import KGEncoder
from ontobert.graph_embedding.KnowledgeGraph import KnowledgeGraph


class GraphEmbedding(tf.keras.layers.Layer):
    def __init__(
        self,
        kg: KnowledgeGraph,
        entities_trainable=True,
        relations_trainable=True,
        entity_initializer: tf.keras.initializers.Initializer = None,
        relation_initializer: tf.keras.initializers.Initializer = None,
        seed=100,
        *args,
        **kwargs,
    ):

        super(GraphEmbedding, self).__init__()

        self.seed = seed
        tf.random.set_seed(seed)

        if entity_initializer is None:
            self.entity_initializer = GlorotUniform(seed=self.seed)
        else:
            self.entity_initializer = entity_initializer

        if relation_initializer is None:
            self.relation_initializer = GlorotUniform(seed=self.seed)
        else:
            self.relation_initializer = relation_initializer

        self.kg = kg

        self.entities_trainable = entities_trainable
        self.relations_trainable = relations_trainable

        self.kg_encoder = KGEncoder(kg)

        self.n_entities = len(kg.entities)
        self.n_relations = len(kg.relations)

    @property
    def entities(self) -> tf.Tensor:
        return self.kg_encoder.entities

    @property
    def relations(self) -> tf.Tensor:
        return self.kg_encoder.relations

    def call(self, raw_triples):
        # TODO: subclasses of GraphEmbedding should be able to implement custom
        # behaviour in case an entity or a relation is not found in the
        # internal vocabulary of the encoder
        #
        # for example, a BERT graph embedding could be able to encode an
        # arbitrary entity, even if it's not in the vocabulary
        #
        # consider moving this in the implementors. For example, a BERT
        # embedding could implement the whole transformer architecture, and
        # forward the triples inside this method. A "simple" embedding class
        # could instead just do the lookup on its internal embeddings.
        return self.lookup_embeddings(self.kg_encoder(raw_triples))

    def lookup_entity(self, entity_labels) -> tf.Tensor:
        return tf.nn.embedding_lookup(self.entity_embeddings, entity_labels)

    def lookup_relation(self, relation_labels) -> tf.Tensor:
        return tf.nn.embedding_lookup(self.relation_embeddings, relation_labels)

    def lookup_embeddings(
        self, labels: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Lookup the latent features associated to a set of label-encoded triples.
        """
        head, rel, tail = tf.unstack(labels, axis=1, num=3)

        entities = tf.concat([head, tail], axis=0)
        entity_embeddings = self.lookup_entity(entities)

        return (
            entity_embeddings[: tf.shape(head)[0], :],
            self.lookup_relation(rel),
            entity_embeddings[tf.shape(head)[0] :, :],
        )

    #        return self.lookup_entity(head), self.lookup_relation(rel), self.lookup_entity(tail)

    def get_config(self):
        return {
            "entities_trainable": self.entities_trainable,
            "relations_trainable": self.relations_trainable,
        }
