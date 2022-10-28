import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import numpy as np
import math
import os
from typing import List, Tuple

# huggingface BERT tokenizer
from transformers import BertTokenizer

from ontobert.graph_embedding.GraphEmbedding import GraphEmbedding

class BertGraphEmbedding(GraphEmbedding):

    tfhub_handle_encoder_cased = (
        "https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/4"
    )
    tfhub_handle_encoder_uncased = (
        "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"
    )

    embeddings_dim = 768  # dimension of BERT output embeddings
    max_length = 512  # max number of input tokens for BERT

    def __init__(
        self,
        use_entity_descriptions=False,  # tells the BERT encoder to use entity descriptions (if available) instead of raw labels
        use_relation_descriptions=False,
        fine_tune=False,
        cased=True,
        *args,
        **kwargs,
    ):

        """
        Arguments:
        * kg: the knowledge graph
        """
        super(BertGraphEmbedding, self).__init__(*args, **kwargs)

        self.use_entity_descriptions = use_entity_descriptions
        self.use_relation_descriptions = use_relation_descriptions

        if cased:
            self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
            self.bert_model = hub.KerasLayer(self.tfhub_handle_encoder_cased, trainable=fine_tune)
        else:
            self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.bert_model = hub.KerasLayer(self.tfhub_handle_encoder_uncased, trainable=fine_tune)

        entity_embeddings_path = os.path.join("embeddings", f"{self.kg.name}")
        relation_embeddings_path = os.path.join("embeddings", f"{self.kg.name}")

        if use_entity_descriptions:
            entity_embeddings_path = os.path.join(entity_embeddings_path, "with_context")
        if use_relation_descriptions:
            relation_embeddings_path = os.path.join(relation_embeddings_path, "with_context")

        if cased:
            entity_embeddings_path = os.path.join(entity_embeddings_path, "cased")
            relation_embeddings_path = os.path.join(relation_embeddings_path, "cased")
        else:
            entity_embeddings_path = os.path.join(entity_embeddings_path, "uncased")
            relation_embeddings_path = os.path.join(relation_embeddings_path, "uncased")

        entity_embeddings_path = os.path.join(entity_embeddings_path, str(len(self.kg.entities)), "entity_embeddings")
        relation_embeddings_path = os.path.join(relation_embeddings_path, str(len(self.kg.relations)), "relation_embeddings")

        if self.entities_trainable:
            self.entity_embeddings = tf.Variable(
                initial_value=self.entity_initializer(
                    (self.n_entities, self.embeddings_dim)
                ),
                name="entity_embeddings",
                trainable=True,
            )
        else:

            if not os.path.exists(entity_embeddings_path):
                print("Pre-computing entity embeddings")
                if use_entity_descriptions:
                    entity_embeddings = self._compute_contextualized_embeddings(
                        self.kg.entity_names.tolist(),
                        self.kg.entity_descriptions.tolist(),
                    )
                else:
                    entity_embeddings = self._compute_embeddings(
                        self.kg.entity_names.tolist()
                    )

                self.entity_embeddings = tf.Variable(
                    entity_embeddings, name="entity_embeddings", trainable=False
                )

                tf.saved_model.save(self.entity_embeddings, entity_embeddings_path)

            else:
                print(
                    f"Loading pre-computed entity embeddings from path: {entity_embeddings_path}"
                )
                self.entity_embeddings = tf.saved_model.load(entity_embeddings_path)

        if self.relations_trainable:
            self.relation_embeddings = tf.Variable(
                initial_value=self.relation_initializer(
                    (self.n_relations, self.embeddings_dim)
                ),
                name="relation_embeddings",
                trainable=True,
            )
        else:

            if not os.path.exists(relation_embeddings_path):
                print("Pre-computing relation embeddings")

                if use_relation_descriptions:
                    relation_embeddings = self._compute_contextualized_embeddings(
                        self.kg.relation_names.tolist(),
                        self.kg.relation_descriptions.tolist(),
                    )
                else:
                    relation_embeddings = self._compute_embeddings(
                        self.kg.relation_names.tolist()
                    )


                self.relation_embeddings = tf.Variable(
                    relation_embeddings,
                    name="relation_embeddings",
                    trainable=False,
                )

                tf.saved_model.save(self.relation_embeddings, relation_embeddings_path)
            else:
                print(
                    f"Loading pre-computed relation embeddings from path: {relation_embeddings_path}"
                )
                self.relation_embeddings = tf.saved_model.load(relation_embeddings_path)

        tf.print(tf.shape(self.entity_embeddings))
        tf.print(self.entity_embeddings)
        tf.print(tf.shape(self.relation_embeddings))
        tf.print(self.relation_embeddings)

    def _compute_embeddings(self, sentences):
        num_tokens = tf.reduce_sum(
            self.bert_tokenizer(
                sentences,
                add_special_tokens=False,
                return_attention_mask=True,
                return_token_type_ids=False,
                return_tensors="tf",
                padding=True,
            )["attention_mask"],
            axis=-1,
        )

        preprocess = tf.expand_dims(self._tokenize(sentences), 1)


        def compute_embedding(idx):
            progress.add(1)
            return tf.reduce_mean(
                self.bert_model(
                    {
                        "input_word_ids": preprocess[idx][:, 0],
                        "input_type_ids": tf.fill(
                            (
                                1,
                                self.max_length,
                            ),
                            0,
                        ),
                        "input_mask": preprocess[idx][:, 1],
                    }
                )["sequence_output"][0][1 : num_tokens[idx] + 1],
                axis=0
            )

        progress = tf.keras.utils.Progbar(
            preprocess.get_shape().as_list()[0],
            width=30,
            verbose=1,
            interval=0.05,
            unit_name="embedding computation",
        )

        return tf.map_fn(
            compute_embedding,
            tf.range(tf.shape(preprocess)[0]),
            fn_output_signature=tf.TensorSpec(
                shape=(self.embeddings_dim,), dtype=tf.float32
            ),
        )

    def _compute_contextualized_embeddings(self, names: List, descriptions: List) -> tf.Tensor:

        def compute_embedding(idx):
            progress.add(1)

            embedding = tf.cond(
                num_tokens[idx] > 0,
                lambda: tf.reduce_mean(
                    self.bert_model(
                        {
                            "input_word_ids": tf.expand_dims(desc_tokenized[idx, 0], 0),
                            "input_type_ids": tf.fill(
                                (
                                    1,
                                    self.max_length,
                                ),
                                0,
                            ),
                            "input_mask": tf.expand_dims(desc_tokenized[idx, 1], 0),
                        }
                    )["sequence_output"][0][1 : num_tokens[idx] + 1],
                    axis=0,
                ),
                lambda: self.bert_model(
                    {
                        "input_word_ids": tf.expand_dims(desc_tokenized[idx, 0], 0),
                        "input_type_ids": tf.fill(
                            (
                                1,
                                self.max_length,
                            ),
                            0,
                        ),
                        "input_mask": tf.expand_dims(desc_tokenized[idx, 1], 0),
                    }
                )["pooled_output"][0],
            )

            return embedding

        desc_tokenized = self._tokenize(descriptions)

        num_tokens = tf.reduce_sum(
            self.bert_tokenizer(
                names,
                add_special_tokens=False,
                return_attention_mask=True,
                return_token_type_ids=False,
                return_tensors="tf",
                padding=True,
            )["attention_mask"],
            axis=-1,
        )  # returns the number of tokens corresponding
        # to each entity name
        # which is the number of tokens we need
        # to pool in the output embedding for the
        # whole description

        progress = tf.keras.utils.Progbar(
            desc_tokenized.get_shape().as_list()[0],
            width=30,
            verbose=1,
            interval=0.05,
            unit_name="embedding computation",
        )

        # return compute_embedding(preprocess) # enable this if enough GPU memory is available

        return tf.map_fn(
            compute_embedding,
            tf.range(tf.shape(desc_tokenized)[0]),
            fn_output_signature=tf.TensorSpec(
                shape=(self.embeddings_dim,), dtype=tf.float32
            ),
        )

    def _tokenize(self, sentences: List):
        tokenized = self.bert_tokenizer(
            sentences,
            add_special_tokens=True,
            max_length=self.max_length,
            return_attention_mask=True,
            return_token_type_ids=False,
            padding="max_length",
            return_tensors="tf",
        )
        return tf.stack([tokenized["input_ids"], tokenized["attention_mask"]], axis=1)
