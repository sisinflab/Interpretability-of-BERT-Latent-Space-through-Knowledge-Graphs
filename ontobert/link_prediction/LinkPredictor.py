import numpy as np
import tensorflow as tf
from typing import Dict

from ontobert.graph_embedding.models.GraphEmbeddingModel import GraphEmbeddingModel


class LinkPredictor(tf.keras.layers.Layer):
    """

    LinkPredictor implements the ranking protocol described by Bordes et
    al. (2013). Specifically, given a set of triples, each triple is
    "ranked" against all the possible triples generated corrupting its head
    and tail. The method returns two tensors:

    * head_ranks: the ranks of each of the input triples against all the
    triples generated corrupting the head;

    * tail_ranks: the ranks of each of the input triples against all the
    triples generated corrupting the tail;

    Parameters:

    - raw_triples: non-encoded triples of a kg (identifiers);

    - filter_out (optional): a set of raw triples that the evaluator should
      remove from the generated triples (usually this is set to filter out
      all the positive triples, that is, the union between train, test and
      validation datasets);

    Returns:

    - head_rank: rank of the given (positive) triples among all the
      negative triples generated replacing the heads;
    - tail_ranks: same, but with tails instead of heads;

    """

    def __init__(self, model: GraphEmbeddingModel):
        # model should be already fit
        super(LinkPredictor, self).__init__()
        self.model = model

    def call(
        self, raw_triples: tf.Tensor, filter_out: np.array = None, max_concurrency=3
    ) -> tf.Tensor:

        if len(filter_out) > 0:
            filter_triples = self.model.graph_embedding.kg_encoder(filter_out)

        triples = self.model.graph_embedding.kg_encoder(
            raw_triples
        )  # we encode the input triples with integer identifiers

        entities = tf.range(self.model.graph_embedding.n_entities, dtype=tf.int64)

        triples_scores = self.model.score(triples)

        def _rank(triple_id):

            triple = triples[triple_id]

            triple_score = triples_scores[triple_id]

            head, rel, tail = tf.unstack(triple)

            if len(filter_out) > 0:

                filter_head, filter_rel, filter_tail = tf.unstack(
                    filter_triples, axis=1, num=3
                )

                filter_out_tail = tf.boolean_mask(
                    filter_triples,
                    tf.logical_and(
                        tf.equal(filter_rel, rel), tf.equal(filter_head, head)
                    ),
                )  # all the entities that should be filtered from appearing in
                # the tail of generated triples

                _, _, filter_out_tail = tf.unstack(filter_out_tail, axis=1)

                filter_out_head = tf.boolean_mask(
                    filter_triples,
                    tf.logical_and(
                        tf.equal(filter_rel, rel), tf.equal(filter_tail, tail)
                    ),
                )

                filter_out_head, _, _ = tf.unstack(filter_out_head, axis=1)

                entities_head = tf.squeeze(
                    tf.sparse.to_dense(
                        tf.sets.difference(
                            tf.expand_dims(entities, 0),
                            tf.expand_dims(filter_out_head, 0),
                        )
                    )
                )

                entities_tail = tf.squeeze(
                    tf.sparse.to_dense(
                        tf.sets.difference(
                            tf.expand_dims(entities, 0),
                            tf.expand_dims(filter_out_tail, 0),
                        )
                    )
                )
            else:
                entities_head = entities_tail = entities

            head_triples = tf.stack(
                (
                    entities_head,
                    tf.fill(tf.shape(entities_head), rel),
                    tf.fill(tf.shape(entities_head), tail),
                ),
                axis=1,
            )  # all the negative triples with replaced head

            tail_triples = tf.stack(
                (
                    tf.fill(tf.shape(entities_tail), head),
                    tf.fill(tf.shape(entities_tail), rel),
                    entities_tail,
                ),
                axis=1,
            )  # all the negative triples with replaced tail

            neg_triples = tf.concat([head_triples, tail_triples], axis=0)

            #            head_scores = self.model.score(head_triples)
            #            tail_scores = self.model.score(tail_triples)
            neg_scores = self.model.score(neg_triples)

            head_scores = neg_scores[: tf.shape(head_triples)[0]]
            tail_scores = neg_scores[tf.shape(head_triples)[0] :]

            head_rank = (
                tf.reduce_sum(tf.cast(tf.greater(head_scores, triple_score), tf.int32))
                + 1
            )

            tail_rank = (
                tf.reduce_sum(tf.cast(tf.greater(tail_scores, triple_score), tf.int32))
                + 1
            )

            return head_rank, tail_rank

        @tf.function
        def _rank_triples(triples):
            return tf.map_fn(
                _rank,
                elems=tf.range(tf.shape(triples)[0]),
                fn_output_signature=(tf.int32, tf.int32),
                parallel_iterations=max_concurrency,  # tune this
            )

        return _rank_triples(triples)

    def mean_rank(self, raw_triples, filter_out=[]) -> np.array:
        head_ranks, tail_ranks = self(raw_triples, filter_out)

        mean_rank = np.mean(np.concatenate([head_ranks, tail_ranks]))

        return mean_rank

    def evaluate(self, raw_triples, filter_out=[]) -> Dict:

        head_ranks, tail_ranks = self(raw_triples, filter_out)

        head_hits_10 = np.sum(head_ranks <= 10) / raw_triples.shape[0] * 100
        tail_hits_10 = np.sum(tail_ranks <= 10) / raw_triples.shape[0] * 100
        head_hits_5 = np.sum(head_ranks <= 5) / raw_triples.shape[0] * 100
        tail_hits_5 = np.sum(tail_ranks <= 5) / raw_triples.shape[0] * 100

        hits_10 = (head_hits_10 + tail_hits_10) / 2
        hits_5 = (head_hits_5 + tail_hits_5) / 2

        mean_head_rank = np.mean(head_ranks)
        mean_tail_rank = np.mean(tail_ranks)
        mean_rank = (mean_head_rank + mean_tail_rank) / 2

        return {
            "mean_head_rank": mean_head_rank,
            "mean_tail_rank": mean_tail_rank,
            "mean_rank": mean_rank,
            "head_hits_10": head_hits_10,
            "tail_hits_10": tail_hits_10,
            "head_hits_5": head_hits_5,
            "tail_hits_5": tail_hits_5,
            "hits_10": hits_10,
            "hits_5": hits_5,
        }
