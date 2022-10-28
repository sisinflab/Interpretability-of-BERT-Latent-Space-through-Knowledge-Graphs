import time
import json
import sys
import tensorflow as tf
from sklearn.model_selection import ParameterGrid
import numpy as np

from typing import Dict

from ontobert.grid_search import EarlyStopping
from ontobert.link_prediction import LinkPredictor
from ontobert.graph_embedding import (
    SimpleGraphEmbedding,
    BertGraphEmbedding,
    KnowledgeGraph,
)
from ontobert.data import load_kg, load_kg_split, Mode

from ontobert.graph_embedding.models import (
    GraphEmbeddingModel,
    dispatch_kge_model,
)

from task_utils import enable_determinism, parse_argv_into_config

tf.config.threading.set_inter_op_parallelism_threads(1)

OPTIMIZERS = {
    "sgd": tf.keras.optimizers.SGD,
    "adam": tf.keras.optimizers.Adam,
    "adagrad": tf.keras.optimizers.Adagrad,
}


def evaluate_model(
    model: GraphEmbeddingModel, test_set: np.ndarray, filter_out: np.ndarray
):

    """
    Evaluates a link prediction task based on the ranking protocol proposed
    by Bordes et al. (2013). An optional filter_out parameter can be provided
    to filter triples from the test dataset (useful to filter out golden
    triples in the ranking procedure).
    """

    lp = LinkPredictor(model=model)

    metrics = lp.evaluate(test_set)
    print("Results (raw):")
    print(metrics)

    metrics_filtered = lp.evaluate(test_set, filter_out=filter_out)
    print("Results (filtered):")
    print(metrics_filtered)

    return metrics, metrics_filtered


def select_best_model(
    kg: KnowledgeGraph,
    configuration: Dict,
    train_set: np.ndarray,
    validation_set: np.ndarray,
):

    """
    Selects the best knowledge graph model based on grid search with early
    stopping criterion. The target metric for the early stopping is the mean
    rank.
    """

    combinations = list(ParameterGrid(configuration["train_params"]))
    best_model = None
    best_mr = np.inf
    best_params = []
    grid_results = {}

    print("Performing model selection with grid search")

    print(f"Parameters combinations ({len(combinations)}): {combinations}")

    for combination_params in combinations:
        #        combination_params = default_params | params

        print(f"Training with parameters combination: {combination_params}")

        kge = create_graph_embedding_layer(kg, configuration, combination_params)

        model = dispatch_kge_model(
            name=configuration["model"], graph_embedding=kge, **combination_params
        )

        model.compile(
            optimizer=OPTIMIZERS[configuration["optimizer"].lower()](
                learning_rate=combination_params["learning_rate"]
            ),
            loss=None,
        )

        early_stopping_callback = EarlyStopping(
            validation_set=validation_set, delay=configuration["early_stopping_delay"]
        )

        print(model.trainable_weights)

        model.fit(
            train_set,
            batch_size=combination_params["batch_size"],
            epochs=configuration["max_epochs"],
            callbacks=[early_stopping_callback],
        )

        if len(combinations) == 1:
            best_model = model
            best_params = {}
            grid_results = {}
            break

        lp = LinkPredictor(model=model)

        curr_mr = lp.mean_rank(validation_set)

        grid_results[str(combination_params)] = curr_mr

        if curr_mr < best_mr:
            best_mr = curr_mr
            best_params = combination_params
            best_model = model

    return best_model, best_params, grid_results


def create_graph_embedding_layer(kg: KnowledgeGraph, configuration: Dict, params: Dict):
    if configuration["embedding_mode"].lower() == "bert":
        kge = BertGraphEmbedding(
            kg=kg,
            entities_trainable=False,
            relations_trainable=configuration["train_relations"],
            use_entity_descriptions=configuration["use_entity_descriptions"],
            use_relation_descriptions=configuration["use_relation_descriptions"],
        )
    elif configuration["embedding_mode"].lower() == "trainable":
        kge = SimpleGraphEmbedding(
            kg=kg,
            entity_shape=params["entity_shape"],
            relation_shape=params["relation_shape"],
        )
    else:
        raise NotImplementedError(
            "Only embedding mode supported are BERT and" "Trainable"
        )

    return kge


enable_determinism()

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


configuration = parse_argv_into_config(sys.argv)

print(f"Test configuration: {configuration}")

# Load the KG data

if configuration["filter_missing"].lower() == "none":
    filter_out_missing_names = False
    filter_out_missing_descriptions = False
elif configuration["filter_missing"].lower() == "labels":
    filter_out_missing_names = True
    filter_out_missing_descriptions = False
elif configuration["filter_missing"].lower() == "labels,descriptions":
    filter_out_missing_names = True
    filter_out_missing_descriptions = True
else:
    raise ValueError(
        "Configuration argument provided for 'filter_missing'" "strategy is invalid"
    )

kg = load_kg(
    configuration["dataset"],
    Mode.Full,
    filter_out_missing_names=filter_out_missing_names,
    filter_out_missing_descriptions=filter_out_missing_descriptions,
)

X_full, X_train, X_test, X_valid = load_kg_split(
    configuration["dataset"],
    filter_out_missing_names=filter_out_missing_names,
    filter_out_missing_descriptions=filter_out_missing_descriptions,
)

print(f"Full dataset has shape {X_full.shape}")
print(f"Train set has shape {X_train.shape}")
print(f"Validation set has shape {X_valid.shape}")
print(f"Test set has shape {X_test.shape}")


# Perform grid search to select best model

model, best_params, grid_results = select_best_model(
    kg=kg,
    configuration=configuration,
    train_set=X_train,
    validation_set=X_valid,
)

configuration["best_params"] = best_params
configuration["grid_results"] = grid_results

# Evaluate best model

metrics, metrics_filtered = evaluate_model(model, X_test, X_full)

configuration["results"] = metrics
configuration["results_filtered"] = metrics_filtered

# Save evaluation results to file

t = time.localtime()
timestamp = time.strftime("%b-%d-%Y_%H%M", t)
with open(f"results-{timestamp}.json", "w+") as f:
    json.dump(configuration, f, indent=4)
