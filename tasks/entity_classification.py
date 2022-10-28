import pandas as pd
import numpy as np
import tensorflow as tf
import sys
from sklearn.model_selection import train_test_split
from task_utils import enable_determinism, parse_argv_into_config

from ontobert.data import load_kg, Mode

from ontobert.entity_classification import MultiBinaryClassifier
from ontobert.graph_embedding import BertGraphEmbedding, GraphEmbedding

configuration = parse_argv_into_config(sys.argv)

enable_determinism()

# Load knowledge graph

kg = load_kg(
    configuration.get("dataset"),
    Mode.Full,
    filter_out_missing_names=True,
    filter_out_missing_descriptions=True,
)


# Extract ontological classes

classes = kg.entity_classes
df: pd.DataFrame = pd.DataFrame(classes)
df.drop_duplicates(inplace=True)
print(df.describe())


# Select the top classes (TOP_CLASSES) for frequency

top_classes = (
    df.groupby(1)
    .count()
    .sort_values(0, ascending=False)
    .iloc[: configuration.get("top_classes")]
    .index
)
print(df.groupby(1).count().sort_values(0, ascending=False)[:20])

print(top_classes)


# extract only entities that have at least one class in top_classes

df = df[np.isin(df[1], top_classes)]


# Group classes by entity
df = df.groupby(0)[1].apply(list).reset_index()

# select only combinations of classes that have more than 1 occurrence

vc = df[1].value_counts()
print(vc[vc > 1])
df = df[np.isin(df[1], vc[vc > 1].index)]

print(df)

# Train-test split

train_df, test_df = train_test_split(
    df,
    test_size=configuration.get("test_split"),
    stratify=df[1].values,
)

# test_df, val_df = train_test_split(
#    test_df,
#    test_size=0.5,
#    stratify=test_df[1].values,
# )
# TODO: ValueError: The least populated class in y has only 1 member, which is too few. The minimum number of groups for any class cannot be less than 2.

val_df = test_df.sample(frac=0.5)
test_df.drop(val_df.index, inplace=True)


print(f"Number of entities in training set: {len(train_df)}")
print(f"Number of entities in test set: {len(test_df)}")
print(f"Number of entities in validation set: {len(val_df)}")

# Multi-hot encoding

classes = tf.ragged.constant(train_df[1].values)
lookup = tf.keras.layers.StringLookup(output_mode="multi_hot", num_oov_indices=0)
lookup.adapt(classes)

kge = BertGraphEmbedding(
    kg=kg,
    entities_trainable=False,
    relations_trainable=True,
    use_entity_descriptions=True,
    use_relation_descriptions=False,
    cased=True,
)


def prepare_dataset(kge: GraphEmbedding, dataframe: pd.DataFrame):
    X = kge.lookup_entity(kge.kg_encoder.entity_lookup(dataframe[0].values))
    y = lookup(tf.ragged.constant(dataframe[1].values)).numpy()
    return X, y


X_train, y_train = prepare_dataset(kge, train_df)
X_test, y_test = prepare_dataset(kge, test_df)
X_val, y_val = prepare_dataset(kge, val_df)


model = MultiBinaryClassifier(
    labels=lookup.get_vocabulary(),
    decision_threshold=configuration.get("decision_threshold", 0.5),
)

model.fit_best(
    X_train,
    y_train,
    X_val,
    y_val,
    param_grid=configuration["train_params"],
    equal_sampling=configuration.get("equal_sampling", True),
)

model.evaluate(X_test, y_test)

model.save()
