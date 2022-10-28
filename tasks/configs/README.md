The purpose of this README is to document the configuration parameters for the
YAML configuration files to be used when running the tasks.

# Link prediction task

* dataset: the dataset to be used for the task. Available values are "FB15K-237" and
  "FB15K";
* filter_missing: can be either one of the following:
    - 'Labels,Descriptions': in this case the dataset is filtered to remove all
      the triples that contains entities whose textual label or description is
      missing from the dataset;
    - 'Labels': in this case the dataset is filtered to remove all
      the triples that contains entities whose textual label is missing;
    - 'Labels': in this case the dataset is filtered to remove all
      the triples that contains entities whose textual description is missing;
* model: The knowledge graph embedding model to be trained. Available values
  are 'TransE', 'TransH' and 'DistMult';
* embedding_mode: can be one of the following:
    - 'Trainable': the knowledge graph embeddings are computed using the
      standard procedure described by the authors of the previously defined
      model;
    - 'BERT': the knowledge graph embeddings for the entities are computed
      using BERT;
* train_relations: if the embedding_mode is set to 'BERT', the relations are
  not computed using BERT, but they are computed using the standard procedure
  (as if embedding_mode was set to 'Trainable');
* use_entity_descriptions: if set to True, and embedding_mode is 'BERT', the
  BERT embeddings for the entities are computed using the complete textual
  description as input to the BERT model;
* max_epochs: the maximum number of epochs to use for training;
* early_stopping_delay: the first epoch where early stopping based on mean rank
  should be applied to select the best model;
* optimizer: the optimizer algorithm to be used during training. Allowed values
  are 'SGD', 'AdaGrad' and 'Adam';
* train_params: a list of model-dependent hyperparameter; consult the model
  definition (ontobert/graph_embedding/models) for further specifications;


# Entity classification task

* dataset: the dataset to be used for the task. Available values are
  "FB15K-237" and "FB15K". FB15K is recommended since FB15K-237 is a subset of
  the latter;
* top_classes: the number of trained binary classifiers; The classes are
  selected prioritizing the most frequent classes in the dataset;
* equal_sampling: whether or not the training set should be filtered in
  order to equalize the amount of negative and positive samples;
* test_split: the percentage of samples to allocate to the test set;
* decision_threshold: the value used by the binary classifier to perform a
  decision;
* train_params: a parameter grid used to select the best binary classifiers
    - learning_rate: a list of candidate learning rates;
    - hidden_layers: a list of tuples of variable length, containing the number
      of hidden layers in each layer of the binary classifier;
    - batch_size: a list of integers, containing the candidate batch sizes to
      use during model selection;
