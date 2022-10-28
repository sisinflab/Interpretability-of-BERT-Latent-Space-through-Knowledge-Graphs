# Interpretability of BERT Latent Space through Knowledge Graphs
Here the code we implemented to interpret and explain the BERT language model through the latent space it generates. The work identifies a feasibility study of analyzing BERT's latent semantic space using a knowledge graph.

# Ontobert

## Prerequisites

* Python 3.9
* Tensorflow 2.7.0
* CUDA 11.7.0
* cuDNN 8.4.1

## Running with Docker

Build the image with:

```
docker build --tag ontobert
```

And run it with:

```
docker run -it --gpus all ontobert
```

## How To

Before running tasks, modify the configuration files (.yml files in the
tasks/configs directory) as needed.

To run the link prediction task:

```
python tasks/link_prediction.py --config tasks/configs/link_prediction.yml
```

To run the entity_classification task:

```
python tasks/entity_classification.py --config tasks/configs/entity_classification.yml
```


## References

* Bordes, Antoine, et al. Translating embeddings for modeling multi-relational
  data. Proceedings of NIPS, 2013.

* Wang, Z., Zhang, J., Feng, J., & Chen, Z. (2014). Knowledge Graph Embedding
  by Translating on Hyperplanes. Proceedings of the AAAI Conference on
  Artificial Intelligence, 28(1). https://doi.org/10.1609/aaai.v28i1.8870

* Embedding Entities and Relations for Learning and Inference in Knowledge
  Bases, Bishan Yang, Scott Wen-tau Yih, Xiaodong He, Jianfeng Gao and Li Deng,
  ICLR 2015.
