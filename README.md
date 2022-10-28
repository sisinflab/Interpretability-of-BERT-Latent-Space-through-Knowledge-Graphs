# Interpretability of BERT Latent Space through Knowledge Graphs
Here the code we implemented to interpret and explain the BERT language model through the latent space it generates. The work identifies a feasibility study of analyzing BERT's latent semantic space using a knowledge graph.

## Citation

If you use our work for your activities, please cite with the following:

```
@inproceedings{10.1145/3511808.3557617,
author = {Anelli, Vito Walter and Biancofiore, Giovanni Maria and De Bellis, Alessandro and Di Noia, Tommaso and Di Sciascio, Eugenio},
title = {Interpretability of BERT Latent Space through Knowledge Graphs},
year = {2022},
isbn = {9781450392365},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3511808.3557617},
doi = {10.1145/3511808.3557617},
booktitle = {Proceedings of the 31st ACM International Conference on Information &amp; Knowledge Management},
pages = {3806–3810},
location = {Atlanta, GA, USA},
series = {CIKM '22}
}
```

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

