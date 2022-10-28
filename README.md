# Interpretability of BERT Latent Space through Knowledge Graphs

Here the code we implemented to interpret and explain the BERT language model through the latent space it generates. The work identifies a feasibility study of analyzing BERT's latent semantic space using a knowledge graph.

## Abstract

The advent of pretrained language have renovated the ways of handling natural languages, improving the quality of systems that rely on them. BERT played a crucial role in revolutionizing the Natural Language Processing (NLP) area.
However, the deep learning framework it implements lacks interpretability. 
Thus, recent research efforts aimed to explain what BERT learns from the text sources exploited to pre-train its linguistic model.

In this paper, we analyze the latent vector space resulting from the BERT context-aware word embeddings. We focus on assessing whether regions of the BERT vector space hold an explicit meaning attributable to a Knowledge Graph (KG).
First, we prove the existence of explicitly meaningful areas through the Link Prediction (LP) task. Then, we demonstrate these regions being linked to explicit ontology concepts of a KG by learning classification patterns.
To the best of our knowledge, this is the first attempt at interpreting the BERT learned linguistic knowledge through a KG relying on its pretrained context-aware word embeddings.

## Citation

If you use our work for your activities, please cite with the following:

```
@inproceedings{10.1145/3511808.3557617,
author = {Vito Walter Anelli and Giovanni Maria Biancofiore and Alessandro De Bellis and Tommaso Di Noia and Eugenio Di Sciascio},
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

## Contacts

To any questions about our work, contact the authors of these work:

* Vito Walter Anelli [vitowalter.anelli@poliba.it](mailto:vitowalter.anelli@poliba.it)
* Giovanni Maria Biancofiore[^1] [giovannimaria.biancofiore@poliba.it](mailto:giovannimaria.biancofiore@poliba.it)
* Alessandro De Bellis[^1] [alessandro.debellis@poliba.it](mailto:alessandro.debellis@poliba.it)
* Tommaso Di Noia [tommaso.dinoia@poliba.it](mailto:tommaso.dinoia@poliba.it)
* Eugenio Di Sciascio [eugenio.disciascio@poliba.it](mailto:eugenio.disciascio@poliba.it)

[^1]: Corresponding auhtors.

# Running the code

Here the instruction to reproduce our experiments discussed in our paper.

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

Run it with:

```
docker run -it --gpus all ontobert
```

## Running with python venv

Create the python virtual environment and install the requirements.txt.

Before running tasks, modify the configuration files (.yml files in the
tasks/configs directory) as needed.

To reproduce our configurtation, just follow these instructions without modifying the .yml files.

To run the link prediction task:

```
python tasks/link_prediction.py --config tasks/configs/link_prediction.yml
```

To run the entity_classification task:

```
python tasks/entity_classification.py --config tasks/configs/entity_classification.yml
```

