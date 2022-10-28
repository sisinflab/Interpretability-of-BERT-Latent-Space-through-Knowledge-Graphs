from ontobert.data import Mode
from ontobert.graph_embedding import KnowledgeGraph
from ontobert.data.fb15k import (
    _load_fb15k,
    _load_fb15k_desc,
    _load_fb15k_names,
)
from ontobert.data.fb15k237 import _load_fb15k237
from ontobert.data import _load_ontology_classes


def load_kg_split(
    dataset, filter_out_missing_names=True, filter_out_missing_descriptions=True
):
    X_full = load_kg(
        dataset,
        Mode.Full,
        filter_out_missing_names=filter_out_missing_names,
        filter_out_missing_descriptions=filter_out_missing_descriptions,
    ).raw_triples
    X_train = load_kg(
        dataset,
        Mode.Train,
        filter_out_missing_names=filter_out_missing_names,
        filter_out_missing_descriptions=filter_out_missing_descriptions,
    ).raw_triples
    X_test = load_kg(
        dataset,
        Mode.Test,
        filter_out_missing_names=filter_out_missing_names,
        filter_out_missing_descriptions=filter_out_missing_descriptions,
    ).raw_triples
    X_valid = load_kg(
        dataset,
        Mode.Valid,
        filter_out_missing_names=filter_out_missing_names,
        filter_out_missing_descriptions=filter_out_missing_descriptions,
    ).raw_triples

    return X_full, X_train, X_test, X_valid


def load_kg(dataset: str, mode: Mode, **kwargs) -> KnowledgeGraph:
    if dataset.lower() == "fb15k":
        return KnowledgeGraph(
            name="FB15K",
            raw_triples=_load_fb15k(mode),
            entity_descriptions_map=_load_fb15k_desc(),
            entity_names_map=_load_fb15k_names(),
            entity_classes=_load_ontology_classes("fb15k"),
            **kwargs,
        )
    elif dataset.lower() == "fb15k-237":
        return KnowledgeGraph(
            name="FB15K-237",
            raw_triples=_load_fb15k237(mode),
            entity_descriptions_map=_load_fb15k_desc(),
            entity_names_map=_load_fb15k_names(),
            entity_classes=_load_ontology_classes("fb15k"),
            **kwargs,
        )
    else:
        raise NotImplementedError(
            "Only datasets currently supported are FB15K and FB15K-237"
        )
