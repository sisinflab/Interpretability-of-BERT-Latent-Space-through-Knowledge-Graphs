from ontobert.graph_embedding.models import (
    TransE,
    TransH,
    DistMult,
)

_MODELS = {
    "TransE": TransE,
    "TransH": TransH,
    "DistMult": DistMult,
}

def dispatch_kge_model(name:str, graph_embedding, **params):
    try:
        return _MODELS.get(name)(graph_embedding=graph_embedding, **params)
    except:
        raise NotImplementedError("The requested graph embedding model is not"\
                                  "supported")
