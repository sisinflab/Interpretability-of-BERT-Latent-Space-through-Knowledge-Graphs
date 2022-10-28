import numpy as np
from typing import List, Dict


def load_tsv(filepath: str, as_dict=False) -> np.array:
    """
    Returns a numpy array from a tab-separated values file.
    """

    if as_dict:
        data: Dict[str, str] = {}
    else:
        data: List[List[str]] = []

    with open(filepath, "r", encoding="utf8") as f:
        for line in f:
            l = line.replace("\n", "").split("\t")
            if as_dict:
                data[l[0]] = l[1]
            else:
                data.append(l)

    if as_dict:
        return data
    else:
        return np.array(data)
