import os
import numpy as np
from typing import Dict
from ontobert.data import load_tsv, Mode, _data_path


def _load_fb15k(mode: Mode) -> np.ndarray:
    """
    Returns a numpy array representation of the FB15K graph dataset.
    """

    if mode == Mode.Train:
        filename = "freebase_mtr100_mte100-train.txt"
    elif mode == Mode.Test:
        filename = "freebase_mtr100_mte100-test.txt"
    elif mode == Mode.Valid:
        filename = "freebase_mtr100_mte100-valid.txt"
    else:
        return np.concatenate(
            [_load_fb15k(Mode.Train), _load_fb15k(Mode.Valid), _load_fb15k(Mode.Test)],
            axis=0,
        )

    filepath = os.path.join(_data_path(), "kg", "fb15k", filename)

    return load_tsv(filepath)


def _load_fb15k_desc() -> Dict[str, str]:
    """
    Returns a dictionary of textual descriptions for each of the fb15k MIDs.
    """
    filepath = os.path.join(_data_path(), "kg", "fb15k", "mid2desc.tsv")
    names = _load_fb15k_names()
    descriptions = load_tsv(filepath, as_dict=True)

    labeled_descriptions = {
        key: f"{names[key]} is {descriptions[key]}" for key in descriptions
    }

    return labeled_descriptions


def _load_fb15k_names() -> Dict[str, str]:
    """
    Returns a dictionary of textual labels for each of the fb15k MIDs.
    """
    filepath = os.path.join(_data_path(), "kg", "fb15k", "mid2name.tsv")
    return load_tsv(filepath, as_dict=True)
