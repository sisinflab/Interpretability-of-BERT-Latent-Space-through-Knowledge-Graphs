import os
import numpy as np
from ontobert.data import Mode, load_tsv, _data_path


def _load_fb15k237(mode: Mode) -> np.ndarray:
    """
    Returns a numpy array representation of the FB15K graph dataset.
    """

    if mode == Mode.Train:
        filename = "train.txt"
    elif mode == Mode.Test:
        filename = "test.txt"
    elif mode == Mode.Valid:
        filename = "valid.txt"
    else:
        return np.concatenate(
            [
                _load_fb15k237(Mode.Train),
                _load_fb15k237(Mode.Valid),
                _load_fb15k237(Mode.Test),
            ],
            axis=0,
        )

    filepath = os.path.join(_data_path(), "kg", "fb15k-237", filename)

    return load_tsv(filepath)
