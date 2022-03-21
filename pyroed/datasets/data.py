import gzip

import numpy as np
import torch


def load_tf_data(data_dir="./pyroed/datasets"):
    """
    Return tuple (x, y) of numpy arrays for PBX4 transcription factor data.

    Reference: Barrera, Luis A., et al. "Survey of variation in human transcription
    factors reveals prevalent DNA binding changes." Science 351.6280 (2016): 1450-1454.
    """
    xy = np.load(gzip.GzipFile(data_dir + "/tf_bind_8-PBX4_REF_R2.npy.gz", "rb"))
    x, y = xy[:, :-1], xy[:, -1]
    assert x.shape[0] == y.shape[0]
    assert x.ndim == 2
    return {
        "sequences": torch.tensor(x, dtype=torch.long),
        "responses": torch.tensor(y, dtype=torch.float),
        "batch_ids": torch.zeros(len(x), dtype=torch.long),
    }
