import gzip

import numpy as np
import torch


def load_tf_data(data_dir="./pyroed/datasets"):
    """
    Return tuple (x, y) of numpy arrays for PBX4 transcription factor data.
    """
    xy = np.load(gzip.GzipFile(data_dir + "/tf_bind_8-PBX4_REF_R2.npy.gz", "rb"))
    x, y = xy[:, :-1], xy[:, -1]
    assert x.shape[0] == y.shape[0]
    assert x.ndim == 2
    return {
        "sequences": torch.tensor(x, dtype=torch.long),
        "batch_id": torch.zeros(len(x), dtype=torch.long),
        "response": torch.tensor(y, dtype=torch.float),
    }