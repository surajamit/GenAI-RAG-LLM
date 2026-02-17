import os
import random
import numpy as np
import torch


def set_global_seed(seed: int = 42):
    """
    Ensures deterministic behavior.

    Used across:
    - retrieval
    - training
    - evaluation
    """

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
