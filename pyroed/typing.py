from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple

import torch

Schema = OrderedDict[str, List[Optional[str]]]
Coefs = Dict[Tuple[str, ...], torch.Tensor]
Features = List[List[str]]
GibbsBlocks = List[List[str]]
Constraints = List[Callable]
