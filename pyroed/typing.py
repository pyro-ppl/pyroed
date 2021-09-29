from typing import Callable, Dict, List, Optional, Tuple

import torch

# Cannot use OrderedDict yet https://stackoverflow.com/questions/41207128
Schema = Dict[str, List[Optional[str]]]
Coefs = Dict[Tuple[str, ...], torch.Tensor]
Features = List[List[str]]
GibbsBlocks = List[List[str]]
Constraints = List[Callable]
