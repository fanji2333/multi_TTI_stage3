from __future__ import annotations

import numpy as np
from torch import nn

from omnisafe.typing import InitFunction


def initialize_layer(init_function: InitFunction, layer: nn.Linear, gain: float=np.sqrt(2)) -> None:
    """Initialize the layer with the given initialization function.

    The ``init_function`` can be chosen from: ``kaiming_uniform``, ``xavier_normal``, ``glorot``,
    ``xavier_uniform``, ``orthogonal``.

    Args:
        init_function (InitFunction): The initialization function.
        layer (nn.Linear): The layer to be initialized.
    """
    if init_function == 'kaiming_uniform':
        nn.init.kaiming_uniform_(layer.weight, a=np.sqrt(5))
    elif init_function == 'xavier_normal':
        nn.init.xavier_normal_(layer.weight)
    elif init_function in ['glorot', 'xavier_uniform']:
        nn.init.xavier_uniform_(layer.weight)
    elif init_function == 'orthogonal':
        nn.init.orthogonal_(layer.weight, gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0.0)
    else:
        raise TypeError(f'Invalid initialization function: {init_function}')

