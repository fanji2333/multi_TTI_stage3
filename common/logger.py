from typing import Union, Optional, Any

import os
from omnisafe.common.logger import Logger
from omnisafe.utils.distributed import dist_statistics_scalar
from omnisafe.utils.config import Config
from collections import deque
import torch
import numpy as np

class MyLogger(Logger):

    def __init__(self,
                 output_dir: str,
                 exp_name: str,
                 output_fname: str = 'progress.csv',
                 seed: int = 0,
                 use_tensorboard: bool = True,
                 use_wandb: bool = False,
                 config: Optional[Config] = None,
                 models: Optional[list[torch.nn.Module]] = None,
                 ) -> None:
        super().__init__(
            output_dir,
            exp_name,
            output_fname,
            seed,
            use_tensorboard,
            use_wandb,
            config,
            models,
        )

        self._last_ep_data: dict[str, float] = {}

    def register_key(
            self,
            key: str,
            window_length: Optional[int] = None,
            min_and_max: bool = False,
            delta: bool = False,
    ) -> None:
        """Register a key to the logger.

        The logger can record the following data:

        .. code-block:: text

            ----------------------------------------------------
            |       Name            |            Value         |
            ----------------------------------------------------
            |    Train/Epoch        |             25           |
            |  Metrics/EpCost/Min   |            22.38         |
            |  Metrics/EpCost/Max   |            25.48         |
            |  Metrics/EpCost/Mean  |            23.93         |
            ----------------------------------------------------

        Args:
            key (str): The name of the key.
            window_length (int or None, optional): The length of the window. Defaults to None.
            min_and_max (bool, optional): Whether to record the min and max value. Defaults to False.
            delta (bool, optional): Whether to record the delta value. Defaults to False.
        """
        assert key not in self._current_row, f'Key {key} has been registered'
        self._current_row[key] = 0
        self._last_ep_data[key] = 0
        if min_and_max:
            self._current_row[f'{key}/Min'] = 0
            self._current_row[f'{key}/Max'] = 0
            self._current_row[f'{key}/Std'] = 0
            self._headers_minmax[key] = True
            self._headers_minmax[f'{key}/Min'] = False
            self._headers_minmax[f'{key}/Max'] = False
            self._headers_minmax[f'{key}/Std'] = False

        else:
            self._headers_minmax[key] = False

        if delta:
            self._current_row[f'{key}/Delta'] = 0
            self._headers_delta[key] = True
            self._headers_delta[f'{key}/Delta'] = False
            self._headers_minmax[f'{key}/Delta'] = False
        else:
            self._headers_delta[key] = False

        if window_length is not None:
            self._data[key] = deque(maxlen=window_length)
            self._headers_windows[key] = window_length
        else:
            self._data[key] = []
            self._headers_windows[key] = None

    def store(
        self,
        data: Optional[dict[str, Any]] = None,
        /,
        **kwargs: Union[Any, float, np.ndarray, torch.Tensor],
    ) -> None:
        """Store the data to the logger.

        .. note ::
            The data stored in ``data`` will be updated by ``kwargs``.

        Args:
            data (dict[str, float | np.ndarray | torch.Tensor] or None, optional): The data to
                be stored. Defaults to None.

        Keyword Args:
            kwargs (int, float, np.ndarray, or torch.Tensor): The data to be stored.
        """
        if data is not None:
            kwargs.update(data)
        for key, val in kwargs.items():
            assert key in self._current_row, f'Key {key} has not been registered'
            if isinstance(val, (int, float)):
                self._data[key].append(val)
                self._last_ep_data[key] = val
            elif isinstance(val, torch.Tensor):
                self._data[key].append(val.mean().item())
                self._last_ep_data[key] = val.mean().item()
            elif isinstance(val, np.ndarray):
                self._data[key].append(val.mean())
                self._last_ep_data[key] = val.mean()
            else:
                raise ValueError(f'Unsupported type {type(val)}')

    def get_stats(
        self,
        key: str,
        min_and_max: bool = False,
        last_ep: bool = False,
    ) -> tuple[float, ...]:
        """Get the statistics of the key.

        Args:
            key (str): The key to be registered.
            min_and_max (bool, optional): Whether to record the min and max value of the key.
                Defaults to False.

        Returns:
            The mean value of the key or (mean, min, max, std) of the key.
        """
        assert key in self._current_row, f'Key {key} has not been registered'
        vals = self._data[key]
        if isinstance(vals, deque):
            vals = list(vals)

        if min_and_max:
            mean, std, min_val, max_val = dist_statistics_scalar(
                torch.tensor(vals).to(os.getenv('OMNISAFE_DEVICE', 'cpu')),
                with_min_and_max=True,
            )
            return mean.item(), min_val.mean().item(), max_val.mean().item(), std.item()

        if last_ep:
            return (self._last_ep_data[key],)

        mean, std = dist_statistics_scalar(  # pylint: disable=unbalanced-tuple-unpacking
            torch.tensor(vals).to(os.getenv('OMNISAFE_DEVICE', 'cpu')),
        )
        return (mean.item(),)