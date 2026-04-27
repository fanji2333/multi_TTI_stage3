import torch

from omnisafe.typing import DEVICE_CPU


class BaseBuffer:

    def __init__(
        self,
        obs_space_dim: int,
        act_space_dim: int,
        cost_dim: int,
        size: int,
        device: torch.device = DEVICE_CPU,
    ) -> None:

        self._device: torch.device = device

        obs_buf = torch.zeros((size, obs_space_dim), dtype=torch.float32, device=device)
        act_buf = torch.zeros((size, act_space_dim), dtype=torch.float32, device=device)

        self.data: dict[str, torch.Tensor] = {
            'obs': obs_buf,
            'act': act_buf,
            'reward': torch.zeros(size, dtype=torch.float32, device=device),
            # TODO：修改了cost维度
            'cost': torch.zeros((size, cost_dim), dtype=torch.float32, device=device),
            'done': torch.zeros(size, dtype=torch.float32, device=device),
        }
        self._size: int = size

    def device(self) -> torch.device:

        return self._device

    def size(self) -> int:

        return self._size

    def __len__(self) -> int:

        return self._size

    def add_field(self, name: str, shape: tuple[int, ...], dtype: torch.dtype) -> None:

        self.data[name] = torch.zeros((self._size, *shape), dtype=dtype, device=self._device)

    def store(self, **data: torch.Tensor) -> None:
        pass
