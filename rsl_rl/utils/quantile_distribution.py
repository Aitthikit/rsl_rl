from abc import ABC, abstractmethod
import torch


class Distribution(ABC):
    def __init__(self, params: torch.Tensor) -> None:
        self._params = params

    @abstractmethod
    def sample(self, sample_count: int = 1) -> torch.Tensor:
        """Sample from the distribution.

        Args:
            sample_count: The number of samples to draw.
        Returns:
            A tensor of shape (sample_count, *parameter_shape).
        """
        pass

class QuantileDistribution(Distribution):
    def sample(self, sample_count: int = 1) -> torch.Tensor:
        idx = torch.randint(
            self._params.shape[-1], (*self._params.shape[:-1], sample_count), device=self._params.device
        )
        samples = torch.take_along_dim(self._params, idx, -1)

        return samples, idx