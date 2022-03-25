import torch
from . import config

class Transform:
    
    def __init__(self) -> None:
        pass

class Normalize(Transform):

    @config("Dataset.Normalize")
    def __init__(self, min_value : float = 0, max_value : float = 100) -> None:
        super().__init__()
        assert max_value > min_value
        self.min_value = min_value
        self.max_value = max_value
        self.value_range = max_value - min_value

    def __call__(self, input : torch.Tensor) -> torch.Tensor:
        return (input - self.min_value) / self.value_range

class Standardize(Transform):

    @config("Dataset.Standardize")
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, input : torch.Tensor) -> torch.Tensor:
        return (input - torch.mean(input)) / torch.std(input)

class Unsqueeze(Transform):

    @config("Dataset.Unsqueeze")
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, input : torch.Tensor) -> torch.Tensor:
        return torch.unsqueeze(input, dim=0)

class ATensorCast(Transform):

    @config("Dataset.Float32")
    def __init__(self, dtype : str = "float32") -> None:
        super().__init__()
        self.dtype = getattr(torch, dtype)

    def __call__(self, input : torch.Tensor) -> torch.Tensor:
        return input.type(self.dtype)

