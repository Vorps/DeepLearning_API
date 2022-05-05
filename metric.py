import torch

class Metric(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

class Dice(Metric):
    def __init__(self, smooth : float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth
    
    def flatten(self, tensor : torch.Tensor) -> torch.Tensor:
        C = tensor.size(1)
        return tensor.permute((1, 0) + tuple(range(2, tensor.dim()))).contiguous().view(C, -1)

    def dice_per_channel(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input = self.flatten(input)
        target = self.flatten(target)
        return (2.*(input * target).sum() + self.smooth)/(input.sum() + target.sum() + self.smooth)

    def forward(self, input: torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        input = torch.nn.functional.one_hot(input.type(torch.int64)).permute(0, len(input.shape), *[i+1 for i in range(len(input.shape)-1)]).float()
        target = torch.nn.functional.one_hot(target.type(torch.int64)).permute(0, len(target.shape), *[i+1 for i in range(len(target.shape)-1)]).float()
        return torch.mean(self.dice_per_channel(input, target))