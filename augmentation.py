from typing import Callable, Dict, List, Optional, Tuple
import importlib
import torch
from abc import ABC, abstractmethod
import numpy as np
import SimpleITK as sitk
from torch.functional import Tensor

from DeepLearning_API.config import config
from DeepLearning_API.utils import _getModule, NeedDevice
import torch.nn.functional as F

class DataAugmentation(NeedDevice, ABC):

    def __init__(self) -> None:
        self.lastIndex = None
        self.who_function: Callable = lambda: None
        self.who : torch.Tensor
        self.nb : int

    def load(self, nb: int, prob: float):
        self.who_function = lambda : torch.rand(nb) < prob

    def state_init(self, index : int, shape: List[int], cache_attribute: Dict[str, torch.Tensor]):
        if self.lastIndex != index:
            self.who = self.who_function()
            self.nb = len([i for i in self.who if i])
            self._state_init(shape, cache_attribute)

    @abstractmethod
    def _state_init(self, shape: List[int], cache_attribute: Dict[str, torch.Tensor]):
        pass
    
    def __call__(self, input : torch.Tensor) -> torch.Tensor:
        input[self.who] = self._compute(input[self.who])
        return input

    @abstractmethod
    def _compute(self, input : torch.Tensor) -> torch.Tensor:
        pass

class Prob():

    @config()
    def __init__(self, prob: float = 1.0) -> None:
        self.prob = prob

class DataAugmentationsList():

    @config()
    def __init__(self, nb : int = 10, dataAugmentations: Dict[str, Prob] = {"default:RandomElastixTransform" : Prob(1)}) -> None:
        self.nb = nb
        self.dataAugmentations : List[DataAugmentation] = []
        self.dataAugmentationsLoader = dataAugmentations

    def load(self, key: str, device: torch.device):
        for augmentation, prob in self.dataAugmentationsLoader.items():
            module, name = _getModule(augmentation, "augmentation")
            dataAugmentation: DataAugmentation = getattr(importlib.import_module(module), name)(config = None, DL_args="Trainer.Dataset.augmentations.{}.dataAugmentations".format(key))
            dataAugmentation.load(self.nb, prob.prob)
            self.dataAugmentations.append(dataAugmentation)
            dataAugmentation.setDevice(device)     
            

class RandomRotateTransform(DataAugmentation):

    @config("RandomRotateTransform")
    def __init__(self, min_angles: List[float] = [0, 0, 0], max_angles: List[float] = [360, 360, 360]) -> None:
        super().__init__()
        assert len(min_angles) == len(max_angles)
        assert len(min_angles) == 3 or len(min_angles) == 1

        self.range_angle = [(min_angle, max_angle) for min_angle, max_angle in zip(min_angles, max_angles)]
        
    def _rotation3DMatrix(self, rotation : torch.Tensor) -> torch.Tensor:
        A = torch.tensor([[torch.cos(rotation[2]), -torch.sin(rotation[2]), 0], [torch.sin(rotation[2]), torch.cos(rotation[2]), 0], [0, 0, 1]])
        B = torch.tensor([[torch.cos(rotation[1]), 0, torch.sin(rotation[1])], [0, 1, 0], [-torch.sin(rotation[1]), 0, torch.cos(rotation[1])]])
        C = torch.tensor([[1,0,0], [0, torch.cos(rotation[0]), -torch.sin(rotation[0])], [0, torch.sin(rotation[0]), torch.cos(rotation[0])]])
        return torch.cat((A.mm(B).mm(C), torch.zeros((3, 1))), dim=1)

    def _rotation2DMatrix(self, rotation : torch.Tensor) -> torch.Tensor:
        return torch.cat((torch.tensor([[torch.cos(rotation[0]), -torch.sin(rotation[0])], [torch.sin(rotation[0]), torch.cos(rotation[0])]]), torch.zeros((2, 1))), dim=1)
    
    def _state_init(self, shape: List[int], cache_attribute: Dict[str, torch.Tensor]):
        func = self._rotation3DMatrix if len(self.range_angle)== 3 else self._rotation2DMatrix
        angles = torch.rand((self.nb, len(self.range_angle)))*torch.tensor([max_angle-min_angle for min_angle, max_angle in self.range_angle])+torch.tensor([min_angle for min_angle, _ in self.range_angle])/360*2*torch.pi
        self.matrix = torch.cat([torch.unsqueeze(func(value), dim=0) for value in angles], dim=0)
    
    def _compute(self, input : torch.Tensor) -> torch.Tensor:
        return F.grid_sample(input.to(self.device).type(torch.float32), F.affine_grid(self.matrix, list(input.shape), align_corners=True).to(self.device), align_corners=True, mode="bilinear", padding_mode="border").type(input.dtype).cpu()

class RandomElastixTransform(DataAugmentation):

    @config("RandomElastixTransform")
    def __init__(self, grid_spacing = 16, max_displacement = 16) -> None:
        super().__init__()
        self.grid_spacing = grid_spacing
        self.max_displacement = max_displacement
    
    @staticmethod
    def _formatLoc(new_locs, shape):
        for i in range(len(shape)):
            new_locs[..., i] = 2 * (new_locs[..., i] / (shape[i] - 1) - 0.5)
        new_locs = new_locs[..., [2, 1, 0]]
        return new_locs

    def _state_init(self, shape: List[int], cache_attribute: Dict[str, torch.Tensor]):
        shape = shape
        dim = len(shape)

        grid_physical_spacing = [self.grid_spacing]*dim
        image_physical_size = [size*spacing for size, spacing in zip(shape, cache_attribute["Spacing"])]
        mesh_size = [int(image_size/grid_spacing + 0.5) for image_size,grid_spacing in zip(image_physical_size, grid_physical_spacing)]
        self.displacement_fields : List[torch.Tensor] = []
        ref_image = sitk.GetImageFromArray(np.zeros(shape))
        ref_image.SetOrigin([float(v) for v in cache_attribute["Origin"]])
        ref_image.SetSpacing([float(v) for v in cache_attribute["Spacing"]])
        ref_image.SetDirection([float(v) for v in cache_attribute["Direction"]])

        bspline_transform = sitk.BSplineTransformInitializer(image1 = ref_image, transformDomainMeshSize = mesh_size, order=3)
        displacement_filter = sitk.TransformToDisplacementFieldFilter()
        displacement_filter.SetReferenceImage(ref_image)
        
        vectors = [torch.arange(0, s) for s in shape]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.float).permute(0, 2, 3, 4, 1)
        
        for _ in range(self.nb):
            control_points = torch.rand(*[size+3 for size in mesh_size], dim)
            control_points -= 0.5
            control_points *= 2*self.max_displacement
            bspline_transform.SetParameters(control_points.flatten().tolist())
            new_locs = grid+torch.unsqueeze(torch.from_numpy(sitk.GetArrayFromImage(displacement_filter.Execute(bspline_transform))), 0).type(torch.float32)
            self.displacement_fields.append(RandomElastixTransform._formatLoc(new_locs, shape))

        self.displacement_field = torch.cat(self.displacement_fields, dim=0)

    def _compute(self, input : torch.Tensor) -> torch.Tensor:
        return F.grid_sample(input.type(torch.float32).to(self.device), self.displacement_field.to(self.device), align_corners=True, mode="bilinear", padding_mode="border").type(input.dtype).cpu()

class RandomFlipTransform(DataAugmentation):

    @config("RandomFlipTransform")
    def __init__(self, flip: List[float] = [0.5, 0.25 ,0.25]) -> None:
        super().__init__()
        self.flip = flip
        self.dim_flip : torch.Tensor

    def _state_init(self, shape: List[int], cache_attribute: Dict[str, torch.Tensor]):
        self.dim_flip = torch.rand((self.nb, len(self.flip))) < torch.tensor(self.flip)

    def _compute(self, input : torch.Tensor) -> torch.Tensor:
        for i in range(input.shape[0]):
            input[i] = torch.flip(input[i], tuple([i+1 for i, v in enumerate(self.dim_flip[i]) if v]))
        return input