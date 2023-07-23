import importlib
import torch
from abc import ABC, abstractmethod
import numpy as np
import SimpleITK as sitk
from torch.functional import Tensor

from DeepLearning_API.config import config
from DeepLearning_API.utils import _getModule, NeedDevice, Attribute
import torch.nn.functional as F
from typing import Callable
import os

class Prob():

    @config()
    def __init__(self, prob: float = 1.0) -> None:
        self.prob = prob

class DataAugmentationsList():

    @config()
    def __init__(self, nb : int = 10, dataAugmentations: dict[str, Prob] = {"default:RandomElastixTransform" : Prob(1)}) -> None:
        self.nb = nb
        self.dataAugmentations : list[DataAugmentation] = []
        self.dataAugmentationsLoader = dataAugmentations

    def load(self, key: str):
        for augmentation, prob in self.dataAugmentationsLoader.items():
            module, name = _getModule(augmentation, "augmentation")
            dataAugmentation: DataAugmentation = getattr(importlib.import_module(module), name)(config = None, DL_args="{}.Dataset.augmentations.{}.dataAugmentations".format(os.environ["DEEP_LEARNING_API_ROOT"], key))
            dataAugmentation.load(prob.prob)
            self.dataAugmentations.append(dataAugmentation)
    
    def to(self, device: torch.device):
        for dataAugmentation in self.dataAugmentations:
            dataAugmentation.setDevice(device)

class DataAugmentation(NeedDevice, ABC):

    def __init__(self) -> None:
        self.who_index: dict[int, list[int]] = {}
        self.shape_index: dict[int, list[list[int]]] = {}
        self.prob: float = 0

    def load(self, prob: float):
        self.prob = prob

    def state_init(self, index: int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        if index not in self.who_index:
            self.who_index[index] = torch.where(torch.rand(len(shapes)) < self.prob)[0].tolist()
        else:
            return self.shape_index[index]
        for i, shape in enumerate(self._state_init(index, [shapes[i] for i in self.who_index[index]], [caches_attribute[i] for i in self.who_index[index]])):
            shapes[self.who_index[index][i]] = shape
        self.shape_index[index] = shapes
        return self.shape_index[index]
    
    @abstractmethod
    def _state_init(self, index : int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        pass

    def __call__(self, index: int, inputs : list[torch.Tensor]) -> list[torch.Tensor]:
        for i, result in enumerate(self._compute(index, [inputs[i] for i in self.who_index[index]])):
            inputs[self.who_index[index][i]] = result
        return inputs
    
    @abstractmethod
    def _compute(self, index: int, inputs : list[torch.Tensor]) -> list[torch.Tensor]:
        pass

    def inverse(self, index: int, a: int, input : torch.Tensor) -> torch.Tensor:
        if a in self.who_index[index]:
            input = self._inverse(index, a, input)
        return input
        
    @abstractmethod
    def _inverse(self, index: int, a: int, input : torch.Tensor) -> torch.Tensor:
        pass

class RandomRotateTransform(DataAugmentation):

    @config("RandomRotateTransform")
    def __init__(self, min_angles: list[float] = [0, 0, 0], max_angles: list[float] = [360, 360, 360]) -> None:
        super().__init__()
        assert len(min_angles) == len(max_angles)
        assert len(min_angles) == 3 or len(min_angles) == 1
        self.range_angle = [(min_angle, max_angle) for min_angle, max_angle in zip(min_angles, max_angles)]
        self.matrix: dict[int, list[torch.Tensor]] = {}
        
    def _rotation3DMatrix(self, rotation : torch.Tensor) -> torch.Tensor:
        A = torch.tensor([[torch.cos(rotation[2]), -torch.sin(rotation[2]), 0], [torch.sin(rotation[2]), torch.cos(rotation[2]), 0], [0, 0, 1]])
        B = torch.tensor([[torch.cos(rotation[1]), 0, torch.sin(rotation[1])], [0, 1, 0], [-torch.sin(rotation[1]), 0, torch.cos(rotation[1])]])
        C = torch.tensor([[1,0,0], [0, torch.cos(rotation[0]), -torch.sin(rotation[0])], [0, torch.sin(rotation[0]), torch.cos(rotation[0])]])
        return torch.cat((A.mm(B).mm(C), torch.zeros((3, 1))), dim=1)

    def _rotation2DMatrix(self, rotation : torch.Tensor) -> torch.Tensor:
        return torch.cat((torch.tensor([[torch.cos(rotation[0]), -torch.sin(rotation[0])], [torch.sin(rotation[0]), torch.cos(rotation[0])]]), torch.zeros((2, 1))), dim=1)
    
    def _state_init(self, index : int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        func = self._rotation3DMatrix if len(self.range_angle)== 3 else self._rotation2DMatrix
        angles = torch.rand((len(shapes), len(self.range_angle)))*torch.tensor([max_angle-min_angle for min_angle, max_angle in self.range_angle])+torch.tensor([min_angle for min_angle, _ in self.range_angle])/360*2*torch.pi
        self.matrix[index] = [torch.unsqueeze(func(value), dim=0) for value in angles]
        return shapes
    
    def _compute(self, index: int, inputs : list[torch.Tensor]) -> list[torch.Tensor]:
        results = []
        for input, matrix in zip(inputs, self.matrix[index]):
            results.append(F.grid_sample(input.to(self.device).type(torch.float32), F.affine_grid(matrix, list(input.shape), align_corners=True).to(self.device), align_corners=True, mode="bilinear", padding_mode="border").type(input.dtype).cpu())
        return results
    
    def _inverse(self, index: int, a: int, input : torch.Tensor) -> torch.Tensor:
        pass

class RandomElastixTransform(DataAugmentation):

    @config("RandomElastixTransform")
    def __init__(self, grid_spacing: int = 16, max_displacement: int = 16) -> None:
        super().__init__()
        self.grid_spacing = grid_spacing
        self.max_displacement = max_displacement
        self.displacement_fields: dict[int, list[torch.Tensor]] = {}
    
    @staticmethod
    def _formatLoc(new_locs, shape):
        for i in range(len(shape)):
            new_locs[..., i] = 2 * (new_locs[..., i] / (shape[i] - 1) - 0.5)
        new_locs = new_locs[..., [2, 1, 0]]
        return new_locs

    def _state_init(self, index : int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        print("Compute Displacement Field for index {}".format(index))
        self.displacement_fields[index] = []
        for i, (shape, cache_attribute) in enumerate(zip(shapes, caches_attribute)):
            shape = shape
            dim = len(shape)

            grid_physical_spacing = [self.grid_spacing]*dim
            image_physical_size = [size*spacing for size, spacing in zip(shape, cache_attribute["Spacing"])]
            mesh_size = [int(image_size/grid_spacing + 0.5) for image_size,grid_spacing in zip(image_physical_size, grid_physical_spacing)]
            
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
        
            control_points = torch.rand(*[size+3 for size in mesh_size], dim)
            control_points -= 0.5
            control_points *= 2*self.max_displacement
            bspline_transform.SetParameters(control_points.flatten().tolist())
            new_locs = grid+torch.unsqueeze(torch.from_numpy(sitk.GetArrayFromImage(displacement_filter.Execute(bspline_transform))), 0).type(torch.float32)
            self.displacement_fields[index].append(RandomElastixTransform._formatLoc(new_locs, shape))
            print("Compute in progress : {:.2f} %".format((i+1)/len(shapes)*100))
        return shapes
    
    def _compute(self, index: int, inputs : list[torch.Tensor]) -> list[torch.Tensor]:
        results = []
        for input, displacement_field in zip(inputs, self.displacement_fields[index]):
            results.append(F.grid_sample(input.type(torch.float32).to(self.device).unsqueeze(0), displacement_field.to(self.device), align_corners=True, mode="bilinear", padding_mode="border").type(input.dtype).squeeze(0).cpu())
        return results
    
    def _inverse(self, index: int, a: int, input : torch.Tensor) -> torch.Tensor:
        pass

class RandomFlipTransform(DataAugmentation):

    @config("RandomFlipTransform")
    def __init__(self, flip: list[float] = [0.5, 0.25 ,0.25]) -> None:
        super().__init__()
        self.flip = flip
        self.dim_flip : dict[int, torch.Tensor] = {}

    def _state_init(self, index : int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        self.dim_flip[index] = torch.rand((len(shapes), len(self.flip))) < torch.tensor(self.flip)
        return shapes
    
    def _compute(self, index: int, inputs : list[torch.Tensor]) -> torch.Tensor:
        results = []
        for input, dim_flip in zip(inputs, self.dim_flip[index]):
            results.append(torch.flip(input, tuple([i+1 for i, v in enumerate(dim_flip) if v])))
        return results
    
    def _inverse(self, index: int, a: int, input : torch.Tensor) -> torch.Tensor:
        pass

class PermuteTransform(DataAugmentation):

    @config("PermuteTransform")
    def __init__(self, dims: list[str] = ["1|0|2", "2|0|1"]) -> None:
        super().__init__()
        self.dims = [[0]+[int(d)+1 for d in dim.split("|")] for dim in dims]

    def _state_init(self, index : int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        for i, shape in enumerate(shapes):
            shapes[i] = [shape[it-1] for it in self.dims[i][1:]]
        return shapes
    
    def _compute(self, index: int, inputs : list[torch.Tensor]) -> torch.Tensor:
        results = []
        for i, input in enumerate(inputs):
            results.append(input.permute(tuple(self.dims[i])))
        return results
    
    def _inverse(self, index: int, a: int, input : torch.Tensor) -> torch.Tensor:
        return input.permute(tuple(np.argsort(self.dims[a])))