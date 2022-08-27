from typing import List
import importlib
import torch
from abc import ABC, abstractmethod
import numpy as np
import SimpleITK as sitk

from DeepLearning_API.config import config
from DeepLearning_API.utils import _getModule, NeedDevice
import torch.nn.functional as F

class DataAugmentation(NeedDevice, ABC):

    def __init__(self, nb : int = 1) -> None:
        self.nb = nb
        self.lastIndex = None

    def setDevice(self, device: torch.device):
        return super().setDevice(device)
    
    def state_init(self, index : int, dataset : object):
        if self.lastIndex != index:
            self._state_init(dataset)

    @abstractmethod
    def _state_init(self, dataset):
        pass

    def __call__(self, input : torch.Tensor) -> List[torch.Tensor]:
        return [input]

class DataAugmentationLoader:

    @config()
    def __init__(self) -> None:
        pass
        
    def getDataAugmentation(self, classpath : str = "") -> DataAugmentation:
        module, name = _getModule(classpath, "augmentation")
        return getattr(importlib.import_module(module), name)(config = None, DL_args="Trainer.Dataset.augmentations")

class RandomGeometricTransform(DataAugmentation):

    @config("RandomGeometricTransform")
    def __init__(self, nb: int = 0, grid_spacing = 16, max_displacement = 16) -> None:
        super().__init__(nb)
        self.grid_spacing = grid_spacing
        self.max_displacement = max_displacement
    
    @staticmethod
    def _formatLoc(new_locs, shape):
        for i in range(len(shape)):
            new_locs[..., i] = 2 * (new_locs[..., i] / (shape[i] - 1) - 0.5)
        new_locs = new_locs[..., [2, 1, 0]]
        return new_locs
    
    def _state_init(self, dataset):
        shape = dataset.getShape()
        dim = len(shape)

        grid_physical_spacing = [self.grid_spacing]*dim
        image_physical_size = [size*spacing for size, spacing in zip(shape, dataset.getSpacing())]
        mesh_size = [int(image_size/grid_spacing + 0.5) for image_size,grid_spacing in zip(image_physical_size, grid_physical_spacing)]
        self.displacement_fields : List[torch.Tensor] = []
        ref_image = dataset.to_image(np.zeros(shape))
        bspline_transform = sitk.BSplineTransformInitializer(image1 = ref_image, transformDomainMeshSize = mesh_size, order=3)
        displacement_filter = sitk.TransformToDisplacementFieldFilter()
        displacement_filter.SetReferenceImage(ref_image)
        
        vectors = [torch.arange(0, s) for s in shape]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.float).permute(0, 2, 3, 4, 1)
        
        self.displacement_fields.append(RandomGeometricTransform._formatLoc(grid, shape))

        for _ in range(self.nb):
            control_points = torch.rand(*[size+3 for size in mesh_size], dim)
            control_points -= 0.5
            control_points *= 2*self.max_displacement
            bspline_transform.SetParameters(control_points.flatten().tolist())
            new_locs = grid+torch.unsqueeze(torch.from_numpy(sitk.GetArrayFromImage(displacement_filter.Execute(bspline_transform))), 0).type(torch.float32)
            self.displacement_fields.append(RandomGeometricTransform._formatLoc(new_locs, shape))
        
    def __call__(self, input : torch.Tensor) -> List[torch.Tensor]:
        results : List[torch.Tensor] = []
        for displacement_field in self.displacement_fields:
            results.append(torch.squeeze(torch.squeeze(F.grid_sample(torch.unsqueeze(torch.unsqueeze(input.type(torch.float32).to(self.device), 0), 0), displacement_field.to(self.device), align_corners=True, mode="bilinear", padding_mode="border"))).type(input.dtype).cpu())
        return results

class MaskAugmentation(DataAugmentation):
    
    def __init__(self, path_mask) -> None:
        self.mask = sitk.ReadImage(path_mask)

    def _state_init(self, dataset):
        pass
    
    def __call__(self, input : torch.Tensor) -> List[torch.Tensor]:
        raise NotImplemented