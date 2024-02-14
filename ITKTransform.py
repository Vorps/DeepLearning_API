import SimpleITK as sitk
from typing import Union
import numpy as np
import torch
import torch.nn.functional as F
from DeepLearning_API.utils import _resample

def _openTransform(transform_files: dict[Union[str, sitk.Transform], bool]) -> list[sitk.Transform]:
    transforms: list[sitk.Transform] = []

    for transform_file, invert in transform_files.items():
        if isinstance(transform_file, str):
            transform = sitk.ReadTransform(transform_file+".itk.txt")
        else:
            transform = transform_file
        if len(transform.GetParameters()) == 6:
            transform = sitk.Euler3DTransform(transform)
            if invert:
                transform = sitk.Euler3DTransform(transform.GetInverse())
            transforms.append(transform)
        elif len(transform.GetParameters()) == 12:
            transform = sitk.AffineTransform(transform)
            if invert:
                transform = sitk.AffineTransform(transform.GetInverse())
            transforms.append(transform)
        else:
            transform = sitk.BSplineTransform(transform)
            if invert:
                transformToDisplacementFieldFilter = sitk.TransformToDisplacementFieldFilter()
                #transformToDisplacementFieldFilter.SetReferenceImage(image)
                displacementField = transformToDisplacementFieldFilter.Execute(transform)
                iterativeInverseDisplacementFieldImageFilter = sitk.IterativeInverseDisplacementFieldImageFilter()
                iterativeInverseDisplacementFieldImageFilter.SetNumberOfIterations(20)
                inverseDisplacementField = iterativeInverseDisplacementFieldImageFilter.Execute(displacementField)
                transform = sitk.DisplacementFieldTransform(inverseDisplacementField)
            transforms.append(transform)
    return transforms

def _openRigidTransform(transform_files: dict[Union[str, sitk.Transform], bool]) -> tuple[np.ndarray, np.ndarray]:
    transforms = _openTransform(transform_files)
    matrix_result = np.identity(3)
    translation_result = np.array([0,0,0])

    for transform in transforms:
        matrix = np.linalg.inv(np.array(transform.GetMatrix(), dtype=np.double).reshape((3,3)))
        translation = -np.asarray(transform.GetTranslation(), dtype=np.double)
        center = np.asarray(transform.GetCenter(), dtype=np.double)
        translation_center = np.linalg.inv(matrix).dot(matrix.dot(translation-center)+center)
        translation_result = np.linalg.inv(matrix_result).dot(translation_center)+translation_result
        matrix_result = matrix.dot(matrix_result)
    return np.linalg.inv(matrix_result), -translation_result

def composeTransform(transform_files : dict[Union[str, sitk.Transform], bool]) -> None:#sitk.CompositeTransform:
    transforms = _openTransform(transform_files)
    result = sitk.CompositeTransform(transforms)
    return result

def flattenTransform(transform_files: dict[Union[str, sitk.Transform], bool]) -> sitk.AffineTransform:
    [matrix, translation] = _openRigidTransform(transform_files)
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(matrix.flatten())
    transform.SetTranslation(translation)
    return transform

def apply_to_image_RigidTransform(image: sitk.Image, transform_files: dict[Union[str, sitk.Transform], bool]) -> sitk.Image:
    [matrix, translation] = _openRigidTransform(transform_files)
    matrix = np.linalg.inv(matrix)
    translation = -translation
    data = sitk.GetArrayFromImage(image)
    result = sitk.GetImageFromArray(data)
    result.SetDirection(matrix.dot(np.array(image.GetDirection()).reshape((3,3))).flatten())
    result.SetOrigin(matrix.dot(np.array(image.GetOrigin())+translation))
    result.SetSpacing(image.GetSpacing())
    return result

def apply_to_data_Transform(data: np.ndarray, transform_files: dict[Union[str, sitk.Transform], bool]) -> sitk.Image:
    transforms = composeTransform(transform_files)
    result = np.copy(data)
    _LPS = lambda matrix: np.array([-matrix[0], -matrix[1], matrix[2]], dtype=np.double)
    for i in range(data.shape[0]):
        result[i, :] =  _LPS(transforms.TransformPoint(np.asarray(_LPS(data[i, :]), dtype=np.double)))
    return result

def resampleITK(image_reference : sitk.Image, image : sitk.Image, transform_files : dict[Union[str, sitk.Transform], bool], mask = True, defaultPixelValue: Union[float, None] = None):
    return sitk.Resample(image, image_reference, composeTransform(transform_files), sitk.sitkNearestNeighbor if mask else sitk.sitkBSpline, (defaultPixelValue if defaultPixelValue is not None else (0 if mask else -1024)))

def parameterMap_to_transform(path_src: str) -> sitk.Transform:
    transform = sitk.ReadParameterFile("{}.0.txt".format(path_src))
    
    if transform["Transform"][0] == "EulerTransform":
        result = sitk.Euler3DTransform()
        parameters = np.array([float(i) for i in transform["TransformParameters"]])
        fixedParameters = np.array([float(i) for i in transform["CenterOfRotationPoint"]]+[0])
    elif transform["Transform"][0] == "AffineTransform":
        result = sitk.AffineTransform(3)
        parameters = np.array([float(i) for i in transform["TransformParameters"]])
        fixedParameters = np.array([float(i) for i in transform["CenterOfRotationPoint"]]+[0])
    else:
        result = sitk.BSplineTransform()
        
        parameters = np.array([float(i) for i in transform["TransformParameters"]])
        GridSize = np.array([int(i) for i in transform["GridSize"]])
        GridOrigin = np.array([float(i) for i in transform["GridOrigin"]])
        GridSpacing = np.array([float(i) for i in transform["GridSpacing"]])
        GridDirection = np.array(np.array([float(i) for i in transform["GridDirection"]])).reshape((3,3)).T.flatten() 
        fixedParameters = np.concatenate([GridSize, GridOrigin, GridSpacing, GridDirection])

    result.SetFixedParameters(fixedParameters)
    result.SetParameters(parameters)
    return result

def resampleIsotropic(image: sitk.Image, spacing : list[float] = [1., 1., 1.]) -> sitk.Image:
    resize_factor = [y/x for x,y in zip(spacing, image.GetSpacing())]
    result = sitk.GetImageFromArray(_resample(torch.tensor(sitk.GetArrayFromImage(image)).unsqueeze(0), [int(size*factor) for size, factor in zip(image.GetSize(), resize_factor)]).squeeze(0).numpy())
    result.SetDirection(image.GetDirection())
    result.SetOrigin(image.GetOrigin())
    result.SetSpacing(spacing)
    return result

def resampleResize(image: sitk.Image, size : list[int] = [100,512,512]):
    result =  sitk.GetImageFromArray(_resample(torch.tensor(sitk.GetArrayFromImage(image)).unsqueeze(0), size).squeeze(0).numpy())
    result.SetDirection(image.GetDirection())
    result.SetOrigin(image.GetOrigin())
    result.SetSpacing([x/y*z for x,y,z in zip(image.GetSize(), size, image.GetSpacing())])
    return result