from DeepLearning_API.utils import DatasetUtils, Attribute
import numpy as np
import SimpleITK as sitk
import os
import shutil

def test_h5(datasetUtils: DatasetUtils):
    """H5 Test"""
    shape = (10,20,30)
    data = np.zeros(shape)
    image = sitk.GetImageFromArray(data)
    direction = (0,1,0,1,0,0,0,0,1)
    image.SetDirection(direction)
    origin = (10,20,30)
    image.SetOrigin(origin)
    spacing = (1,2,3)
    image.SetSpacing(spacing)
    euler3D_0 = (10.0,20.0,30.0,5.0,6.0,7.0)
    transform_0 = sitk.Euler3DTransform()
    transform_0.SetParameters(euler3D_0)

    euler3D_1 = (50.0,18.0,30.0,5.0,6.0,7.0)
    transform_1 = sitk.Euler3DTransform()
    transform_1.SetParameters(euler3D_1)

    compositeTransform = sitk.CompositeTransform([transform_0, transform_1])

    landmarks = np.random.random((50, 3))

    attribute = Attribute({"Test": "1"})
    datasetUtils.write("Group", "Data1", data, Attribute(attribute))
    datasetUtils.write("Group", "Data2", data, Attribute(attribute))
    datasetUtils.write("Group/Group2", "Image", image, Attribute(attribute))
    datasetUtils.write("Group1/Group2", "Transform", transform_0, Attribute(attribute))
    datasetUtils.write("Group1/Group2", "CompositeTransform", compositeTransform, Attribute(attribute))
    datasetUtils.write("Deformation", "Landmarks", landmarks)


    assert datasetUtils.isGroupExist("Group")
    assert datasetUtils.isGroupExist("Group/Group2")
    assert datasetUtils.isGroupExist("Group1/Group2")
    assert datasetUtils.isGroupExist("*/Group2")
    assert datasetUtils.isGroupExist("Deformation")

    assert datasetUtils.isDatasetExist("Group", "Data1")
    assert datasetUtils.isDatasetExist("Group", "Data2")
    assert datasetUtils.isDatasetExist("Group/Group2", "Image")
    assert datasetUtils.isDatasetExist("*/Group2", "Image")
    assert datasetUtils.isDatasetExist("Group1/Group2", "Transform")
    assert datasetUtils.isDatasetExist("Deformation", "Landmarks")

    assert not datasetUtils.isGroupExist("Group/Group1")
    assert not datasetUtils.isDatasetExist("Test/Group1", "Test")
    assert not datasetUtils.isDatasetExist("Group/Group1", "Test")
    assert not datasetUtils.isDatasetExist("*/Group1", "Test")

    assert datasetUtils.getSize("Group") == 2
    assert datasetUtils.getSize("Group/Group2") == 1
    assert datasetUtils.getSize("Group1/Group2") == 2
    assert datasetUtils.getSize("*/Group2") == 3
    assert datasetUtils.getSize("Deformation") == 1

    assert datasetUtils.getNames("Group") == ["Data1", "Data2"]
    assert datasetUtils.getNames("Group/Group2") == ["Image"]
    assert set(datasetUtils.getNames("*/Group2")) == {"Image", "CompositeTransform", "Transform"}
    assert datasetUtils.getNames("Group1/Group2") == ["CompositeTransform", "Transform"]
    assert datasetUtils.getNames("Deformation") == ["Landmarks"]

    s, a = datasetUtils.getInfos("Group", "Data1")
    assert shape == s
    assert attribute == a

    s, a = datasetUtils.getInfos("Group", "Data2")
    assert shape == s
    assert attribute == a

    s, a = datasetUtils.getInfos("Group/Group2", "Image")
    assert tuple([1]+list(shape)) == s

    s, a = datasetUtils.getInfos("*/Group2", "Image")
    assert tuple([1]+list(shape)) == s

    s, a = datasetUtils.getInfos("Group1/Group2", "Transform")
    assert a == {"0:FixedParameters_0": "(0.0, 0.0, 0.0, 0.0)", "Test": "1", "0:Transform_0": "Euler3DTransform_double_3_3"}
    assert s == (1,6)

    s, a = datasetUtils.getInfos("Group1/Group2", "CompositeTransform")
    assert a == {'0:FixedParameters_0': '(0.0, 0.0, 0.0, 0.0)', '1:FixedParameters_0': '(0.0, 0.0, 0.0, 0.0)', 'Test': '1', '0:Transform_0': 'Euler3DTransform_double_3_3', '1:Transform_0': 'Euler3DTransform_double_3_3'}
    assert s == (2,6)

    s, a = datasetUtils.getInfos("Deformation", "Landmarks")
    assert a == {}
    assert s == (50,3)

    d, a = datasetUtils.readData("Group", "Data1")
    assert np.array_equal(data, d)
    assert attribute == a

    i = datasetUtils.readImage("Group/Group2", "Image")
    assert i == image

    i = datasetUtils.readImage("*/Group2", "Image")
    assert i == image

    t = datasetUtils.readTransform("Group1/Group2", "Transform")
    assert isinstance(t, sitk.Euler3DTransform) and t.GetParameters() == euler3D_0

    t = datasetUtils.readTransform("Group1/Group2", "CompositeTransform")
    assert isinstance(t, sitk.CompositeTransform) and isinstance(t.GetNthTransform(0), sitk.Euler3DTransform) and isinstance(t.GetNthTransform(1), sitk.Euler3DTransform) and t.GetNthTransform(0).GetParameters() == euler3D_0 and t.GetNthTransform(1).GetParameters() == euler3D_1

    d, a = datasetUtils.readData("Deformation", "Landmarks")
    assert np.array_equal(landmarks, d)
    assert {} == a


def test(datasetUtils: DatasetUtils):
    """H5 Test"""
    shape = (10,20,30)
    data = np.zeros(shape)
    image = sitk.GetImageFromArray(data)
    direction = (0,1,0,1,0,0,0,0,1)
    image.SetDirection(direction)
    origin = (10,20,30)
    image.SetOrigin(origin)
    spacing = (1,2,3)
    image.SetSpacing(spacing)

    RGB_image = sitk.Image([128,100], sitk.sitkVectorUInt8, 3)

    euler3D_0 = (10.0,20.0,30.0,5.0,6.0,7.0)
    transform_0 = sitk.Euler3DTransform()
    transform_0.SetParameters(euler3D_0)

    euler3D_1 = (50.0,18.0,30.0,5.0,6.0,7.0)
    transform_1 = sitk.Euler3DTransform()
    transform_1.SetParameters(euler3D_1)

    compositeTransform = sitk.CompositeTransform([transform_0, transform_1])
    landmarks = np.random.random((50, 3))


    attribute = Attribute({"Test": "1"})
    try:
        datasetUtils.write("Group", "Data1", data, Attribute(attribute))
        assert False
    except:
        pass
    datasetUtils.write("Group/Group2", "Image", image, Attribute(attribute))
    datasetUtils.write("Group", "RGBImage", RGB_image, Attribute(attribute))
    datasetUtils.write("Group1/Group2", "Transform", transform_0, Attribute(attribute))
    datasetUtils.write("Group1/Group2", "CompositeTransform", compositeTransform, Attribute(attribute))
    datasetUtils.write("Deformation", "Landmarks", landmarks)


    assert datasetUtils.isGroupExist("Group")
    assert datasetUtils.isGroupExist("Group/Group2")
    assert datasetUtils.isGroupExist("Group1/Group2")
    assert datasetUtils.isGroupExist("*/Group2")
    assert datasetUtils.isGroupExist("Deformation")

    assert datasetUtils.isDatasetExist("Group", "RGBImage")
    assert not datasetUtils.isDatasetExist("Group", "Data2")
    assert datasetUtils.isDatasetExist("Group/Group2", "Image")
    assert datasetUtils.isDatasetExist("Group1/Group2", "Transform")
    assert datasetUtils.isDatasetExist("*/Group2", "Image")
    assert not datasetUtils.isDatasetExist("*/Group1", "Test")
    assert datasetUtils.isDatasetExist("Deformation", "Landmarks")

    assert not datasetUtils.isGroupExist("Group/Group1")
    assert not datasetUtils.isDatasetExist("Test/Group1", "Test")
    assert not datasetUtils.isDatasetExist("Group/Group1", "Test")
    
    assert datasetUtils.getSize("Group") == 1
    assert datasetUtils.getSize("Group/Group2") == 1
    assert datasetUtils.getSize("Group1/Group2") == 2
    assert datasetUtils.getSize("*/Group2") == 3
    assert datasetUtils.getSize("Deformation") == 1

    assert datasetUtils.getNames("Group") == ["RGBImage"]
    assert datasetUtils.getNames("Group/Group2") == ["Image"]
    assert datasetUtils.getNames("Group1/Group2") == ["CompositeTransform", "Transform"]
    assert datasetUtils.getNames("*/Group2") == ['CompositeTransform', 'Transform', 'Image']
    assert datasetUtils.getNames("Deformation") == ["Landmarks"]

    s, a = datasetUtils.getInfos("Group/Group2", "Image")
    assert tuple([1]+list(shape)) == s

    s, a = datasetUtils.getInfos("*/Group2", "Image")
    assert tuple([1]+list(shape)) == s

    s, a = datasetUtils.getInfos("Group", "RGBImage")
    assert (3,128,100) == s

    s, a = datasetUtils.getInfos("Group1/Group2", "Transform")
    assert a == {'0:Transform_0': 'Euler3DTransform_double_3_3', '0:FixedParameters_0': "(0.0, 0.0, 0.0, 0.0)"}
    assert s == (1,6)

    s, a = datasetUtils.getInfos("Group1/Group2", "CompositeTransform")
    assert a == {'0:Transform_0': 'Euler3DTransform_double_3_3', '0:FixedParameters_0': "(0.0, 0.0, 0.0, 0.0)", '1:Transform_0': 'Euler3DTransform_double_3_3', '1:FixedParameters_0': "(0.0, 0.0, 0.0, 0.0)"}
    assert s == (2,6)

    s, a = datasetUtils.getInfos("Deformation", "Landmarks")
    assert a == {}
    assert s == (50,3)

    i = datasetUtils.readImage("Group/Group2", "Image")
    assert i == image

    i = datasetUtils.readImage("*/Group2", "Image")
    assert i == image

    i = datasetUtils.readImage("Group", "RGBImage")

    t = datasetUtils.readTransform("Group1/Group2", "Transform")
    assert isinstance(t, sitk.Euler3DTransform) and t.GetParameters() == euler3D_0

    t = datasetUtils.readTransform("Group1/Group2", "CompositeTransform")
    assert isinstance(t, sitk.CompositeTransform) and isinstance(t.GetNthTransform(0), sitk.Euler3DTransform) and isinstance(t.GetNthTransform(1), sitk.Euler3DTransform) and t.GetNthTransform(0).GetParameters() == euler3D_0 and t.GetNthTransform(1).GetParameters() == euler3D_1
    
    d, a = datasetUtils.readData("Deformation", "Landmarks")
    assert {} == a

if __name__ == "__main__":
    if os.path.exists("h5_test.h5"):
        os.remove("h5_test.h5")
    test_h5(DatasetUtils("h5_test", "h5"))
    test_h5(DatasetUtils("h5_test", "h5"))

    if os.path.exists("h5_test/"):
        shutil.rmtree("h5_test/")
    test_h5(DatasetUtils("h5_test/", "h5"))
    test_h5(DatasetUtils("h5_test/", "h5"))
    if os.path.exists("mha_test/"):
        shutil.rmtree("mha_test/")
    test(DatasetUtils("mha_test", "mha"))
    if os.path.exists("nii_gz_test/"):
        shutil.rmtree("nii_gz_test/")
    test(DatasetUtils("nii_gz_test", "nii.gz"))
    test(DatasetUtils("nii_gz_test", "nii.gz"))