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
    euler3D = (10.0,20.0,30.0,5.0,6.0,7.0)
    transform = sitk.Euler3DTransform()
    transform.SetParameters(euler3D)

    attribute = Attribute({"Test": "1"})
    datasetUtils.write("Group", "Data1", data, Attribute(attribute))
    datasetUtils.write("Group", "Data2", data, Attribute(attribute))
    datasetUtils.write("Group/Group2", "Image", image, Attribute(attribute))
    datasetUtils.write("Group1/Group2", "Transform", transform, Attribute(attribute))

    assert datasetUtils.isGroupExist("Group")
    assert datasetUtils.isGroupExist("Group/Group2")
    assert datasetUtils.isGroupExist("Group1/Group2")

    assert datasetUtils.isDatasetExist("Group", "Data1")
    assert datasetUtils.isDatasetExist("Group", "Data2")
    assert datasetUtils.isDatasetExist("Group/Group2", "Image")
    assert datasetUtils.isDatasetExist("Group1/Group2", "Transform")

    assert not datasetUtils.isGroupExist("Group/Group1")
    assert not datasetUtils.isDatasetExist("Test/Group1", "Test")
    assert not datasetUtils.isDatasetExist("Group/Group1", "Test")

    assert datasetUtils.getSize("Group") == 2
    assert datasetUtils.getSize("Group/Group2") == 1
    assert datasetUtils.getSize("Group1/Group2") == 1

    assert datasetUtils.getNames("Group") == ["Data1", "Data2"]
    assert datasetUtils.getNames("Group/Group2") == ["Image"]
    assert datasetUtils.getNames("Group1/Group2") == ["Transform"]

    s, a = datasetUtils.getInfos("Group", "Data1")
    assert shape == s
    assert attribute == a

    s, a = datasetUtils.getInfos("Group", "Data2")
    assert shape == s
    assert attribute == a

    s, a = datasetUtils.getInfos("Group/Group2", "Image")
    assert tuple([1]+list(shape)) == s

    s, a = datasetUtils.getInfos("Group1/Group2", "Transform")
    assert a == {"FixedParameters_0": "(0.0, 0.0, 0.0, 0.0)", "Test": "1", "Transform_0": "Euler3DTransform_double_3_3"}
    assert s == (6,)

    d, a = datasetUtils.readData("Group", "Data1")
    assert np.array_equal(data, d)
    assert attribute == a

    i = datasetUtils.readImage("Group/Group2", "Image")
    assert i == image

    t = datasetUtils.readTransform("Group1/Group2", "Transform")
    assert isinstance(t, sitk.Euler3DTransform) and t.GetParameters() == euler3D

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

    euler3D = (10.0,20.0,30.0,5.0,6.0,7.0)
    transform = sitk.Euler3DTransform()
    transform.SetParameters(euler3D)

    attribute = Attribute({"Test": "1"})
    try:
        datasetUtils.write("Group", "Data1", data, Attribute(attribute))
        assert False
    except:
        pass
    datasetUtils.write("Group/Group2", "Image", image, Attribute(attribute))
    datasetUtils.write("Group", "RGBImage", RGB_image, Attribute(attribute))
    datasetUtils.write("Group1/Group2", "Transform", transform, Attribute(attribute))


    assert datasetUtils.isGroupExist("Group")
    assert datasetUtils.isGroupExist("Group/Group2")
    assert datasetUtils.isGroupExist("Group1/Group2")

    assert datasetUtils.isDatasetExist("Group", "RGBImage")
    assert not datasetUtils.isDatasetExist("Group", "Data2")
    assert datasetUtils.isDatasetExist("Group/Group2", "Image")
    assert datasetUtils.isDatasetExist("Group1/Group2", "Transform")

    assert not datasetUtils.isGroupExist("Group/Group1")
    assert not datasetUtils.isDatasetExist("Test/Group1", "Test")
    assert not datasetUtils.isDatasetExist("Group/Group1", "Test")

    assert datasetUtils.getSize("Group") == 1
    assert datasetUtils.getSize("Group/Group2") == 1
    assert datasetUtils.getSize("Group1/Group2") == 1

    assert datasetUtils.getNames("Group") == ["RGBImage"]
    assert datasetUtils.getNames("Group/Group2") == ["Image"]
    assert datasetUtils.getNames("Group1/Group2") == ["Transform"]

    s, a = datasetUtils.getInfos("Group/Group2", "Image")
    assert tuple([1]+list(shape)) == s

    s, a = datasetUtils.getInfos("Group", "RGBImage")
    assert (3,128,100) == s

    i = datasetUtils.readImage("Group/Group2", "Image")
    assert i == image

    i = datasetUtils.readImage("Group", "RGBImage")

    t = datasetUtils.readTransform("Group1/Group2", "Transform")
    assert isinstance(t, sitk.Euler3DTransform) and t.GetParameters() == euler3D

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