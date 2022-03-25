import SimpleITK as sitk
import h5py
import numpy as np
import torch

class HDF5():

    class Dataset():

        def __init__(self, dataset : h5py.Dataset) -> None:
            self.dataset = dataset
            self.name = dataset.name
            self.attrs = dataset.attrs
            self.loaded = False
        
        def read(self, i : int, n : int) -> torch.Tensor:
            shape = [self.dataset.shape[0]//n] + list(self.dataset.shape)[1:]
            self.data = np.zeros(shape, self.dataset.dtype)
            self.dataset.read_direct(self.data, np.s_[i*shape[0]:(i+1)*shape[0], ...])
            self.data = torch.from_numpy(self.data)
            self.loaded = True

    def __init__(self, filename : str) -> None:
        self.filename = filename

    def __enter__(self):
        self.h5 = h5py.File(self.filename, 'r')
        self.size = int(self.h5.attrs["Size"])
        return self
    
    def __exit__(self, type, value, traceback):
        self.h5.close()

    def getDatasets(self, path, index):
        return [HDF5.Dataset(self.h5[path][list(self.h5[path])[i]]) for i in index]

def data_to_dataset(h5, name, data, origin, spacing, direction):
    dataset = h5.create_dataset(name, data=data, compression='gzip', chunks=True)
    dataset.attrs['Origin'] = origin
    dataset.attrs['Spacing'] = spacing
    dataset.attrs['Direction'] = direction
    
def image_to_dataset(h5, name, image):
    data_to_dataset(h5, name, sitk.GetArrayFromImage(image), origin=image.GetOrigin(), spacing=image.GetSpacing(), direction=image.GetDirection())

