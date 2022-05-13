import os

MODELS_DIRECTORY = lambda : os.environ["DL_API_MODELS_DIRECTORY"]
CHECKPOINTS_DIRECTORY =lambda : os.environ["DL_API_CHECKPOINTS_DIRECTORY"]
PREDICTIONS_DIRECTORY =lambda : os.environ["DL_API_PREDICTIONS_DIRECTORY"]
STATISTICS_DIRECTORY = lambda : os.environ["DL_API_STATISTICS_DIRECTORY"]
SETUPS_DIRECTORY = lambda : os.environ["DL_API_SETUPS_DIRECTORY"]
CONFIG_FILE = lambda : os.environ["DEEP_LEARNING_API_CONFIG_FILE"]

from .utils import memoryInfo, gpuInfo, cpuInfo, data_to_dataset, image_to_dataset, dataset_to_data, dataset_to_image, memoryForecast, getMemory, getAvailableDevice, getDevice, DatasetUtils, _getModule
from .config import config
from .transform import Transform, TransformLoader
from .HDF5 import HDF5
from .dataset import DataSet, DataTrain, DataPrediction, Group
from .metric import Dice
from .criterion import Loss
from .trainer import Trainer, State
from .predictor import Predictor
from . import networks