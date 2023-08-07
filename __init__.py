import os
import datetime

MODELS_DIRECTORY = lambda : os.environ["DL_API_MODELS_DIRECTORY"]
CHECKPOINTS_DIRECTORY =lambda : os.environ["DL_API_CHECKPOINTS_DIRECTORY"]
URL_MODEL =lambda : os.environ["DL_API_URL_MODEL"]
PREDICTIONS_DIRECTORY =lambda : os.environ["DL_API_PREDICTIONS_DIRECTORY"]
METRICS_DIRECTORY =lambda : os.environ["DL_API_METRICS_DIRECTORY"]
STATISTICS_DIRECTORY = lambda : os.environ["DL_API_STATISTICS_DIRECTORY"]
SETUPS_DIRECTORY = lambda : os.environ["DL_API_SETUPS_DIRECTORY"]
CONFIG_FILE = lambda : os.environ["DEEP_LEARNING_API_CONFIG_FILE"]

DATE = lambda : datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")