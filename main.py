from enum import Enum
import sys
import argparse
import os

import numpy as np

from DeepLearning_API import Trainer, State, Predictor, CONFIG_FILE

def main():
    parser = argparse.ArgumentParser(description="DeepLearing API",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("type", type=State, choices=list(State))
    parser.add_argument("-c", "--config", default="None", help="Configuration file location")
    parser.add_argument("-models_dir", "--MODELS_DIRECTORY", default="./Models/", help="Models location")
    parser.add_argument("-checkpoints_dir", "--CHECKPOINTS_DIRECTORY", default="./Checkpoints/", help="Checkpoints location")
    parser.add_argument("-predictions_dir", "--PREDICTIONS_DIRECTORY", default="./Predictions/", help="Predictions location")
    parser.add_argument("-statistics_dir", "--STATISTICS_DIRECTORY", default="./Statistics/", help="Statistics location")
    parser.add_argument("-setups_dir", "--SETUPS_DIRECTORY", default="./Setups/", help="Setups location")
    
    args = parser.parse_args()
    config = vars(args)

    os.environ["DL_API_MODELS_DIRECTORY"] = config["MODELS_DIRECTORY"]
    os.environ["DL_API_CHECKPOINTS_DIRECTORY"] = config["CHECKPOINTS_DIRECTORY"]
    os.environ["DL_API_PREDICTIONS_DIRECTORY"] = config["PREDICTIONS_DIRECTORY"]
    os.environ["DL_API_STATISTICS_DIRECTORY"] = config["STATISTICS_DIRECTORY"]
    os.environ["DL_API_SETUPS_DIRECTORY"] = config["SETUPS_DIRECTORY"]
    os.environ["DEEP_LEANING_API_CONFIG_MODE"] = "Done"
    
    if config["config"] == "None":
        os.environ["DEEP_LEARNING_API_CONFIG_FILE"] = "Config.yml" if config["type"]  is not State.PREDICTION else "Prediction.yml"
    else:
        os.environ["DEEP_LEARNING_API_CONFIG_FILE"] = config["config"]

    if config["type"] is not State.PREDICTION:
        os.environ["DEEP_LEARNING_API_ROOT"] = "Trainer"
        with Trainer(config = CONFIG_FILE()) as trainer:
            trainer.train(config["type"])

    else:
        os.environ["DEEP_LEARNING_API_ROOT"] = "Predictor"
        with Predictor(config = CONFIG_FILE()) as predictor:
            predictor.predict()

if __name__ == "__main__":
    main()