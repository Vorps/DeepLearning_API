import argparse
import os
from subprocess import call
from DeepLearning_API import CONFIG_FILE
from DeepLearning_API.trainer import Trainer
from DeepLearning_API.predictor import Predictor
from DeepLearning_API.utils import State

def train(state : State):
    os.environ["DEEP_LEARNING_API_ROOT"] = "Trainer"
    with Trainer(config = CONFIG_FILE()) as trainer:
        trainer.train(state)

def predict():
    os.environ["DEEP_LEARNING_API_ROOT"] = "Predictor"
    with Predictor(config = CONFIG_FILE()) as predictor:
        predictor.predict()

def main():
    parser = argparse.ArgumentParser(description="DeepLearing API",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    api_args = parser.add_argument_group('API arguments')
    api_args.add_argument("type", type=State, choices=list(State))
    api_args.add_argument('-y', action='store_true', help="Accept overwrite")
    api_args.add_argument("-c", "--config", type=str, default="None", help="Configuration file location")
    api_args.add_argument("-models_dir", "--MODELS_DIRECTORY", type=str, default="./Models/", help="Models location")
    api_args.add_argument("-checkpoints_dir", "--CHECKPOINTS_DIRECTORY", type=str, default="./Checkpoints/", help="Checkpoints location")
    api_args.add_argument("-url", "--URL_MODEL", type=str, default="", help="URL Model")
    api_args.add_argument("-predictions_dir", "--PREDICTIONS_DIRECTORY", type=str, default="./Predictions/", help="Predictions location")
    api_args.add_argument("-statistics_dir", "--STATISTICS_DIRECTORY", type=str, default="./Statistics/", help="Statistics location")
    api_args.add_argument("-setups_dir", "--SETUPS_DIRECTORY", type=str, default="./Setups/", help="Setups location")
    api_args.add_argument('--resubmit', action='store_true', help='Automatically resubmit job just before timout')
    
    args = parser.parse_args()
    config = vars(args)

    os.environ["DL_API_MODELS_DIRECTORY"] = config["MODELS_DIRECTORY"]
    os.environ["DL_API_CHECKPOINTS_DIRECTORY"] = config["CHECKPOINTS_DIRECTORY"]
    os.environ["DL_API_PREDICTIONS_DIRECTORY"] = config["PREDICTIONS_DIRECTORY"]
    os.environ["DL_API_STATISTICS_DIRECTORY"] = config["STATISTICS_DIRECTORY"]
    os.environ["DL_API_URL_MODEL"] = config["URL_MODEL"]

    os.environ["DL_API_SETUPS_DIRECTORY"] = config["SETUPS_DIRECTORY"]

    os.environ["DL_API_OVERWRITE"] = "{}".format(config["y"])
    os.environ["DEEP_LEANING_API_CONFIG_MODE"] = "Done"
    
    if config["config"] == "None":
        os.environ["DEEP_LEARNING_API_CONFIG_FILE"] = "Config.yml" if config["type"]  is not State.PREDICTION else "Prediction.yml"
    else:
        os.environ["DEEP_LEARNING_API_CONFIG_FILE"] = config["config"]
    
    if config["type"] is not State.PREDICTION:
        train(config["type"])
    else:
        predict()

if __name__ == "__main__":
    main()
 