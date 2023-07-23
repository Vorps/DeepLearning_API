from DeepLearning_API.config import config
import os

class Metric():

    @config("Metric")
    def __init__(self, metric_name: str = "name") -> None:
        if os.environ["DEEP_LEANING_API_CONFIG_MODE"] != "Done":
            exit(0)
        self.metric_name = metric_name

    def __enter__(self):
        return self
    
    def __exit__(self, type, value, traceback):
        pass

    def measure(self):
        print(self.metric_name)