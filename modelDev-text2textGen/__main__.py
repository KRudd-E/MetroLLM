from src.pipeline import FinetunePipeline
from src.utils.misc import get_config

if __name__ == "__main__":
    config = get_config("modelDev-text2textGen/config.yaml")
    pipeline = FinetunePipeline(config)
    pipeline.run()