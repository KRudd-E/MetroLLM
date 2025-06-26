if __name__ == "__main__":
    from src.utils.misc import get_config
    config = get_config("modelDev-text2textGen/config.yaml")
    from src.pipeline import FinetunePipeline
    pipeline = FinetunePipeline(config)
    pipeline.run()