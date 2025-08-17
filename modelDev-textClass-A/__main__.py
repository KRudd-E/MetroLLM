from src.pipeline import FineTunePipeline

if __name__ == "__main__":
    pipeline = FineTunePipeline()
    pipeline.run()
    
# NB: config_path = 'modelDev/config.yaml' 
# To alter, see modelDev/src/misc/utils.py