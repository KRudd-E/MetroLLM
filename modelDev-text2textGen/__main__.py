from src.pipeline import FinetunePipeline

if __name__ == "__main__":
    pipeline = FinetunePipeline()
    pipeline.run()
    
# NB: config_path = 'modelDev/config.yaml' 
# To alter, see modelDev/src/misc/utils.py