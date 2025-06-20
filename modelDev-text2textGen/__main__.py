from src.pipeline import FinetunePipeline
import yaml

def main():
    config = load_config()
    pipeline = FinetunePipeline(config)
    pipeline.run()

def load_config(path="modelDev-text2textGen/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    main()