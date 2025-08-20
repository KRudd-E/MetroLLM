
from src.model_wrapper import ClassificationWrapper


class Evaluator:
    def __init__(self, config):
        self.config = config

    def run(self):
        """Run basic evaluation."""
        # Load trained model (no weights needed for evaluation)
        model_wrapper = ClassificationWrapper("evaluate", self.config, pos_weights=None)
        
        print("Evaluation setup complete. Model loaded successfully.")
        # Add your evaluation logic here
        
        return