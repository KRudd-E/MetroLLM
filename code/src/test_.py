""" 
For testing the model against the test dataset.
"""

class Tester:
    def __init__(self, config):
        self.config = config

    def run_tests(self):
        print("Running tests...")
        # Load test data, run predictions