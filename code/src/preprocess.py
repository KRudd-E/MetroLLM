"""
For handling all data preprocessing
"""

class DataPreprocessor:
    def __init__(self, config):
        self.config = config

    def prepare_data(self):
        print("Preprocessing data from:", self.config.data_path)
        # Implement data loading, cleaning, and splitting