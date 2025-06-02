"""
This module defines the Definitions class, which reformats the data into a useful format for modelDev.
"""

class Definitions:
    def __init__(self, config):
        self.config = config
        if self.config.get('debug', True):
            print(f'Definitions initialized with config:\n {self.config}')

    def run(self):
        
        pass