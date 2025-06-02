"""
This module defines the Companies class, which reformats the data into a useful format for modelDev.
"""

class Companies_Reformat:
    def __init__(self, config):
        self.config = config
        if self.config.get('debug', True):
            print(f'Companies_Reformat initialized with config:\n {self.config}')

    def run(self):
        
        pass