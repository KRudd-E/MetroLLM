"""
This module defines the Definitions class, which reformats the data into a useful format for modelDev.
"""
import time

class Definitions_Reformat:
    def __init__(self, config):
        self.config = config
        if self.config.get('debug', True):
            print(f'Definitions_Reformat initialized with config:\n {self.config}')

    def run(self):
        time.sleep(2)
        print('Definitions_Reformat is empty!')
        time.sleep(2)
        pass