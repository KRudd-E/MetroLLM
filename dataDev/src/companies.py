"""
This module defines the Companies class, which reformats the data into a useful format for modelDev.
"""
import time 

class Companies_Reformat:
    def __init__(self, config):
        self.config = config
        if self.config.get('debug', True):
            print(f'Companies_Reformat initialized with config:\n {self.config}')

    def run(self):
        time.sleep(2)
        print('Companies_Reformat is empty!')
        time.sleep(2)
        pass