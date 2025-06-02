"""
This module defines the CaseStudies class, which reformats the data into a useful format for modelDev.
"""

class CaseStudies:
    def __init__(self, config):
        self.config = config
        if self.config.get('debug', True):
            print(f'CaseStudies initialized with config:\n {self.config}')

    def run(self):
        
        pass