"""
Controller class for top-level management of the data processing.
"""

import time

class Controller:
    def __init__(self, config):
        self.config = config
    
    def run(self):
        #************ Applications ************#
        if self.config['run_applications_db'] == True:
            print("\n>> Processing Case Studies Database...")
            from src.applications import Applications_Reformat
            applications = Applications_Reformat(self.config['applications_db'])
            applications.run()
            print(">> Applications completed.\n")
            time.sleep(self.config['sleep'])
        
        #************ Definitions ************#
        if self.config['run_definitions_db'] == True:
            print(">> Processing Definitions Database...")
            from src.definitions import Definitions_Reformat
            definitions = Definitions_Reformat(self.config['definitions_db'])
            definitions.run()
            print(">> Definitions completed.\n")
            time.sleep(self.config['sleep'])

        #************ Companies ************#
        if self.config['run_companies_db'] == True:
            print(">> Processing Companies Database...")
            from src.companies import Companies_Reformat
            companies = Companies_Reformat(self.config['companies_db'])
            companies.run()
            print(">> Companies completed.\n")
            time.sleep(self.config['sleep'])

        print(">> Data processing completed.\n")