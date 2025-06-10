"""
Controller class for top-level management of the data processing.
"""

from src.applications import Applications_Reformat
from src.definitions import Definitions_Reformat        
from src.companies import Companies_Reformat
import time

class Controller:
    def __init__(self, config):
        self.config = config
        self.applications = Applications_Reformat(self.config['applications_db'])
        self.definitions = Definitions_Reformat(self.config['definitions_db'])
        self.companies = Companies_Reformat(self.config['companies_db'])
    
    def run(self):
        #************ Applications ************#
        if self.config['run_applications_db'] == True:
            print("\n>> Processing Applications")
            self.applications.run()
            print(">> Completed Applications\n")
            time.sleep(self.config['sleep'])
        
        #************ Definitions ************#
        if self.config['run_definitions_db'] == True:
            print(">> Processing Definitions")
            self.definitions.run()
            print(">> Completed Definitions\n")
            time.sleep(self.config['sleep'])

        #************ Companies ************#
        if self.config['run_companies_db'] == True:
            print(">> Processing Companies")
            self.companies.run()
            print(">> Completed Companies\n")
            time.sleep(self.config['sleep'])

        print(">> dataPre Done.\n")