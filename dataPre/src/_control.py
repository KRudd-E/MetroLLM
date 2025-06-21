from src.applications import Applications_Reformat
from src.definitions import Definitions_Reformat        
from src.companies import Companies_Reformat
from src.utils import applicationsDB_initialisation_query, companiesDB_initialisation_query, definitionsDB_initialisation_query

class Controller:
    def __init__(self, config):
        self.config = config
    
    def run(self):
        #************ Applications ************#
        if self.config['run_applications_db'] == True:
            applicationsDB_initialisation_query()
            print("\n>> Processing Applications")
            applications = Applications_Reformat(self.config['applications_db'])
            applications.run()
            print(">> Completed Applications\n")
        
        #************ Definitions ************#
        if self.config['run_definitions_db'] == True:
            definitionsDB_initialisation_query()
            print(">> Processing Definitions")
            definitions = Definitions_Reformat(self.config['definitions_db'])
            definitions.run()
            print(">> Completed Definitions\n")

        #************ Companies ************#
        if self.config['run_companies_db'] == True:
            companiesDB_initialisation_query()
            print(">> Processing Companies")
            companies = Companies_Reformat(self.config['companies_db'])
            companies.run()
            print(">> Completed Companies\n")