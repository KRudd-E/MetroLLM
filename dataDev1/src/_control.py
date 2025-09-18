from src.applications   import Applications_Reformat
from src.companies      import Companies_Reformat
from src.definitions    import Definitions_Reformat        
from src.utils import (applicationsDB_initialisation_query, 
                       companiesDB_initialisation_query, 
                       definitionsDB_initialisation_query,
                       dataDev1_query,
                       parser)

class Controller:
    def __init__(self, config):
        self.config = config
    
    def run(self):
        data_source = parser()
        dataDev1_query(data_source)
        
        #** Applications **#
        if data_source == 'Applications':
            applicationsDB_initialisation_query()
            applications = Applications_Reformat(self.config['applications_db'])
            applications.run()

        #** Companies **#
        if data_source == 'Companies':
            companiesDB_initialisation_query()
            companies = Companies_Reformat(self.config['companies_db'])
            companies.run()
        
        #** Definitions **#
        if data_source == 'Definitions':
            definitionsDB_initialisation_query()
            definitions = Definitions_Reformat(self.config['definitions_db'])
            definitions.run()