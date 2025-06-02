"""
Controller class for top-level management of the data processing.
"""

class Controller:
    def __init__(self, config):
        self.config = config
    
    def run(self):
        #************ Case Studies ************#
        if self.config['run_caseStudies_db'] == True:
            print("\n>> Running Case Studies Database Processing")
            from src.caseStudy import CaseStudies
            case_studies = CaseStudies(self.config['caseStudies_db'])
            case_studies.run()
            print(">> Case Studies completed.\n")
        
        #************ Definitions ************#
        if self.config['run_definitions_db'] == True:
            print(">> Running Definitions Database Processing")
            from src.definition import Definitions
            definitions = Definitions(self.config['definitions_db'])
            definitions.run()
            print(">> Definitions completed.\n")
        
        #************ Companies ************#
        if self.config['run_companies_db'] == True:
            print(">> Running Companies Database Processing")
            from src.company import Companies
            companies = Companies(self.config['companies_db'])
            companies.run()
            print(">> Companies completed.")