"""
Controller class for top-level management of the data processing.
"""

class Controller:
    def __init__(self, config):
        self.config = config
    
    def run(self):
        #************ Case Studies ************#
        if self.config['run_caseStudies_db'] == True:
            print("\n>> Processing Case Studies Database...")
            from src.caseStudies import CaseStudies_Reformat
            case_studies = CaseStudies_Reformat(self.config['caseStudies_db'])
            case_studies.run()
            print(">> Case Studies completed.\n")
        
        #************ Definitions ************#
        if self.config['run_definitions_db'] == True:
            print(">> Processing Definitions Database...")
            from src.definitions import Definitions_Reformat
            definitions = Definitions_Reformat(self.config['definitions_db'])
            definitions.run()
            print(">> Definitions completed.\n")
        
        #************ Companies ************#
        if self.config['run_companies_db'] == True:
            print(">> Processing Companies Database...")
            from src.companies import Companies_Reformat
            companies = Companies_Reformat(self.config['companies_db'])
            companies.run()
            print(">> Companies completed.")

        print("\n>> Data processing completed.\n")