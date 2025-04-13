""" 
For storing settings and paramters used in the dataDev script
"""


class Config:
    def __init__(self):

        self.caseStudes_db: bool = True
        self.defitions_db: bool = True
        self.companies_db: bool = True
        
        self.data_path = '../data/original'
        self.final_data_dir = '../data/final'
        


