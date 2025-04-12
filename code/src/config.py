"""

This file holds configuration settings - like hyperparamters - 
so they can be easily managed and updated.

"""

class Config:
    def __init__(self):
        
        #******** Preprocessing Hyperparameters ********#

        #Sources
        self.caseStudes_db: bool = True
        self.defitions_db: bool = True
        self.companies_db: bool = True

        self.data_path: str = '../data/' #relative



        #******** Training Hyperparameters ********#

        self.pretrained_model: str = 'DeepSeek-r1'
        self.learning_rate: float = 5e-5
        self.batch_size: int = 32




        #******** Evaluation Hyperparameters ********#


