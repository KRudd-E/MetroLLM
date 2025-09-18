from src.utils import get_config, pureAPI_query, check_config
from src.retrieve import Retriever
from src.ai import AI_Assister

import os
import pandas as pd
from tqdm import tqdm
from openpyxl import load_workbook
from time import strftime


class Pipeline:
    def __init__(self):
        self.config = get_config()
        self.starting_datetime = strftime("%Y-%m-%d %H-%M-%S")
        self.ai_assist = AI_Assister(self.config)
        self.retriever = Retriever(self.config)

    
    def run(self):

        #**** Create DataFrame and ID ***#
        df = pd.read_csv(os.path.join(os.getcwd() + self.config['input_dir']))
        df['Pred_task'] = ''

        #**** Iterate over folders and text files ****#
        for row in tqdm(df.iterrows(), total=len(df), desc="Processing Instances", position=0, colour='blue'):
            
            text = row[1]['Text']

            #**** Generate Task(s) ****#
            task_dict: dict = self.retriever.retrieve_multiple(
                names=['task'],
                options={
                    'task': self.config['task_list']
                },
                prompt=self.config['task_prompt'].format(
                    task_list=self.config['task_list'],
                    txt=text
                ),
                tries=3,
                expect_result=True,
            )
        
            #**** Add to DataFrame ****#
            df.at[row[0], 'Pred_task'] = task_dict.get('task', '')
        
        
        
        #** Save DataFrame to CSV ***#
        output_path = os.path.join(os.getcwd() +  self.config['output_dir'] + f'output_{self.config["model"]}.csv')
        df.to_csv(output_path, index=False, quotechar='"', quoting=1)


    @staticmethod
    def clean_value(val: str | list) -> str:
        """ Cleans the value to be added to the DataFrame.
        """
        if isinstance(val, list):
            if len(val) == 0:
                return ''
            elif len(val) == 1:
                return str(val[0])
            else:
                return ' | '.join(str(v) for v in val)
        return val if val is not None else ''

