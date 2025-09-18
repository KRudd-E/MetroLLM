import os
import pandas as pd
from tqdm import tqdm
import uuid
from src.utils.retrieve import Retriever
from src.utils.format_txt import format_txt
from src.utils.patterns import PATTERNS
import random
import re
import csv

class TextGen:
    def __init__(self, config, src):
        self.config = config
        self.retriever = Retriever()
        print('\n')
        if src == 'applications'  : self.applicationsDB_Gen(config['applicationsDB'])
        if src == 'definitions'   : self.definitionsDB_Gen(config['definitionsDB'])
        if src == 'companies'     : print("Companies DB generation not yet implemented."); exit()
        
    def applicationsDB_Gen(self, config):
        
        #** 1 - JSON QA pairs **#
        if config['JSON_pairs']['enabled']:
            
            df1 = pd.read_csv(os.path.join(os.getcwd() + config['JSON_pairs']['source_dir'])).set_index('id')
            f_df1 = pd.DataFrame(columns=['id', 'pattern', 'input', 'output'])

            for row in tqdm(df1.iterrows(), total=len(df1), desc="Processing applications", position=0, dynamic_ncols=True, colour='blue'):
                for choice in ['task', 'sector', 'user']:

                    input_txt = config['JSON_pairs']['input'][choice].format(txt=row[1]['Text']).strip().replace("\n","    ")
                    output_txt = config['JSON_pairs']['output'][choice].format(task=row[1]['Task'], sector=row[1]['Sector'], user=row[1]['Sector']) \
                        .strip().replace("€","{").replace("£","}").replace("\n","  ")
                    id = str(uuid.uuid4())
                    f_df1 = pd.concat([f_df1, pd.DataFrame([[id, 'JSON-Pair', input_txt, output_txt]], columns=['id', 'pattern', 'input', 'output'])], ignore_index=True)


            f_df1.to_csv(os.path.join(os.getcwd() + config['JSON_pairs']['output_dir']), index=False)
        
        
        
        #** 2 - General QA pairs from text windows **#
        if config['general_pairs']['enabled']:
            
            #** Prepare data **#
            df2 = pd.read_csv(os.path.join(os.getcwd() + config['general_pairs']['source_dir'])).set_index('id').sort_values(by='file_name')
            if config['general_pairs']['starting_file']:
                df2 = df2[df2['file_name'] >= config['general_pairs']['starting_file']]
            if config['general_pairs']['ending_file']:
                df2 = df2[df2['file_name'] <= config['general_pairs']['ending_file']]
            
            #** Prepare output **#
            if config['general_pairs']['append_or_overwrite'] == 'append':
                self.append_manager(config['general_pairs']['output_dir'], 'id,file_name,pattern,contextual,input,output')
            if config['general_pairs']['append_or_overwrite'] == 'overwrite':
                self.overwrite_manager(config['general_pairs']['output_dir'], 'id,file_name,pattern,contextual,input,output')
            
           #** Iterate through case studies **#
            for row in tqdm(df2.iterrows(), total=len(df2), desc="Processing applications", position=0, dynamic_ncols=True, colour='blue'):
                
                try:
                    row[1]['text'] = format_txt(row[1]['text'])
                except Exception as e:
                    tqdm.write(f"Error formatting text for {row[1]['file_name']}: {e}")
                    continue
                
                if self.bad_text(row[1]['text'], row[1]['file_name'], config['general_pairs']['bad_files']) == True:
                    continue

                
                #** Iterate through text windows **#
                for i in tqdm(range((len(row[1]['text']) - config['general_pairs']['window_size']) // config['general_pairs']['stride']), 
                              desc=f"Processing {row[1]['file_name']}", position=1, leave=False, dynamic_ncols=True, colour='yellow'):
                    
                    window = row[1]['text'][i*config['general_pairs']['stride']:(i*config['general_pairs']['stride'] + config['general_pairs']['window_size'])]

                    if len(window) < config['general_pairs']['min_text_size']:
                        continue
                    
                    #** Select random pattern **#
                    pattern_choices = random.choice(list(PATTERNS.items()))
                    pattern_name: str = pattern_choices[0]
                    pattern_list: list = pattern_choices[1]
                    pattern_choice: tuple = pattern_list[random.randint(0, len(pattern_list)-1)]
                    
                    keys: list = self.keys_in_pattern(str(pattern_choice))
                    
                    #** Select appropriate prompt **#
                    if pattern_name in config['general_pairs']['general_patterns']:
                        prompt = config['general_pairs']['general_prompt'].format(keys=keys, pattern=pattern_choice, text=window).strip()
                        contextual = False
                    elif pattern_name in config['general_pairs']['contextual_patterns']:
                        prompt = config['general_pairs']['contextual_prompt'].format(keys=keys, pattern=pattern_choice, text=window).strip()
                        contextual = True
                    else:
                        tqdm.write(f"Pattern {pattern_name} not found in general or contextual patterns.")
                        exit()

                    #** Retrieve pattern values **#
                    output = self.retriever.retrieve_multiple(
                        names=keys,
                        options=None,
                        prompt=prompt,
                        tries=3,
                        expect_result=True,
                        print_responses=False,
                    )
                    
                    #** Clean output **#
                    for k, v in output.items():
                        if isinstance(v, list):
                            v = ' '.join(v)

                        v = v.replace("\n", " ").strip()

                        # # For removing pattern fragments from output 
                        # for frag in pattern_choice:
                        #     if isinstance(frag, str):
                        #         v = v.replace(frag.strip(), "").strip()

                        output[k] = v
                    
                    for o in output.values():
                        if o == '' or o is None:
                            tqdm.write(f"Skipping due to empty output: {output}")
                            break
                    else:
                        pass
                    
                    #** Prepare text and write to file **#
                    input_txt = pattern_choice[0].format(**output).replace("\n","  ").strip()
                    output_txt = pattern_choice[1].format(**output).replace("\n","  ").strip()
                    id = str(uuid.uuid4())
                    
                    if len(input_txt) > 3 and len(output_txt) > 0:
                        with open(os.path.join(os.getcwd() + config['general_pairs']['output_dir']), 'a', encoding='utf-8', newline='') as fo:
                            writer = csv.writer(fo, quoting=csv.QUOTE_ALL)
                            writer.writerow([
                                f"A{id}",
                                row[1]['file_name'],
                                pattern_name,
                                contextual,
                                input_txt,
                                output_txt
                            ])
                    else:
                        tqdm.write(f"Skipping due to short input/output. Input length: {len(input_txt)}, Output length: {len(output_txt)}")
                        continue
            
            print(f"\nApplications DB generation complete.")


    def definitionsDB_Gen(self, config):
        
        #** 3- Definitions QA pairs **#
        f_df = pd.DataFrame(columns=['id', 'input', 'output'])

        df = pd.read_csv(os.path.join(os.getcwd() + config['source_dir'])).set_index('id')
        for row in tqdm(df.iterrows(), total=len(df), desc="Processing definitions", position=0, dynamic_ncols=True, colour='blue'):
            for _ in range(config['pairs_per_definition']):
                
                input_txt = config['input'][random.randint(0, len(config['input'])-1)].format(name=row[1]['name']).strip()
                output_txt = config['output'][random.randint(0, len(config['output'])-1)].format(name=row[1]['name'], definition=row[1]['definition']).strip()
                id = str(uuid.uuid4())

                f_df = pd.concat([f_df, pd.DataFrame([[id, input_txt, output_txt]], columns=['id', 'input', 'output'])], ignore_index=True)
        
        f_df.to_csv(os.path.join(os.getcwd() + config['output_dir']), index=False)
        print(f"\nDefinitions DB generation complete.")

    
    
    @staticmethod
    def keys_in_pattern(pattern) -> list:
        keys = re.findall(r'\{(.*?)\}', pattern)
        return keys


    @staticmethod
    def bad_text(text: str, file_name: str, bad_files: list, min_text_len: int = 200) -> bool:
        if len(text) < min_text_len:
            return True
        if file_name in bad_files:
            return True
        return False


    @staticmethod
    def append_manager(output_dir: str, csv_header: str) -> None:
        """ Manage the output file for appending new data.
        """
        
        #** Check if the output file exists, if not create it with the header **#
        if not os.path.exists(os.path.join(os.getcwd() + output_dir)):
            with open(os.path.join(os.getcwd() + output_dir), 'w', encoding='utf-8') as f:
                f.write(f"{csv_header}\n")
        
        #** If it exists, check if the header is correct **#
        else:
            with open(os.path.join(os.getcwd() + output_dir), 'r', encoding='utf-8') as f:
                if f.readline().strip().replace('"','').replace("'","") != f"{csv_header}":
                    raise ValueError("Output file format is incorrect. Please check the file.")
                
                
    @staticmethod
    def overwrite_manager(output_dir: str, csv_header: str) -> None:
        """Manage the output file for overwriting with new data.
        """
        
        #** Check if the output file exists, if so, remove it **#
        if os.path.exists(os.path.join(os.getcwd() + output_dir)):
            os.remove(os.path.join(os.getcwd() + output_dir))
        
        #** Create the output file and write the header **#
        with open(os.path.join(os.getcwd() + output_dir), 'w', encoding='utf-8') as f:
            f.write(f"{csv_header}\n")