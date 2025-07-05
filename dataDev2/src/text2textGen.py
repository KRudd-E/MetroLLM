import os
import random
import pandas as pd
import random
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from ollama import Client
from src.utils.utils import t2t_app_query, t2t_def_query
from src.utils.patterns import PATTERNS
import uuid
import json
from datetime import datetime

class Text2TextGen:
    def __init__(self, config, src=None):
        self.config = config
        if src.lower() == 'applications' or src.lower() == 'a': self.applicationsDB_Gen(config['applicationsDB'])
        if src.lower() == 'definitions' or src.lower() == 'd': self.definitionsDB_Gen(config['definitionsDB'])
        if src.lower() == 'companies' or src.lower() == 'c': self.companiesDB_Gen(config['companiesDB'])
        # NB: avoided using self.config for readability. 


    def applicationsDB_Gen(self, config):
        
        # Confirmation message
        t2t_app_query(config['starting_subfolder'], config['output_dir'], config['append_or_overwrite'])
        
        # Get Subfolder directories and names
        subdirs = sorted([x[0] for x in os.walk(os.getcwd() + config['source_dir'])][1:])
        subdir_names = sorted([x[1] for x in os.walk(os.getcwd() + config['source_dir']) if x[1] != []][0])
        
        # Adjust subfolder array if starting_subfolder is set
        if config['starting_subfolder']:
            subdir_names, subdirs = self.starting_subfolder_manager(config['starting_subfolder'], subdir_names, subdirs)
        
        # Append/Overwrite handling
        if config['append_or_overwrite'] == 'overwrite': self.overwrite_manager(config, config['output_dir'], config['log_dir'], config['csv_header'])
        elif config['append_or_overwrite'] == 'append': self.append_manager(config, config['output_dir'], config['log_dir'], config['csv_header'])
        else: self.unknown_method(config['append_or_overwrite'])

        # Subfolder Loop
        for subfolder_dir, subfolder_name in tqdm(zip(subdirs, subdir_names),
                                                    total=len(subdirs),
                                                    desc="Processing subfolders",
                                                    position=0,
                                                    dynamic_ncols=True,
                                                    colour='blue'):
            # Get text files
            text_files = sorted([f for f in os.listdir(subfolder_dir) if f.endswith('.txt')])
            
            # Text File Loop
            for text_file in tqdm(text_files,
                                    desc=f"Processing {subfolder_name}",
                                    total=len(text_files),
                                    position=1,
                                    leave=False,
                                    dynamic_ncols=True,
                                    colour='green'):
                # Get text
                with open(os.path.join(subfolder_dir + '/' + text_file), 'r', encoding='utf-8') as f:
                    content = f.read()
                    f.close()
                
                # Window Loop
                for i in tqdm(range((len(content) - config['window_size']) // config['stride']),
                                desc=f"Processing {text_file}",
                                position=2,
                                leave=False,
                                dynamic_ncols=True,
                                colour='yellow'):

                    # Prepare window and AI prompt
                    window = content[i*config['stride']:(i*config['stride'] + config['window_size'])]

                    rnd_task, rnd_pattern_list = random.choice(list(PATTERNS.items()))
                    p1, p2, p3 = random.sample(rnd_pattern_list, 3) # Three random patterns from the list
                    p1_in, p1_out = self.pattern_cleanup(p1[0]), self.pattern_cleanup(p1[1])
                    p2_in, p2_out = self.pattern_cleanup(p2[0]), self.pattern_cleanup(p2[1])
                    p3_in, p3_out = self.pattern_cleanup(p3[0]), self.pattern_cleanup(p3[1])

                    ai_txt = config['prompt'].format(
                        p1_in=p1_in, p1_out=p1_out,
                        p2_in=p2_in, p2_out=p2_out,
                        p3_in=p3_in, p3_out=p3_out,
                        window=window
                    )
                    
                    # Generate output
                    response = self.ai_chat(
                        model_source=config['model_source'],
                        model=config['model'],
                        input=ai_txt
                    )

                    # Process response
                    pairs = [line for line in response.split('\n') if "=>" in line]
                    for pair in pairs:
                        try: # As some reponses are poorly formatted. 
                            input_txt, output_txt = pair.split("=>", 1)
                            input, output = self.output_cleanup(input_txt), self.output_cleanup(output_txt)
                            if rnd_task and input_txt and output_txt and output_txt not in input_txt and input_txt not in output_txt:
                                with open(os.getcwd() + config['output_dir'], 'a', encoding='utf-8') as fo:
                                    fo.write(f"\"A{uuid.uuid4().int:0>39}\",\"{subfolder_name}\",\"{text_file}\",\"{rnd_task}\",\"{input}\",\"{output}\"\n")
                                    fo.close()
                        except: pass

                        #! Watch for cot_stream_general_input_inversion

    def definitionsDB_Gen(self, config):
        t2t_def_query(config['append_or_overwrite'], config['output_dir'], config['starting_definition'])
        
        # Method handling
        if config['append_or_overwrite'] == 'overwrite': self.overwrite_manager(config, config['output_dir'], config['log_dir'], config['csv_header'])
        elif config['append_or_overwrite'] == 'append': self.append_manager(config, config['output_dir'], config['log_dir'], config['csv_header'])
        else: self.unknown_method(config['append_or_overwrite'])
  
        # Read definitions from CSV
        df = pd.read_csv(os.path.join(os.getcwd() + config['source_dir'])).set_index('id')
    
        # Manage starting definition specificiation
        if config['starting_definition']:
            if config['starting_definition'] not in df['name'].values:
                raise ValueError(f"Starting definition '{config['starting_definition']}' not found in definitionsDB.")
            df = df.loc[config['starting_definition']:]

        # Iterate over definition columns
        for row in tqdm(df.iterrows(), total=len(df), desc="Processing definitions", position=0, dynamic_ncols=True, colour='blue'):
            for _ in range(config['pairs_per_definition']):
                
                ai_txt = config['prompt'].format(
                    name=row[1]['name'],
                    definition=row[1]['definition']
                )
                
                response = self.ai_chat(
                    model_source=config['model_source'],
                    model=config['model'],
                    input=ai_txt
                )
                pairs = [line for line in response.split('\n') if "=>" in line]
                # Process response
                for pair in pairs:
                    try:
                        input_txt, output_txt = response.split("=>", 1)
                        input, output = self.output_cleanup(input_txt), self.output_cleanup(output_txt)
                        if input and output and output not in input and input not in output:
                            with open(os.getcwd() + config['output_dir'], 'a', encoding='utf-8') as fo:
                                fo.write(f"\"D{uuid.uuid4().int:0>39}\",\"define\",\"{row[1]['name']}\",\"{input}\",\"{output}\"\n")
                                fo.close()
                    except: 
                        tqdm.write(f"Error processing response for '{row[1]['name']}': {response}")
                        pass
                
        
        # Reformat
        df = pd.DataFrame({
            "id": ["D" + str(idx) for idx in df.index],
            "task": "definition",
            "input": ["Define: " + name for name in df["name"]],
            "output": [definition for definition in df["definition"]]
        }).set_index('id')

        # Write to CSV
        df.to_csv(os.path.join(os.getcwd() + config['output_dir']), 
                  index=True, encoding='utf-8', quotechar='"', quoting=1)


    def companiesDB_Gen(self, config):
        print("CompaniesDB processing is not implemented yet.")
        pass




    def get_API_key(self) -> None:
        """Get API key from config or .env file. Writes to self.config['api_key'].
        """
        if not self.config['api_key']:
            try:
                load_dotenv()
                self.config['api_key'] = os.getenv('API_KEY')
                if not self.config['api_key']:
                    raise ValueError("API key not found in .env file.")
                    
            except Exception as e:
                print(f"\nError loading API key: {e}")
                exit()
                
    def ai_chat(self, model_source: str, model: str, input: str) -> None:
        """ Sends a chat request to the specified model and returns the response.
        Args:
            model_source (string): Model source. Either 'ollama' or 'OpenAI'.
            model (string): Model name.
            txt (string): The text to send to the model.

        Returns:
            string: The response from the model.
        """
        if model_source == 'ollama':
            client = Client(host='http://localhost:11434')
            response = client.chat(
                model=model,
                messages=[{'role': 'user', 'content': f'{input}/n'}]
            )
            return response.message.content.strip()

        elif model_source == 'OpenAI':
            self.get_API_key()
            client = OpenAI(api_key=self.config['api_key'])
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user",
                    "content": input,
                }])
            return completion.choices[0].message.content

        else:
            print(f"\nUnknown model source: {model_source}")
            exit()


    @staticmethod
    def starting_subfolder_manager(starting_subfolder: str, subdir_names: list, subdirs: list) -> tuple:
        """ Adjusts subdirs and subdir_names to start from the specified starting_subfolder.
        """
        if starting_subfolder not in subdir_names:
            raise ValueError(f"Starting subfolder '{starting_subfolder}' not found in subdir_names\n Check config.yaml and data.")
        sbf_idx = subdir_names.index(starting_subfolder)
        subdirs = subdirs[sbf_idx:]
        subdir_names = subdir_names[sbf_idx:]    
        return subdir_names, subdirs

    @staticmethod
    def output_cleanup(output: str) -> str:
        """ Cleans output string by removing various unwanted characters and formatting.
        """
        if not output: return ''
        if output.startswith('['): output = output[1:]
        if output.endswith(']'): output = output[:-1]
        if output.startswith('<'): output = output[1:]
        if output.endswith('>'): output = output[:-1]
        if output.startswith(('1.', '2.', '3.', '4.','5.','6.','7.','8.')): output = output[2:] # Remove numbering
        for _ in range(output.count('{')):
            try:
                
                p1, p2 = output.split('{', 1), output.split('}', 1)
                output = p1[0] + '.' + p2[1]
            except: pass
        return output.replace('<input>', '').replace('<output>', '').strip().replace('"', '""')


    @staticmethod
    def pattern_cleanup(pattern: str) -> str:
        """ Cleans a pattern string by removing newlines, quotes, and extra spaces.
        """
        return pattern.replace('\n', ' ').replace("'", '').replace('"', '').strip()


    def overwrite_manager(self, config: dict, output_dir: str, log_dir: str, csv_header: str) -> None:
        """Overwrite the output file if it exists and write the CSV header.

        Args:
            output_file (str): The path to the output file.
            csv_header (str): The CSV header to write.
        """
        # Check if the output file exists, if so, remove it
        if os.path.exists(os.path.join(os.getcwd() + output_dir)):
            os.remove(os.path.join(os.getcwd() + output_dir))
        # Create the output file and write the header
        with open(os.path.join(os.getcwd() + output_dir), 'w', encoding='utf-8') as f:
            f.write(f"{csv_header}\n")
        # Log the configuration
        self.log_updater(self, log_dir, output_dir, config)



    def append_manager(self, config: dict, output_dir: str, log_dir: str, csv_header: str) -> None:
        """Append to the output file if it exists and write the CSV header if not.

        Args:
            output_file (str): The path to the output file.
            csv_header (str): The CSV header to write.

        Returns:
            int: The next available ID for new entries.
        """
        # Check if the output file exists, if not create it with the header
        if not os.path.exists(os.path.join(os.getcwd() + output_dir)):
            with open(os.path.join(os.getcwd() + output_dir), 'w', encoding='utf-8') as f:
                f.write(f"{csv_header}\n")
        #  If it exists, check if the header is correct
        else:
            with open(os.path.join(os.getcwd() + output_dir), 'r', encoding='utf-8') as f:
                if f.readline().strip().replace('"','').replace("'","") != f"{csv_header}":
                    raise ValueError("Output file format is incorrect. Please check the file.")
        # Log the configuration
        self.log_updater(self, log_dir, output_dir, config)
                    

    @staticmethod
    def unknown_method(method: str) -> None:
        raise ValueError(f"Unknown method: {method}. Please use 'overwrite' or 'append'.")
    
    
    @staticmethod
    def log_updater(self, log_dir: str, output_dir: str, config: dict) -> None:
        """Updates the log file with the current configuration and timestamp.
        """
        # Get existing log data or create new 
        if os.path.exists(os.path.join(os.getcwd() + log_dir)):
            with open(os.path.join(os.getcwd() + log_dir), 'r', encoding='utf-8') as f:
                try:
                    z = json.load(f)
                except json.JSONDecodeError:
                    z = {}
        else:
            z = {}
        
        row_count = self.get_file_row_count(output_dir)
        # Update log with starting line, timestamp, and config
        dt = {f'linestart~{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}': row_count,
              f'config~{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}': config}
        z.update(dt)

        # Write updated log to file
        with open(os.path.join(os.getcwd() + log_dir), 'w', encoding='utf-8') as f:
            json.dump(z, f, indent=4)
    
    @staticmethod
    def get_file_row_count(file_path: str) -> int:
        """Returns the number of rows in a file."""
        if not os.path.exists(os.path.join(os.getcwd() + file_path)):
            return 0
        with open(os.path.join(os.getcwd() + file_path), 'r', encoding='utf-8') as f:
            return sum(1 for line in f)