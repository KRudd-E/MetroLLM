import os
import random
import pandas as pd
import random
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from ollama import Client
from src.utils import t2t_app_query, t2t_def_query
from src.patterns import PATTERNS
import uuid
import json
from datetime import datetime

class Text2TextGen:
    def __init__(self, config):
        self.config = config
        if self.config['run_applicationsDB']: self.applicationsDB_Gen(config=config['applicationsDB'])
        if self.config['run_definitionsDB']: self.definitionsDB_Gen(config=config['definitionsDB'])
        if self.config['run_companiesDB']: self.companiesDB_Gen(config=config['companiesDB'])
        # NB: avoided using self.config for readability. 


    def applicationsDB_Gen(self, config):
        
        # Confirmation message
        t2t_app_query(config['beginning_subfolder'], config['output_dir'], config['method'])
        
        # Get Subfolder directories and names
        subdirs = sorted([x[0] for x in os.walk(os.getcwd() + config['source_dir'])][1:])
        subdir_names = sorted([x[1] for x in os.walk(os.getcwd() + config['source_dir']) if x[1] != []][0])
        
        # Adjust subfolder array if applicationsDB_applicationsDB_beginning_subfolder is set
        if config['beginning_subfolder']:
            try:
                sbf_idx = subdir_names.index(config['beginning_subfolder'])
                subdirs = subdirs[sbf_idx:]
                subdir_names = subdir_names[sbf_idx:]
                print(f"Beginning at subfolder: {config['beginning_subfolder']}")
            except IndexError:
                print(f"Error with beginning_subfolder: {config['beginning_subfolder']}")
                exit()
        
        # Method handling
        if config['method'] == 'overwrite': self.overwrite_manager(self, config, config['output_dir'], config['log_dir'], config['csv_header'])
        elif config['method'] == 'append': self.append_manager(self, config, config['output_dir'], config['log_dir'], config['csv_header'])
        else: self.unknown_method(config['method'])

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
                                    dynamic_ncols=True):
                # Get text
                with open(os.path.join(subfolder_dir + '/' + text_file), 'r', encoding='utf-8') as f:
                    content = f.read()
                    f.close()
                
                # Window Loop
                for i in tqdm(range((len(content) - config['window_size']) // config['stride']),
                                desc=f"Processing {text_file}",
                                position=2,
                                leave=False,
                                dynamic_ncols=True):

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
                            if  rnd_task and input_txt and output_txt and output_txt not in input_txt and input_txt not in output_txt:
                                with open(os.getcwd() + config['output_dir'], 'a', encoding='utf-8') as fo:
                                    fo.write(f"\"A{uuid.uuid4().int:0>39}\",\"{subfolder_name}\",\"{text_file}\",\"{rnd_task}\",\"{input}\",\"{output}\"\n")
                                    fo.close()
                        except ValueError: pass

                        #! Watch for cot_stream_general_input_inversion

    def definitionsDB_Gen(self, config):
        t2t_def_query(config['method'], config['output_dir'])
        
        # Method handling
        if config['method'] == 'overwrite': self.overwrite_manager(config['output_dir'], config['log_dir'], config['csv_header'])
        elif config['method'] == 'append': self.append_manager(config['output_dir'], config['log_dir'], config['csv_header'])
        else: self.unknown_method(config['method'])

        # Read definitions from CSV
        df = pd.read_csv(os.path.join(os.getcwd() + config['source_dir'])).set_index('id')
        
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

    @staticmethod
    def output_cleanup(output: str) -> str:
        """ Cleans output string by removing various unwanted characters and formatting.
        """
        if not output: return ''
        if output.startswith('['): output = output[1:]
        if output.startswith(('1.', '2.', '3.', '4.','5.','6.','7.','8.')): output = output[2:] # Remove numbering
        try:
            for _ in range(output.count('{')):
                p1, p2 = output.split('{', 1), output.split('}', 1)
                output = p1[0] + '.' + p2[1]
        except ValueError: pass
        return output.replace('<input>', '').replace('<output>', '').strip().replace('"', '""')


    @staticmethod
    def pattern_cleanup(pattern: str) -> str:
        """ Cleans a pattern string by removing newlines, quotes, and extra spaces.
        """
        return pattern.replace('\n', ' ').replace("'", '').replace('"', '').strip()


    @staticmethod
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
        self.log_updater(log_dir, config)


    @staticmethod
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
                if f.readline().strip() != f"{csv_header}":
                    raise ValueError("Output file format is incorrect. Please check the file.")
        # Log the configuration
        self.log_updater(log_dir, config)
                    

    @staticmethod
    def unknown_method(method: str) -> None:
        raise ValueError(f"Unknown method: {method}. Please use 'overwrite' or 'append'.")


    def ai_chat(self, model_source: str, model: str, input: str) -> None:
        """
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
            # Get API key from config or .env file
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
    def log_updater(log_dir: str, config: dict) -> None:
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
        # Update log with current configuration and timestamp
        dt = {datetime.now().strftime("%Y-%m-%d %H:%M:%S"): config}
        z.update(dt)
        # Write updated log to file
        with open(os.path.join(os.getcwd() + log_dir), 'w', encoding='utf-8') as f:
            json.dump(z, f, indent=4)