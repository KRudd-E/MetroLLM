import os
from tqdm import tqdm
from dotenv import load_dotenv
import time
from openai import OpenAI
from ollama import Client
import random

class Text2TextGen:
    def __init__(self, config):
        self.config = config

        while True:
            if not self.config['beginning_subfolder']:
                user_input = input(f"You are about to {self.config['method'].upper()} '{self.config['output_file']}'. Do you wish to continue? (y/n): ")
            else:
                user_input = input(f"You are about to {self.config['method'].upper()} '{self.config['output_file']}', beginning at subfolder '{self.config['beginning_subfolder']}'. Do you wish to continue? (y/n): ")
            if user_input in ['Y', 'y', 'N', 'n']:
                if user_input in ['N', 'n']:
                    print("Exiting the script.")
                    exit()
                break
            else:
                print("Invalid input. Please enter Y or N.")

        if self.config['debug']:
            print("Text2TextGen initialized with config:\n", self.config)


    def run(self):

        #*** ApplicationsDB ***#
        if self.config['use_applicationsDB']:
            
            # Get the list of subfolders in the data path
            subfolder_directories = sorted([x[0] for x in os.walk(os.getcwd() + self.config['data_path'])][1:])
            subfolder_names = sorted([x[1] for x in os.walk(os.getcwd() + self.config['data_path']) if x[1] != []][0])
            
            # Adjust subfolder array if beginning_subfolder is set
            if self.config['beginning_subfolder']:
                try:
                    sbf_idx = subfolder_names.index(self.config['beginning_subfolder'])
                    subfolder_directories = subfolder_directories[sbf_idx:]
                    subfolder_names = subfolder_names[sbf_idx:]
                    print(f">> Beginning at subfolder: {self.config['beginning_subfolder']}")
                except IndexError:
                    print(f"Beginning subfolder {self.config['beginning_subfolder']} is out of range. Using all subfolders.")
                    exit()
            
            if self.config['debug']:
                print(f">> Found {len(subfolder_directories)} subfolders in {self.config['data_path']}")

            # Overwrite setup
            if self.config['method'] == 'overwrite':
                if os.path.exists(self.config['output_file']):
                    os.remove(self.config['output_file'])
                
                with open(os.getcwd() + self.config['output_file'], 'w', encoding='utf-8') as f:
                    f.write(f"{self.config['csv_header']}\n")
            # Append setup
            elif self.config['method'] == 'append':
                if not os.path.exists(os.path.join(os.getcwd() + self.config['output_file'])):
                    with open(os.path.join(os.getcwd() + self.config['output_file']), 'w', encoding='utf-8') as f:
                        f.write(f"{self.config['csv_header']}\n")
                else:
                    with open(os.path.join(os.getcwd() + self.config['output_file']), 'r', encoding='utf-8') as f:
                        if f.readline().strip() != f"{self.config['csv_header']}":
                            print("Output file format is incorrect. Please check the file.")
                            exit()

            else:
                print(f"Unknown method: {self.config['method']}")
                print("Please use either 'overwrite' or 'append'.")
                exit()

            id = 1
            # get text files
            for subfolder_dir, subfolder_name in tqdm(zip(subfolder_directories, subfolder_names),
                                                      total=len(subfolder_directories),
                                                      desc="Processing subfolders",
                                                      position=0,
                                                      leave=True,
                                                      dynamic_ncols=True):
                text_files = sorted([f for f in os.listdir(subfolder_dir) if f.endswith('.txt')])
                if self.config['debug']:
                    tqdm.write(f">> Processing {len(text_files):03} text files in {subfolder_name}.")
                

                for text_file in tqdm(text_files,
                                      desc=f"Processing {subfolder_name}",
                                      total=len(text_files),
                                      position=1,
                                      leave=False,
                                      dynamic_ncols=True):

                    with open(subfolder_dir + '/' + text_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        f.close()
                        for i in tqdm(range((len(content) - self.config['window_size'])// self.config['stride']),
                                      desc=f"Processing {text_file}",
                                      position=2,
                                      leave=False,
                                      dynamic_ncols=True):

                            # Prepare text
                            window = content[i*self.config['stride']:(i*self.config['stride'] + self.config['window_size'])]
                            t1, t2, t3 = random.sample(self.config['tasks'], 3) # Three random tasks from the list
                            ai_txt = self.config['prompt'].format(
                                task1=t1,
                                task2=t2,
                                task3=t3,
                                window=window
                            )
                            
                            # Generate output
                            response = self.ai_chat(
                                source=self.config['source'],
                                model=self.config['model'],
                                input=ai_txt 
                            )

                            pairs = [line for line in response.split('\n') if "=>" in line and ":" in line]
                            for pair in pairs:
                                try:
                                    task_part, io_part = pair.split(":", 1)
                                    input_txt, output_txt = io_part.split("=>", 1)
                                    task = task_part.replace('<task>', '').replace('<input>', '').replace('<output>', '').lower().strip().replace('"', '""').replace('<', '').replace('>', '')
                                    input_txt = input_txt.replace('<task>', '').replace('<input>', '').replace('<output>', '').strip().replace('"', '""')
                                    output_txt = output_txt.replace('<task>', '').replace('<input>', '').replace('<output>', '').strip().replace('"', '""')
                                    if task and input_txt and output_txt:
                                        with open(os.getcwd() + self.config['output_file'], 'a', encoding='utf-8') as fo:
                                            fo.write(f"A{id:08},\"{subfolder_name}\",\"{text_file}\",\"{task}\",\"{input_txt}\",\"{output_txt}\"\n")
                                            fo.close()
                                    else:
                                        if self.config['debug']:
                                            tqdm.write(f"Skipping empty pair: {pair}")
                                    id += 1

                                except ValueError:
                                    if self.config['debug']:
                                        tqdm.write(f"Skipping badly formatted pair: {pair}")

                            



        #*** DefinitionsDB ***#
        if self.config['use_definitionsDB']:
            print("DefinitionsDB processing is not implemented yet.")
            pass
        
        #*** CompaniesDB ***#
        if self.config['use_companiesDB']:
            print("CompaniesDB processing is not implemented yet.")
            pass
        pass
    
    
    def ai_chat(self, source, model, input):
        """
        Args:
            source (string): Model source. Either 'ollama' or 'OpenAI'.
            model (string): Model name.
            txt (string): The text to send to the model.

        Returns:
            string: The response from the model.
        """
        if source == 'ollama':
            client = Client(host='http://localhost:11434')
            response = client.chat(
                model=model,
                messages=[{'role': 'user', 'content': f'{input}/n'}]
            )
            return response.message.content.strip()

        elif source == 'OpenAI':
            # Get API key from config or .env file
            if not self.config['api_key']:
                try:
                    load_dotenv()
                    self.config['api_key'] = os.getenv('API_KEY')
                    if not self.config['api_key']:
                        raise ValueError("API key not found in .env file.")
                        
                except Exception as e:
                    print(f"\nError loading API key: {e}")
                    exit()

            client = OpenAI(api_key=self.config['api_key'])
            completion = client.chat.completions.create(
                model=self.config['model'],
                messages=[{"role": "user",
                    "content": input,
                }])
            return completion.choices[0].message.content

        else:
            print(f"\nUnknown model source: {source}")
            exit()