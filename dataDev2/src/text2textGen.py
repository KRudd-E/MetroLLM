import os
import random
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from ollama import Client
from src.utils import t2t_app_query, t2t_def_query

class Text2TextGen:
    def __init__(self, config):
        self.config = config

        if self.config['debug']:
            print("Text2TextGen initialized with config:\n", self.config)


    def run(self):

        #*** ApplicationsDB ***#
        if self.config['run_applicationsDB']:
            t2t_app_query(self)
            
            # Get the list of subfolders in the data path
            subfolder_directories = sorted([x[0] for x in os.walk(os.getcwd() + self.config['applicationsDB']['source'])][1:])
            subfolder_names = sorted([x[1] for x in os.walk(os.getcwd() + self.config['applicationsDB']['source']) if x[1] != []][0])
            
            # Adjust subfolder array if applicationsDB_applicationsDB_beginning_subfolder is set
            if self.config['applicationsDB']['beginning_subfolder']:
                try:
                    sbf_idx = subfolder_names.index(self.config['applicationsDB']['beginning_subfolder'])
                    subfolder_directories = subfolder_directories[sbf_idx:]
                    subfolder_names = subfolder_names[sbf_idx:]
                    print(f">> Beginning at subfolder: {self.config['applicationsDB']['beginning_subfolder']}")
                except IndexError:
                    print(f">> Beginning subfolder {self.config['applicationsDB']['beginning_subfolder']} is bad. Using all subfolders.")
                    exit()
            
            if self.config['debug']:
                print(f">> Found {len(subfolder_directories)} subfolders in {self.config['applicationsDB']['source']}")

            # Overwrite setup
            if self.config['applicationsDB']['method'] == 'overwrite':
                self.overwrite_manager(self.config['applicationsDB']['output'], self.config['applicationsDB']['csv_header'])
            elif self.config['applicationsDB']['method'] == 'append':
                id = self.append_manager(self.config['applicationsDB']['output'], self.config['applicationsDB']['csv_header'])
            else:
                self.unknown_method(self.config['applicationsDB']['method'])

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
                        for i in tqdm(range((len(content) - self.config['applicationsDB']['window_size'])// self.config['applicationsDB']['stride']),
                                      desc=f"Processing {text_file}",
                                      position=2,
                                      leave=False,
                                      dynamic_ncols=True):

                            # Prepare text
                            window = content[i*self.config['applicationsDB']['stride']:(i*self.config['applicationsDB']['stride'] + self.config['applicationsDB']['window_size'])]
                            t1, t2, t3 = random.sample(self.config['applicationsDB']['tasks'], 3) # Three random tasks from the list
                            ai_txt = self.config['applicationsDB']['prompt'].format(
                                task1=t1,
                                task2=t2,
                                task3=t3,
                                window=window
                            )
                            
                            # Generate output
                            response = self.ai_chat(
                                model_source=self.config['applicationsDB']['model_source'],
                                model=self.config['applicationsDB']['model'],
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
                                        with open(os.getcwd() + self.config['applicationsDB']['output'], 'a', encoding='utf-8') as fo:
                                            fo.write(f"\"A{id:08}\",\"{subfolder_name}\",\"{text_file}\",\"{task}\",\"{input_txt}\",\"{output_txt}\"\n")
                                            fo.close()
                                    else:
                                        if self.config['debug']:
                                            tqdm.write(f"Skipping empty pair: {pair}")
                                    id += 1

                                except ValueError:
                                    if self.config['debug']:
                                        tqdm.write(f"Skipping badly formatted pair: {pair}")


        #*** DefinitionsDB ***#
        if self.config['run_definitionsDB']:
            t2t_def_query(self)
            
            # Method handling
            if self.config['definitionsDB']['method'] ==  'overwrite':
                self.overwrite_manager(self.config['definitionsDB']['output'], self.config['definitionsDB']['csv_header'])
            elif self.config['definitionsDB']['method'] == 'append':
                id = self.append_manager(self.config['definitionsDB']['output'], self.config['definitionsDB']['csv_header'])
            else:
                self.unknown_method(self.config['definitionsDB']['method'])

            df = pd.read_csv(os.path.join(os.getcwd() + self.config['definitionsDB']['source'])).set_index('id')
            
            df = pd.DataFrame({
                "id": ["D" + str(idx) for idx in df.index],
                "task": "define",
                "input": ["define " + name for name in df["name"]],
                "output": [definition for definition in df["definition"]]
            }).set_index('id')

            df.to_csv(os.path.join(os.getcwd() + self.config['definitionsDB']['output']), index=True, encoding='utf-8', quotechar='"', quoting=1)

        
        #*** CompaniesDB ***#
        if self.config['run_companiesDB']:
            print("CompaniesDB processing is not implemented yet.")
            pass
        pass
    

    
    def overwrite_manager(self, output_file: str, csv_header: str) -> None:
        """Overwrite the output file if it exists and write the CSV header.

        Args:
            output_file (str): The path to the output file.
            csv_header (str): The CSV header to write.
        """
        # Check if the output file exists, if so, remove it
        if os.path.exists(os.path.join(os.getcwd() + output_file)):
            os.remove(os.path.join(os.getcwd() + output_file))
        # Create the output file and write the header
        with open(os.path.join(os.getcwd() + output_file), 'w', encoding='utf-8') as f:
            f.write(f"{csv_header}\n")

    def append_manager(self, output_file: str, csv_header: str) -> int:
        """Append to the output file if it exists and write the CSV header if not.

        Args:
            output_file (str): The path to the output file.
            csv_header (str): The CSV header to write.

        Returns:
            int: The next available ID for new entries.
        """
        # Check if the output file exists, if not create it with the header
        if not os.path.exists(os.path.join(os.getcwd() + output_file)):
            with open(os.path.join(os.getcwd() + output_file), 'w', encoding='utf-8') as f:
                f.write(f"{csv_header}\n")
        #  If it exists, check if the header is correct
        else:
            with open(os.path.join(os.getcwd() + output_file), 'r', encoding='utf-8') as f:
                if f.readline().strip() != f"{csv_header}":
                    print("Output file format is incorrect. Please check the file.")
                    exit()
        # Get the next id according to the file, or set to 1
        with open(os.path.join(os.getcwd() + output_file), 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if lines:
                last_line = lines[-1]
                id = int(last_line.split(',')[0][1:]) + 1
            else:
                id = 1
        return id

    def unknown_method(method: str) -> None:
        """Handle unknown method types.

        Args:
            method (str): The unknown method type.
        """
        print(f"Unknown method: {method}")
        print("Please use either 'overwrite' or 'append'.")
        exit()


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
                model=model,
                messages=[{"role": "user",
                    "content": input,
                }])
            return completion.choices[0].message.content

        else:
            print(f"\nUnknown model source: {model_source}")
            exit()