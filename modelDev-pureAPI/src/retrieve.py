from src.ai import AI_Assister
from tqdm import tqdm
import ast
import re

class Retriever:
    def __init__(self, config):
        self.config = config
        self.ai_chat = AI_Assister(config).ai_chat
    
    
    # multiple inputs and respective lists
    # multiple inputs and some lists
    # multiple inputs and no list
        
    
    def retrieve_multiple(self, names: list, options: dict | None, prompt: str, print_responses: bool = False, tries: int = 2, expect_result: bool = False) -> dict:
        """
        Retrieves selected information from a case study text.

        Args:
            names (list): List of field names to retrieve.
            options (dict | None): Optional filtering options for each field.
            prompt (str): Prompt to send to the model.
            print_responses (bool): OPTIONAL - Whether to print the model responses for debugging.
            tries (int): Number of attempts to retrieve the information.
            expect_result (bool): If True, expects all fields to have non-empty values.

        Returns:
            dict: Mapping of each name to a list of retrieved values.
        """
        if options:
            assert set(names) == set(options.keys()), "Names and options keys do not match."

        vals = {name: [] for name in names}
        options_list = self.flatten_top_level_values(options) if options else {}

        for i in range(tries):  # Retry loop
            try:
                response = self.ai_chat(
                    model_source=self.config['model_source'],
                    model=self.config['model'],
                    input=prompt
                )
                if print_responses:
                    tqdm.write(f"Response {i+1} for {names}:\n{response}")
                # Attempt to extract the first full JSON/dict-like block
                match = re.search(r'\{.*?\}', response, re.DOTALL)
                if not match:
                    raise ValueError("No JSON-like dictionary found in response.")

                parsed_response = ast.literal_eval(match.group(0))
                if not isinstance(parsed_response, dict):
                    raise ValueError("Parsed content is not a dictionary.")

                for key, value in parsed_response.items():
                    if key not in names:
                        continue

                    # Normalize to list of strings
                    if isinstance(value, list):
                        result = [str(v).strip() for v in value]
                    elif value is None:
                        result = []
                    else:
                        result = [str(value).strip()]

                    # Apply filtering if options exist
                    if options_list.get(key):
                        vals[key] = [v for v in result if v in options_list[key]]
                    else:
                        vals[key] = result

                if expect_result and all(vals[name] for name in names):
                    return vals
                elif not expect_result:
                    return vals

            except Exception as e:
                tqdm.write(f"Error retrieving values for {names}: {e}\n{response}")

        return vals  # fallback
                
    
    
    @staticmethod
    def flatten_top_level_values(data: dict) -> dict:
        """Flattens the top-level values of a dictionary, collecting all values from nested dictionaries and lists.
        
        Maintains the top-level keys and collects all values beneath each key into a single list,
        regardless of how deeply nested the structure is or whether values are single items or lists.

        Args:
            data (dict): The input dictionary with potentially nested structures.

        Returns:
            dict: A new dictionary with top-level keys and all nested values collected into lists.
        """
        def collect_all_values(val):
            """Recursively collect all values from nested structures."""
            collected = []
            
            if isinstance(val, dict):
                # If it's a dictionary, recursively collect values from all nested items
                for nested_val in val.values():
                    collected.extend(collect_all_values(nested_val))
            elif isinstance(val, list):
                # If it's a list, recursively collect values from each item
                for item in val:
                    collected.extend(collect_all_values(item))
            else:
                # If it's a single value, add it to the collection
                collected.append(val)
            
            return collected

        result = {}
        for top_key, nested_structure in data.items():
            # Collect all values under this top-level key
            result[top_key] = collect_all_values(nested_structure)
        return result




    # def retrieve_two(self, config, names: tuple, prompt: str, case_study_text: str):
    #     """ ENSURE NAMES ARE ALPHABETICALLY ORDERED.
    #     """
    #     vals = []
    #     for i in range(2):
    #         response = self.ai_chat(
    #             model_source=config['model_source'],
    #             model=config['model'],
    #             input=prompt.format(txt=case_study_text)
    #         )
    #         try:
    #             response = response[response.find('{')+1:response.rfind('}')]
    #             response = response.strip().replace('"','').replace(',','').split('\n')
    #             response = sorted([x.strip() for x in response])
    #             for i in range(len(names)):
    #                 if [s for s in response if names[i] in s]:
    #                     name = response[i].split(':')[1].strip()
    #                     vals += name
            

            
    #             if name1 and name2:
    #                 return name1, name2
    #         except Exception:
    #             pass
    #     return name1, name2
    
    # def retrieve_level_name(self, config, case_study_text) -> tuple:
    #     input_text = config['level_name_prompt'].format(
    #         txt=case_study_text
    #     )
    #     name, level = 'Unknown', 'Unknown'
    #     # Loop to ensure we get a valid response
    #     for i in range(2):
    #         response = self.ai_chat(
    #             model_source=config['model_source'],
    #             model=config['model'],
    #             input=input_text
    #         )
    #         # Chops anything before and including the first '{', and after and including the last '}'.
    #         response = response[response.find('{')+1:response.rfind('}')]

    #         try:
    #             response = response[response.find('{')+1:response.rfind('}')]
    #             response = response.strip().replace('"','').replace(',','').split('\n')
    #             response = sorted([x.strip() for x in response])

    #             if [s for s in response if 'level' in s]:
    #                 level = response[1].split(':')[1].strip()
                
    #             if [s for s in response if 'name' in s]:
    #                 name = response[0].split(':')[1].strip()
                
    #             if level and name:
    #                 return level, name
    #         except Exception:
    #             pass
    #     return level, name
    
    
    # def retrieve_location_date(self, config, case_study_text) -> tuple:
    #     input_text = config['location_date_prompt'].format(
    #         txt=case_study_text
    #     )
    #     date, location = 'Unknown', 'Unknown'
    #     for i in range(2):
    #         response = self.ai_chat(
    #             model_source=config['model_source'],
    #             model=config['model'],
    #             input=input_text
    #         )
    #         try:
    #             response = response[response.find('{')+1:response.rfind('}')]
    #             response = response.strip().replace('"','').replace(',','').split('\n')
    #             response = sorted([x.strip() for x in response])
                
    #             if [s for s in response if 'country' in s]:
    #                 location = response[0].split(':')[1].strip()
                
    #             if [s for s in response if 'date' in s]:
    #                 date = response[1].split(':')[1].strip()
                
    #             if location and date:
    #                 return location, date
    #         except Exception:
    #             pass
    #     return date, location

        
    # def retrieve_sector(self, config, case_study_text):
        
    #     input_text = config['sector_prompt'].format(
    #         sector_list=self.config['sector_list_v2'],
    #         txt=case_study_text
    #     )
    #     for i in range(2):
    #         response = self.ai_chat(
    #             model_source=config['model_source'],
    #             model=config['model'],
    #             input=input_text
    #         )
    #         sector: list | None = self.find_in_list(response, config['sector_list_v2'])
    #         if sector:
    #             return sector
    #     return 'Unknown'
    

    
    
    # def retrieve_tasks(self, config, case_study_text):
        
    #     input_text = config['tasks_prompt'].format(
    #         task_list=self.config['task_list'],
    #         txt=case_study_text
    #     )
    #     for i in range(2):
    #         response = self.ai_chat(
    #             model_source=config['model_source'],
    #             model=config['model'],
    #             input=input_text
    #         )
    #         tasks: list | None = self.find_in_list(response, config['task_list'])
    #         if tasks:
    #             if len(tasks) == 1:
    #                 task_1 = tasks[0]
    #                 task_2 = "Unknown"
    #             elif len(tasks) == 2:
    #                 task_1 = tasks[0]
    #                 task_2 = tasks[1]

    #             return task_1, task_2
    #     return 'Unknown', 'Unknown'


    
    # def retrieve_object_keywords(self):
    #     pass

    # def retrieve_measurement_metrics(self):
    #     # measuremment extend & measurement tolerance
    #     pass
        
    # def retrieve_system_surface_interaction(self):
    #     pass
    
    # def retrieve_measurement_object_properties(self):
    #     pass
    
    # def retrieve_task_opertation(self):
    #     pass
    
    # def retrieve_environment(self):
    #     pass
    
    # def retrieve_tools_methods(self):
    #     pass
    
    # def retreieve_application_info(self):
    #     pass
        
    
    # @staticmethod
    # def find_in_list(output: str, item_list: list) -> Optional[list]:
    #     """Manages the identification of items from the output."""
    #     identified_items = []
    #     for item in item_list:
    #         if type(item) is dict:
    #             for key, value in item.items():
    #                 if type(value) is list:
    #                     for i in value:
    #                         if i in output:
    #                             identified_items.append(i)
    #                 elif type(value) is str:
    #                     if value in output:
    #                         identified_items.append(value)
                
    #         elif type(item) is str:
    #             if item in output:
    #                 identified_items.append(item)
        
    #     return identified_items if identified_items else None
    
    
    
        
    # # def get_tasks(self, output: str, tasks: list) -> list:
    # #     """Extracts tasks from the AI model output.
    # #     Defintely not the best way to do this, but it works for now.
    # #     """
    # #     output_tasks = []
    # #     for task_group in tasks:
    # #         for key, tasks in task_group.items():
    # #             for task in tasks:
    # #                 if task in output:
    # #                     output_tasks.append(task)
    # #     return output_tasks
    
    
    # # def get_sector(self, output: str, sectors: list) -> str | None:
    # #     """Extracts sector from the AI model output.
    # #     """
    # #     tqdm.write(f"Sectors shape: {len(sectors)} - {type(sectors)}")
    # #     for sector in sectors:
    # #         #tqdm.write(f"Sector: {sector} - {type(sector)}")
    # #         time.sleep(0.1)
    # #         if sector in output:
    # #             tqdm.write(f"Sector: {sector} - {type(sector)}")
    # #             return str(sector)
    # #     return None
    
    
        
    
    # # def retrieve_multiple(self, names: list, prompt: str, potential_vals: list) -> list:
    # #     """
    # #     Requires the names to be in alphabetical order.
    # #     Requires the prompt to contain
    # #     """
    # #     vals = []
    # #     for i in range(2):
    # #         response = self.ai_chat(
    # #             model_source=self.config['model_source'],
    # #             model=self.config['model'],
    # #             input=prompt
    # #         )
    # #         try:
    # #             response = response[response.find('{')+1:response.rfind('}')]
    # #             response = response.strip().replace('"','').replace(',','').split('\n')
    # #             response = sorted([x.strip() for x in response])
                
    # #             for i in range(len(names)):
    # #                 if [s for s in response if names[i] in s]:
    # #                     name = response[i].split(':')[1].strip()
    # #                     vals.append(name)
                
    # #             if vals:
    # #                 return vals
    # #         except Exception:
    # #             pass
                
    # #     return 
                
        