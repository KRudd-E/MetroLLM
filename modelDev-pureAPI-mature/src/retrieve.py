from src.ai import AI_Assister
from tqdm import tqdm
import ast
import re

class Retriever:
    def __init__(self, config):
        self.config = config
        self.ai_chat = AI_Assister(config).ai_chat
    
    
    def retrieve_multiple(self, names: list, options: dict | None, prompt: str, print_responses: bool = False, tries: int = 2, expect_result: bool = False) -> dict:
        """
        Retrieves selected information from a case study text.

        Args:
            names (list): List of field names to retrieve.
            options (dict | None): Optional filtering options for each field.
            prompt (str): Prompt to send to the model.
            print_responses (bool): Whether to print the model responses for debugging.
            tries (int): Number of attempts to retrieve the information.
            expect_result (bool): If True, expects all fields to have non-empty values.
            model_source (str): Model source, either 'OpenAI' or 'ollama'.
            model (str): Model name.

        Returns:
            dict: Mapping of each name to a list of retrieved values.
        """
        
        if options:
            assert set(names) == set(options.keys()), "Names and options keys do not match."

        vals = {name: [] for name in names}
        options_list = self.flatten_top_level_values(options) if options else {}

        for i in range(tries):
            try:
                response = self.ai_chat(
                    model_source=self.config['model_source'],
                    model=self.config['model'],
                    input=prompt
                )      
                          
                if print_responses:
                    tqdm.write(f"Response {i+1} for {names}:\n{response}")
               
                #** Extract JSON-like dictionary from response **#
                match = re.search(r'\{.*?\}', response, re.DOTALL)
                if not match:
                    raise ValueError("No JSON-like dictionary found in response.")

                parsed_response = ast.literal_eval(match.group(0))
                if not isinstance(parsed_response, dict):
                    raise ValueError("Parsed content is not a dictionary.")

                
                for key, value in parsed_response.items():
                    if key not in names:
                        continue

                    #** Normalize value to a list **#
                    if isinstance(value, list):
                        result = [str(v).strip() for v in value]
                    elif value is None:
                        result = []
                    else:
                        result = [str(value).strip()]

                    #** Apply options filtering if provided **#
                    if options_list.get(key):
                        vals[key] = [v for v in result if v in options_list[key]]
                    else:
                        vals[key] = result

                if expect_result and all(vals[name] for name in names):
                    return vals
                elif not expect_result:
                    return vals

            except Exception as e:
                tqdm.write(f"Error retrieving values for {names}: {e}")

        # fallback
        return vals
    
    
    
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
                #** If dict, recurse into values **#
                for nested_val in val.values():
                    collected.extend(collect_all_values(nested_val))
            
            elif isinstance(val, list):
                #** If list, recurse into each item **#
                for item in val:
                    collected.extend(collect_all_values(item))
            
            else:
                #** If single value, add to list **#
                collected.append(val)
            
            return collected

        #** Collect values **#
        result = {}
        for top_key, values in data.items():
            result[top_key] = collect_all_values(values)
        return result
