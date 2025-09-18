from dotenv import load_dotenv
import os
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from ollama import Client
import time

class AI_Assister:
    def __init__(self, config):
        self.config = config
        self.load_environment_variables()
    
    
    def load_environment_variables(self):
        """ Loads environment variables from a .env file. """
        load_dotenv()
        self.config['api_key'] = os.getenv('API_KEY')


    def ai_chat(self, model_source: str, model: str, input: str) -> str:
        """ Sends a chat request to the specified model and returns the response.
        Args:
            model_source (string): Model source. Either 'ollama' or 'OpenAI'.
            model (string): Model name.
            input (string): The text to send to the model.

        Returns:
            string: The response from the model.
        """
        
        #** OpenAI **#
        if model_source == 'OpenAI':
            client = OpenAI(api_key=self.config['api_key'])

            for i in range(5):
                try:
                    completion = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": input}]
                    )
                    break
                except RateLimitError:
                    time.sleep(0.4 ** i)
            else:
                exit(f"\nFailed to get completion after retries from OpenAI API for model {model}.")

            if not completion.choices:
                exit(f"\nNo choices returned from OpenAI API for model {model}.")

            return str(completion.choices[0].message.content)
        
        #** Ollama **#
        elif model_source == 'ollama':
            client = Client(host='http://localhost:11434')
            response = client.chat(
                model=model,
                messages=[{'role': 'user', 'content': f'{input}/n'}]
            )
            return str(response.message.content).strip()

        
        else:
            print(f"\nUnknown model source: {model_source}")
            exit()
