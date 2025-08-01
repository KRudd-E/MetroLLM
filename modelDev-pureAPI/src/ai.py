from dotenv import load_dotenv
import os
from dotenv import load_dotenv
from openai import OpenAI
from ollama import Client

class AI_Assister:
    def __init__(self, config):
        self.config = config
        self.load_environment_variables()
        self.get_API_key()
    
    
    def load_environment_variables(self):
        """ Loads environment variables from a .env file. """
        load_dotenv()
        self.config['api_key'] = os.getenv('API_KEY')
        
    
    def ai_chat(self, model_source: str, model: str, input: str) -> str:
        """ Sends a chat request to the specified model and returns the response.
        Args:
            model_source (string): Model source. Either 'ollama' or 'OpenAI'.
            model (string): Model name.
            txt (string): The text to send to the model.

        Returns:
            string: The response from the model.
        """
        if model_source == 'OpenAI':
            client = OpenAI(api_key=self.config['api_key'])
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user",
                    "content": input,
                }])
            return str(completion.choices[0].message.content)
        
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