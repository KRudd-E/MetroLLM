""" 
For testing the model against the test dataset.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer

class Tester:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.output_dir, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(config.output_dir, trust_remote_code=True)

    def run_tests(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config.device)
        self.model.to(self.config.device)
        output = self.model.generate(**inputs, max_length=512)
        decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
        print("Generated:", decoded)