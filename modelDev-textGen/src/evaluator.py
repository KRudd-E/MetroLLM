import json
import torch
from tqdm import tqdm
import numpy as np
from datasets import load_dataset


class MMLU_Evaluator:
    def __init__(self, model_wrapper, config, mmlu_subset="test"):
        self.config = config
        
        self.model = model_wrapper.get_model()
        self.tokenizer = model_wrapper.get_tokenizer()
        self.device = model_wrapper.get_device()
        self.batch_size = config["eval_args"]['batch_size']
        
        self.dataset = load_dataset("TIGER-Lab/MMLU-Pro", split=mmlu_subset)

    def evaluate(self):
        print("Evaluating.\n")

        #** Get data components **#
        examples = list(self.dataset)
        prompts = [self.format_prompt(ex) for ex in examples]
        answers = [ex["answer"] for ex in examples]
        num_choices = [len(ex["choices"]) for ex in examples]

        all_predictions = []
        self.model.eval()
        self.model.to(self.device)

        for i in tqdm(range(0, len(prompts), self.batch_size)):
            
            #** Prepare batch **#
            batch_prompts = prompts[i:i+self.batch_size]
            batch_num_choices = num_choices[i:i+self.batch_size]
            
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config["mmlu_eval_args"]["max_length"]
            ).to(self.device)
            
            #** Generate outputs **#
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config["mmlu_eval_args"]["batch_size"],
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
                #** Decode and extract predictions **#
                generated_texts = []
                for j, output in enumerate(outputs):
                    input_length = inputs['input_ids'][j].shape[0]
                    generated_part = output[input_length:]
                    generated_text = self.tokenizer.decode(generated_part, skip_special_tokens=True)
                    # print(f"Generated text: {generated_text}")
                    generated_texts.append(generated_text.strip())
                    
                for j, gen in enumerate(generated_texts):
                    pred = self.extract_choice(gen, batch_num_choices[j])
                    all_predictions.append(pred)

        #** Compute accuracy **#
        correct = [int(p == a) for p, a in zip(all_predictions, answers)]
        accuracy = np.mean(correct) * 100
        results = {"accuracy": round(accuracy, 2)}
        
        #** Print and save results **#
        print(results)
        with open(self.config["mmlu_eval_args"]["output_dir"], "w") as f:
            json.dump(results, f, indent=4)


    def format_prompt(self, example):
        #** Get components of data **#
        context = example.get("context", "")
        question = example["question"]
        choices = example["choices"] # i.e., options

        #** Format for use as prompt **#
        prompt = ""

        if context:
            prompt += context.strip() + "\n"
        prompt += question.strip() + "\n"
        for idx, option in enumerate(choices):
            letter = chr(ord('A') + idx)
            prompt += f"{letter}. {option.strip()}\n"
        prompt += "Answer:"

        return prompt

    def extract_choice(self, generated_text, num_choices):
        #** Extract the first valid choice **#
        for char in generated_text.strip():
            if char.upper() in [chr(ord('A') + i) for i in range(num_choices)]:
                return char.upper() #? A, B, C...
        #** No effective response **#
        return "?"



class Task_Evaluator:
    def __init__(self, model_wrapper, dataset, config):
        self.config = config
        
        self.model = model_wrapper.get_model()
        self.tokenizer = model_wrapper.get_tokenizer()
        self.device = model_wrapper.get_device()
        self.batch_size = config["task_eval_args"]['batch_size']
        
        self.dataset = dataset


    def evaluate(self):
        print(self.dataset)
        pass