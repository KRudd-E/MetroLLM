import json
import torch
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
import random
import re
import os


class MMLU_Evaluator:
    def __init__(self, model_wrapper, config, mmlu_subset="test"):
        self.config = config
        
        self.model = model_wrapper.get_model()
        self.tokenizer = model_wrapper.get_tokenizer()
        self.device = model_wrapper.get_device()
        self.batch_size = config['batch_size']
        
        self.dataset = load_dataset("TIGER-Lab/MMLU-Pro", split=mmlu_subset)
        self.eval_split = mmlu_subset

        self.categories = ['computer science', 'math', 'chemistry', 'engineering', 'law', 
                           'biology', 'health', 'physics', 'business', 'philosophy', 
                           'economics', 'other', 'psychology', 'history']

        #** Per-category prompts **#
        self.prompts = {c: '' for c in self.categories}
        
        for d in self.dataset:
            self.prompts[d['category']] += (                # type: ignore
                'Q: ' + d['question'] + '\n' +              # type: ignore
                self.form_options(d['options']) + '\n' +    # type: ignore
                d['cot_content'] + '\n\n'                   # type: ignore
            )
            
        # src:      https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
        # example:  https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/blob/main/run_gpt4o.py

    def evaluate(self):
        
        if self.config['run'] == False:
            print("MMLU evaluation skipped.\n"); return
        else:
            print("Evaluating on MMLU dataset.\n")

        per_category_accuracy = {c: [0, 0] for c in self.categories}
        success, fail = 0, 0
        answers = []
        
        self.model.eval()
        self.model.to(self.device)
        
        for entry in tqdm(self.dataset):
            
            #** Prepare input **#
            prefix = self.prompts[entry['category']]
            query = prefix + 'Q: ' + entry['question'] + '\n' + self.form_options(entry['options']) + '\nAnswer:'
            
            inputs = self.tokenizer(
                query, return_tensors="pt", truncation=True,
                max_length=self.config["max_length"]
            ).to(self.device)

            #** Generate output **#
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config["max_length"],
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                gen = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

            entry['solution'] = gen
            answers.append(entry)

            #** Extract prediction **#
            prediction = self.get_prediction(gen, len(entry['options']))
            if entry["answer"] == prediction:
                success += 1
                per_category_accuracy[entry['category']][0] += 1
            else:
                fail += 1
                per_category_accuracy[entry['category']][1] += 1

            print("Overall accuracy:",success / (success + fail))


        #** Raw results **#
        with open(os.path.join(self.config["output_dir"] + "MMLU_raw.json"), "w") as f:
            json.dump(answers, f, indent=2)

        #** Save & Print per-category accuracy **#
        accuracies = {k: (v[0] / (v[0] + v[1]) if (v[0] + v[1]) > 0 else 0) for k, v in per_category_accuracy.items()}
        with open(os.path.join(self.config["output_dir"] + "MMLU.json"), "w") as f:
            json.dump(accuracies, f, indent=2)
        print("\nPer-category accuracies:\n", accuracies)
        
        
    @staticmethod
    def form_options(options: list):
        #** Format multiple-choice **#
        option_str = 'Options are:\n'
        opts = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        for opt, o in zip(options, opts):
            option_str += f'({o}): {opt}\n'
        return option_str

    @staticmethod
    def get_prediction(output, num_choices):
        
        #** Look for "the answer is X" **#
        pattern = r"answer is \(?([A-J])\)?"
        match = re.search(pattern, output, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        #** Fallback: find 1st A, B, C ... **#
        for char in output:
            if char.upper() in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']:
                return char.upper()
        
        #** Fail **#
        tqdm.write("No valid answer found.")
        return random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])




class Test_Set_Evaluator:
    """ Evaluator for the post-training test set. """
    def __init__(self, model_wrapper, config, dataset):
        self.config = config
        
        self.dataset = dataset
        self.model = model_wrapper.get_model()
        self.tokenizer = model_wrapper.get_tokenizer()
        self.device = model_wrapper.get_device()
        
        self.batch_size = config['batch_size']
        
    
    def evaluate(self):
        
        if self.config['run'] == False:
            print("Test Set evaluation skipped.\n")
            return
        
        print("Dataset:", self.dataset)
        
        all_predictions = []
        self.model.eval()
        self.model.to(self.device)
        
        for i in tqdm(range(0, len(self.dataset), self.batch_size)):
            batch = self.dataset[i:i+self.batch_size]

            #** Prepare batch **#
            input_prompt = self.config['task_prompt'].format(
                task_list=self.config['task_list'],
                txt=batch['input'] #??
            )
            
            inputs = self.tokenizer(
                input_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config["max_length"]
            ).to(self.device)

            #** Generate outputs **#
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config["max_length"],
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
                #** Decode and extract predictions **#
                generated_texts = []
                for j, output in enumerate(outputs):
                    input_length = inputs['input_ids'][j].shape[0]
                    generated_part = output[input_length:]
                    generated_text = self.tokenizer.decode(generated_part, skip_special_tokens=True)
                    generated_texts.append(generated_text.strip())
                    
                all_predictions.extend(generated_texts)
        
        #** Compute metrics **#
        
        # perplexity
        
        # accuracy
        
        # F1-score
        





class Task_Evaluator:
    def __init__(self, model_wrapper, config):
        self.config = config
        
        self.model = model_wrapper.get_model()
        self.tokenizer = model_wrapper.get_tokenizer()
        self.device = model_wrapper.get_device()
        self.batch_size = config["task_eval_args"]['batch_size']
        
        self.dataset = None


    def evaluate(self):
        print("Evaluating on specific tasks.\n")
        
        
        
        
        # all_predictions = []
        # self.model.eval()
        # self.model.to(self.device)
        
        # for i in tqdm(range(0, len(self.dataset), self.batch_size)):
        #     batch = self.dataset[i:i+self.batch_size]

        #     #** Prepare batch **#
        #     input_prompt = self.config['task_prompt'].format(
        #         task_list=self.config['task_list'],
        #         txt=batch['input'] #??
        #     )
            
        #     inputs = self.tokenizer(
        #         input_prompt,
        #         return_tensors="pt",
        #         padding=True,
        #         truncation=True,
        #         max_length=self.config["max_length"]
        #     ).to(self.device)

        #     #** Generate outputs **#
        #     with torch.no_grad():
        #         outputs = self.model.generate(
        #             **inputs,
        #             max_new_tokens=self.config["max_length"],
        #             do_sample=False,
        #             pad_token_id=self.tokenizer.pad_token_id
        #         )
                
        #         #** Decode and extract predictions **#
        #         generated_texts = []
        #         for j, output in enumerate(outputs):
        #             input_length = inputs['input_ids'][j].shape[0]
        #             generated_part = output[input_length:]
        #             generated_text = self.tokenizer.decode(generated_part, skip_special_tokens=True)
        #             generated_texts.append(generated_text.strip())
                    
        #         all_predictions.extend(generated_texts)
        