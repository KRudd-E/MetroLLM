import json
import torch
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
import random
import re
import os
import pandas as pd


class MMLU_Evaluator:
    def __init__(self, model_wrapper, config, mmlu_subset="test"):
        self.config = config
        
        self.model = model_wrapper.get_model()
        self.tokenizer = model_wrapper.get_tokenizer()
        self.device = model_wrapper.get_device()
        self.batch_size = config['batch_size']
        
        self.full_dataset = load_dataset("TIGER-Lab/MMLU-Pro")
        self.dataset = self.full_dataset[mmlu_subset]
        self.eval_split = mmlu_subset

        self.categories = ['computer science', 'math', 'chemistry', 'engineering', 'law', 
                           'biology', 'health', 'physics', 'business', 'philosophy', 
                           'economics', 'other', 'psychology', 'history']

        #** Per-category prompts **#
        self.prompts = {c: '' for c in self.categories}
        
        val_split = self.full_dataset["validation"]
        self.prompts = {c: '' for c in self.categories}
        for d in val_split:
            self.prompts[d['category']] += (                # type: ignore
                'Q: ' + d['question'] + '\n' +              # type: ignore
                self.form_options(d['options']) + '\n' +    # type: ignore
                d['cot_content'] + '\n\n'                   # type: ignore
            )
            
        # src:      https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
        # example:  https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/blob/main/run_gpt4o.py


    def evaluate(self):
        if not self.config["run"]:
            print("MMLU evaluation skipped.\n")
            return

        print(f"Evaluating on MMLU {self.eval_split} split with {len(self.dataset)} samples.\n")    # type: ignore

        per_category_accuracy = {c: [0, 0] for c in self.categories}
        success, fail = 0, 0
        answers = []
        batch_prompts = []

        self.model.eval()

        self.model.to(self.device)

        for i in tqdm(range(0, len(self.dataset), self.batch_size)):  # type: ignore

            # ** Batch prep **
            batch = self.dataset.select(range(i, min(i + self.batch_size, len(self.dataset))))  # type: ignore
            batch_entries = [dict(row) for row in batch]
            
            for entry in batch_entries:
                prefix = self.prompts[entry['category']]
                query = prefix + 'Q: ' + entry['question'] + '\n' + self.form_options(entry['options']) + '\nAnswer:'
                batch_prompts.append(query)

            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config["max_length"]
            ).to(self.device)

            # ** Generate outputs **
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config["max_new_tokens"],
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            # ** Decode batch outputs **
            gen_texts = self.tokenizer.batch_decode(
                outputs[:, inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            # ** Process predictions **
            for entry, gen in zip(batch_entries, gen_texts):
                gen = gen.strip()
                entry['solution'] = gen
                answers.append(entry)

                prediction = self.get_prediction(gen, len(entry['options']))
                if entry["answer"] == prediction:
                    success += 1
                    per_category_accuracy[entry['category']][0] += 1
                else:
                    fail += 1
                    per_category_accuracy[entry['category']][1] += 1

            print("Overall accuracy:", success / (success + fail))

        # ** Save results **
        with open(os.path.join(self.config["output_dir"], "MMLU_raw.json"), "w") as f:
            json.dump(answers, f, indent=2)

        accuracies = {k: (v[0] / (v[0] + v[1]) if (v[0] + v[1]) > 0 else 0) for k, v in per_category_accuracy.items()}
        with open(os.path.join(self.config["output_dir"], "MMLU.json"), "w") as f:
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
        
        #** Standardize **#
        out = output.strip().upper()

        #** Look for patterns **#
        patterns = [
            r"ANSWER:? ?\(?([A-J])\)?",
            r"OPTION:? ?\(?([A-J])\)?",
            r"THE CORRECT ANSWER (?:IS|:) ?\(?([A-J])\)?",
            r"\(([A-J])\)\s*(?:IS CORRECT|IS THE ANSWER)",
        ]
        for pat in patterns:
            match = re.search(pat, out, flags=re.IGNORECASE)
            if match:
                return match.group(1).upper()

        #** Fallback: look for first standalone letter **#
        for char in out:
            if char in list("ABCDEFGHIJ")[:num_choices]:
                return char

        #** Random **#
        tqdm.write(f"No valid answer found in: {output[:100]}...")
        return random.choice(list("ABCDEFGHIJ")[:num_choices])
        
    def _to_serializable(self, obj):
        # Recursively convert numpy arrays to lists for JSON serialization
        if isinstance(obj, dict):
            return {k: self._to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._to_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

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
            inputs = self.tokenizer(
                batch['input'],
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
        self.batch_size = config['batch_size']
        
        self.dataset = pd.read_csv(self.config['data_dir'])


    def evaluate(self):
        print("Evaluating on specific tasks.\n")
        
        print("Dataset:", self.dataset.head())
        print("Dataset size:", len(self.dataset))
        print("Data columns:", self.dataset.columns)
        
        for i in tqdm(range(0, len(self.dataset), self.batch_size)):
            batch = self.dataset[i:i+self.batch_size]
            
            input_prompt = self.config['task_prompt'].format(
                task_list=self.config['task_list'],
                txt=batch['Text']
            )
            
            inputs = self.tokenizer(
                input_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config["max_length"]
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config["max_length"],
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
                generated_texts = []
                for j, output in enumerate(outputs):
                    input_length = inputs['input_ids'][j].shape[0]
                    generated_part = output[input_length:]
                    generated_text = self.tokenizer.decode(generated_part, skip_special_tokens=True)
                    generated_texts.append(generated_text.strip())
                    
                print("Generated texts:", generated_texts)
            
