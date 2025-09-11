import json, torch, os, re, random
from tqdm import tqdm
import pandas as pd
import numpy as np
from datasets import load_dataset
from src.utils.retrieve import Retriever
import gc

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


        #** Clip dataset **#
        clip_percentage = config['data_reduction']  # Default to 100% if not provided
        if clip_percentage < 1.0:
            clipped_data = []
            for category in self.categories:
                category_data = [entry for entry in self.dataset if entry['category'] == category] # type: ignore
                clip_size = min(len(category_data), max(1, int(len(category_data) * clip_percentage)))
                clipped_data.extend(random.sample(category_data, clip_size))
                self.dataset = clipped_data
                
        # src:      https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
        # example:  https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/blob/main/run_gpt4o.py


    def evaluate(self):
        if not self.config["run"]:
            print("\nMMLU evaluation skipped.\n")
            return

        print(f"Evaluating on MMLU {self.eval_split} split with {len(self.dataset)} samples.\n")    # type: ignore
        print(f"Max new tokens: {self.config['max_new_tokens']}\n")

        per_category_accuracy = {c: [0, 0] for c in self.categories}
        success, fail = 0, 0
        answers = []

        self.model.eval()
        self.model.to(self.device)

        for i in tqdm(range(0, len(self.dataset), self.batch_size)):  # type: ignore

            #** Batch prep **#
            batch = self.dataset[i:min(i + self.batch_size, len(self.dataset))]  # type: ignore
            batch_entries = [dict(row) for row in batch]

            batch_prompts = []  # Reset for each batch to avoid memory buildup
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

            #** Generate outputs **#
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=self.config["max_new_tokens"],
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            #** Decode batch outputs **#
            gen_texts = self.tokenizer.batch_decode(
                outputs[:, inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            #** Process predictions **#
            for entry, gen in zip(batch_entries, gen_texts):
                
                if self.config['print_generated']:
                    prnt = gen.replace('\n', ' ')
                    tqdm.write(f"Generated text: {prnt}")
                
                gen = gen.strip()
                entry['generated_text'] = gen
                entry['solution'] = gen
                
                prediction = self.get_prediction(gen, len(entry['options']))
                entry['predicted_answer'] = prediction
                entry['correct_answer'] = entry["answer"]
                entry['is_correct'] = (entry["answer"] == prediction)
                
                answers.append(entry)

                if entry["answer"] == prediction:
                    success += 1
                    per_category_accuracy[entry['category']][0] += 1
                else:
                    fail += 1
                    per_category_accuracy[entry['category']][1] += 1

            print("Overall accuracy:", success / (success + fail))
            torch.cuda.empty_cache()
            del batch_prompts, batch_entries, inputs, outputs, gen_texts 
            gc.collect()

        #** Save results **#
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
    def get_prediction(output: str, num_choices: int) -> str:
        
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

        #** Fallback: look for LAST standalone letter **#
        last_match = None
        for match in re.finditer(r'\b([A-J])\b', out):
            if match.group(1).upper() in list("ABCDEFGHIJ")[:num_choices]:
                last_match = match.group(1).upper()
        if last_match:
            return last_match

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
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=self.config["max_new_tokens"],
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
        
        self.dataset = pd.read_csv(self.config['data_dir'])


    def evaluate(self):
        
        if self.config['run'] == False:
            print("Task evaluation skipped.\n")
            return

        all_predicted = []
        all_actual = []
        all_names = []
        all_ids = []
        retriever = Retriever(self.config)

        for idx in tqdm(range(len(self.dataset))):
            
            #** Prepare input **#
            row = self.dataset.iloc[idx]

            prompt = self.config['task_prompt'].format(
                task_list=self.config['task_list'],
                txt=row['Text']
            )

            inputs = self.tokenizer(
                [prompt],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config["max_length"]
            ).to(self.device)
            
            #** Loop until a valid output is found or max tries reached **#
            i=1
            while True:
                #** Generate output **#
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.config["max_new_tokens"] + i*2,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id
                    )

                #** Decode output **#
                generated_text = self.tokenizer.decode(
                    outputs[0, inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                ).strip()
                
                #** Print generated text **#
                if self.config['print_generated']:
                    prnt = generated_text.replace('\n', ' ')
                    tqdm.write(f"Generated text (attempt {i}): {prnt}")

                #** Retrieve task **#
                vals: dict = retriever.retrieve_multiple(
                    names=['task'],
                    options={'task': self.config['task_list']},
                    response=generated_text,
                )
                
                #** Check valid output and exit loop **#
                if vals.get('task') and len(vals['task']) < 3:
                    all_predicted.append(vals['task'])
                    all_actual.append(row['Task'])
                    all_names.append(row['Name'])
                    all_ids.append(row['id'])
                    break
                
                #** Check against max tries **#
                
                if i == self.config['max_tries']:
                    tqdm.write(f"Failed to extract task after 15 attempts  --   id: {row['id']}  name: {row['Name']}")
                    all_predicted.append('ERR')
                    all_actual.append(row['Task'])
                    all_names.append(row['Name'])
                    all_ids.append(row['id'])
                    break
                
                i += 1
            
                      
            torch.cuda.empty_cache()


        #** Save results **#
        results_df = pd.DataFrame({
            'predicted_label': all_predicted,
            'actual_label': all_actual,
            'name': all_names,
            'id': all_ids
        })
        output_path = os.path.join(self.config['output_dir'] +  'task_eval_results.csv')
        results_df.to_csv(output_path, index_label='id', index=False)
        print(f"Saved predictions and labels to {output_path}")
