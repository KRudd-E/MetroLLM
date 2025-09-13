# evaluator.py
# Evaluator classes for three cases: MMLU, post-training test set, and task classification.

import json, torch, os, re, random, gc
import pandas as pd
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from src.utils.retrieve import Retriever


class MMLU_Evaluator:
    def __init__(self, model_wrapper, config, mmlu_subset="test"):
        import random

        self.config = config
        self.model = model_wrapper.get_model()
        self.tokenizer = model_wrapper.get_tokenizer()
        self.device = model_wrapper.get_device()
        self.batch_size = config['batch_size']

        #** Load MMLU dataset **#
        self.full_dataset = load_dataset("TIGER-Lab/MMLU-Pro")
        self.dataset = list(self.full_dataset[mmlu_subset])  # convert to list for sampling
        self.eval_split = mmlu_subset

        self.categories = [
            "computer science", "math", "chemistry", "engineering", "law",
            "biology", "health", "physics", "business", "philosophy",
            "economics", "other", "psychology", "history"
        ]

        #** Data reduction **#
        clip_percentage = self.config['data_reduction']
        if clip_percentage < 1.0:
            random.seed(config.get("seed", 42))
            clipped = []
            for cat in self.categories:
                cat_items = [x for x in self.dataset if x["category"] == cat]
                if not cat_items:
                    continue
                clip_size = max(1, int(len(cat_items) * clip_percentage))
                clipped.extend(random.sample(cat_items, clip_size))
            self.dataset = clipped

        #** Few-shot prompts **#
        val_split = list(self.full_dataset["validation"])
        few_shot_k = config.get("few_shot_k", 3)  # default: 3 shots
        random.seed(config.get("seed", 42))

        self.prompts = {}
        for cat in self.categories:
            val_items = [v for v in val_split if v["category"] == cat]
            random.shuffle(val_items)
            examples = val_items[:few_shot_k]

            #** Build prompt **#
            prompt = (
                "You are a helpful multiple-choice assistant.\n"
                "Answer with a single letter (A-J) and NOTHING ELSE — no explanation, no chain-of-thought.\n\n"
            )

            for ex in examples:
                answer = ex["answer"]

                if isinstance(answer, int):
                    ans_letter = "ABCDEFGHIJ"[answer]
                else:
                    ans_letter = str(answer).strip().upper()

                prompt += f"Q: {ex['question']}\n"
                prompt += self.form_options(ex["options"])
                prompt += f"A: ({ans_letter})\n\n"

            self.prompts[cat] = prompt      
            
        # src:      https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
        # example:  https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/blob/main/run_gpt4o.py


    def evaluate(self):
        if not self.config["run"]:
            print("\nMMLU evaluation skipped.\n")
            return

        print(f"Evaluating on MMLU {self.eval_split} split with {len(self.dataset)} samples.\n")    # type: ignore

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
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    #stopping_criteria=None 
                )

            #** Decode batch outputs **#
            gen_texts = self.tokenizer.batch_decode(
                outputs[:, inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            #** Process predictions **#
            for entry, raw_gen in zip(batch_entries, gen_texts):
                
                if self.config['print_generated']:
                    tqdm.write(f"Generated: {raw_gen.replace(chr(10), ' ')}")
                            
                gen = raw_gen.strip()
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

            tqdm.write(f"Overall accuracy: {success / (success + fail)}")
            
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
            r"THE ANSWER IS[:\s]*\(?([A-J])\)?",
            r"THE CORRECT ANSWER (?:IS|:) ?\(?([A-J])\)?",
            r"ANSWER[:\s]*\(?([A-J])\)?",
            r"\(([A-J])\)\s*(?:IS CORRECT|IS THE ANSWER)?",  # e.g. (B) is correct
            r"^\(?([A-J])\)?\s*$",                          # entire output is just "(B)" or "B"
            r"^([A-J])[\.\:\)]\s",                          # starts with "A." or "A:"
        ]

        for pat in patterns:
            matches = re.findall(pat, out, flags=re.IGNORECASE)
            if matches:
                # choose the last valid match (prefer the last explicit answer)
                for m in reversed(matches):
                    if m and m in list("ABCDEFGHIJ")[:num_choices]:
                        return m

        #** Fallback: look for LAST standalone letter **#
        all_letters = [m.group(1).upper() for m in re.finditer(r'\b([A-J])\b', out)]
        for c in reversed(all_letters):
            if c in list("ABCDEFGHIJ")[:num_choices]:
                return c

        #** Random **#
        tqdm.write(f"No valid answer found in generation: {output[:200]!r} ... falling back to random.")
        return random.choice(list("ABCDEFGHIJ")[:num_choices])



#*----------------------------------------------------------------------------*#


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
        


#*----------------------------------------------------------------------------*#


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
