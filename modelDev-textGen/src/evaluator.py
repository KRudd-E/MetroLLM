# evaluator.py
# Evaluator classes for three cases: MMLU, post-training test set, and task classification.

import json, torch, os, re, random, gc
import pandas as pd
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from src.utils.retrieve import Retriever
from transformers import GenerationConfig
import math
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


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
        self.dataset = list(self.full_dataset[mmlu_subset]) 
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

            batch_prompts = []
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

            # tqdm.write(f"tokenizer.eos_token={self.tokenizer.eos_token!r} id={self.tokenizer.eos_token_id!r} "
            #           f"tokenizer.pad_token={self.tokenizer.pad_token!r} model.eos_id={getattr(self.model.config,'eos_token_id', None)!r}")


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
                    pad_token_id=self.tokenizer.pad_token_id,
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
            r"\(([A-J])\)\s*(?:IS CORRECT|IS THE ANSWER)?", # e.g. (B) is correct
            r"^\(?([A-J])\)?\s*$",                          # entire output is just "(B)" or "B"
            r"^([A-J])[\.\:\)]\s",                          # starts with "A." or "A:"
        ]

        #** Try each pattern in reverse order **#
        for pat in patterns:
            matches = re.findall(pat, out, flags=re.IGNORECASE)
            if matches:
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
    """ Evaluator for the post-training test set (no batching, one-by-one). """
    def __init__(self, model_wrapper, config, dataset):
        self.config = config
        
        #** Data reduction **#
        reduction = self.config.get('data_percentage', 1.0)
        num_samples = max(1, int(len(dataset) * reduction))
        self.dataset = dataset.shuffle(seed=config.get("seed", 42)).select(range(num_samples))

        self.model = model_wrapper.get_model()
        self.tokenizer = model_wrapper.get_tokenizer()
        self.device = model_wrapper.get_device()

    def evaluate(self):
        if not self.config['run']:
            print("Test Set evaluation skipped.\n")
            return

        print(f"Evaluating on test set with {len(self.dataset)}.\n")

        all_predictions = []
        all_ground_truth = []
        all_inputs = []

        self.model.eval()
        self.model.to(self.device)

        for idx in tqdm(range(len(self.dataset))):
            
            #** Prepare input **#
            item = self.dataset[idx]
            
            input_ids = torch.tensor([item['input_ids']], dtype=torch.long, device=self.device)
            attention_mask = torch.tensor([item['attention_mask']], dtype=torch.long, device=self.device)
            
            gen_config = GenerationConfig(
                max_new_tokens=min(len(item["labels"]), self.config["max_new_tokens"]),
                do_sample=self.config["do_sample"],
                temperature=self.config['temperature'],
                top_p=self.config['top_p'],
                top_k=self.config['top_k'],
                num_return_sequences=self.config['num_return_sequences'],
                early_stopping=self.config['early_stopping'],
                repetition_penalty=self.config['repetition_penalty'],
                no_repeat_ngram_size=self.config['no_repeat_ngram_size'],
                num_beams=self.config['num_beams'],
                #return_dict_in_generate=self.config['return_dict_in_generate'],
                remove_invalid_values=self.config['remove_invalid_values'],
                renormalize_logits=self.config['renormalize_logits'],
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                # bos_token_id=self.tokenizer.bos_token_id,
            )

            #** Generate prediction **#
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=gen_config,
                )

            #** Decode prediction **#
            input_len = attention_mask.sum().item()
            gen_ids = outputs[0, input_len:]
            gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            all_predictions.append(gen_text)

            #** Decode ground truth **#
            if "labels" in item:
                labels = item["labels"]
                if isinstance(labels, (list, tuple, np.ndarray, torch.Tensor)):
                    if isinstance(labels, torch.Tensor):
                        labels = labels.tolist()
                    labels = [x for x in labels if isinstance(x, int) and x >= 0]
                    gt_text = self.tokenizer.decode(labels, skip_special_tokens=True).strip()
                else:
                    gt_text = str(labels)
            else:
                gt_text = item.get("output", item.get("target", ""))
            all_ground_truth.append(gt_text)

            #** Decode input **#
            input_text = self.tokenizer.decode(item['input_ids'], skip_special_tokens=True).strip()
            all_inputs.append(input_text)
            
            tqdm.write(f"\nGT: {gt_text}")
            tqdm.write(f"Pred: {gen_text}\n")

            #** Clear memory **#
            torch.cuda.empty_cache()
            gc.collect()


        #** Initialize scorers **#
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        smoothing = SmoothingFunction().method1

        perps = []
        bleus = []
        rouge1_f = []
        rouge2_f = []
        rougeL_f = []

        for pred, gt, item in zip(all_predictions, all_ground_truth, self.dataset):
            #** BLEU **#
            try:
                ref_tokens = gt.split()
                cand_tokens = pred.split()
                bleu_score = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothing)
            except Exception:
                bleu_score = 0.0
            bleus.append(float(bleu_score)) # type: ignore

            #** ROUGE **#
            try:
                r = scorer.score(gt, pred)
                rouge1_f.append(float(r['rouge1'].fmeasure))
                rouge2_f.append(float(r['rouge2'].fmeasure))
                rougeL_f.append(float(r['rougeL'].fmeasure))
            except Exception:
                rouge1_f.append(0.0)
                rouge2_f.append(0.0)
                rougeL_f.append(0.0)

            #** Perplexity **#
            perp_val = None
            labels = item.get("labels", None)
            if labels is not None and isinstance(labels, (list, tuple, np.ndarray, torch.Tensor)):
                try:
                    if isinstance(labels, torch.Tensor):
                        lbls = labels.tolist()
                    else:
                        lbls = list(labels)
                    lbls = [x for x in lbls if isinstance(x, int) and x >= 0]
                    if len(lbls) > 0:
                        lbl_tensor = torch.tensor([lbls], dtype=torch.long, device=self.device)
                        with torch.no_grad():
                            out = self.model(input_ids=lbl_tensor, labels=lbl_tensor)
                            loss = out.loss.item()
                            perp_val = float(math.exp(loss)) if not math.isinf(loss) else float("inf")
                    else: 
                        perp_val = None
                except Exception:
                    perp_val = None
            perps.append(perp_val)

        #** Aggregate metrics **#
        valid_perps = [p for p in perps if p is not None]
        avg_perplexity = float(np.mean(valid_perps)) if len(valid_perps) > 0 else None
        avg_bleu = float(np.mean(bleus))
        avg_rouge1 = float(np.mean(rouge1_f))
        avg_rouge2 = float(np.mean(rouge2_f))
        avg_rougeL = float(np.mean(rougeL_f))

        #** Save detailed results **#
        detailed_results = pd.DataFrame({
            'input': all_inputs,
            'ground_truth': all_ground_truth,
            'prediction': all_predictions,
            'correct': [pred == truth for pred, truth in zip(all_predictions, all_ground_truth)],
            'perplexity': perps,
            'bleu': bleus,
            'rouge1_f': rouge1_f,
            'rouge2_f': rouge2_f,
            'rougeL_f': rougeL_f,
        })
        output_path = os.path.join(self.config["output_dir"] + "test_set_detailed_results.csv")
        detailed_results.to_csv(output_path, index=False)
        print(f"Detailed results saved to: {output_path}")

        #** Save summary metrics **#
        results = {
            'avg_perplexity': avg_perplexity,
            'avg_bleu': avg_bleu,
            'avg_rouge1_f': avg_rouge1,
            'avg_rouge2_f': avg_rouge2,
            'avg_rougeL_f': avg_rougeL,
            'total_samples': len(all_predictions)
        }
        metrics_path = os.path.join(self.config["output_dir"] + "test_set_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Summary metrics saved to: {metrics_path}")

        print("\nTest Set Evaluation Results:")
        if avg_perplexity:
            print(f"Avg Perplexity: {avg_perplexity:.4f}")
        print(f"Avg BLEU: {avg_bleu:.4f}")
        print(f"Avg ROUGE-1 F: {avg_rouge1:.4f}")
        print(f"Avg ROUGE-2 F: {avg_rouge2:.4f}")
        print(f"Avg ROUGE-L F: {avg_rougeL:.4f}")
        print(f"Total Samples: {len(all_predictions)}")

        return results
        


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
                format='"task": ["<one task>", ...optional second task...]',
                txt=row['Text'],
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
                gen_config = GenerationConfig(
                    max_new_tokens=self.config["max_new_tokens"] + i*2,
                    do_sample=self.config["do_sample"],
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    bos_token_id=self.tokenizer.bos_token_id,
                )
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        generation_config=gen_config,
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
                    tqdm.write(f"Failed to extract task after {self.config['max_tries']} attempts  --   id: {row['id']}  name: {row['Name']}")
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
