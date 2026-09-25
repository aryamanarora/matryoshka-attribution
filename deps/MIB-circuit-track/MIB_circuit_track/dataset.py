from typing import Optional
import random

from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import pandas as pd
from transformers import PreTrainedTokenizer

def collate_EAP(xs):
    clean, corrupted, labels = zip(*xs)
    clean = list(clean)
    corrupted = list(corrupted)
    return clean, corrupted, labels


class EAPDataset(Dataset):
    def __init__(self, filepath:str, task:str='greater-than'):
        self.task = task
        self.df = pd.read_csv(filepath)

    def __len__(self):
        return len(self.df)
    
    def shuffle(self):
        self.df = self.df.sample(frac=1)

    def head(self, n: int):
        self.df = self.df.head(n)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        if self.task == 'greater-than':
            return row['clean'], row['corrupted'], row['correct_idx']
        elif self.task == 'ioi':
            return row['clean'], row['corrupted'], [row['correct_idx'], row['incorrect_idx']]
        elif self.task == 'ewok':
            return row['Context1'], row['Context2'], [row['Target1'], row['Target2']]
        else:
            raise ValueError(f'Got invalid task: {self.task}')
    
    def to_dataloader(self, batch_size: int):
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_EAP)
    

class HFEAPDataset(Dataset):
    task: str 
    tokenizer: PreTrainedTokenizer
    control: bool
    model_name: Optional[str]
    dataset: Dataset

    def __init__(self, url:str, tokenizer: PreTrainedTokenizer, split:str="train", task:str='ioi', num_examples:Optional[int]=None,
                 control:Optional[bool]=False, counterfactual_type:Optional[str]=None,
                 example_domain:Optional[str]=None, model_name: Optional[str] = None, hf_token: Optional[str] = None):      
        self.task = task
        self.tokenizer = tokenizer
        self.control = control
        self.model_name = model_name

        self.counterfactual_type = counterfactual_type
        if task == 'mcqa':
            self.dataset = load_dataset(url, '4_answer_choices', split=split, token=hf_token)
            if self.counterfactual_type is None:
                self.counterfactual_type = "symbol_counterfactual"
        elif task.startswith('arc'):
            self.dataset = load_dataset(url, split=split, token=hf_token)
            if self.counterfactual_type is None:
                self.counterfactual_type = "symbol_counterfactual"
        elif task == 'ewok':
            self.dataset = load_dataset(url, split="test")
            self.example_domain = "social-properties" if example_domain is None else example_domain
        elif task.startswith('arithmetic'):
            self.dataset = load_dataset(url, split=split, token=hf_token)
            self.example_domain = "-" if "subtraction" in task else example_domain if example_domain is not None else "+"
        elif task == 'greater-than':
            assert model_name is not None, "For greater-than you must specify the model name, but it is None"
            self.dataset = load_dataset(url, split=split, token=hf_token)
        else:
            self.dataset = load_dataset(url, split=split, token=hf_token)
        
        self.dataset = self.filter_dataset()
        #self.dataset = self.shuffle()
        if num_examples:
            self.head(num_examples)
        
        # for when `control is True`
        self.answer_map = {}
        self.seed_offset = 0


    def __len__(self):
        return len(self.dataset)
    
    def shuffle(self):
        return self.dataset.shuffle()
    
    def head(self, n: int):
        if n <= len(self.dataset):
            self.dataset = self.dataset.select(range(n))
        else:
            print("Warning: `num_examples` is greater than the size of the dataset! Returning the full dataset.")
    
    def tail(self, n: int):
        return [self.dataset[i] for i in range(len(self.dataset)-n, len(self.dataset))]
        

    def filter_dataset(self):
        if self.task == 'ioi':
            filtered_dataset = self.dataset.filter(
                lambda x: len(self.tokenizer(f" {x['metadata']['indirect_object']}", add_special_tokens=False).input_ids) == 
                          len(self.tokenizer(f" {x['metadata']['subject']}", add_special_tokens=False).input_ids) and
                          len(self.tokenizer(f" {x['metadata']['indirect_object']}", add_special_tokens=False).input_ids) ==
                          len(self.tokenizer(f" {x['metadata']['random_c']}", add_special_tokens=False).input_ids)
            )
        elif self.task == 'mcqa':
            filtered_dataset = self.dataset.filter(
                lambda x: len(self.tokenizer(x["choices"]["label"][x["answerKey"]], add_special_tokens=False).input_ids) ==
                          len(self.tokenizer(str(x[self.counterfactual_type]["choices"]["label"][x[self.counterfactual_type]["answerKey"]]),
                                             add_special_tokens=False).input_ids)
            )
        elif self.task == 'ewok':
            filtered_dataset = self.dataset.filter(
                lambda x: len(self.tokenizer(x["Target1"], add_special_tokens=False).input_ids) ==
                          len(self.tokenizer(x["Target2"], add_special_tokens=False).input_ids) and
                          x["Domain"] == self.example_domain
            )
        elif self.task.startswith('arithmetic'):
            filtered_dataset = self.dataset.filter(
                lambda x: len(self.tokenizer(str(x["label"]), add_special_tokens=False).input_ids) == 1 and
                          x["random_counterfactual"] is not None and
                          x["random_counterfactual"]["prompt"] is not None and x["operator"] == self.example_domain and
                          len(self.tokenizer(str(x["random_counterfactual"]["label"]), add_special_tokens=False).input_ids) == 1
            )
        elif self.task == 'greater-than':
            filtered_dataset = self.dataset.filter(
                lambda x: len(self.tokenizer(x["clean"], add_special_tokens=False).input_ids) ==
                          len(self.tokenizer(x["corrupted"], add_special_tokens=False).input_ids)
            )
        elif self.task.startswith('arc'):
            filtered_dataset = self.dataset.filter(
                lambda x: len(self.tokenizer(x["choices"]["label"][x["answerKey"]], add_special_tokens=False).input_ids) ==
                          len(self.tokenizer(str(x[self.counterfactual_type]["choices"]["label"][x[self.counterfactual_type]["answerKey"]]),
                                             add_special_tokens=False).input_ids)
            )
        else:
            raise ValueError(f"Unrecognized task: {self.task}")

        return filtered_dataset
    
    def __getitem__(self, index):
        def _make_control_answer(answer_idx, offset=0):
            if offset != 0:
                self.seed_offset += offset
            random.seed(index + self.seed_offset)

            if answer_idx not in self.answer_map:
                random_token = random.randint(1000, self.tokenizer.vocab_size-1000)
                existing_random_answers = set(self.answer_map.values())
                # keep resampling until we obtain a unique answer. maintains bijectivity
                while random_token in existing_random_answers:
                    self.seed_offset += 1
                    random.seed(index + self.seed_offset)
                    random_token = random.randint(1000, self.tokenizer.vocab_size-1000)
                self.answer_map[answer_idx] = random_token
                
            new_answer_idx = self.answer_map[answer_idx]
            return new_answer_idx

        row = self.dataset[index]
        if self.task == 'ioi':
            counterfactual_col = 's2_io_flip_counterfactual' if self.counterfactual_type is None else self.counterfactual_type
            correct_idx = self.tokenizer(f" {row['metadata']['indirect_object']}", add_special_tokens=False).input_ids[0]
            incorrect_idx = self.tokenizer(f" {row['metadata']['subject']}", add_special_tokens=False).input_ids[0]
            if self.control:
                correct_idx = _make_control_answer(correct_idx)
                incorrect_idx = _make_control_answer(incorrect_idx, offset=1)
            return row["prompt"], row[counterfactual_col]["prompt"], [correct_idx, incorrect_idx]
        
        elif self.task == 'mcqa' or self.task.startswith('arc'):
            clean_prompt = row["prompt"]
            correct_idx = self.tokenizer(row["choices"]["label"][row["answerKey"]], add_special_tokens=False).input_ids[0]
            # counterfactual_cols = [k for k in list(row.keys()) if "_counterfactual" in k]
            counterfactual_col = row[self.counterfactual_type]
            corrupted_prompt = counterfactual_col["prompt"]
            incorrect_ans = str(counterfactual_col["choices"]["label"][counterfactual_col["answerKey"]])
            incorrect_idx = self.tokenizer(incorrect_ans, add_special_tokens=False).input_ids[0]
            if self.control:
                correct_idx = _make_control_answer(correct_idx)
                incorrect_idx = _make_control_answer(incorrect_idx, offset=1)
            return clean_prompt, corrupted_prompt, [correct_idx, incorrect_idx]

        elif self.task.startswith('arithmetic'):
            clean_prompt = row["prompt"]
            corrupted_prompt = row["random_counterfactual"]["prompt"]
            correct_idx = self.tokenizer(str(row["label"]), add_special_tokens=False).input_ids[0]
            incorrect_idx = self.tokenizer(str(row["random_counterfactual"]["label"]), add_special_tokens=False).input_ids[0]
            if self.control:
                correct_idx = _make_control_answer(correct_idx)
                incorrect_idx = _make_control_answer(incorrect_idx, offset=1)
            return clean_prompt, corrupted_prompt, [correct_idx, incorrect_idx]

        elif self.task == 'ewok':
            clean_prompt = row["Context1"] + " " + row["Target1"]
            counterfactual_prompt = row["Context1"] + " " + row["Target2"]
            correct_idxs = self.tokenizer(row["Target1"], add_special_tokens=False).input_ids
            incorrect_idxs = self.tokenizer(row["Target2"], add_special_tokens=False).input_ids
            if self.control:
                raise NotImplementedError("Error: controls are not implemented for multi-token outputs.")
            return clean_prompt, counterfactual_prompt, [correct_idxs, incorrect_idxs]

        elif self.task == 'greater-than':
            year = row['year']
            if 'gpt2' in self.model_name:
                label = [int(year[2:])]
            elif 'Llama-3' in self.model_name:
                label = [int(year[:3]), int(year[3:])]
            elif 'gemma-2' in self.model_name or 'Qwen' in self.model_name:
                label = [int(year[2]), int(year[3])]
            else:
                raise ValueError(f"Unrecognized model name: {self.model_name}")
            return row['clean'], row['corrupted'], label

    def to_dataloader(self, batch_size: int):
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_EAP)
