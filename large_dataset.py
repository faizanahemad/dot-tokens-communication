# datasets.py  

from typing import Any, List, Optional, Tuple
import torch  
import os
os.environ["HF_TOKEN"] = "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"
from datasets import Dataset, IterableDataset  
from datasets import load_dataset  
from transformers import AutoTokenizer  
from tqdm import tqdm  
import re  
  
class BaseDataset(IterableDataset):  
    def __init__(self, dataset_name, tokenizer_name, max_input_length, max_output_length, split, subset_size=None, **extra_kwargs):  
        self._dataset_name = dataset_name  
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)  
        # Check if the tokenizer has no pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self._max_input_length = max_input_length  
        self._max_output_length = max_output_length  
        self._split = split  
        self._subset_size = subset_size  
        self._data = self.load_data()
        self._extra_kwargs = extra_kwargs
        self._num_rows = len(self._data)
        self._ex_iterable = None
        self._info = self._data.info
        
    @property
    def info(self):
        return self._info
    
    @info.setter
    def info(self, value):
        self._info = value
    
    @property
    def num_rows(self):
        return self._num_rows
    
    @num_rows.setter
    def num_rows(self, value):
        self._num_rows = value
        
    
    @property
    def subset_size(self):
        return self._subset_size
    
    @subset_size.setter
    def subset_size(self, value):
        self._subset_size = value
    
    
    
    @property
    def max_input_length(self):
        return self._max_input_length
    
    @max_input_length.setter
    def max_input_length(self, value):
        self._max_input_length = value
        
    @property
    def max_output_length(self):
        return self._max_output_length
    
    @max_output_length.setter
    def max_output_length(self, value):
        self._max_output_length = value
        
    @property
    def features(self) -> Optional[Any]:
        return self._features if hasattr(self, '_features') else None
    
    @features.setter
    def features(self, value):
        self._features = value
    
    
    
    @property
    def extra_kwargs(self):
        return self._extra_kwargs
    
    @extra_kwargs.setter
    def extra_kwargs(self, value):
        self._extra_kwargs = value
    
    
    
    @property
    def dataset_name(self):
        return self._dataset_name
    
    @dataset_name.setter
    def dataset_name(self, value):
        self._dataset_name = value
    
    
    
    @property
    def tokenizer(self):
        return self._tokenizer
    
    @tokenizer.setter
    def tokenizer(self, value):
        self._tokenizer = value
    
    
    
    @property
    def use_cache(self):
        return self._use_cache
    
    @use_cache.setter
    def use_cache(self, value):
        self._use_cache = value
    
    
    
    @property
    def cached_data(self):
        return self._cached_data
    
    @cached_data.setter
    def cached_data(self, value):
        self._cached_data = value
    
    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, value):
        self._data = value
    
    
    @property
    def split(self):
        return self._split

    @split.setter
    def split(self, value):
        self._split = value
        
    
    def load_data(self):  
        raise NotImplementedError("This method should be implemented by subclasses.")  
  
    def preprocess_data(self):  
        raise NotImplementedError("This method should be implemented by subclasses.")  
  
    def __len__(self):  
        return len(self.cached_data)  
  
    def __getitem__(self, idx):  
        if self.use_cache and self.cached_data:
            return self.cached_data[idx]
        else:
            raise NotImplementedError("This method should be implemented by subclasses.")
    
    def __iter__(self):
        if self.use_cache and self.cached_data:
            for item in self.cached_data:
                yield item
        else:
            raise NotImplementedError("This method should be implemented by subclasses.")
  
    def extract_answer(self, text):  
        raise NotImplementedError("This method should be implemented by subclasses.")  
  
    def get_evaluation_metrics(self):  
        # Returns a dictionary of metric functions specific to the dataset  
        raise NotImplementedError("This method should be implemented by subclasses.")  

    
class ConfigurableDataset(BaseDataset):  
    def __init__(self, dataset_name, tokenizer_name, max_input_length, max_output_length, split, subset_size=None, use_cache=True, metrics=None, **extra_kwargs):  
        self.dataset_name = dataset_name
        self.subset_name = extra_kwargs.get("subset_name", None)
        self.use_cache = use_cache  
        self.metrics = metrics or {}  # Store custom metrics  
        
        self.extra_kwargs = extra_kwargs  
        self.instruction_column = extra_kwargs.get("instruction_column", None)  
        self.instruction = extra_kwargs.get("instruction", None)  
        assert self.instruction is not None or self.instruction_column is not None, "Either instruction or instruction_column must be provided"  
        self.input_columns = extra_kwargs.get("input_columns")  
        self.output_columns = extra_kwargs.get("output_columns")  
        self.answer_parser = extra_kwargs.get("answer_parser", lambda x: x.strip().lower())  
        self.input_parser = extra_kwargs.get("input_parser", lambda x: " ".join([x[col][0] if isinstance(x[col], list) else str(x[col]) for col in self.input_columns]))
        super().__init__(dataset_name, tokenizer_name, max_input_length, max_output_length, split, subset_size, **extra_kwargs)  
        
        self.extract_answer = lambda x: self.answer_parser(x).strip().lower().replace("Answer:", "").replace("answer:", "").strip()
        self.cached_data = None  
        if self.use_cache:  
            self.preprocess_data()  
  
    @property
    def column_names(self) -> List[str]:
        return ['input_ids', 'attention_mask', 'labels', 'labels_attention_mask', 'reference_answer']
    
    @property
    def shape(self) -> Tuple[int, int]:
        return (len(self.column_names), len(self))
    
    
    @property
    def num_columns(self) -> int:
        return len(self.column_names)
    
    def load_data(self):  
        dataset = load_dataset(self.dataset_name, "main" if self.subset_name is None else self.subset_name, trust_remote_code=True)  
        tokenizer = self.tokenizer
        max_input_length = self.max_input_length
        max_output_length = self.max_output_length
        data = dataset[self.split]  
        output_columns = self.output_columns
        output_parser = lambda x: " ".join([str(x[col]) for col in output_columns])
        input_columns = self.input_columns
        input_parser = lambda x: " ".join([str(x[col]) for col in input_columns])
        data = data.filter(lambda x: len(tokenizer.encode(input_parser(x))) <= max_input_length and len(tokenizer.encode(output_parser(x))) <= max_output_length)
        if self.subset_size:  
            data = data.shuffle(seed=42).select(range(min(self.subset_size, len(data))))  
        return data  
  
    def preprocess_data(self):  
        if not self.use_cache:  
            return  
  
        self.cached_data = []  
        for item in tqdm(self.data, desc=f"Preprocessing {self.dataset_name} {self.split} data"):  
            processed_item = self.process_item(item)  
            self.cached_data.append(processed_item)  
  
    def process_item(self, item, padding="max_length"):  
        # Construct the prompt  
        prompt = self.construct_prompt(item)  
  
        # Tokenize the prompt with left padding  
        self.tokenizer.padding_side = 'left'  
        self.tokenizer.truncation_side = 'left'  
        encoded_prompt = self.tokenizer(  
            prompt,  
            max_length=self.max_input_length,  
            padding=padding,  
            truncation=True,  
            return_tensors='pt',  
            add_special_tokens=True  
        )  
        input_ids = encoded_prompt['input_ids'].squeeze()  
        attention_mask = encoded_prompt['attention_mask'].squeeze()  
  
        # Construct the target output  
        target = self.construct_target(item)  
  
        # Tokenize the target with right padding  
        self.tokenizer.padding_side = 'right'  
        self.tokenizer.truncation_side = 'right'  
        encoded_target = self.tokenizer(  
            target,  
            max_length=self.max_output_length,  
            padding=padding,  
            truncation=True,  
            return_tensors='pt',  
            add_special_tokens=True  
        )  
        labels = encoded_target['input_ids'].squeeze()  
        labels_attention_mask = encoded_target['attention_mask'].squeeze()  
  
        # Apply the answer parser to the reference answer  
        parsed_reference_answer = self.extract_answer(target)  
  
        return {  
            'input_ids': input_ids,  
            'attention_mask': attention_mask,  
            'labels': labels,  
            'labels_attention_mask': labels_attention_mask,  
            'reference_answer': parsed_reference_answer,
            "prompt": prompt,  
            "target": target,
            "dataset_name": self.dataset_name,
        }  
  
    def construct_prompt(self, item):  
        instruction = self.instruction if self.instruction else (item[self.instruction_column]  if self.instruction_column else "")
        instruction = f"{instruction}\n\n" if instruction.strip() != "" else ""
        inputs = self.input_parser({col: item[col] for col in self.input_columns}).strip()
        prompt = f"{instruction}{inputs}\nAnswer:\n"  
        # print(prompt, flush=True)
        return prompt
  
    def construct_target(self, item):  
        target = " ".join([item[col][0] if isinstance(item[col], list) else str(item[col]) for col in self.output_columns])  
        # print(f"Target: {target}")
        return target.strip().replace("Answer:", "").replace("answer:", "").strip()
  
  
    def _get_single_item(self, idx):  
        if self.use_cache and self.cached_data:  
            return self.cached_data[idx]  
        else:  
            item = self.data[idx]  
            return self.process_item(item)  
        
    def __iter__(self):
        if self.use_cache and self.cached_data:
            for item in self.cached_data:
                yield item
        else:
            for item in self.data:
                yield self.process_item(item)
                
    def __call__(self, *args, **kwargs):
        return self.__iter__()
        
    def __getitem__(self, idx):  
        # Implement indexing and slicing  
        if isinstance(idx, int):  
            return self._get_single_item(idx)  
        elif isinstance(idx, slice):  
            return [self._get_single_item(i) for i in range(*idx.indices(len(self)))]  
  
    def __len__(self):  
        return len(self.cached_data) if self.use_cache and self.cached_data else len(self.data)  
  
    def get_evaluation_metrics(self):  
        def exact_match(predictions, references):  
            correct = sum(pred == ref for pred, ref in zip(predictions, references) if pred is not None and ref is not None)  
            total = sum(1 for pred, ref in zip(predictions, references) if pred is not None and ref is not None)  
            return {'exact_match': correct / total if total > 0 else 0}  
        
        default_metrics = {'exact_match': exact_match}  
        default_metrics.update(self.metrics)  # Add custom metrics  
        return default_metrics  
    
    def get_largest_input_ids_length(self):
        if self.use_cache and self.cached_data:
            lengths = [len(item['input_ids']) for item in self.cached_data]
        else:
            lengths = []
            for item in tqdm(self.data, desc="Processing items", unit="item"):
                processed_item = self.process_item(item, padding=False)
                input_ids_length = len(processed_item['input_ids'])
                lengths.append(input_ids_length)
        
        lengths = sorted(lengths)
        n = len(lengths)
        
        return {
            'largest': lengths[-1],
            'median': lengths[n // 2] if n % 2 != 0 else (lengths[n // 2 - 1] + lengths[n // 2]) / 2,
            '75_percentile': lengths[int(n * 0.75)],
            '90_percentile': lengths[int(n * 0.90)],
            '95_percentile': lengths[int(n * 0.95)],
            '98_percentile': lengths[int(n * 0.98)]
        }
    
    
    def get_largest_labels_length(self):
        if self.use_cache and self.cached_data:
            lengths = [len(item['labels']) for item in self.cached_data]
        else:
            lengths = []
            for item in tqdm(self.data, desc="Processing items", unit="item"):
                processed_item = self.process_item(item, padding=False)
                labels_length = len(processed_item['labels'])
                lengths.append(labels_length)
        
        lengths = sorted(lengths)
        n = len(lengths)
        
        return {
            'largest': lengths[-1],
            'median': lengths[n // 2] if n % 2 != 0 else (lengths[n // 2 - 1] + lengths[n // 2]) / 2,
            '75_percentile': lengths[int(n * 0.75)],
            '90_percentile': lengths[int(n * 0.90)],
            '95_percentile': lengths[int(n * 0.95)],
            '98_percentile': lengths[int(n * 0.98)]
        }
    

def gsm8k_answer_parser(text):  
    pos = text.find('####')  
    if pos == -1:  
        return None  
    answer = text[pos+4:].strip()  
    return ''.join(answer.split()).lower().strip().replace("Answer:", "").replace("answer:", "").strip()

def reference_contained(predictions, references):  
    correct = 0  
    total = 0  
    for pred, ref in zip(predictions, references):  
        total += 1  
        if pred is None or ref is None:  
            continue  
        pred = pred.strip().lower().replace("\n", " ")  
        ref = ref.strip().lower().replace("\n", " ")  
        if ref.lower() in pred.lower():  
            correct += 1  
    accuracy = correct / total if total > 0 else 0  
    return {'reference_contained': accuracy}  
  


def custom_target_constructor_for_gsm8k_reasoning(item):  
    return f"{item['generation']}\n#### {item['short_answer']}"  

def custom_target_constructor_for_lmsys_arena_human_preference_55k(item):
    response = f"{item['response_a'] if item['winner_model_a']==1 else item['response_b']}"
    return response.replace('["', "").replace('"]', "").strip()




import re  
  
def competition_math_answer_parser(text):  
    # Find the content inside the \boxed{} command  
    boxed_match = re.search(r'\\boxed\{(.*?)\}', text)  
    if boxed_match:  
        return boxed_match.group(1).strip().replace("Answer:", "").replace("answer:", "").strip()
    else:  
        # If no \boxed{} is found, return the last sentence as a fallback  
        sentences = text.split('.')  
        return sentences[-1].strip().replace("Answer:", "").replace("answer:", "").strip()

math_instruction = """Solve this mathematical problem.
Important:
1. Think step by step and show your work, explaining each step.
2. Provide the final answer in a LaTeX \\boxed{} command.
3. Only put the final result in \\boxed{}, in LaTeX format. The \\boxed{} command should contain only the final numerical or algebraic result in latex format.

Examples:
\\boxed{42}
\\boxed{\\frac{3}{4}}

Solve the problem:
"""
def custom_prompt_constructor_math(item):  
    return f"{math_instruction}\nProblem Type: {item['type']}\nProblem: {item['problem']}\n"


    
    
def get_dataset(dataset_name, tokenizer_name, max_input_length, max_output_length, split, subset_size=None, **extra_kwargs):
    if dataset_name == "Rowan/hellaswag":
        hellaswag_dataset = ConfigurableDataset(
            dataset_name="Rowan/hellaswag",
            tokenizer_name=tokenizer_name,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            split="train",
            subset_size=subset_size,
            instruction="Choose the correct ending for the given context and situation.",
            input_columns=["activity_label", "ctx", "endings"],
            output_columns=["label"],
            input_parser=lambda x: x["activity_label"] + "\n" + x["ctx"] + "\n" + "\nPossible Endings:\n" + "\n".join([f"{i}. {ending}" for i, ending in enumerate(x["endings"])]),
            answer_parser=lambda x: x.strip().lower(),
            use_cache=False,
            subset_name="default",
            metrics={'reference_contained': reference_contained}
        )
        
        hellaswag_test_dataset = ConfigurableDataset(
            dataset_name="Rowan/hellaswag",
            tokenizer_name=tokenizer_name,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            split="validation",
            subset_size=subset_size,
            instruction="Choose the correct ending for the given context and situation.",
            input_columns=["activity_label", "ctx", "endings"],
            output_columns=["label"],
            input_parser=lambda x: x["activity_label"] + "\n" + x["ctx"] + "\n" + "\nPossible Endings:\n" + "\n".join([f"{i}. {ending}" for i, ending in enumerate(x["endings"])]),
            answer_parser=lambda x: x.strip().lower(),
            use_cache=False,
            subset_name="default",
            metrics={'reference_contained': reference_contained}
        )
        if split == "train":
            return hellaswag_dataset
        elif split == "test":
            return hellaswag_test_dataset
    
    if dataset_name == "Isotonic/human_assistant_conversation":
        isotonic_human_assistant_conversation_dataset = ConfigurableDataset(
            dataset_name="Isotonic/human_assistant_conversation",
            tokenizer_name=tokenizer_name,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            split="train",
            subset_size=subset_size,
            instruction="",
            input_columns=["prompt"],
            output_columns=["response"],
            input_parser=lambda x: x["prompt"].replace("Assistant:", "").replace("Output:", "").replace("Human: ", "").replace("User: ", "").strip(),
            answer_parser=lambda x: x.strip().lower(),
            use_cache=False,
            subset_name="default",
            metrics={'reference_contained': reference_contained}
        )
        if split == "train":
            return isotonic_human_assistant_conversation_dataset
        elif split == "test":
            raise ValueError("No test split available for this dataset")
    
    if dataset_name == "flammenai/casual-conversation-DPO":
        casual_conversation_dpo_dataset = ConfigurableDataset(
            dataset_name="flammenai/casual-conversation-DPO",
            tokenizer_name=tokenizer_name,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            split="train",
            subset_size=subset_size,
            instruction="",
            input_columns=["prompt"],
            output_columns=["chosen"],
            answer_parser=lambda x: x.strip().lower(),
            use_cache=False,
            subset_name="default",
            metrics={'reference_contained': reference_contained}
        )
        if split == "train":
            return casual_conversation_dpo_dataset
        elif split == "test":
            raise ValueError("No test split available for this dataset")
        
    if dataset_name == "flammenai/casual-conversation-DPO-rejected":
        casual_conversation_dpo_dataset = ConfigurableDataset(
            dataset_name="flammenai/casual-conversation-DPO",
            tokenizer_name=tokenizer_name,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            split="train",
            subset_size=subset_size,
            instruction="",
            input_columns=["prompt"],
            output_columns=["rejected"],
            answer_parser=lambda x: x.strip().lower(),
            use_cache=False,
            subset_name="default",
            metrics={'reference_contained': reference_contained}
        )
        if split == "train":
            return casual_conversation_dpo_dataset
        elif split == "test":
            raise ValueError("No test split available for this dataset")
    
    if dataset_name == "hendrycks/competition_math":
        math_train_dataset = ConfigurableDataset(  
            dataset_name="hendrycks/competition_math",  
            tokenizer_name=tokenizer_name,  
            max_input_length=max_input_length,  # Increased due to potentially longer problems  
            max_output_length=max_output_length,  # Increased to accommodate detailed solutions  
            split="train",  
            subset_size=subset_size,  
            instruction=math_instruction,  
            input_columns=["problem", "type"],  
            output_columns=["solution"],  
            input_parser=custom_prompt_constructor_math,
            answer_parser=competition_math_answer_parser,  
            use_cache=False,  
            subset_name="default",  
            metrics={'reference_contained': reference_contained}  
        )  
        
        math_test_dataset = ConfigurableDataset(  
            dataset_name="hendrycks/competition_math",  
            tokenizer_name=tokenizer_name,  
            max_input_length=max_input_length,  
            max_output_length=max_output_length,  
            split="test",  
            subset_size=subset_size,  
            instruction=math_instruction,  
            input_columns=["problem", "type"],  
            output_columns=["solution"],  
            input_parser=custom_prompt_constructor_math,
            answer_parser=competition_math_answer_parser,  
            use_cache=False,  
            subset_name="default",  
            metrics={'reference_contained': reference_contained}  
        )  
 
        
        if split == "train":
            return math_train_dataset
        elif split == "test":
            return math_test_dataset

    
    if dataset_name == "gsm8k":
        gsm8k_train_dataset = ConfigurableDataset(  
            dataset_name="gsm8k",  
            tokenizer_name=tokenizer_name,  # You can replace this with your preferred tokenizer  
            max_input_length=max_input_length,  
            max_output_length=max_output_length,  
            split="train",  # or "test" depending on your needs  
            subset_size=subset_size,  # Set this to a number if you want to use only a subset of the data  
            instruction="Solve the following math problem step by step. Show your work and provide the final answer after '#### '.",  
            input_columns=["question"],  
            output_columns=["answer"],  
            answer_parser=gsm8k_answer_parser,
            use_cache=False,
            subset_name="main",
            metrics={'reference_contained': reference_contained} 
            
        )  
        
        gsm8k_test_dataset = ConfigurableDataset(  
            dataset_name="gsm8k",  
            tokenizer_name=tokenizer_name,  # You can replace this with your preferred tokenizer  
            max_input_length=max_input_length,  
            max_output_length=max_output_length,  
            split="test",  # or "test" depending on your needs  
            subset_size=subset_size,  # Set this to a number if you want to use only a subset of the data  
            instruction="Solve the following math problem step by step. Show your work and provide the final answer after '#### '.",  
            input_columns=["question"],  
            output_columns=["answer"],  
            answer_parser=gsm8k_answer_parser,
            use_cache=False,
            subset_name="main",
            metrics={'reference_contained': reference_contained} 
            
        )  
        if split == "train":
            return gsm8k_train_dataset
        elif split == "test":
            return gsm8k_test_dataset
        
    if dataset_name == "gsm8k-reasoning":
        gsm8k_reasoning_dataset = ConfigurableDataset(  
            dataset_name="thesven/gsm8k-reasoning",  
            tokenizer_name=tokenizer_name,  # You can replace this with your preferred tokenizer  
            max_input_length=max_input_length,  
            max_output_length=max_output_length,  
            split="train",  # Only 'train' split is available in this dataset  
            subset_size=subset_size,    # Set this to a number if you want to use only a subset of the data  
            instruction_column="system_prompt",  
            input_columns=["question"],  
            output_columns=["generation", "short_answer"],  
            answer_parser=lambda x: x.split("####")[-1].strip().lower() if "####" in x else x.strip().lower(),  
            use_cache=False,  
            subset_name="default",
            metrics={'reference_contained': reference_contained}  
        )  
        gsm8k_reasoning_dataset.construct_target = custom_target_constructor_for_gsm8k_reasoning
        
        if split == "train":
            return gsm8k_reasoning_dataset
        elif split == "test":
            raise ValueError("No test split available for this dataset")
            
    
    elif dataset_name == "synthetic-gsm8k":
        synthetic_gsm8k_dataset = ConfigurableDataset(  
            dataset_name="gretelai/synthetic-gsm8k-reflection-405b",  
            tokenizer_name=tokenizer_name,  # Using the same tokenizer as in your example  
            max_input_length=max_input_length,  
            max_output_length=max_output_length,  
            split="train",  # Assuming 'train' split is available, adjust if needed  
            subset_size=subset_size,    # Set this to a number if you want to use only a subset of the data  
            instruction="Solve the following math problem step by step. Show your work and provide the final answer after '#### '.",  
            input_columns=["question"],  
            output_columns=["answer"],  
            answer_parser=lambda x: x.split("####")[-1].strip().lower() if "####" in x else x.strip().lower(),  
            use_cache=False,  
            subset_name="default",  
            metrics={'reference_contained': reference_contained}  
        )  
        
        synthetic_gsm8k_test_dataset = ConfigurableDataset(  
            dataset_name="gretelai/synthetic-gsm8k-reflection-405b",  
            tokenizer_name=tokenizer_name,  # Using the same tokenizer as in your example  
            max_input_length=max_input_length,  
            max_output_length=max_output_length,  
            split="test",  # Assuming 'train' split is available, adjust if needed  
            subset_size=subset_size,    # Set this to a number if you want to use only a subset of the data  
            instruction="Solve the following math problem step by step. Show your work and provide the final answer after '#### '.",  
            input_columns=["question"],  
            output_columns=["answer"],  
            answer_parser=lambda x: x.split("####")[-1].strip().lower() if "####" in x else x.strip().lower(),  
            use_cache=False,  
            subset_name="default",  
            metrics={'reference_contained': reference_contained}  
        )  
        if split == "train":
            return synthetic_gsm8k_dataset
        elif split == "test":
            return synthetic_gsm8k_test_dataset
    elif dataset_name == "argilla/distilabel-intel-orca-dpo-pairs":
        distilabel_intel_orca_dpo_pairs_dataset = ConfigurableDataset(
            dataset_name="argilla/distilabel-intel-orca-dpo-pairs",
            tokenizer_name=tokenizer_name,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            split="train",
            subset_size=subset_size,
            instruction_column="system",
            input_columns=["input"],
            output_columns=["chosen"],
            answer_parser=lambda x: x.strip().lower(),
            use_cache=False,
            subset_name="default",
            metrics={'reference_contained': reference_contained}
        )
        if split == "train":
            return distilabel_intel_orca_dpo_pairs_dataset
        elif split == "test":
            raise ValueError("No test split available for this dataset")
    elif dataset_name == "lmsys/lmsys-arena-human-preference-55k":
        lmsys_arena_human_preference_55k_dataset = ConfigurableDataset(
            dataset_name="lmsys/lmsys-arena-human-preference-55k",
            tokenizer_name=tokenizer_name,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            split="train",
            subset_size=subset_size,
            instruction="",
            input_columns=["prompt"],
            output_columns=["response_a", "response_b"],
            input_parser=lambda x: x["prompt"].replace('["', "").replace('"]', "").strip(),
            answer_parser=lambda x: x.strip().lower(),
            use_cache=False,
            subset_name="default",
            metrics={'reference_contained': reference_contained}
        )
        lmsys_arena_human_preference_55k_dataset.construct_target = custom_target_constructor_for_lmsys_arena_human_preference_55k
        if split == "train":
            return lmsys_arena_human_preference_55k_dataset
        elif split == "test":
            raise ValueError("No test split available for this dataset")
        
    elif dataset_name == "mosaicml/dolly_hhrlhf":
        dolly_hhrlhf_dataset = ConfigurableDataset(
            dataset_name="mosaicml/dolly_hhrlhf",
            tokenizer_name=tokenizer_name,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            split="train",
            subset_size=subset_size,
            instruction="",
            input_columns=["prompt"],
            output_columns=["response"],
            answer_parser=lambda x: x.strip().lower(),
            input_parser=lambda x: x["prompt"].replace("Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction:", "").replace("### Response:", "").replace("Human: ", "").replace("User: ", "").strip(),
            use_cache=False,
            subset_name="default",
            metrics={'reference_contained': reference_contained}
        )
        if split == "train":
            return dolly_hhrlhf_dataset
        elif split == "test":
            raise ValueError("No test split available for this dataset")
        
    elif dataset_name == "tatsu-lab/alpaca_eval":
        alpaca_eval_dataset = ConfigurableDataset(
            dataset_name="tatsu-lab/alpaca_eval",
            tokenizer_name=tokenizer_name,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            split="eval",
            subset_size=subset_size,
            instruction="",
            input_columns=["instruction"],
            output_columns=["output"],
            answer_parser=lambda x: x.strip().lower(),
            use_cache=False,
            subset_name="alpaca_eval_gpt4_baseline",
            metrics={'reference_contained': reference_contained}
        )
        if split == "train":
            return alpaca_eval_dataset
        elif split == "test":
            raise ValueError("No test split available for this dataset")
        
    elif dataset_name == "Norquinal/claude_evol_instruct_100k":
        claude_evol_instruct_100k_dataset = ConfigurableDataset(
            dataset_name="Norquinal/claude_evol_instruct_100k",
            tokenizer_name=tokenizer_name,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            split="train",
            subset_size=subset_size,
            instruction="",
            input_columns=["instruction"],
            output_columns=["output"],
            answer_parser=lambda x: x.strip().lower(),
            use_cache=False,
            subset_name="default",
            metrics={'reference_contained': reference_contained}
        )
        if split == "train":
            return claude_evol_instruct_100k_dataset
        elif split == "test":
            raise ValueError("No test split available for this dataset")
        
    if dataset_name == "mlabonne/Evol-Instruct-Python-26k":
        mlabonne_evol_instruct_python_26k_dataset = ConfigurableDataset(
            dataset_name="mlabonne/Evol-Instruct-Python-26k",
            tokenizer_name=tokenizer_name,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            split="train",
            subset_size=subset_size,
            instruction="",
            input_columns=["instruction"],
            output_columns=["output"],
            answer_parser=lambda x: x.strip().lower(),
            use_cache=False,
            subset_name="default",
            metrics={'reference_contained': reference_contained}
        )
        if split == "train":
            return mlabonne_evol_instruct_python_26k_dataset
        elif split == "test":
            raise ValueError("No test split available for this dataset")
        
    if dataset_name == "iamtarun/python_code_instructions_18k_alpaca":
        iamtarun_python_code_instructions_18k_alpaca_dataset = ConfigurableDataset(
            dataset_name="iamtarun/python_code_instructions_18k_alpaca",
            tokenizer_name=tokenizer_name,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            split="train",
            subset_size=subset_size,
            instruction="",
            input_columns=["instruction", "input"],
            output_columns=["output"],
            answer_parser=lambda x: x.strip().lower(),
            use_cache=False,
            subset_name="default",
            metrics={'reference_contained': reference_contained}
        )
        if split == "train":
            return iamtarun_python_code_instructions_18k_alpaca_dataset
        elif split == "test":
            raise ValueError("No test split available for this dataset")
        
    if dataset_name == "lighteval/mmlu":
        lighteval_mmlu_dataset = ConfigurableDataset(
            dataset_name="lighteval/mmlu",
            tokenizer_name=tokenizer_name,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            split="auxiliary_train",
            subset_size=subset_size,
            instruction="Choose the answer correctly from the given options. Output the answer as a single number. There are 4 options (0, 1, 2, or 3).",
            input_columns=["question", "choices"],
            output_columns=["answer"],
            input_parser=lambda x: f"{x['question']}\nOptions:\n0. {x['choices'][0]}\n1. {x['choices'][1]}\n2. {x['choices'][2]}\n3. {x['choices'][3]}",
            answer_parser=lambda x: x.strip().lower(),
            use_cache=False,
            subset_name="all",
            metrics={'reference_contained': reference_contained}
        )
        
        lighteval_mmlu_test_dataset = ConfigurableDataset(
            dataset_name="lighteval/mmlu",
            tokenizer_name=tokenizer_name,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            split="test",
            subset_size=subset_size,
            instruction="Choose the answer correctly from the given options. Output the answer as a single number. There are 4 options (0, 1, 2, or 3).",
            input_columns=["question", "choices"],
            output_columns=["answer"],
            input_parser=lambda x: f"{x['question']}\nOptions:\n0. {x['choices'][0]}\n1. {x['choices'][1]}\n2. {x['choices'][2]}\n3. {x['choices'][3]}",
            answer_parser=lambda x: x.strip().lower(),
            use_cache=False,
            subset_name="all",
            metrics={'reference_contained': reference_contained}
        )
        if split == "train":
            return lighteval_mmlu_dataset
        elif split == "test":
            return lighteval_mmlu_test_dataset
        
    if dataset_name == "mlabonne/orpo-dpo-mix-40k-flat":    
        mlabonne_orpo_dpo_mix_40k_flat_dataset = ConfigurableDataset(
            dataset_name="mlabonne/orpo-dpo-mix-40k-flat",
            tokenizer_name=tokenizer_name,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            split="train",
            subset_size=subset_size,
            instruction="",
            input_columns=["prompt"],
            output_columns=["chosen"],
            answer_parser=lambda x: x.strip().lower(),
            use_cache=False,
            subset_name="default",
            metrics={'reference_contained': reference_contained}
        )
        if split == "train":
            return mlabonne_orpo_dpo_mix_40k_flat_dataset
        elif split == "test":
            raise ValueError("No test split available for this dataset")
        
    if dataset_name == "jondurbin/truthy-dpo-v0.1":
        jondurbin_truthy_dpo_v0_1_dataset = ConfigurableDataset(
            dataset_name="jondurbin/truthy-dpo-v0.1",
            tokenizer_name=tokenizer_name,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            split="train",
            subset_size=subset_size,
            instruction_column="system",
            input_columns=["prompt"],
            output_columns=["chosen"],
            answer_parser=lambda x: x.strip().lower(),
            use_cache=False,
            subset_name="default",
            metrics={'reference_contained': reference_contained}
        )
        if split == "train":
            return jondurbin_truthy_dpo_v0_1_dataset
        elif split == "test":
            raise ValueError("No test split available for this dataset")
        
    if dataset_name == "argilla/distilabel-math-preference-dpo":
        argilla_distilabel_math_preference_dpo_dataset = ConfigurableDataset(
            dataset_name="argilla/distilabel-math-preference-dpo",
            tokenizer_name=tokenizer_name,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            split="train",
            subset_size=subset_size,
            instruction="",
            input_columns=["instruction"],
            output_columns=["chosen_response"],
            answer_parser=lambda x: x.strip().lower(),
            use_cache=False,
            subset_name="default",
            metrics={'reference_contained': reference_contained}
        )
        if split == "train":
            return argilla_distilabel_math_preference_dpo_dataset
        elif split == "test":
            raise ValueError("No test split available for this dataset")
        
    if dataset_name == "facebook/Self-taught-evaluator-DPO-data":
        facebook_self_taught_evaluator_dpo_data_dataset = ConfigurableDataset(
            dataset_name="facebook/Self-taught-evaluator-DPO-data",
            tokenizer_name=tokenizer_name,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            split="train",
            subset_size=subset_size,
            instruction="",
            input_columns=["src"],
            output_columns=["tgt_chosen"],
            answer_parser=lambda x: x.strip().lower(),
            use_cache=False,
            subset_name="default",
            metrics={'reference_contained': reference_contained}
        )
        if split == "train":
            return facebook_self_taught_evaluator_dpo_data_dataset
        elif split == "test":
            raise ValueError("No test split available for this dataset")
        
    if dataset_name == "argilla/magpie-ultra-v0.1":
        argilla_magpie_ultra_v0_1_dataset = ConfigurableDataset(
            dataset_name="argilla/magpie-ultra-v0.1",
            tokenizer_name=tokenizer_name,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            split="train",
            subset_size=subset_size,
            instruction="",
            input_columns=["instruction"],
            output_columns=["response"],
            answer_parser=lambda x: x.strip().lower(),
            use_cache=False,
            subset_name="default",
            metrics={'reference_contained': reference_contained}
        )
        if split == "train":
            return argilla_magpie_ultra_v0_1_dataset
        elif split == "test":
            raise ValueError("No test split available for this dataset")
        
    if dataset_name == "arcee-ai/EvolKit-20k":
        arcee_ai_evolkit_20k_dataset = ConfigurableDataset(
            dataset_name="arcee-ai/EvolKit-20k",
            tokenizer_name=tokenizer_name,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            split="train",
            subset_size=subset_size,
            instruction="",
            input_columns=["conversations"],
            output_columns=["conversations"],
            input_parser=lambda sample: "\n".join([f"{conv['value']}\n" if conv['from'] == 'human' else f"Answer:\n{conv['value']}\n\n" for conv in sample['conversations'][:-1]]),
            answer_parser=lambda x: x.strip().lower(),
            use_cache=False,
            subset_name="default",
            metrics={'reference_contained': reference_contained}
        )
        arcee_ai_evolkit_20k_dataset.construct_target = lambda sample: sample['conversations'][-1]['value']
        
        if split == "train":
            return arcee_ai_evolkit_20k_dataset
        elif split == "test":
            raise ValueError("No test split available for this dataset")
        
    if dataset_name == "WizardLMTeam/WizardLM_evol_instruct_V2_196k":
        wizardlmteam_wizardlm_evol_instruct_v2_196k_dataset = ConfigurableDataset(
            dataset_name="WizardLMTeam/WizardLM_evol_instruct_V2_196k",
            tokenizer_name=tokenizer_name,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            split="train",
            subset_size=subset_size,
            instruction="",
            input_columns=["conversations"],
            output_columns=["conversations"],
            input_parser=lambda sample: "\n".join([f"{conv['value']}\n" if conv['from'] == 'human' else f"Answer:\n{conv['value']}\n\n" for conv in sample['conversations'][:-1]]),
            answer_parser=lambda x: x.strip().lower(),
            use_cache=False,
            subset_name="default",
            metrics={'reference_contained': reference_contained}
        )
        wizardlmteam_wizardlm_evol_instruct_v2_196k_dataset.construct_target = lambda sample: sample['conversations'][-1]['value'] 
        if split == "train":
            return wizardlmteam_wizardlm_evol_instruct_v2_196k_dataset
        elif split == "test":
            raise ValueError("No test split available for this dataset")
        
    if dataset_name == "TIGER-Lab/MMLU-Pro":
        tiger_lab_mmlu_pro_test_dataset = ConfigurableDataset(
            dataset_name="TIGER-Lab/MMLU-Pro",
            tokenizer_name=tokenizer_name,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            split="test",
            subset_size=subset_size,
            instruction="Choose the most appropriate answer from the given options. Output the answer as a single letter. There could be upto 10 options from A to J, output the answer as a single letter (A, B, C, D, E, F, G, H, I, or J).",
            input_columns=["question", "options"],
            output_columns=["answer"],
            input_parser=lambda x: f"{x['question']}\nOptions:\n" + "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(x['options'])]),
            answer_parser=lambda x: x.strip().lower(),
            use_cache=False,
            subset_name="default",
            metrics={'reference_contained': reference_contained}
        )
        if split == "train":
            tiger_lab_mmlu_pro_dataset = ConfigurableDataset(
                dataset_name="TIGER-Lab/MMLU-Pro",
                tokenizer_name=tokenizer_name,
                max_input_length=max_input_length,
                max_output_length=max_output_length,
                split="validation",
                subset_size=subset_size,
                instruction="Choose the most appropriate answer from the given options. Output the answer as a single letter. There could be upto 10 options from A to J, output the answer as a single letter (A, B, C, D, E, F, G, H, I, or J).",
                input_columns=["question", "options"],
                output_columns=["answer"],
                input_parser=lambda x: f"{x['question']}\nOptions:\n" + "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(x['options'])]),
                answer_parser=lambda x: x.strip().lower(),
                use_cache=False,
                subset_name="default",
                metrics={'reference_contained': reference_contained}
            )
            return tiger_lab_mmlu_pro_dataset
        elif split == "test":
            return tiger_lab_mmlu_pro_test_dataset
        
    if dataset_name == "TIGER-Lab/MMLU-Pro-COT":
        
        tiger_lab_mmlu_pro_test_dataset = ConfigurableDataset(
            dataset_name="TIGER-Lab/MMLU-Pro",
            tokenizer_name=tokenizer_name,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            split="test",
            subset_size=subset_size,
            instruction="Answer the following multiple choice question by thinking step by step. Your reply should follow the format: [YOUR THOUGHTS] <answer>[YOUR CHOICE]</answer>. There might be upto 10 options from A to J, output the answer as a single letter (A, B, C, D, E, F, G, H, I, or J).",
            input_columns=["question", "options"],
            output_columns=["cot_content", "answer"],
            answer_parser=lambda x: x.split('<answer>')[-1].split('</answer>')[0].strip().lower() if '<answer>' in x and '</answer>' in x else x.strip().lower(),
            input_parser=lambda x: f"{x['question']}\nOptions:\n" + "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(x['options'])]),
            use_cache=False,
            subset_name="default",
            metrics={'reference_contained': reference_contained}
        )
        construct_target = lambda x: f"{x['cot_content']} <answer>{x['answer']}</answer>"
        
        tiger_lab_mmlu_pro_test_dataset.construct_target = construct_target
        if split == "train":
            tiger_lab_mmlu_pro_dataset = ConfigurableDataset(
                dataset_name="TIGER-Lab/MMLU-Pro",
                tokenizer_name=tokenizer_name,
                max_input_length=max_input_length,
                max_output_length=max_output_length,
                split="validation",
                subset_size=subset_size,
                instruction="Answer the following multiple choice question by thinking step by step. Your reply should follow the format: [YOUR THOUGHTS] <answer>[YOUR CHOICE]</answer>. There might be upto 10 options from A to J, output the answer as a single letter (A, B, C, D, E, F, G, H, I, or J).",
                input_columns=["question", "options"],
                output_columns=["cot_content", "answer"],
                input_parser=lambda x: f"{x['question']}\nOptions:\n" + "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(x['options'])]),
                answer_parser=lambda x: x.split('<answer>')[-1].split('</answer>')[0].strip().lower() if '<answer>' in x and '</answer>' in x else x.strip().lower(),
                use_cache=False,
                subset_name="default",
                metrics={'reference_contained': reference_contained}
            )
            tiger_lab_mmlu_pro_dataset.construct_target = construct_target
            return tiger_lab_mmlu_pro_dataset
        elif split == "test":
            return tiger_lab_mmlu_pro_test_dataset
        
    if dataset_name == "lighteval/MATH-Hard":
        lighteval_math_hard_dataset = ConfigurableDataset(
            dataset_name="lighteval/MATH-Hard",
            tokenizer_name=tokenizer_name,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            split="train",
            subset_size=subset_size,
            instruction=math_instruction,
            input_columns=["problem", "type"],
            output_columns=["solution"],
            answer_parser=competition_math_answer_parser,
            input_parser=custom_prompt_constructor_math,
            use_cache=False,
            subset_name="default",
            metrics={'reference_contained': reference_contained}
        )
        lighteval_math_hard_test_dataset = ConfigurableDataset(
            dataset_name="lighteval/MATH-Hard",
            tokenizer_name=tokenizer_name,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            split="test",
            subset_size=subset_size,
            instruction=math_instruction,
            input_columns=["problem", "type"],
            output_columns=["solution"],
            answer_parser=competition_math_answer_parser,
            input_parser=custom_prompt_constructor_math,
            use_cache=False,
            subset_name="default",
            metrics={'reference_contained': reference_contained}
        )
        if split == "train":
            return lighteval_math_hard_dataset
        elif split == "test":
            return lighteval_math_hard_test_dataset
    
    if dataset_name == "tau/commonsense_qa":
        tau_commonsense_qa_dataset = ConfigurableDataset(
            dataset_name="tau/commonsense_qa",
            tokenizer_name=tokenizer_name,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            split="train",
            subset_size=subset_size,
            instruction="",
            input_columns=["question", "choices"],
            output_columns=["answerKey"],
            input_parser=lambda x: f"{x['question']}\nOptions:\n" + "\n".join([f"{label}. {text}" for label, text in zip(x['choices']['label'], x['choices']['text'])]),
            answer_parser=lambda x: x.strip().lower(),
            use_cache=False,
            subset_name="default",
            metrics={'reference_contained': reference_contained}
        )
        
        tau_commonsense_qa_test_dataset = ConfigurableDataset(  
            dataset_name="tau/commonsense_qa",
            tokenizer_name=tokenizer_name,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            split="validation",
            subset_size=subset_size,
            instruction="",
            input_columns=["question", "choices"],
            output_columns=["answerKey"],
            input_parser=lambda x: f"{x['question']}\nOptions:\n" + "\n".join([f"{label}. {text}" for label, text in zip(x['choices']['label'], x['choices']['text'])]),
            answer_parser=lambda x: x.strip().lower(),
            use_cache=False,
            subset_name="default",
            metrics={'reference_contained': reference_contained}
        )
        if split == "train":
            return tau_commonsense_qa_dataset
        elif split == "test":
            return tau_commonsense_qa_test_dataset
        
    if dataset_name == "amanrangapur/Fin-Fact":
        amanrangapur_fin_fact_dataset = ConfigurableDataset(
            dataset_name="amanrangapur/Fin-Fact",
            tokenizer_name=tokenizer_name,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            split="train",
            subset_size=subset_size,
            instruction="Fact check the following claim and then answer with true or false only.",
            input_columns=["claim"],
            output_columns=["label"],
            answer_parser=lambda x: x.strip().lower(),
            use_cache=False,
            subset_name="default",
            metrics={'reference_contained': reference_contained}
        )
        
        if split == "train":
            return ValueError("No train split available for this dataset")
        elif split == "test":
            return amanrangapur_fin_fact_dataset
        
        
    if dataset_name == "EleutherAI/truthful_qa_mc":
        eleutherai_truthful_qa_mc_dataset = ConfigurableDataset(
            dataset_name="EleutherAI/truthful_qa_mc",
            tokenizer_name=tokenizer_name,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            split="validation",
            subset_size=subset_size,
            instruction="Answer the following multiple choice question and choose the answer correctly from the given options. Output the answer as a single letter. There will be 4 options (A, B, C, or D) of which only one is correct.",
            input_columns=["question", "choices"],
            output_columns=["label"],
            input_parser=lambda x: f"Question: {x['question']}\nChoices: {'\n'.join([f'{chr(65+i)}. {choice}' for i, choice in enumerate(x['choices'])])}",
            answer_parser=lambda x: x.strip().lower(),
            use_cache=False,
            subset_name="multiple_choice",
            metrics={'reference_contained': reference_contained}
        )
        if split == "train":
            raise ValueError("No train split available for this dataset")
        elif split == "test":
            return eleutherai_truthful_qa_mc_dataset
        
    if dataset_name == "FinanceMTEB/financial_phrasebank":
        finance_mteb_financial_phrasebank_dataset = ConfigurableDataset(
            dataset_name="FinanceMTEB/financial_phrasebank",
            tokenizer_name=tokenizer_name,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            split="train",
            subset_size=subset_size,
            instruction="Classify the following financial phrase into one of the following categories: 'Positive', 'Negative', or 'Neutral'. Your output should be one of the following: positive, negative, or neutral.",
            input_columns=["text"],
            output_columns=["label_text"],
            answer_parser=lambda x: x.strip().lower(),
            use_cache=False,
            subset_name="default",
            metrics={'reference_contained': reference_contained}
        )
        
        finance_mteb_financial_phrasebank_test_dataset = ConfigurableDataset(
            dataset_name="FinanceMTEB/financial_phrasebank",
            tokenizer_name=tokenizer_name,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            split="test",
            subset_size=subset_size,
            instruction="Classify the following financial phrase into one of the following categories: 'Positive', 'Negative', or 'Neutral'. Your output should be one of the following: positive, negative, or neutral.",
            input_columns=["text"],
            output_columns=["label_text"],
            answer_parser=lambda x: x.strip().lower(),
            use_cache=False,
            subset_name="default",
            metrics={'reference_contained': reference_contained}
        )
        
        if split == "train":
            return finance_mteb_financial_phrasebank_dataset
        elif split == "test":
            return finance_mteb_financial_phrasebank_test_dataset
        
        
        
from datasets import concatenate_datasets  

# Assume that ConfigurableDataset and BaseDataset are already defined.  
  
class CombinedConfigurableDataset:  
    def __init__(self, datasets):  
        """  
        Initialize the CombinedConfigurableDataset with a list of ConfigurableDataset instances.  
          
        Parameters:  
        - datasets (list): A list of ConfigurableDataset instances to combine.  
        """  
        self.datasets = datasets  
        self.dataset_lengths = [len(ds) for ds in datasets if ds is not None]  
        self.cumulative_lengths = self._compute_cumulative_lengths()  
        self.total_length = sum(self.dataset_lengths)  
        self.tokenizer = datasets[0].tokenizer
        self.extract_answer = lambda x: x.strip().lower()
        # Optionally, store other attributes or metadata if needed.  
      
    def _compute_cumulative_lengths(self):  
        """  
        Compute cumulative lengths to map global indices to local dataset indices.  
          
        Returns:  
        - cumulative_lengths (list): A list containing the cumulative lengths.  
        """  
        cumulative_lengths = []  
        total = 0  
        for length in self.dataset_lengths:  
            total += length  
            cumulative_lengths.append(total)  
        return cumulative_lengths  
  
    def __len__(self):  
        """  
        Return the total number of samples across all datasets.  
        """  
        return self.total_length  
  
    def __getitem__(self, idx):  
        """  
        Retrieve an item by global index, mapping it to the appropriate dataset.  
          
        Parameters:  
        - idx (int): The global index of the item.  
          
        Returns:  
        - item (dict): The retrieved item from the appropriate dataset.  
        """  
        if idx < 0 or idx >= self.total_length:  
            raise IndexError("Index out of range")  
          
        # Find the dataset the index belongs to  
        dataset_idx = self._find_dataset_idx(idx)  
        if dataset_idx is None:  
            raise IndexError("Index out of range")  
          
        # Adjust index to local dataset index  
        if dataset_idx == 0:  
            local_idx = idx  
        else:  
            local_idx = idx - self.cumulative_lengths[dataset_idx - 1]  
          
        # Retrieve item from the correct dataset  
        item = self.datasets[dataset_idx][local_idx]  
        return item  
  
    def _find_dataset_idx(self, idx):  
        """  
        Find the index of the dataset that contains the given global index.  
          
        Parameters:  
        - idx (int): The global index.  
          
        Returns:  
        - dataset_idx (int): The index of the dataset in self.datasets.  
        """  
        for i, cumulative_length in enumerate(self.cumulative_lengths):  
            if idx < cumulative_length:  
                return i  
        return None  # Should not reach here if idx is within range  
  
    def __iter__(self):  
        """  
        Iterate over all items in all datasets sequentially.  
          
        Yields:  
        - item (dict): The next item in the sequence.  
        """  
        for dataset in self.datasets:  
            for item in dataset:  
                yield item  
                
    def get_evaluation_metrics(self):  
        def exact_match(predictions, references):  
            correct = sum(pred == ref for pred, ref in zip(predictions, references) if pred is not None and ref is not None)  
            total = sum(1 for pred, ref in zip(predictions, references) if pred is not None and ref is not None)  
            return {'exact_match': correct / total if total > 0 else 0}  
        
        default_metrics = {'exact_match': exact_match}  
        return default_metrics  

def create_pretraining_dataset(  
    dataset_name,
    tokenizer_name,  
    max_input_length,  
    max_output_length,  
    split="train",  # Default to 'train' split  
    subset_size=None,  
    **extra_kwargs
):  
    """  
    Create a pretraining dataset by combining multiple ConfigurableDataset instances.  
      
    Parameters:  
    - tokenizer_name (str): The name of the tokenizer to use.  
    - max_input_length (int): Maximum length of the input sequences.  
    - max_output_length (int): Maximum length of the output sequences.  
    - subset_size (int or None): If set, limits each dataset to a subset.  
    - split (str): The dataset split to use ('train' or 'test').  
      
    Returns:  
    - combined_dataset (CombinedConfigurableDataset): The combined dataset.  
    """  
    # List of datasets to include  
    if split == "train": 
        dataset_names = [  
            "hendrycks/competition_math",  
            "Rowan/hellaswag",
            "gsm8k",  
            # "flammenai/casual-conversation-DPO-rejected",
            "flammenai/casual-conversation-DPO",
            
            "Isotonic/human_assistant_conversation",
            # "gsm8k-reasoning",  
            # "synthetic-gsm8k",  
            "argilla/distilabel-intel-orca-dpo-pairs",  
           "lmsys/lmsys-arena-human-preference-55k",  
            "mosaicml/dolly_hhrlhf",  
            "tatsu-lab/alpaca_eval",  
            "Norquinal/claude_evol_instruct_100k",  
            # "mlabonne/Evol-Instruct-Python-26k",
            "mlabonne/orpo-dpo-mix-40k-flat",  
            # "jondurbin/truthy-dpo-v0.1",  
            "argilla/distilabel-math-preference-dpo",  
            # "facebook/Self-taught-evaluator-DPO-data",  
            "argilla/magpie-ultra-v0.1",  
            # "arcee-ai/EvolKit-20k",  
            
            # "WizardLMTeam/WizardLM_evol_instruct_V2_196k",  
            "TIGER-Lab/MMLU-Pro",  
            "TIGER-Lab/MMLU-Pro",  
            "lighteval/MATH-Hard",  
            "tau/commonsense_qa",  
            "FinanceMTEB/financial_phrasebank",
            "lighteval/mmlu",
            "lighteval/mmlu",
            # "TIGER-Lab/MMLU-Pro-COT"  
        ]  
    elif split == "test":
        dataset_names = [  
            "Rowan/hellaswag",
            "hendrycks/competition_math",  
            "gsm8k",  
            "synthetic-gsm8k",  
            "lighteval/mmlu",
            "TIGER-Lab/MMLU-Pro", 
            "TIGER-Lab/MMLU-Pro-COT", 
            "lighteval/MATH-Hard",  
            "tau/commonsense_qa",  
            "amanrangapur/Fin-Fact",
            "EleutherAI/truthful_qa_mc",
            "FinanceMTEB/financial_phrasebank"  
        ]  
        
      
    datasets = []  
    for dataset_name in dataset_names:  
        try:  
            # Assume get_dataset loads and returns a ConfigurableDataset instance  
            dataset = get_dataset(  
                dataset_name,  
                tokenizer_name,  
                max_input_length,  
                max_output_length,  
                split,  
                subset_size  
            )  
            datasets.append(dataset)  
            print(f"Successfully added {dataset_name} with length {len(dataset)} and subset size {subset_size} to the pretraining dataset.")  
        except Exception as e:  
            print(f"Error loading {dataset_name}: {str(e)}")  
      
    if not datasets:  
        raise ValueError("No datasets were successfully loaded.")  
      
    # Create the combined dataset using the CombinedConfigurableDataset class  
    combined_dataset = CombinedConfigurableDataset(datasets)  
    print(f"Created pretraining dataset with {len(combined_dataset)} samples.")  
    return combined_dataset  


if __name__ == "__main__":
    tokenizer_name = "meta-llama/Llama-3.2-1B-Instruct"
    
    pretraining_dataset = create_pretraining_dataset(  
        tokenizer_name="meta-llama/Llama-3.2-1B-Instruct",  
        max_input_length=1024,  
        max_output_length=2048,  
        subset_size=None  # Set to a number if you want to limit the size of each dataset  
    )  
    print("Length of pretraining dataset: ", len(pretraining_dataset))
    # example
    sample = pretraining_dataset[0]
    print(f"Input IDs shape: {sample['input_ids'].shape}")
    print(f"Attention Mask shape: {sample['attention_mask'].shape}")
    print(f"Labels shape: {sample['labels'].shape}")
    print(f"Labels Attention Mask shape: {sample['labels_attention_mask'].shape}")
    print(f"Reference Answer: {sample['reference_answer']}")
    
    

    
    alpaca_eval_dataset = get_dataset("tatsu-lab/alpaca_eval", tokenizer_name, 2048, 2048, "train", 100)
    sample = alpaca_eval_dataset[0]  
    print(f"Input IDs shape: {sample['input_ids'].shape}")  
    print(f"Attention Mask shape: {sample['attention_mask'].shape}")  
    print(f"Labels shape: {sample['labels'].shape}")  
    print(f"Labels Attention Mask shape: {sample['labels_attention_mask'].shape}")  
    print(f"Reference Answer: {sample['reference_answer']}")
    
    # To get the evaluation metrics:
    metrics = alpaca_eval_dataset.get_evaluation_metrics()
    print(f"Largest input IDs length: {alpaca_eval_dataset.get_largest_input_ids_length()}")
    print(f"Largest labels length: {alpaca_eval_dataset.get_largest_labels_length()}")
    
    gsm8k_dataset = get_dataset("gsm8k", tokenizer_name, 2048, 2048, "train", 100)
    sample = gsm8k_dataset[0]  
    print(f"Input IDs shape: {sample['input_ids'].shape}")  
    print(f"Attention Mask shape: {sample['attention_mask'].shape}")  
    print(f"Labels shape: {sample['labels'].shape}")  
    print(f"Labels Attention Mask shape: {sample['labels_attention_mask'].shape}")  
    print(f"Reference Answer: {sample['reference_answer']}")  
    
    # To get the evaluation metrics:  
    metrics = gsm8k_dataset.get_evaluation_metrics()  
    print(f"Available metrics: {list(metrics.keys())}")  
    print(f"Largest input IDs length: {gsm8k_dataset.get_largest_input_ids_length()}")
    print(f"Largest labels length: {gsm8k_dataset.get_largest_labels_length()}")
    
    gsm8k_reasoning_dataset = get_dataset("gsm8k-reasoning", tokenizer_name, 2048, 2048, "train", 100)
    sample = gsm8k_reasoning_dataset[0]  
    print(f"Input IDs shape: {sample['input_ids'].shape}")  
    print(f"Attention Mask shape: {sample['attention_mask'].shape}")  
    print(f"Labels shape: {sample['labels'].shape}")  
    print(f"Labels Attention Mask shape: {sample['labels_attention_mask'].shape}")  
    print(f"Reference Answer: {sample['reference_answer']}")  
    
    # To get the evaluation metrics:  
    metrics = gsm8k_reasoning_dataset.get_evaluation_metrics()  
    print(f"Largest input IDs length: {gsm8k_reasoning_dataset.get_largest_input_ids_length()}")
    print(f"Largest labels length: {gsm8k_reasoning_dataset.get_largest_labels_length()}")



    synthetic_gsm8k_dataset = get_dataset("synthetic-gsm8k", tokenizer_name, 2048, 2048, "train", 100)
    sample = synthetic_gsm8k_dataset[0]  
    print(f"Input IDs shape: {sample['input_ids'].shape}")  
    print(f"Attention Mask shape: {sample['attention_mask'].shape}")  
    print(f"Labels shape: {sample['labels'].shape}")  
    print(f"Labels Attention Mask shape: {sample['labels_attention_mask'].shape}")  
    print(f"Reference Answer: {sample['reference_answer']}")  
    
    # To get the evaluation metrics:
    metrics = synthetic_gsm8k_dataset.get_evaluation_metrics()
    print(f"Largest input IDs length: {synthetic_gsm8k_dataset.get_largest_input_ids_length()}")
    print(f"Largest labels length: {synthetic_gsm8k_dataset.get_largest_labels_length()}")

    math_train_dataset = get_dataset("hendrycks/competition_math", tokenizer_name, 2048, 2048, "train", 100)
    math_test_dataset = get_dataset("hendrycks/competition_math", tokenizer_name, 2048, 2048, "test", 100)
    for dataset_name, dataset in [("Train", math_train_dataset), ("Test", math_test_dataset)]:  
        print(f"\n{dataset_name} Dataset Sample:")  
        sample = dataset[0]  
        print(f"Input IDs shape: {sample['input_ids'].shape}")  
        print(f"Attention Mask shape: {sample['attention_mask'].shape}")  
        print(f"Labels shape: {sample['labels'].shape}")  
        print(f"Labels Attention Mask shape: {sample['labels_attention_mask'].shape}")  
        print(f"Reference Answer: {sample['reference_answer']}")  
    
        print(f"\nLargest input IDs length: {dataset.get_largest_input_ids_length()}")  
        print(f"Largest labels length: {dataset.get_largest_labels_length()}")  
    
        # Decode a sample input and output  
        tokenizer = dataset.tokenizer  
        decoded_input = tokenizer.decode(sample['input_ids'])  
        decoded_output = tokenizer.decode(sample['labels'])  
        print(f"\nSample Input (truncated to 500 chars):\n{decoded_input[:500]}...")  
        print(f"\nSample Output (truncated to 500 chars):\n{decoded_output[:500]}...")  
        
    distilabel_intel_orca_dpo_pairs_dataset = get_dataset("argilla/distilabel-intel-orca-dpo-pairs", tokenizer_name, 2048, 2048, "train", 100)
    sample = distilabel_intel_orca_dpo_pairs_dataset[0]  
    print(f"Input IDs shape: {sample['input_ids'].shape}")  
    print(f"Attention Mask shape: {sample['attention_mask'].shape}")  
    print(f"Labels shape: {sample['labels'].shape}")  
    print(f"Labels Attention Mask shape: {sample['labels_attention_mask'].shape}")  
    print(f"Reference Answer: {sample['reference_answer']}")    
    
    # To get the evaluation metrics:
    metrics = distilabel_intel_orca_dpo_pairs_dataset.get_evaluation_metrics()
    print(f"Largest input IDs length: {distilabel_intel_orca_dpo_pairs_dataset.get_largest_input_ids_length()}")
    print(f"Largest labels length: {distilabel_intel_orca_dpo_pairs_dataset.get_largest_labels_length()}")
    
    lmsys_arena_human_preference_55k_dataset = get_dataset("lmsys/lmsys-arena-human-preference-55k", tokenizer_name, 2048, 2048, "train", 100)  
    sample = lmsys_arena_human_preference_55k_dataset[0]  
    print(f"Input IDs shape: {sample['input_ids'].shape}")  
    print(f"Attention Mask shape: {sample['attention_mask'].shape}")  
    print(f"Labels shape: {sample['labels'].shape}")  
    print(f"Labels Attention Mask shape: {sample['labels_attention_mask'].shape}")  
    print(f"Reference Answer: {sample['reference_answer']}")    
    
    # To get the evaluation metrics:
    metrics = lmsys_arena_human_preference_55k_dataset.get_evaluation_metrics()
    print(f"Largest input IDs length: {lmsys_arena_human_preference_55k_dataset.get_largest_input_ids_length()}")
    print(f"Largest labels length: {lmsys_arena_human_preference_55k_dataset.get_largest_labels_length()}")
    
    
    dolly_hhrlhf_dataset = get_dataset("mosaicml/dolly_hhrlhf", tokenizer_name, 2048, 2048, "train", 100)
    sample = dolly_hhrlhf_dataset[0]  
    print(f"Input IDs shape: {sample['input_ids'].shape}")  
    print(f"Attention Mask shape: {sample['attention_mask'].shape}")  
    print(f"Labels shape: {sample['labels'].shape}")  
    print(f"Labels Attention Mask shape: {sample['labels_attention_mask'].shape}")  
    print(f"Reference Answer: {sample['reference_answer']}")
    
    # To get the evaluation metrics:
    metrics = dolly_hhrlhf_dataset.get_evaluation_metrics()
    print(f"Largest input IDs length: {dolly_hhrlhf_dataset.get_largest_input_ids_length()}")
    print(f"Largest labels length: {dolly_hhrlhf_dataset.get_largest_labels_length()}")
    
    
    
    claude_evol_instruct_100k_dataset = get_dataset("Norquinal/claude_evol_instruct_100k", tokenizer_name, 2048, 2048, "train", 100)
    sample = claude_evol_instruct_100k_dataset[0]  
    print(f"Input IDs shape: {sample['input_ids'].shape}")  
    print(f"Attention Mask shape: {sample['attention_mask'].shape}")  
    print(f"Labels shape: {sample['labels'].shape}")  
    print(f"Labels Attention Mask shape: {sample['labels_attention_mask'].shape}")  
    print(f"Reference Answer: {sample['reference_answer']}")
    
    # To get the evaluation metrics:
    metrics = claude_evol_instruct_100k_dataset.get_evaluation_metrics()
    print(f"Largest input IDs length: {claude_evol_instruct_100k_dataset.get_largest_input_ids_length()}")
    print(f"Largest labels length: {claude_evol_instruct_100k_dataset.get_largest_labels_length()}")
    
    
    mlabonne_evol_instruct_python_26k_dataset = get_dataset("mlabonne/Evol-Instruct-Python-26k", tokenizer_name, 2048, 2048, "train", 100)
    sample = mlabonne_evol_instruct_python_26k_dataset[0]  
    print(f"Input IDs shape: {sample['input_ids'].shape}")  
    print(f"Attention Mask shape: {sample['attention_mask'].shape}")  
    print(f"Labels shape: {sample['labels'].shape}")  
    print(f"Labels Attention Mask shape: {sample['labels_attention_mask'].shape}")  
    print(f"Reference Answer: {sample['reference_answer']}")
    
    # To get the evaluation metrics:
    metrics = mlabonne_evol_instruct_python_26k_dataset.get_evaluation_metrics()
    print(f"Largest input IDs length: {mlabonne_evol_instruct_python_26k_dataset.get_largest_input_ids_length()}")
    print(f"Largest labels length: {mlabonne_evol_instruct_python_26k_dataset.get_largest_labels_length()}")
    
    
    iamtarun_python_code_instructions_18k_alpaca_dataset = get_dataset("iamtarun/python_code_instructions_18k_alpaca", tokenizer_name, 2048, 2048, "train", 100)
    sample = iamtarun_python_code_instructions_18k_alpaca_dataset[0]  
    print(f"Input IDs shape: {sample['input_ids'].shape}")  
    print(f"Attention Mask shape: {sample['attention_mask'].shape}")  
    print(f"Labels shape: {sample['labels'].shape}")  
    print(f"Labels Attention Mask shape: {sample['labels_attention_mask'].shape}")  
    print(f"Reference Answer: {sample['reference_answer']}")
    
    # To get the evaluation metrics:
    metrics = iamtarun_python_code_instructions_18k_alpaca_dataset.get_evaluation_metrics()
    print(f"Largest input IDs length: {iamtarun_python_code_instructions_18k_alpaca_dataset.get_largest_input_ids_length()}")
    print(f"Largest labels length: {iamtarun_python_code_instructions_18k_alpaca_dataset.get_largest_labels_length()}")
    
    lighteval_mmlu_dataset = get_dataset("lighteval/mmlu", tokenizer_name, 2048, 2048, "train", 100)
    sample = lighteval_mmlu_dataset[0]  
    print(f"Input IDs shape: {sample['input_ids'].shape}")  
    print(f"Attention Mask shape: {sample['attention_mask'].shape}")  
    print(f"Labels shape: {sample['labels'].shape}")  
    print(f"Labels Attention Mask shape: {sample['labels_attention_mask'].shape}")  
    print(f"Reference Answer: {sample['reference_answer']}")
    
    # To get the evaluation metrics:
    metrics = lighteval_mmlu_dataset.get_evaluation_metrics()
    print(f"Largest input IDs length: {lighteval_mmlu_dataset.get_largest_input_ids_length()}")
    print(f"Largest labels length: {lighteval_mmlu_dataset.get_largest_labels_length()}")
    
    
    mlabonne_orpo_dpo_mix_40k_flat_dataset = get_dataset("mlabonne/orpo-dpo-mix-40k-flat", tokenizer_name, 2048, 2048, "train", 100)
    sample = mlabonne_orpo_dpo_mix_40k_flat_dataset[0]  
    print(f"Input IDs shape: {sample['input_ids'].shape}")  
    print(f"Attention Mask shape: {sample['attention_mask'].shape}")  
    print(f"Labels shape: {sample['labels'].shape}")  
    print(f"Labels Attention Mask shape: {sample['labels_attention_mask'].shape}")  
    print(f"Reference Answer: {sample['reference_answer']}")
    
    # To get the evaluation metrics:
    metrics = mlabonne_orpo_dpo_mix_40k_flat_dataset.get_evaluation_metrics()
    print(f"Largest input IDs length: {mlabonne_orpo_dpo_mix_40k_flat_dataset.get_largest_input_ids_length()}")
    print(f"Largest labels length: {mlabonne_orpo_dpo_mix_40k_flat_dataset.get_largest_labels_length()}")
    
    
    jondurbin_truthy_dpo_v0_1_dataset = get_dataset("jondurbin/truthy-dpo-v0.1", tokenizer_name, 2048, 2048, "train", 100)
    sample = jondurbin_truthy_dpo_v0_1_dataset[0]  
    print(f"Input IDs shape: {sample['input_ids'].shape}")  
    print(f"Attention Mask shape: {sample['attention_mask'].shape}")  
    print(f"Labels shape: {sample['labels'].shape}")  
    print(f"Labels Attention Mask shape: {sample['labels_attention_mask'].shape}")  
    print(f"Reference Answer: {sample['reference_answer']}")
    
    # To get the evaluation metrics:
    metrics = jondurbin_truthy_dpo_v0_1_dataset.get_evaluation_metrics()
    print(f"Largest input IDs length: {jondurbin_truthy_dpo_v0_1_dataset.get_largest_input_ids_length()}")
    print(f"Largest labels length: {jondurbin_truthy_dpo_v0_1_dataset.get_largest_labels_length()}")
    
    
    argilla_distilabel_math_preference_dpo_dataset = get_dataset("argilla/distilabel-math-preference-dpo", tokenizer_name, 2048, 2048, "train", 100)
    sample = argilla_distilabel_math_preference_dpo_dataset[0]  
    print(f"Input IDs shape: {sample['input_ids'].shape}")  
    print(f"Attention Mask shape: {sample['attention_mask'].shape}")  
    print(f"Labels shape: {sample['labels'].shape}")  
    print(f"Labels Attention Mask shape: {sample['labels_attention_mask'].shape}")  
    print(f"Reference Answer: {sample['reference_answer']}")
    
    # To get the evaluation metrics:
    metrics = argilla_distilabel_math_preference_dpo_dataset.get_evaluation_metrics()
    print(f"Largest input IDs length: {argilla_distilabel_math_preference_dpo_dataset.get_largest_input_ids_length()}")
    print(f"Largest labels length: {argilla_distilabel_math_preference_dpo_dataset.get_largest_labels_length()}")
    
    
    facebook_self_taught_evaluator_dpo_data_dataset = get_dataset("facebook/Self-taught-evaluator-DPO-data", tokenizer_name, 2048, 2048, "train", 100)
    sample = facebook_self_taught_evaluator_dpo_data_dataset[0]  
    print(f"Input IDs shape: {sample['input_ids'].shape}")  
    print(f"Attention Mask shape: {sample['attention_mask'].shape}")  
    print(f"Labels shape: {sample['labels'].shape}")  
    print(f"Labels Attention Mask shape: {sample['labels_attention_mask'].shape}")  
    print(f"Reference Answer: {sample['reference_answer']}")
    
    
    # To get the evaluation metrics:
    metrics = facebook_self_taught_evaluator_dpo_data_dataset.get_evaluation_metrics()
    print(f"Largest input IDs length: {facebook_self_taught_evaluator_dpo_data_dataset.get_largest_input_ids_length()}")
    print(f"Largest labels length: {facebook_self_taught_evaluator_dpo_data_dataset.get_largest_labels_length()}")
    
    
    argilla_magpie_ultra_v0_1_dataset = get_dataset("argilla/magpie-ultra-v0.1", tokenizer_name, 2048, 2048, "train", 100)
    sample = argilla_magpie_ultra_v0_1_dataset[0]  
    print(f"Input IDs shape: {sample['input_ids'].shape}")  
    print(f"Attention Mask shape: {sample['attention_mask'].shape}")  
    print(f"Labels shape: {sample['labels'].shape}")  
    print(f"Labels Attention Mask shape: {sample['labels_attention_mask'].shape}")  
    print(f"Reference Answer: {sample['reference_answer']}")
    
    # To get the evaluation metrics:
    metrics = argilla_magpie_ultra_v0_1_dataset.get_evaluation_metrics()
    print(f"Largest input IDs length: {argilla_magpie_ultra_v0_1_dataset.get_largest_input_ids_length()}")
    print(f"Largest labels length: {argilla_magpie_ultra_v0_1_dataset.get_largest_labels_length()}")
    
    
    arcee_ai_evolkit_20k_dataset = get_dataset("arcee-ai/EvolKit-20k", tokenizer_name, 2048, 2048, "train", 100)
    sample = arcee_ai_evolkit_20k_dataset[0]  
    print(f"Input IDs shape: {sample['input_ids'].shape}")  
    print(f"Attention Mask shape: {sample['attention_mask'].shape}")  
    print(f"Labels shape: {sample['labels'].shape}")  
    print(f"Labels Attention Mask shape: {sample['labels_attention_mask'].shape}")  
    print(f"Reference Answer: {sample['reference_answer']}")
    
    # To get the evaluation metrics:
    metrics = arcee_ai_evolkit_20k_dataset.get_evaluation_metrics() 
    print(f"Largest input IDs length: {arcee_ai_evolkit_20k_dataset.get_largest_input_ids_length()}")
    print(f"Largest labels length: {arcee_ai_evolkit_20k_dataset.get_largest_labels_length()}")
    
    wizardlmteam_wizardlm_evol_instruct_v2_196k_dataset = get_dataset("WizardLMTeam/WizardLM_evol_instruct_V2_196k", tokenizer_name, 2048, 2048, "train", 100)
    sample = wizardlmteam_wizardlm_evol_instruct_v2_196k_dataset[0]  
    print(f"Input IDs shape: {sample['input_ids'].shape}")  
    print(f"Attention Mask shape: {sample['attention_mask'].shape}")  
    print(f"Labels shape: {sample['labels'].shape}")  
    print(f"Labels Attention Mask shape: {sample['labels_attention_mask'].shape}")  
    print(f"Reference Answer: {sample['reference_answer']}")
    
    # To get the evaluation metrics:
    metrics = wizardlmteam_wizardlm_evol_instruct_v2_196k_dataset.get_evaluation_metrics()
    print(f"Largest input IDs length: {wizardlmteam_wizardlm_evol_instruct_v2_196k_dataset.get_largest_input_ids_length()}")
    print(f"Largest labels length: {wizardlmteam_wizardlm_evol_instruct_v2_196k_dataset.get_largest_labels_length()}")
    
    
    mmlu_pro_dataset = get_dataset("TIGER-Lab/MMLU-Pro", tokenizer_name, 2048, 2048, "test", 100)
    sample = mmlu_pro_dataset[0]  
    print(f"Input IDs shape: {sample['input_ids'].shape}")  
    print(f"Attention Mask shape: {sample['attention_mask'].shape}")  
    print(f"Labels shape: {sample['labels'].shape}")  
    print(f"Labels Attention Mask shape: {sample['labels_attention_mask'].shape}")  
    print(f"Reference Answer: {sample['reference_answer']}")
    
    # To get the evaluation metrics:
    metrics = mmlu_pro_dataset.get_evaluation_metrics()
    print(f"Largest input IDs length: {mmlu_pro_dataset.get_largest_input_ids_length()}")
    print(f"Largest labels length: {mmlu_pro_dataset.get_largest_labels_length()}")
    
    mmlu_pro_COT_dataset = get_dataset("TIGER-Lab/MMLU-Pro-COT", tokenizer_name, 2048, 2048, "test", 100)
    sample = mmlu_pro_COT_dataset[0]  
    print(f"Input IDs shape: {sample['input_ids'].shape}")  
    print(f"Attention Mask shape: {sample['attention_mask'].shape}")  
    print(f"Labels shape: {sample['labels'].shape}")  
    print(f"Labels Attention Mask shape: {sample['labels_attention_mask'].shape}")  
    print(f"Reference Answer: {sample['reference_answer']}")
    
    # To get the evaluation metrics:
    metrics = mmlu_pro_COT_dataset.get_evaluation_metrics()
    print(f"Largest input IDs length: {mmlu_pro_COT_dataset.get_largest_input_ids_length()}")
    print(f"Largest labels length: {mmlu_pro_COT_dataset.get_largest_labels_length()}")
    
    
    lighteval_math_hard_dataset = get_dataset("lighteval/MATH-Hard", tokenizer_name, 2048, 2048, "train", 100)
    sample = lighteval_math_hard_dataset[0]  
    print(f"Input IDs shape: {sample['input_ids'].shape}")  
    print(f"Attention Mask shape: {sample['attention_mask'].shape}")  
    print(f"Labels shape: {sample['labels'].shape}")  
    print(f"Labels Attention Mask shape: {sample['labels_attention_mask'].shape}")  
    print(f"Reference Answer: {sample['reference_answer']}")    
    
    # To get the evaluation metrics:
    metrics = lighteval_math_hard_dataset.get_evaluation_metrics()
    print(f"Largest input IDs length: {lighteval_math_hard_dataset.get_largest_input_ids_length()}")
    print(f"Largest labels length: {lighteval_math_hard_dataset.get_largest_labels_length()}")
    
    
    tau_instruct_dataset = get_dataset("tau/commonsense_qa", tokenizer_name, 2048, 2048, "train", 100)
    sample = tau_instruct_dataset[0]  
    print(f"Input IDs shape: {sample['input_ids'].shape}")  
    print(f"Attention Mask shape: {sample['attention_mask'].shape}")  
    print(f"Labels shape: {sample['labels'].shape}")  
    print(f"Labels Attention Mask shape: {sample['labels_attention_mask'].shape}")  
    print(f"Reference Answer: {sample['reference_answer']}")
    
    # To get the evaluation metrics:
    metrics = tau_instruct_dataset.get_evaluation_metrics()
    print(f"Largest input IDs length: {tau_instruct_dataset.get_largest_input_ids_length()}")
    print(f"Largest labels length: {tau_instruct_dataset.get_largest_labels_length()}")
    
    # "EleutherAI/truthful_qa_mc"
    eleutherai_truthful_qa_mc_dataset = get_dataset("EleutherAI/truthful_qa_mc", tokenizer_name, 2048, 2048, "test", 100)
    sample = eleutherai_truthful_qa_mc_dataset[0]  
    print(f"Input IDs shape: {sample['input_ids'].shape}")  
    print(f"Attention Mask shape: {sample['attention_mask'].shape}")  
    print(f"Labels shape: {sample['labels'].shape}")  
    print(f"Labels Attention Mask shape: {sample['labels_attention_mask'].shape}")  
    print(f"Reference Answer: {sample['reference_answer']}")
    
    # To get the evaluation metrics:
    metrics = eleutherai_truthful_qa_mc_dataset.get_evaluation_metrics()
    print(f"Largest input IDs length: {eleutherai_truthful_qa_mc_dataset.get_largest_input_ids_length()}")
    print(f"Largest labels length: {eleutherai_truthful_qa_mc_dataset.get_largest_labels_length()}")
    
    # "amanrangapur/Fin-Fact" 
    amanrangapur_fin_fact_dataset = get_dataset("amanrangapur/Fin-Fact", tokenizer_name, 2048, 2048, "test", 100)
    sample = amanrangapur_fin_fact_dataset[0]  
    print(f"Input IDs shape: {sample['input_ids'].shape}")  
    print(f"Attention Mask shape: {sample['attention_mask'].shape}")  
    print(f"Labels shape: {sample['labels'].shape}")  
    print(f"Labels Attention Mask shape: {sample['labels_attention_mask'].shape}")  
    print(f"Reference Answer: {sample['reference_answer']}")
    
    # To get the evaluation metrics:
    metrics = amanrangapur_fin_fact_dataset.get_evaluation_metrics()
    print(f"Largest input IDs length: {amanrangapur_fin_fact_dataset.get_largest_input_ids_length()}")
    print(f"Largest labels length: {amanrangapur_fin_fact_dataset.get_largest_labels_length()}")
    
    # "FinanceMTEB/financial_phrasebank"
    finance_mteb_financial_phrasebank_dataset = get_dataset("FinanceMTEB/financial_phrasebank", tokenizer_name, 2048, 2048, "train", 100)
    sample = finance_mteb_financial_phrasebank_dataset[0]  
    print(f"Input IDs shape: {sample['input_ids'].shape}")  
    print(f"Attention Mask shape: {sample['attention_mask'].shape}")  
    print(f"Labels shape: {sample['labels'].shape}")  
    print(f"Labels Attention Mask shape: {sample['labels_attention_mask'].shape}")  
    print(f"Reference Answer: {sample['reference_answer']}")
    
    # To get the evaluation metrics:
    metrics = finance_mteb_financial_phrasebank_dataset.get_evaluation_metrics()
    print(f"Largest input IDs length: {finance_mteb_financial_phrasebank_dataset.get_largest_input_ids_length()}")
    print(f"Largest labels length: {finance_mteb_financial_phrasebank_dataset.get_largest_labels_length()}")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
      
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    