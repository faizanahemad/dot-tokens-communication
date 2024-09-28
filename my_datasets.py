# datasets.py  

import torch  
from torch.utils.data import Dataset  
from datasets import load_dataset  
from transformers import AutoTokenizer  
from tqdm import tqdm  
import re  
  
class BaseDataset(Dataset):  
    def __init__(self, dataset_name, tokenizer_name, max_input_length, max_output_length, split, subset_size=None, **extra_kwargs):  
        self.dataset_name = dataset_name  
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)  
        # Check if the tokenizer has no pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_input_length = max_input_length  
        self.max_output_length = max_output_length  
        self.split = split  
        self.subset_size = subset_size  
        self.data = self.load_data()  
        self.cached_data = self.preprocess_data()  
        self.extra_kwargs = extra_kwargs
  
    def load_data(self):  
        raise NotImplementedError("This method should be implemented by subclasses.")  
  
    def preprocess_data(self):  
        raise NotImplementedError("This method should be implemented by subclasses.")  
  
    def __len__(self):  
        return len(self.cached_data)  
  
    def __getitem__(self, idx):  
        return self.cached_data[idx]  
  
    def extract_answer(self, text):  
        raise NotImplementedError("This method should be implemented by subclasses.")  
  
    def get_evaluation_metrics(self):  
        # Returns a dictionary of metric functions specific to the dataset  
        raise NotImplementedError("This method should be implemented by subclasses.")  
  
class GSM8KDataset(BaseDataset):  
    def load_data(self):  
        dataset = load_dataset("gsm8k", "main")  
        data = dataset[self.split]  
        if self.subset_size:  
            data = data.select(range(self.subset_size))  
        return data  
  
    def preprocess_data(self):  
        self.cached_data = []  
        for item in tqdm(self.data, desc=f"Preprocessing GSM8K {self.split} data"):  
            question = item['question']  
            answer = item['answer']  
  
            # Construct the prompt  
            prompt = f"Solve the following math problem step by step. Show your work and provide the final answer after '#### '.\n\nQuestion: {question}\n\nStep by Step working and Answer:"  
  
            # Tokenize the prompt with left padding  
            self.tokenizer.padding_side = 'left'  
            encoded_prompt = self.tokenizer(  
                prompt,  
                max_length=self.max_input_length,  
                padding='max_length',  
                truncation=True,  
                return_tensors='pt',  
                add_special_tokens=True  
            )  
  
            input_ids = encoded_prompt['input_ids'].squeeze()  
            attention_mask = encoded_prompt['attention_mask'].squeeze()  
  
            # Tokenize the answer with right padding  
            self.tokenizer.padding_side = 'right'  
            encoded_answer = self.tokenizer(  
                answer,  
                max_length=self.max_output_length,  
                padding='max_length',  
                truncation=True,  
                return_tensors='pt',  
                add_special_tokens=True  
            )  
  
            labels = encoded_answer['input_ids'].squeeze()  
            labels_attention_mask = encoded_answer['attention_mask'].squeeze()  
  
            self.cached_data.append({  
                'input_ids': input_ids,  
                'attention_mask': attention_mask,  
                'labels': labels,  
                'labels_attention_mask': labels_attention_mask,
                'reference_answer': answer.lower()
            })  
        return self.cached_data  
  
    def extract_answer(self, text):  
        
        pos = text.find('####')  
        if pos == -1:  
            return None  
        answer = text[pos+4:].strip()  
        # import re
        # match = re.search(r'####\s*(\S+)', answer)
        # if match:
        #     answer = match.group(1).strip()
        # else:
        #     return None
        answer = ''.join(answer.split()).lower()  
        # print(text.replace('\n', ' | '), " || ", answer.replace('\n', ' | '))
        return answer
  
    def get_evaluation_metrics(self):  
        # For GSM8K, we use exact match accuracy  
        def exact_match(predictions, references):  
            correct = 0  
            total = 0  
            for pred, ref in zip(predictions, references):  
                if pred is not None and ref is not None and pred == ref:  
                    correct += 1  
                total += 1  
            accuracy = correct / total if total > 0 else 0  
            return {'exact_match': accuracy}  
        
        def reference_contained(predictions, references):
            correct = 0
            total = 0
            for pred, ref in zip(predictions, references):
                total += 1
                if pred is None or ref is None:
                    continue
                pred = pred.strip().lower().replace("\n", " ")
                ref = ref.strip().lower().replace("\n", " ")
                # lower case both pred and ref
                
                if pred is not None and ref is not None and ref.lower() in pred.lower():
                    correct += 1
                
                # print(pred, ref, " | ", ref.lower() in pred.lower(), " | ", correct, total)
            accuracy = correct / total if total > 0 else 0
            # print(f"Accuracy: {accuracy:.2f} ({correct}/{total} correct)")
            return {'reference_contained': accuracy}
        return {'exact_match': exact_match, 'reference_contained': reference_contained} 
  
# datasets.py (continued)  
  

  
class MMLUDataset(BaseDataset):  
    def __init__(  
        self,  
        dataset_name,  
        tokenizer_name,  
        max_input_length,  
        max_output_length,  
        split,  
        subset_size=None,  
        **extra_kwargs
    ):  
        self.include_thinking = extra_kwargs.get("include_thinking", False)
        super().__init__(  
            dataset_name,  
            tokenizer_name,  
            max_input_length,  
            max_output_length,  
            split,  
            subset_size  
        )  
  
    def load_data(self):  
        dataset = load_dataset("lighteval/mmlu", "all")  
        data = dataset[self.split]  
        if self.subset_size:  
            data = data.select(range(self.subset_size))  
        return data  
  
    def construct_prompt(self, question, options, subject):  
        # Base prompt  
        prompt = f"Please answer the following question by selecting the correct option. Subject: {subject}\nQuestion: {question}\nOptions:\n"  
        for idx, option in enumerate(options):  
            option_label = chr(65 + idx)  # Convert 0->A, 1->B, etc.  
            prompt += f"{option_label}. {option}\n"  
  
        if self.include_thinking:  
            prompt += "\nProvide your thought process inside <thinking> and </thinking> tags."  
  
        # Clarify the instruction for the final answer  
        prompt += "\nProvide your final answer inside  <answer> and </answer>  tags."  
  
        return prompt  
  
    def preprocess_data(self):  
        self.cached_data = []  
        for item in tqdm(self.data, desc=f"Preprocessing MMLU {self.split} data"):  
            question = item['question']  
            options = item['options']  
            correct_answer = item['answer']  
            subject = item.get('subject', 'General Knowledge')  # Default subject if not provided  
  
            # Construct the prompt  
            prompt = self.construct_prompt(question, options, subject)  
  
            # Tokenize the prompt with left padding  
            self.tokenizer.padding_side = 'left'  
            encoded_prompt = self.tokenizer(  
                prompt,  
                max_length=self.max_input_length,  
                padding='max_length',  
                truncation=True,  
                return_tensors='pt',  
                add_special_tokens=True  
            )  
            input_ids = encoded_prompt['input_ids'].squeeze()  
            attention_mask = encoded_prompt['attention_mask'].squeeze()  
  
            # Prepare the target answer with  tags  
            answer_text = f"<answer>{correct_answer}</answer>"  
  
            if self.include_thinking:  
                # Optionally include chain-of-thought reasoning placeholder  
                answer_text = f"<thinking>Your thought process here.</thinking>\n{answer_text}"  
  
            # Tokenize the answer  
            encoded_answer = self.tokenizer(  
                answer_text,  
                max_length=self.max_output_length,  
                padding='max_length',  
                truncation=True,  
                return_tensors='pt',  
                add_special_tokens=True  
            )  
            labels = encoded_answer['input_ids'].squeeze()  
  
            self.cached_data.append({  
                'input_ids': input_ids,  
                'attention_mask': attention_mask,  
                'labels': labels,  
                'correct_answer': correct_answer  
            })  
  
    def __len__(self):  
        return len(self.cached_data)  
  
    def __getitem__(self, idx):  
        return self.cached_data[idx]  
  
    def extract_answer(self, output_text):  
        # Use regex to extract answer from between  tags  
        match = re.search(r'<answer>(.*?)</answer>', output_text, re.DOTALL)
        if match:  
            answer = match.group(1).strip()  
            return answer  
        else:  
            # If tags are missing, try to extract the answer heuristically  
            possible_answers = ['A', 'B', 'C', 'D']  
            for option in possible_answers:  
                if option in output_text:  
                    return option  
            return None  # Unable to extract answer  
  
    def accuracy_metric(self, predictions, references):  
        correct = 0  
        total = len(predictions)  
        for pred, ref in zip(predictions, references):  
            pred_answer = self.extract_answer(pred)  
            ref_answer = ref.strip()  
  
            if pred_answer is None:  
                continue  # Skip if unable to extract answer  
  
            # Make comparison flexible to formatting  
            if pred_answer.upper().strip() == ref_answer.upper().strip():  
                correct += 1  
            else:  
                # Additional checks for flexibility  
                pred_answer = pred_answer.upper().strip()  
                ref_answer = ref_answer.upper().strip()  
  
                # Map full words to letters if necessary (e.g., "Option A" -> "A")  
                option_mapping = {  
                    "OPTION A": "A", "OPTION B": "B", "OPTION C": "C", "OPTION D": "D",  
                    "A.": "A", "B.": "B", "C.": "C", "D.": "D"  
                }  
                pred_answer = option_mapping.get(pred_answer, pred_answer)  
                ref_answer = option_mapping.get(ref_answer, ref_answer)  
  
                if pred_answer == ref_answer:  
                    correct += 1  
  
        accuracy = correct / total if total > 0 else 0  
        return accuracy  


class MLQADataset(BaseDataset):  
    def load_data(self):  
        dataset = load_dataset("mlqa", 'en.en')  
        data = dataset[self.split]  
        if self.subset_size:  
            data = data.select(range(self.subset_size))  
        return data  
  
    def preprocess_data(self):  
        self.cached_data = []  
        for item in tqdm(self.data, desc=f"Preprocessing MLQA {self.split} data"):  
            context = item['context']  
            question = item['question']  
            answers = item['answers']['text']  
  
            # For simplicity, we'll use the first answer  
            answer = answers[0] if answers else ""  
  
            # Construct the prompt  
            prompt = f"Read the following passage and answer the question.\n\nPassage: {context}\n\nQuestion: {question}\n\nAnswer:"  
  
            # Tokenize the prompt with left padding  
            self.tokenizer.padding_side = 'left'  
            encoded_prompt = self.tokenizer(  
                prompt,  
                max_length=self.max_input_length,  
                padding='max_length',  
                truncation=True,  
                return_tensors='pt',  
                add_special_tokens=True  
            )  
  
            input_ids = encoded_prompt['input_ids'].squeeze()  
            attention_mask = encoded_prompt['attention_mask'].squeeze()  
  
            # Tokenize the answer with right padding  
            self.tokenizer.padding_side = 'right'  
            encoded_answer = self.tokenizer(  
                answer,  
                max_length=self.max_output_length,  
                padding='max_length',  
                truncation=True,  
                return_tensors='pt',  
                add_special_tokens=True  
            )  
  
            labels = encoded_answer['input_ids'].squeeze()  
            labels_attention_mask = encoded_answer['attention_mask'].squeeze()  
  
            self.cached_data.append({  
                'input_ids': input_ids,  
                'attention_mask': attention_mask,  
                'labels': labels,  
                'labels_attention_mask': labels_attention_mask,  
                'reference_answer': answer.lower()  
            })  
        return self.cached_data  
  
    def extract_answer(self, text):  
        return text.strip().lower()  
  
    def get_evaluation_metrics(self):  
        # For MLQA, we can use substring match for evaluation  
        def substring_match(predictions, references):  
            correct = 0  
            total = 0  
            for pred, ref in zip(predictions, references):  
                if pred is not None and ref is not None and ref in pred:  
                    correct += 1  
                total += 1  
            accuracy = correct / total if total > 0 else 0  
            return {'accuracy': accuracy}  
  
        return {'substring_match': substring_match}  
  
class SQuADDataset(BaseDataset):  
    def load_data(self):  
        dataset = load_dataset("squad")  
        data = dataset[self.split]  
        if self.subset_size:  
            data = data.select(range(self.subset_size))  
        return data  
  
    def preprocess_data(self):  
        self.cached_data = []  
        for item in tqdm(self.data, desc=f"Preprocessing SQuAD {self.split} data"):  
            context = item['context']  
            question = item['question']  
            answers = item['answers']['text']  
  
            # For simplicity, we'll use the first answer  
            answer = answers[0] if answers else ""  
  
            # Construct the prompt  
            prompt = f"Read the following passage and answer the question.\n\nPassage: {context}\n\nQuestion: {question}\n\nAnswer:"  
  
            # Tokenize the prompt with left padding  
            self.tokenizer.padding_side = 'left'  
            encoded_prompt = self.tokenizer(  
                prompt,  
                max_length=self.max_input_length,  
                padding='max_length',  
                truncation=True,  
                return_tensors='pt',  
                add_special_tokens=True  
            )  
  
            input_ids = encoded_prompt['input_ids'].squeeze()  
            attention_mask = encoded_prompt['attention_mask'].squeeze()  
  
            # Tokenize the answer with right padding  
            self.tokenizer.padding_side = 'right'  
            encoded_answer = self.tokenizer(  
                answer,  
                max_length=self.max_output_length,  
                padding='max_length',  
                truncation=True,  
                return_tensors='pt',  
                add_special_tokens=True  
            )  
  
            labels = encoded_answer['input_ids'].squeeze()  
            labels_attention_mask = encoded_answer['attention_mask'].squeeze()  
  
            self.cached_data.append({  
                'input_ids': input_ids,  
                'attention_mask': attention_mask,  
                'labels': labels,  
                'labels_attention_mask': labels_attention_mask,  
                'reference_answer': answer.lower()  
            })  
        return self.cached_data  
  
    def extract_answer(self, text):  
        return text.strip().lower()  
  
    def get_evaluation_metrics(self):  
        # For SQuAD, we can use substring match for evaluation  
        def substring_match(predictions, references):  
            correct = 0  
            total = 0  
            for pred, ref in zip(predictions, references):  
                if pred is not None and ref is not None and ref in pred:  
                    correct += 1  
                total += 1  
            accuracy = correct / total if total > 0 else 0  
            return {'accuracy': accuracy}  
  
        return {'substring_match': substring_match}  
  
class MMLUDataset(BaseDataset):  
    def load_data(self):  
        # Assuming you have the MMLU dataset available  
        dataset = load_dataset("hendrycks_test", "abstract_algebra")  
        data = dataset[self.split]  
        if self.subset_size:  
            data = data.select(range(self.subset_size))  
        return data  
  
    def preprocess_data(self):  
        self.cached_data = []  
        for item in tqdm(self.data, desc=f"Preprocessing MMLU {self.split} data"):  
            question = item['question']  
            answer = item['answer']  
  
            # Construct the prompt  
            prompt = f"Answer the following question:\n\nQuestion: {question}\n\nAnswer choices:\n{item['A']}\n{item['B']}\n{item['C']}\n{item['D']}\n\nYour answer:"  
  
            # Tokenize the prompt with left padding  
            self.tokenizer.padding_side = 'left'  
            encoded_prompt = self.tokenizer(  
                prompt,  
                max_length=self.max_input_length,  
                padding='max_length',  
                truncation=True,  
                return_tensors='pt',  
                add_special_tokens=True  
            )  
  
            input_ids = encoded_prompt['input_ids'].squeeze()  
            attention_mask = encoded_prompt['attention_mask'].squeeze()  
  
            # Tokenize the answer with right padding  
            self.tokenizer.padding_side = 'right'  
            encoded_answer = self.tokenizer(  
                answer,  
                max_length=1,  # Answer is a single character ('A', 'B', 'C', 'D')  
                padding='max_length',  
                truncation=True,  
                return_tensors='pt',  
                add_special_tokens=False  
            )  
  
            labels = encoded_answer['input_ids'].squeeze()  
            labels_attention_mask = encoded_answer['attention_mask'].squeeze()  
  
            self.cached_data.append({  
                'input_ids': input_ids,  
                'attention_mask': attention_mask,  
                'labels': labels,  
                'labels_attention_mask': labels_attention_mask,  
                'reference_answer': answer.lower()  
            })  
        return self.cached_data  
  
    def extract_answer(self, text):  
        return text.strip().lower()[0]  # Return the first character (assumed to be 'A', 'B', 'C', or 'D')  
  
    def get_evaluation_metrics(self):  
        # For MMLU, we use exact match accuracy  
        def exact_match(predictions, references):  
            correct = 0  
            total = 0  
            for pred, ref in zip(predictions, references):  
                if pred is not None and ref is not None and pred[0] == ref[0]:  
                    correct += 1  
                total += 1  
            accuracy = correct / total if total > 0 else 0  
            return {'accuracy': accuracy}  
        return {'exact_match': exact_match}  
  
# Add additional dataset classes here following the same pattern  

class XNLIDataset(BaseDataset):  
    def load_data(self):  
        dataset = load_dataset("xnli", "en")  
        data = dataset[self.split]  
        if self.subset_size:  
            data = data.select(range(self.subset_size))  
        return data  
  
    def preprocess_data(self):  
        self.cached_data = []  
        for item in tqdm(self.data, desc=f"Preprocessing XNLI {self.split} data"):  
            premise = item['premise']  
            hypothesis = item['hypothesis']  
            label = item['label']  # Label: 0 (entailment), 1 (neutral), 2 (contradiction)  
  
            # Construct the prompt  
            prompt = f"Premise: {premise}\nHypothesis: {hypothesis}\n\nDoes the premise entail the hypothesis? Answer 'entailment', 'neutral', or 'contradiction'.\nAnswer:"  
  
            # Tokenize the prompt with left padding  
            self.tokenizer.padding_side = 'left'  
            encoded_prompt = self.tokenizer(  
                prompt,  
                max_length=self.max_input_length,  
                padding='max_length',  
                truncation=True,  
                return_tensors='pt',  
                add_special_tokens=True  
            )  
  
            input_ids = encoded_prompt['input_ids'].squeeze()  
            attention_mask = encoded_prompt['attention_mask'].squeeze()  
  
            # Map labels to textual answers  
            label_map = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}  
            answer = label_map[label]  
  
            # Tokenize the answer with right padding  
            self.tokenizer.padding_side = 'right'  
            encoded_answer = self.tokenizer(  
                answer,  
                max_length=self.max_output_length,  # 'entailment' is the longest label  
                padding='max_length',  
                truncation=True,  
                return_tensors='pt',  
                add_special_tokens=False  
            )  
  
            labels = encoded_answer['input_ids'].squeeze()  
            labels_attention_mask = encoded_answer['attention_mask'].squeeze()  
  
            self.cached_data.append({  
                'input_ids': input_ids,  
                'attention_mask': attention_mask,  
                'labels': labels,  
                'labels_attention_mask': labels_attention_mask,  
                'reference_answer': answer.lower()  
            })  
        return self.cached_data  
  
    def extract_answer(self, text):  
        return text.strip().lower()  
  
    def get_evaluation_metrics(self):  
        # For XNLI, we use exact match accuracy  
        def exact_match(predictions, references):  
            correct = 0  
            total = 0  
            for pred, ref in zip(predictions, references):  
                if pred is not None and ref is not None and pred == ref:  
                    correct += 1  
                total += 1  
            accuracy = correct / total if total > 0 else 0  
            return {'accuracy': accuracy}  
        return {'exact_match': exact_match}  

  
def get_dataset_class(dataset_name):  
    dataset_classes = {  
        'gsm8k': GSM8KDataset,  
        'mlqa': MLQADataset,  
        'squad': SQuADDataset,  
        'mmlu': MMLUDataset,  
        'xnli': XNLIDataset,  
        'fill_the_blank': FillTheBlankDataset,
        'complete_the_sentence': CompleteTheSentenceDataset,
        # Add other datasets here  
    }  
    return dataset_classes.get(dataset_name.lower())  

class FillTheBlankDataset(BaseDataset):  
    def load_data(self):  
        # Manually create the dataset  
        data = ([  
            # Proverbs and common sayings  
            {"sentence": "The quick brown ___ jumps over the lazy dog.", "answer": "fox"},  
            {"sentence": "A stitch in time saves ___.", "answer": "nine"},  
            {"sentence": "An apple a day keeps the ___ away.", "answer": "doctor"},  
            {"sentence": "All that glitters is not ___.", "answer": "gold"},  
            {"sentence": "A journey of a thousand miles begins with a single ___.", "answer": "step"},  
            {"sentence": "Better late than ___.", "answer": "never"},  
            {"sentence": "Birds of a feather flock ___.", "answer": "together"},  
            {"sentence": "Actions speak louder than ___.", "answer": "words"},  
            {"sentence": "Beauty is in the eye of the ___.", "answer": "beholder"},  
            {"sentence": "The early bird catches the ___.", "answer": "worm"},  
            {"sentence": "Don't count your chickens before they ___.", "answer": "hatch"},  
            {"sentence": "Two wrongs don't make a ___.", "answer": "right"},  
            {"sentence": "When in Rome, do as the Romans ___.", "answer": "do"},  
            {"sentence": "The pen is mightier than the ___.", "answer": "sword"},  
            {"sentence": "Every cloud has a silver ___.", "answer": "lining"},  
            {"sentence": "A penny saved is a penny ___.", "answer": "earned"},  
            {"sentence": "Curiosity killed the ___.", "answer": "cat"},  
            {"sentence": "Don't cry over spilled ___.", "answer": "milk"},  
            {"sentence": "The squeaky wheel gets the ___.", "answer": "grease"},  
            {"sentence": "You can't judge a book by its ___.", "answer": "cover"},  
              
            # Wise musings and philosophical statements  
            {"sentence": "To be or not to be, that is the ___.", "answer": "question"},  
            {"sentence": "Knowledge is ___.", "answer": "power"},  
            {"sentence": "Time is ___.", "answer": "money"},  
            {"sentence": "The best things in life are ___.", "answer": "free"},  
            {"sentence": "Where there's a will, there's a ___.", "answer": "way"},  
            {"sentence": "Practice makes ___.", "answer": "perfect"},  
            {"sentence": "Honesty is the best ___.", "answer": "policy"},  
            {"sentence": "Fortune favors the ___.", "answer": "bold"},  
            {"sentence": "Necessity is the mother of ___.", "answer": "invention"},  
            {"sentence": "The only constant is ___.", "answer": "change"},  
              
            # Common facts  
            {"sentence": "The Earth revolves around the ___.", "answer": "Sun"},  
            {"sentence": "Water boils at 100 degrees ___.", "answer": "Celsius"},  
            {"sentence": "The capital of France is ___.", "answer": "Paris"},  
            {"sentence": "The human body has 206 ___.", "answer": "bones"},  
            {"sentence": "Mount Everest is the world's highest ___.", "answer": "mountain"},  
              
            # Literary references  
            
            {"sentence": "It was the best of times, it was the ___ of times.", "answer": "worst"},  
            {"sentence": "All animals are equal, but some animals are more ___ than others.", "answer": "equal"},  
            {"sentence": "The ___ that men do lives after them; The good is oft interred with their bones.", "answer": "evil"},  
            {"sentence": "I think, therefore I ___.", "answer": "am"},  
              
            # More challenging examples  
            {"sentence": "The proof of the ___ is in the eating.", "answer": "pudding"},  
            {"sentence": "A ___ in sheep's clothing.", "answer": "wolf"},  
            {"sentence": "The road to ___ is paved with good intentions.", "answer": "hell"},  
            {"sentence": "Don't throw the baby out with the ___.", "answer": "bathwater"},  
            {"sentence": "A bird in the hand is worth two in the ___.", "answer": "bush"},  
        ]  + [  
            {  
                "sentence": "A penny for your ___.",  
                "answer": "thoughts",  
                "theme": "Proverbs and Common Sayings",  
                "difficulty": "Easy",  
                "cultural_origin": "Western",  
                "time_period": "16th Century"  
            },  
            {  
                "sentence": "The unexamined life is not worth ___.",  
                "answer": "living",  
                "theme": "Philosophical Statements",  
                "difficulty": "Medium",  
                "cultural_origin": "Ancient Greek",  
                "time_period": "Classical"  
            },  
            {  
                "sentence": "It was the best of times, it was the ___ of times.",  
                "answer": "worst",  
                "theme": "Literary References",  
                "difficulty": "Medium",  
                "cultural_origin": "Western",  
                "time_period": "19th Century"  
            },  
            {  
                "sentence": "The chemical symbol for potassium is ___.",  
                "answer": "K",  
                "theme": "Scientific Concepts",  
                "difficulty": "Easy",  
                "cultural_origin": "International",  
                "time_period": "Modern"  
            },  
            {  
                "sentence": "Break the ___.",  
                "answer": "ice",  
                "theme": "Modern Idioms",  
                "difficulty": "Easy",  
                "cultural_origin": "Western",  
                "time_period": "20th Century"  
            },  
        ]  + [  
            # Proverbs and Common Sayings  
            {  
                "sentence": "A penny for your ___.",  
                "answer": "thoughts",  
                "theme": "Proverbs and Common Sayings",  
                "difficulty": "Easy",  
                "cultural_origin": "Western",  
                "time_period": "16th Century"  
            },  
            {  
                "sentence": "A picture is worth a thousand ___.",  
                "answer": "words",  
                "theme": "Proverbs and Common Sayings",  
                "difficulty": "Easy",  
                "cultural_origin": "International",  
                "time_period": "20th Century"  
            },  
            {  
                "sentence": "When in Rome, do as the Romans ___.",  
                "answer": "do",  
                "theme": "Proverbs and Common Sayings",  
                "difficulty": "Easy",  
                "cultural_origin": "Western",  
                "time_period": "Ancient"  
            },  
            {  
                "sentence": "The early bird catches the ___.",  
                "answer": "worm",  
                "theme": "Proverbs and Common Sayings",  
                "difficulty": "Easy",  
                "cultural_origin": "English",  
                "time_period": "17th Century"  
            },  
            {  
                "sentence": "A rolling stone gathers no ___.",  
                "answer": "moss",  
                "theme": "Proverbs and Common Sayings",  
                "difficulty": "Medium",  
                "cultural_origin": "Latin",  
                "time_period": "Ancient"  
            },  
            
            # Philosophical Statements  
            {  
                "sentence": "I think, therefore I ___.",  
                "answer": "am",  
                "theme": "Philosophical Statements",  
                "difficulty": "Medium",  
                "cultural_origin": "French",  
                "time_period": "17th Century"  
            },  
            {  
                "sentence": "The unexamined life is not worth ___.",  
                "answer": "living",  
                "theme": "Philosophical Statements",  
                "difficulty": "Medium",  
                "cultural_origin": "Ancient Greek",  
                "time_period": "Classical"  
            },  
            {  
                "sentence": "To be or not to be, that is the ___.",  
                "answer": "question",  
                "theme": "Philosophical Statements",  
                "difficulty": "Easy",  
                "cultural_origin": "English",  
                "time_period": "16th Century"  
            },  
            {  
                "sentence": "The only thing we have to fear is fear ___.",  
                "answer": "itself",  
                "theme": "Philosophical Statements",  
                "difficulty": "Medium",  
                "cultural_origin": "American",  
                "time_period": "20th Century"  
            },  
            {  
                "sentence": "Cogito, ergo ___.",  
                "answer": "sum",  
                "theme": "Philosophical Statements",  
                "difficulty": "Hard",  
                "cultural_origin": "Latin",  
                "time_period": "17th Century"  
            },  
            
            # Scientific Concepts  
            {  
                "sentence": "The chemical symbol for gold is ___.",  
                "answer": "Au",  
                "theme": "Scientific Concepts",  
                "difficulty": "Medium",  
                "cultural_origin": "International",  
                "time_period": "Modern"  
            },  
            {  
                "sentence": "E = mc^2 is Einstein's theory of ___.",  
                "answer": "relativity",  
                "theme": "Scientific Concepts",  
                "difficulty": "Medium",  
                "cultural_origin": "International",  
                "time_period": "20th Century"  
            },  
            {  
                "sentence": "The powerhouse of the cell is the ___.",  
                "answer": "mitochondria",  
                "theme": "Scientific Concepts",  
                "difficulty": "Medium",  
                "cultural_origin": "International",  
                "time_period": "Modern"  
            },  
            {  
                "sentence": "The speed of light is approximately 300,000 kilometers per ___.",  
                "answer": "second",  
                "theme": "Scientific Concepts",  
                "difficulty": "Easy",  
                "cultural_origin": "International",  
                "time_period": "Modern"  
            },  
            {  
                "sentence": "The atomic number of carbon is ___.",  
                "answer": "6",  
                "theme": "Scientific Concepts",  
                "difficulty": "Hard",  
                "cultural_origin": "International",  
                "time_period": "Modern"  
            },  
            
            # Literary References  
            {  
                "sentence": "It was the best of times, it was the ___ of times.",  
                "answer": "worst",  
                "theme": "Literary References",  
                "difficulty": "Medium",  
                "cultural_origin": "English",  
                "time_period": "19th Century"  
            },  
            {  
                "sentence": "Call me ___.",  
                "answer": "Ishmael",  
                "theme": "Literary References",  
                "difficulty": "Medium",  
                "cultural_origin": "American",  
                "time_period": "19th Century"  
            },  
            {  
                "sentence": "All animals are equal, but some animals are more ___ than others.",  
                "answer": "equal",  
                "theme": "Literary References",  
                "difficulty": "Medium",  
                "cultural_origin": "English",  
                "time_period": "20th Century"  
            },  
            {  
                "sentence": "It was a bright cold day in April, and the clocks were striking ___.",  
                "answer": "thirteen",  
                "theme": "Literary References",  
                "difficulty": "Hard",  
                "cultural_origin": "English",  
                "time_period": "20th Century"  
            },  
            {  
                "sentence": "To be, or not to be: that is the ___.",  
                "answer": "question",  
                "theme": "Literary References",  
                "difficulty": "Easy",  
                "cultural_origin": "English",  
                "time_period": "16th Century"  
            },  
            
            # Historical Facts  
            {  
                "sentence": "The Declaration of Independence was signed in ___.",  
                "answer": "1776",  
                "theme": "Historical Facts",  
                "difficulty": "Easy",  
                "cultural_origin": "American",  
                "time_period": "18th Century"  
            },  
            {  
                "sentence": "The Berlin Wall fell in ___.",  
                "answer": "1989",  
                "theme": "Historical Facts",  
                "difficulty": "Medium",  
                "cultural_origin": "German",  
                "time_period": "20th Century"  
            },  
            {  
                "sentence": "The French Revolution began in ___.",  
                "answer": "1789",  
                "theme": "Historical Facts",  
                "difficulty": "Medium",  
                "cultural_origin": "French",  
                "time_period": "18th Century"  
            },  
            {  
                "sentence": "The first moon landing occurred in ___.",  
                "answer": "1969",  
                "theme": "Historical Facts",  
                "difficulty": "Easy",  
                "cultural_origin": "International",  
                "time_period": "20th Century"  
            },  
            {  
                "sentence": "The Magna Carta was signed in ___.",  
                "answer": "1215",  
                "theme": "Historical Facts",  
                "difficulty": "Hard",  
                "cultural_origin": "English",  
                "time_period": "Medieval"  
            },  
            
            # Geography  
            {  
                "sentence": "The capital of Japan is ___.",  
                "answer": "Tokyo",  
                "theme": "Geography",  
                "difficulty": "Easy",  
                "cultural_origin": "International",  
                "time_period": "Modern"  
            },  
            {  
                "sentence": "The longest river in the world is the ___.",  
                "answer": "Nile",  
                "theme": "Geography",  
                "difficulty": "Easy",  
                "cultural_origin": "International",  
                "time_period": "Ancient"  
            },  
            {  
                "sentence": "The Great Barrier Reef is located off the coast of ___.",  
                "answer": "Australia",  
                "theme": "Geography",  
                "difficulty": "Medium",  
                "cultural_origin": "International",  
                "time_period": "Modern"  
            },  
            {  
                "sentence": "The highest mountain in North America is ___.",  
                "answer": "Denali",  
                "theme": "Geography",  
                "difficulty": "Hard",  
                "cultural_origin": "North American",  
                "time_period": "Modern"  
            },  
            {  
                "sentence": "The Sahara Desert is located primarily in ___.",  
                "answer": "Africa",  
                "theme": "Geography",  
                "difficulty": "Easy",  
                "cultural_origin": "International",  
                "time_period": "Ancient"  
            },  
            
            # Modern Idioms  
            {  
                "sentence": "Break the ___.",  
                "answer": "ice",  
                "theme": "Modern Idioms",  
                "difficulty": "Easy",  
                "cultural_origin": "Western",  
                "time_period": "20th Century"  
            },  
            {  
                "sentence": "It's not rocket ___.",  
                "answer": "science",  
                "theme": "Modern Idioms",  
                "difficulty": "Easy",  
                "cultural_origin": "American",  
                "time_period": "20th Century"  
            },  
            {  
                "sentence": "The ball is in your ___.",  
                "answer": "court",  
                "theme": "Modern Idioms",  
                "difficulty": "Medium",  
                "cultural_origin": "Western",  
                "time_period": "20th Century"  
            },  
            {  
                "sentence": "Barking up the wrong ___.",  
                "answer": "tree",  
                "theme": "Modern Idioms",  
                "difficulty": "Medium",  
                "cultural_origin": "English",  
                "time_period": "19th Century"  
            },  
            {  
                "sentence": "Bite off more than you can ___.",  
                "answer": "chew",  
                "theme": "Modern Idioms",  
                "difficulty": "Medium",  
                "cultural_origin": "American",  
                "time_period": "19th Century"  
            },  
            
            # Mathematics  
            {  
                "sentence": "The square root of 64 is ___.",  
                "answer": "8",  
                "theme": "Mathematics",  
                "difficulty": "Easy",  
                "cultural_origin": "International",  
                "time_period": "Ancient"  
            },  
            {  
                "sentence": "Pi is approximately equal to ___.",  
                "answer": "3.14159",  
                "theme": "Mathematics",  
                "difficulty": "Medium",  
                "cultural_origin": "International",  
                "time_period": "Ancient"  
            },  
            {  
                "sentence": "In a right-angled triangle, the square of the hypotenuse is equal to the sum of the squares of the other two ___.",  
                "answer": "sides",  
                "theme": "Mathematics",  
                "difficulty": "Hard",  
                "cultural_origin": "Greek",  
                "time_period": "Ancient"  
            },  
            {  
                "sentence": "The sum of the angles in a triangle is ___ degrees.",  
                "answer": "180",  
                "theme": "Mathematics",  
                "difficulty": "Easy",  
                "cultural_origin": "International",  
                "time_period": "Ancient"  
            },  
            {  
                "sentence": "The factorial of 5 (5!) is equal to ___.",  
                "answer": "120",  
                "theme": "Mathematics",  
                "difficulty": "Medium",  
                "cultural_origin": "International",  
                "time_period": "Modern"  
            },  
            
            # Technology  
            {  
                "sentence": "HTML stands for Hypertext Markup ___.",  
                "answer": "Language",  
                "theme": "Technology",  
                "difficulty": "Easy",  
                "cultural_origin": "International",  
                "time_period": "Modern"  
            },  
            {  
                "sentence": "The 'www' in a website URL stands for World Wide ___.",  
                "answer": "Web",  
                "theme": "Technology",  
                "difficulty": "Easy",  
                "cultural_origin": "International",  
                "time_period": "Modern"  
            },  
            {  
                "sentence": "The first commercially successful personal computer was the Apple ___.",  
                "answer": "II",  
                "theme": "Technology",  
                "difficulty": "Medium",  
                "cultural_origin": "American",  
                "time_period": "20th Century"  
            },  
            {  
                "sentence": "Moore's Law states that the number of transistors on a microchip doubles about every two ___.",  
                "answer": "years",  
                "theme": "Technology",  
                "difficulty": "Hard",  
                "cultural_origin": "American",  
                "time_period": "20th Century"  
            },  
            {  
                "sentence": "The programming language Python is named after the British comedy group Monty ___.",  
                "answer": "Python",  
                "theme": "Technology",  
                "difficulty": "Medium",  
                "cultural_origin": "Dutch",  
                "time_period": "20th Century"  
            },  
            
            # Music  
            {  
                "sentence": "Beethoven's Symphony No. 5 begins with the famous four-note ___.",  
                "answer": "motif",  
                "theme": "Music",  
                "difficulty": "Medium",  
                "cultural_origin": "German",  
                "time_period": "19th Century"  
            },  
            {  
                "sentence": "The Beatles were originally from ___.",  
                "answer": "Liverpool",  
                "theme": "Music",  
                "difficulty": "Easy",  
                "cultural_origin": "British",  
                "time_period": "20th Century"  
            },  
            {  
                "sentence": "Mozart wrote his first symphony at the age of ___.",  
                "answer": "8",  
                "theme": "Music",  
                "difficulty": "Hard",  
                "cultural_origin": "Austrian",  
                "time_period": "18th Century"  
            },  
            {  
                "sentence": "The 'King of Pop' was the nickname given to Michael ___.",  
                "answer": "Jackson",  
                "theme": "Music",  
                "difficulty": "Easy",  
                "cultural_origin": "American",  
                "time_period": "20th Century"  
            },  
            {  
                "sentence": "The woodwind instrument that looks like a long metal tube is called a ___.",  
                "answer": "flute",  
                "theme": "Music",  
                "difficulty": "Medium",  
                "cultural_origin": "International",  
                "time_period": "Ancient"  
            },  
            
            # Sports  
            {  
                "sentence": "In soccer, a game typically lasts ___ minutes.",  
                "answer": "90",  
                "theme": "Sports",  
                "difficulty": "Easy",  
                "cultural_origin": "International",  
                "time_period": "Modern"  
            },  
            {  
                "sentence": "The Olympic Games are held every ___ years.",  
                "answer": "4",  
                "theme": "Sports",  
                "difficulty": "Easy",  
                "cultural_origin": "International",  
                "time_period": "Ancient"  
            },  
            {  
                "sentence": "In baseball, a ___ is when a batter hits the ball out of the playing field in fair territory.",  
                "answer": "home run",  
                "theme": "Sports",  
                "difficulty": "Medium",  
                "cultural_origin": "American",  
                "time_period": "19th Century"  
            },  
            {  
                "sentence": "The sport of ___ involves sliding down a snow-covered slope on a board attached to a rider's feet.",  
                "answer": "snowboarding",  
                "theme": "Sports",  
                "difficulty": "Medium",  
                "cultural_origin": "American",  
                "time_period": "20th Century"  
            },  
            {  
                "sentence": "In cricket, the term '___ out' refers to when a batsman is dismissed by the wicket-keeper removing the bails after the batsman steps out of the crease.",  
                "answer": "stumped",  
                "theme": "Sports",  
                "difficulty": "Hard",  
                "cultural_origin": "British",  
                "time_period": "18th Century"  
            }  
        ]) 


        
        holdout_test_set = [  
            # Proverbs and common sayings  
            {"sentence": "A rolling stone gathers no ___.", "answer": "moss"},  
            {"sentence": "Absence makes the heart grow ___.", "answer": "fonder"},  
            {"sentence": "Look before you ___.", "answer": "leap"},  
            {"sentence": "Make hay while the sun ___.", "answer": "shines"},  
            {"sentence": "Too many cooks spoil the ___.", "answer": "broth"},  
            {"sentence": "A watched pot never ___.", "answer": "boils"},  
            {"sentence": "When it rains, it ___.", "answer": "pours"},  
            {"sentence": "The grass is always greener on the other side of the ___.", "answer": "fence"},  
            {"sentence": "Don't put all your eggs in one ___.", "answer": "basket"},  
            {"sentence": "A chain is only as strong as its weakest ___.", "answer": "link"},  
        
            # Wise musings and philosophical statements  
            {"sentence": "The unexamined life is not worth ___.", "answer": "living"},  
            {"sentence": "I think, therefore I ___.", "answer": "am"},  
            {"sentence": "Knowledge is knowing a tomato is a fruit; wisdom is not putting it in a ___.", "answer": "salad"},  
            {"sentence": "The greatest wealth is to live content with ___.", "answer": "little"},  
            {"sentence": "Life is what happens when you're busy making other ___.", "answer": "plans"},  
        
            # Common facts  
            {"sentence": "The largest planet in our solar system is ___.", "answer": "Jupiter"},  
            {"sentence": "The chemical symbol for gold is ___.", "answer": "Au"},  
            {"sentence": "Water boils at 100 degrees ___.", "answer": "Celsius"},  
            {"sentence": "The speed of light is approximately 300,000 kilometers per ___.", "answer": "second"},  
            {"sentence": "The currency of Japan is the ___.", "answer": "Yen"},  
        
            # Literary references  
            {"sentence": "Call me ___.", "answer": "Ishmael"},  
            {"sentence": "It was a bright cold day in April, and the clocks were striking ___.", "answer": "thirteen"},  
            {"sentence": "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a ___.", "answer": "wife"},  
            {"sentence": "To be, or not to be: that is the ___.", "answer": "question"},  
            {"sentence": "All that is gold does not ___.", "answer": "glitter"},  
        
            # More challenging examples  
            
            
             
            
            {"sentence": "Blood is thicker than ___.", "answer": "water"}  
        ]  
        if self.split == 'train':
            return data[:self.subset_size] if self.subset_size else data
        elif self.split == 'test':
            return holdout_test_set[:self.subset_size] if self.subset_size else holdout_test_set
        else:
            raise ValueError(f"Invalid split: {self.split}")
  
    def preprocess_data(self):  
        self.cached_data = []  
        for item in tqdm(self.data, desc=f"Preprocessing FillTheBlank {self.split} data"):  
            sentence = item['sentence']  
            answer = item['answer']  
  
            # Construct the prompt  
            prompt = f"Fill in the blank: {sentence}"  
  
            # Tokenize the prompt with left padding  
            self.tokenizer.padding_side = 'left'  
            encoded_prompt = self.tokenizer(  
                prompt,  
                max_length=self.max_input_length,  
                padding='max_length',  
                truncation=True,  
                return_tensors='pt',  
                add_special_tokens=True  
            )  
  
            input_ids = encoded_prompt['input_ids'].squeeze()  
            attention_mask = encoded_prompt['attention_mask'].squeeze()  
  
            # Tokenize the answer with right padding  
            self.tokenizer.padding_side = 'right'  
            encoded_answer = self.tokenizer(  
                answer,  
                max_length=self.max_output_length,  
                padding='max_length',  
                truncation=True,  
                return_tensors='pt',  
                add_special_tokens=True  
            )  
  
            labels = encoded_answer['input_ids'].squeeze()  
            labels_attention_mask = encoded_answer['attention_mask'].squeeze()  
  
            self.cached_data.append({  
                'input_ids': input_ids,  
                'attention_mask': attention_mask,  
                'labels': labels,  
                'labels_attention_mask': labels_attention_mask,  
                'reference_answer': answer.lower()  
            })  
        return self.cached_data  
  
    def extract_answer(self, text):  
        return text.strip().lower()  
  
    def get_evaluation_metrics(self):  
        # For FillTheBlank, we use exact match accuracy  
        def exact_match(predictions, references):  
            correct = 0  
            total = 0  
            for pred, ref in zip(predictions, references):  
                # lower case both pred and ref
                if pred is not None and ref is not None and pred.lower() == ref.lower():  
                    correct += 1  
                total += 1  
            accuracy = correct / total if total > 0 else 0  
            return {'exact_match': accuracy}  
        # reference is contained in the output
        def reference_contained(predictions, references):
            correct = 0
            total = 0
            for pred, ref in zip(predictions, references):
                pred = pred.strip().lower().replace("\n", " ")
                ref = ref.strip().lower().replace("\n", "")
                # lower case both pred and ref
                
                if pred is not None and ref is not None and ref.lower() in pred.lower():
                    correct += 1
                total += 1
                # print(pred, ref, " | ", ref.lower() in pred.lower(), " | ", correct, total)
            accuracy = correct / total if total > 0 else 0
            # print(f"Accuracy: {accuracy:.2f} ({correct}/{total} correct)")
            return {'reference_contained': accuracy}
        return {'exact_match': exact_match, 'reference_contained': reference_contained}  
  
class CompleteTheSentenceDataset(BaseDataset):  
    def load_data(self):  
        # Manually create the dataset  
        data = ([  
            # Proverbs and common sayings  
            {"sentence": "The quick brown fox jumps over the lazy ___.", "completion": "dog"},  
            {"sentence": "A stitch in time saves ___.", "completion": "nine"},  
            {"sentence": "An apple a day keeps the doctor ___.", "completion": "away"},  
            {"sentence": "All that glitters is not ___.", "completion": "gold"},  
            {"sentence": "A journey of a thousand miles begins with a single ___.", "completion": "step"},  
            {"sentence": "Better late than ___.", "completion": "never"},  
            {"sentence": "Birds of a feather flock ___.", "completion": "together"},  
            {"sentence": "Actions speak louder than ___.", "completion": "words"},  
            {"sentence": "Beauty is in the eye of the ___.", "completion": "beholder"},  
            {"sentence": "The early bird catches the ___.", "completion": "worm"},  
            {"sentence": "Don't count your chickens before they ___.", "completion": "hatch"},  
            {"sentence": "Two wrongs don't make a ___.", "completion": "right"},  
            {"sentence": "When in Rome, do as the Romans ___.", "completion": "do"},  
            {"sentence": "The pen is mightier than the ___.", "completion": "sword"},  
            {"sentence": "Every cloud has a silver ___.", "completion": "lining"},  
            {"sentence": "A penny saved is a penny ___.", "completion": "earned"},  
            {"sentence": "Curiosity killed the ___.", "completion": "cat"},  
            {"sentence": "Don't cry over spilled ___.", "completion": "milk"},  
            {"sentence": "The squeaky wheel gets the ___.", "completion": "grease"},  
            {"sentence": "You can't judge a book by its ___.", "completion": "cover"},  
              
            # Wise musings and philosophical statements  
            {"sentence": "To be or not to be, that is the ___.", "completion": "question"},  
            {"sentence": "Knowledge is ___.", "completion": "power"},  
            {"sentence": "Time is ___.", "completion": "money"},  
            {"sentence": "The best things in life are ___.", "completion": "free"},  
            {"sentence": "Where there's a will, there's a ___.", "completion": "way"},  
            {"sentence": "Practice makes ___.", "completion": "perfect"},  
            {"sentence": "Honesty is the best ___.", "completion": "policy"},  
            {"sentence": "Fortune favors the ___.", "completion": "bold"},  
            {"sentence": "Necessity is the mother of ___.", "completion": "invention"},  
            {"sentence": "The only constant is ___.", "completion": "change"},  
              
            # Literary references  
            {"sentence": "It was the best of times, it was the worst of ___.", "completion": "times"},  
            {"sentence": "All animals are equal, but some animals are more equal than ___.", "completion": "others"},  
            {"sentence": "To be, or not to be: that is the ___.", "completion": "question"},  
            {"sentence": "I think, therefore I ___.", "completion": "am"},  
            {"sentence": "Ask not what your country can do for you, ask what you can do for your ___.", "completion": "country"},  
              
            # More challenging examples  
            {"sentence": "The proof of the pudding is in the ___.", "completion": "eating"},  
            {"sentence": "A wolf in sheep's ___.", "completion": "clothing"},  
            {"sentence": "The road to hell is paved with good ___.", "completion": "intentions"},  
            {"sentence": "Don't throw the baby out with the ___.", "completion": "bathwater"},  
            {"sentence": "A bird in the hand is worth two in the ___.", "completion": "bush"},  
            {"sentence": "You can lead a horse to water, but you can't make it ___.", "completion": "drink"},  
            {"sentence": "People who live in glass houses shouldn't throw ___.", "completion": "stones"},  
            {"sentence": "The grass is always greener on the other side of the ___.", "completion": "fence"},  
            {"sentence": "A rolling stone gathers no ___.", "completion": "moss"},  
            {"sentence": "When the cat's away, the mice will ___.", "completion": "play"},  
        ]  + [  
            {"sentence": "A friend in need is a friend ___.", "completion": "indeed", "theme": "Proverbs", "difficulty": "Easy", "cultural_origin": "Global", "time_period": "Traditional"},  
            {"sentence": "An apple a day keeps the doctor ___.", "completion": "away", "theme": "Proverbs", "difficulty": "Easy", "cultural_origin": "English", "time_period": "Traditional"},  
            {"sentence": "Early to bed and early to rise makes a man healthy, wealthy, and ___.", "completion": "wise", "theme": "Proverbs", "difficulty": "Easy", "cultural_origin": "American", "time_period": "18th Century"},  
            {"sentence": "Laughter is the best ___.", "completion": "medicine", "theme": "Proverbs", "difficulty": "Easy", "cultural_origin": "Global", "time_period": "Traditional"},  
            {"sentence": "Where there's a will, there's a ___.", "completion": "way", "theme": "Proverbs", "difficulty": "Easy", "cultural_origin": "English", "time_period": "Traditional"}  
        ]  + [  
            {"sentence": "A bird in the hand is worth two in the ___.", "completion": "bush", "theme": "Proverbs", "difficulty": "Medium", "cultural_origin": "English", "time_period": "Traditional"},  
            {"sentence": "Don't put all your eggs in one ___.", "completion": "basket", "theme": "Proverbs", "difficulty": "Medium", "cultural_origin": "English", "time_period": "Traditional"},  
            {"sentence": "The early bird catches the ___.", "completion": "worm", "theme": "Proverbs", "difficulty": "Medium", "cultural_origin": "English", "time_period": "Traditional"},  
            {"sentence": "You can lead a horse to water, but you can't make it ___.", "completion": "drink", "theme": "Proverbs", "difficulty": "Medium", "cultural_origin": "English", "time_period": "Traditional"},  
            {"sentence": "A penny saved is a penny ___.", "completion": "earned", "theme": "Proverbs", "difficulty": "Medium", "cultural_origin": "American", "time_period": "18th Century"}  
        ]  + [  
            {"sentence": "The best-laid plans of mice and ___ often go awry.", "completion": "men", "theme": "Proverbs", "difficulty": "Hard", "cultural_origin": "Scottish", "time_period": "18th Century"},  
            {"sentence": "A watched pot never ___.", "completion": "boils", "theme": "Proverbs", "difficulty": "Hard", "cultural_origin": "English", "time_period": "Traditional"},  
            {"sentence": "The road to hell is paved with good ___.", "completion": "intentions", "theme": "Proverbs", "difficulty": "Hard", "cultural_origin": "English", "time_period": "Traditional"},  
            {"sentence": "Discretion is the better part of ___.", "completion": "valor", "theme": "Proverbs", "difficulty": "Hard", "cultural_origin": "English", "time_period": "Traditional"},  
            {"sentence": "There's many a slip 'twixt the cup and the ___.", "completion": "lip", "theme": "Proverbs", "difficulty": "Hard", "cultural_origin": "English", "time_period": "Traditional"}  
        ]  + [  
            {"sentence": "Knowledge is ___.", "completion": "power", "theme": "Philosophy", "difficulty": "Easy", "cultural_origin": "Global", "time_period": "Modern"},  
            {"sentence": "The only way to do great work is to love what you ___.", "completion": "do", "theme": "Philosophy", "difficulty": "Easy", "cultural_origin": "American", "time_period": "Modern"},  
            {"sentence": "Life is what happens when you're busy making other ___.", "completion": "plans", "theme": "Philosophy", "difficulty": "Easy", "cultural_origin": "English", "time_period": "20th Century"},  
            {"sentence": "Be the change you wish to see in the ___.", "completion": "world", "theme": "Philosophy", "difficulty": "Easy", "cultural_origin": "Indian", "time_period": "20th Century"},  
            {"sentence": "The future belongs to those who believe in the beauty of their ___.", "completion": "dreams", "theme": "Philosophy", "difficulty": "Easy", "cultural_origin": "American", "time_period": "20th Century"}  
        ]  + [  
            {"sentence": "The unexamined life is not worth ___.", "completion": "living", "theme": "Philosophy", "difficulty": "Medium", "cultural_origin": "Greek", "time_period": "Ancient"},  
            {"sentence": "I think, therefore I ___.", "completion": "am", "theme": "Philosophy", "difficulty": "Medium", "cultural_origin": "French", "time_period": "17th Century"},  
            {"sentence": "We are what we repeatedly ___.", "completion": "do", "theme": "Philosophy", "difficulty": "Medium", "cultural_origin": "Greek", "time_period": "Ancient"},  
            {"sentence": "The only true wisdom is in knowing you know ___.", "completion": "nothing", "theme": "Philosophy", "difficulty": "Medium", "cultural_origin": "Greek", "time_period": "Ancient"},  
            {"sentence": "Man is by nature a political ___.", "completion": "animal", "theme": "Philosophy", "difficulty": "Medium", "cultural_origin": "Greek", "time_period": "Ancient"}  
        ]  + [  
            {"sentence": "The moving finger writes, and having writ, moves ___.", "completion": "on", "theme": "Philosophy", "difficulty": "Hard", "cultural_origin": "Persian", "time_period": "11th Century"},  
            {"sentence": "Cogito, ergo ___.", "completion": "sum", "theme": "Philosophy", "difficulty": "Hard", "cultural_origin": "Latin", "time_period": "17th Century"},  
            {"sentence": "The owl of Minerva spreads its wings only with the falling of the ___.", "completion": "dusk", "theme": "Philosophy", "difficulty": "Hard", "cultural_origin": "German", "time_period": "19th Century"},  
            {"sentence": "Man is condemned to be ___.", "completion": "free", "theme": "Philosophy", "difficulty": "Hard", "cultural_origin": "French", "time_period": "20th Century"},  
            {"sentence": "That which does not kill us makes us ___.", "completion": "stronger", "theme": "Philosophy", "difficulty": "Hard", "cultural_origin": "German", "time_period": "19th Century"}  
        ]  + [  
            {"sentence": "To be, or not to be: that is the ___.", "completion": "question", "theme": "Literature", "difficulty": "Easy", "cultural_origin": "English", "time_period": "16th Century"},  
            {"sentence": "It was the best of times, it was the worst of ___.", "completion": "times", "theme": "Literature", "difficulty": "Easy", "cultural_origin": "English", "time_period": "19th Century"},  
            {"sentence": "Call me ___.", "completion": "Ishmael", "theme": "Literature", "difficulty": "Easy", "cultural_origin": "American", "time_period": "19th Century"},  
            {"sentence": "In a hole in the ground there lived a ___.", "completion": "hobbit", "theme": "Literature", "difficulty": "Easy", "cultural_origin": "English", "time_period": "20th Century"},  
            {"sentence": "The catcher in the ___.", "completion": "rye", "theme": "Literature", "difficulty": "Easy", "cultural_origin": "American", "time_period": "20th Century"}  
        ]  + [  
            {"sentence": "All animals are equal, but some animals are more equal than ___.", "completion": "others", "theme": "Literature", "difficulty": "Medium", "cultural_origin": "English", "time_period": "20th Century"},  
            {"sentence": "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a ___.", "completion": "wife", "theme": "Literature", "difficulty": "Medium", "cultural_origin": "English", "time_period": "19th Century"},  
            {"sentence": "The man in black fled across the desert, and the gunslinger ___.", "completion": "followed", "theme": "Literature", "difficulty": "Medium", "cultural_origin": "American", "time_period": "20th Century"},  
            {"sentence": "All the world's a stage, and all the men and women merely ___.", "completion": "players", "theme": "Literature", "difficulty": "Medium", "cultural_origin": "English", "time_period": "16th Century"},  
            {"sentence": "It was a bright cold day in April, and the clocks were striking ___.", "completion": "thirteen", "theme": "Literature", "difficulty": "Medium", "cultural_origin": "English", "time_period": "20th Century"}  
        ]  + [  
            {"sentence": "Stately, plump Buck Mulligan came from the stairhead, bearing a bowl of lather on which a mirror and a razor lay ___.", "completion": "crossed", "theme": "Literature", "difficulty": "Hard", "cultural_origin": "Irish", "time_period": "20th Century"},  
            {"sentence": "Many years later, as he faced the firing squad, Colonel Aureliano Buendía was to remember that distant afternoon when his father took him to discover ___.", "completion": "ice", "theme": "Literature", "difficulty": "Hard", "cultural_origin": "Colombian", "time_period": "20th Century"},  
            {"sentence": "The sky above the port was the color of television, tuned to a dead ___.", "completion": "channel", "theme": "Literature", "difficulty": "Hard", "cultural_origin": "American", "time_period": "20th Century"},  
            {"sentence": "If you really want to hear about it, the first thing you'll probably want to know is where I was born, and what my lousy childhood was like, and how my parents were occupied and all before they had me, and all that David Copperfield kind of ___.", "completion": "crap", "theme": "Literature", "difficulty": "Hard", "cultural_origin": "American", "time_period": "20th Century"},  
            {"sentence": "He was an old man who fished alone in a skiff in the Gulf Stream and he had gone eighty-four days now without taking a ___.", "completion": "fish", "theme": "Literature", "difficulty": "Hard", "cultural_origin": "American", "time_period": "20th Century"}  
        ]  + [  
            {"sentence": "Every action has an equal and opposite ___.", "completion": "reaction", "theme": "Science", "difficulty": "Easy", "cultural_origin": "English", "time_period": "17th Century"},  
            {"sentence": "Energy can neither be created nor destroyed, only ___.", "completion": "transformed", "theme": "Science", "difficulty": "Easy", "cultural_origin": "Global", "time_period": "19th Century"},  
            {"sentence": "The square of the hypotenuse is equal to the sum of the squares of the other two ___.", "completion": "sides", "theme": "Mathematics", "difficulty": "Easy", "cultural_origin": "Greek", "time_period": "Ancient"},  
            {"sentence": "Matter can neither be created nor ___.", "completion": "destroyed", "theme": "Science", "difficulty": "Easy", "cultural_origin": "Global", "time_period": "18th Century"},  
            {"sentence": "What goes up must come ___.", "completion": "down", "theme": "Science", "difficulty": "Easy", "cultural_origin": "English", "time_period": "17th Century"}  
        ]  + [  
            {"sentence": "In nature, nothing is created, nothing is lost, everything ___.", "completion": "changes", "theme": "Science", "difficulty": "Medium", "cultural_origin": "Greek", "time_period": "Ancient"},  
            {"sentence": "The whole is greater than the sum of its ___.", "completion": "parts", "theme": "Science", "difficulty": "Medium", "cultural_origin": "Greek", "time_period": "Ancient"},  
            {"sentence": "Nature abhors a ___.", "completion": "vacuum", "theme": "Science", "difficulty": "Medium", "cultural_origin": "Greek", "time_period": "Ancient"},  
            {"sentence": "Survival of the ___.", "completion": "fittest", "theme": "Science", "difficulty": "Medium", "cultural_origin": "English", "time_period": "19th Century"},  
            {"sentence": "For every complex problem there is an answer that is clear, simple, and ___.", "completion": "wrong", "theme": "Science", "difficulty": "Medium", "cultural_origin": "American", "time_period": "20th Century"}  
        ]  + [  
            {"sentence": "The entropy of an isolated system always ___.", "completion": "increases", "theme": "Science", "difficulty": "Hard", "cultural_origin": "German", "time_period": "19th Century"},  
            {"sentence": "In quantum mechanics, the position and momentum of a particle cannot be simultaneously measured with arbitrary ___.", "completion": "precision", "theme": "Science", "difficulty": "Hard", "cultural_origin": "German", "time_period": "20th Century"},  
            {"sentence": "The curvature of spacetime is directly proportional to the energy and momentum of whatever matter and radiation are ___.", "completion": "present", "theme": "Science", "difficulty": "Hard", "cultural_origin": "German", "time_period": "20th Century"},  
            {"sentence": "In the limit, as the sample size approaches infinity, the sampling distribution of the mean approaches a normal ___.", "completion": "distribution", "theme": "Mathematics", "difficulty": "Hard", "cultural_origin": "Global", "time_period": "20th Century"},  
            {"sentence": "The integral of the product of two functions is equal to the integral of the product of one function and the derivative of the other, minus the integral of the product of the derivative of the first function and the ___.", "completion": "second", "theme": "Mathematics", "difficulty": "Hard", "cultural_origin": "Global", "time_period": "18th Century"}  
        ]  + [  
            {"sentence": "Break a ___.", "completion": "leg", "theme": "Idioms", "difficulty": "Easy", "cultural_origin": "English", "time_period": "Modern"},  
            {"sentence": "It's raining cats and ___.", "completion": "dogs", "theme": "Idioms", "difficulty": "Easy", "cultural_origin": "English", "time_period": "Modern"},  
            {"sentence": "The ball is in your ___.", "completion": "court", "theme": "Idioms", "difficulty": "Easy", "cultural_origin": "English", "time_period": "Modern"},  
            {"sentence": "Bite off more than you can ___.", "completion": "chew", "theme": "Idioms", "difficulty": "Easy", "cultural_origin": "English", "time_period": "Modern"},  
            {"sentence": "Speak of the ___.", "completion": "devil", "theme": "Idioms", "difficulty": "Easy", "cultural_origin": "English", "time_period": "Modern"}  
        ]  + [  
            {"sentence": "The elephant in the ___.", "completion": "room", "theme": "Idioms", "difficulty": "Medium", "cultural_origin": "English", "time_period": "Modern"},  
            {"sentence": "Barking up the wrong ___.", "completion": "tree", "theme": "Idioms", "difficulty": "Medium", "cultural_origin": "English", "time_period": "Modern"},  
            {"sentence": "Cut to the ___.", "completion": "chase", "theme": "Idioms", "difficulty": "Medium", "cultural_origin": "English", "time_period": "Modern"},  
            {"sentence": "Hit the nail on the ___.", "completion": "head", "theme": "Idioms", "difficulty": "Medium", "cultural_origin": "English", "time_period": "Modern"},  
            {"sentence": "Spill the ___.", "completion": "beans", "theme": "Idioms", "difficulty": "Medium", "cultural_origin": "English", "time_period": "Modern"}  
        ]  + [  
            {"sentence": "The pot calling the kettle ___.", "completion": "black", "theme": "Idioms", "difficulty": "Hard", "cultural_origin": "English", "time_period": "Traditional"},  
            {"sentence": "A diamond in the ___.", "completion": "rough", "theme": "Idioms", "difficulty": "Hard", "cultural_origin": "English", "time_period": "Traditional"},  
            {"sentence": "Caught between Scylla and ___.", "completion": "Charybdis", "theme": "Idioms", "difficulty": "Hard", "cultural_origin": "Greek", "time_period": "Ancient"},  
            {"sentence": "Burning the candle at both ___.", "completion": "ends", "theme": "Idioms", "difficulty": "Hard", "cultural_origin": "English", "time_period": "Traditional"},  
            {"sentence": "The writing is on the ___.", "completion": "wall", "theme": "Idioms", "difficulty": "Hard", "cultural_origin": "English", "time_period": "Traditional"}  
        ])
        
        holdout_test_set = [  
            # Proverbs and common sayings (Easy)  
            {"sentence": "A picture is worth a thousand ___.", "completion": "words", "difficulty": "easy"},  
            {"sentence": "Make hay while the sun ___.", "completion": "shines", "difficulty": "easy"},  
            {"sentence": "The early bird catches the ___.", "completion": "worm", "difficulty": "easy"},  
            {"sentence": "Look before you ___.", "completion": "leap", "difficulty": "easy"},  
            {"sentence": "Better safe than ___.", "completion": "sorry", "difficulty": "easy"},  
        
            # Proverbs and common sayings (Medium)  
            {"sentence": "A chain is only as strong as its weakest ___.", "completion": "link", "difficulty": "medium"},  
            {"sentence": "The apple doesn't fall far from the ___.", "completion": "tree", "difficulty": "medium"},  
            {"sentence": "Don't put all your eggs in one ___.", "completion": "basket", "difficulty": "medium"},  
            {"sentence": "A watched pot never ___.", "completion": "boils", "difficulty": "medium"},  
            {"sentence": "The grass is always greener on the other side of the ___.", "completion": "fence", "difficulty": "medium"},  
        
            # Proverbs and common sayings (Hard)  
            {"sentence": "The cobbler's children have no ___.", "completion": "shoes", "difficulty": "hard"},  
            {"sentence": "A rising tide lifts all ___.", "completion": "boats", "difficulty": "hard"},  
            {"sentence": "The proof of the pudding is in the ___.", "completion": "eating", "difficulty": "hard"},  
            {"sentence": "Don't throw the baby out with the ___.", "completion": "bathwater", "difficulty": "hard"},  
            {"sentence": "The exception proves the ___.", "completion": "rule", "difficulty": "hard"},  
        
            # Wise musings and philosophical statements (Easy)  
            {"sentence": "Knowledge is ___.", "completion": "power", "difficulty": "easy"},  
            {"sentence": "Life is what happens when you're busy making other ___.", "completion": "plans", "difficulty": "easy"},  
            {"sentence": "The journey of a thousand miles begins with a single ___.", "completion": "step", "difficulty": "easy"},  
            {"sentence": "United we stand, divided we ___.", "completion": "fall", "difficulty": "easy"},  
            {"sentence": "To err is human, to forgive ___.", "completion": "divine", "difficulty": "easy"},  
        
            # Wise musings and philosophical statements (Medium)  
            {"sentence": "The unexamined life is not worth ___.", "completion": "living", "difficulty": "medium"},  
            {"sentence": "We are what we repeatedly ___.", "completion": "do", "difficulty": "medium"},  
            {"sentence": "The only true wisdom is in knowing you know ___.", "completion": "nothing", "difficulty": "medium"},  
            {"sentence": "He who has a why to live can bear almost any ___.", "completion": "how", "difficulty": "medium"},  
            {"sentence": "The greatest wealth is to live content with ___.", "completion": "little", "difficulty": "medium"},  
        
            # Wise musings and philosophical statements (Hard)  
            {"sentence": "Cogito, ergo ___.", "completion": "sum", "difficulty": "hard"},  
            {"sentence": "The owl of Minerva spreads its wings only with the falling of the ___.", "completion": "dusk", "difficulty": "hard"},  
            {"sentence": "That which does not kill us makes us ___.", "completion": "stronger", "difficulty": "hard"},  
            {"sentence": "The life which is unexamined is not worth ___.", "completion": "living", "difficulty": "hard"},  
            {"sentence": "Man is condemned to be ___.", "completion": "free", "difficulty": "hard"},  
        
            # Literary references (Easy)  
            {"sentence": "To be, or not to be: that is the ___.", "completion": "question", "difficulty": "easy"},  
            {"sentence": "It was the best of times, it was the worst of ___.", "completion": "times", "difficulty": "easy"},  
            {"sentence": "Call me ___.", "completion": "Ishmael", "difficulty": "easy"},  
            {"sentence": "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a ___.", "completion": "wife", "difficulty": "easy"},  
            {"sentence": "All animals are equal, but some animals are more equal than ___.", "completion": "others", "difficulty": "easy"},  
        
            # Literary references (Medium)  
            {"sentence": "In the beginning God created the heaven and the ___.", "completion": "earth", "difficulty": "medium"},  
            {"sentence": "It was a bright cold day in April, and the clocks were striking ___.", "completion": "thirteen", "difficulty": "medium"},  
            {"sentence": "Two roads diverged in a yellow wood, and I— I took the one less traveled ___.", "completion": "by", "difficulty": "medium"},  
            {"sentence": "All happy families are alike; each unhappy family is unhappy in its own ___.", "completion": "way", "difficulty": "medium"},  
            {"sentence": "Ask not what your country can do for you – ask what you can do for your ___.", "completion": "country", "difficulty": "medium"},  
        
            # Literary references (Hard)  
            {"sentence": "Stately, plump Buck Mulligan came from the stairhead, bearing a bowl of lather on which a mirror and a razor lay ___.", "completion": "crossed", "difficulty": "hard"},  
            {"sentence": "Many years later, as he faced the firing squad, Colonel Aureliano Buendía was to remember that distant afternoon when his father took him to discover ___.", "completion": "ice", "difficulty": "hard"},  
            {"sentence": "The sky above the port was the color of television, tuned to a dead ___.", "completion": "channel", "difficulty": "hard"},  
            {"sentence": "If you really want to hear about it, the first thing you'll probably want to know is where I was born, and what my lousy childhood was like, and how my parents were occupied and all before they had me, and all that David Copperfield kind of ___.", "completion": "crap", "difficulty": "hard"},  
            {"sentence": "He was an old man who fished alone in a skiff in the Gulf Stream and he had gone eighty-four days now without taking a ___.", "completion": "fish", "difficulty": "hard"},  
        
            # More challenging examples (Medium)  
            {"sentence": "The elephant in the ___.", "completion": "room", "difficulty": "medium"},  
            {"sentence": "Barking up the wrong ___.", "completion": "tree", "difficulty": "medium"},  
            {"sentence": "Every dog has its ___.", "completion": "day", "difficulty": "medium"},  
            {"sentence": "The devil is in the ___.", "completion": "details", "difficulty": "medium"},  
            {"sentence": "A penny for your ___.", "completion": "thoughts", "difficulty": "medium"},  
        
            # More challenging examples (Hard)  
            {"sentence": "The road to hell is paved with good ___.", "completion": "intentions", "difficulty": "hard"},  
            {"sentence": "A wolf in sheep's ___.", "completion": "clothing", "difficulty": "hard"},  
            {"sentence": "The pot calling the kettle ___.", "completion": "black", "difficulty": "hard"},  
            {"sentence": "Discretion is the better part of ___.", "completion": "valor", "difficulty": "hard"},  
            {"sentence": "The writing is on the ___.", "completion": "wall", "difficulty": "hard"}  
        ]  
















        if self.split == 'train':
            return data[:self.subset_size] if self.subset_size else data
        elif self.split == 'test':
            return holdout_test_set[:self.subset_size] if self.subset_size else holdout_test_set
        else:
            raise ValueError(f"Invalid split: {self.split}")
  
    def preprocess_data(self):  
        self.cached_data = []  
        for item in tqdm(self.data, desc=f"Preprocessing CompleteTheSentence {self.split} data"):  
            sentence = item['sentence']  
            completion = item['completion']  
  
            # Construct the prompt  
            prompt = f"Complete the sentence: {sentence}"  
  
            # Tokenize the prompt with left padding  
            self.tokenizer.padding_side = 'left'  
            encoded_prompt = self.tokenizer(  
                prompt,  
                max_length=self.max_input_length,  
                padding='max_length',  
                truncation=True,  
                return_tensors='pt',  
                add_special_tokens=True  
            )  
  
            input_ids = encoded_prompt['input_ids'].squeeze()  
            attention_mask = encoded_prompt['attention_mask'].squeeze()  
  
            # Tokenize the completion with right padding  
            self.tokenizer.padding_side = 'right'  
            encoded_completion = self.tokenizer(  
                completion,  
                max_length=self.max_output_length,  
                padding='max_length',  
                truncation=True,  
                return_tensors='pt',  
                add_special_tokens=True  
            )  
  
            labels = encoded_completion['input_ids'].squeeze()  
            labels_attention_mask = encoded_completion['attention_mask'].squeeze()  
  
            self.cached_data.append({  
                'input_ids': input_ids,  
                'attention_mask': attention_mask,  
                'labels': labels,  
                'labels_attention_mask': labels_attention_mask,  
                'reference_answer': completion.lower()  
            })  
        return self.cached_data  
  
    def extract_answer(self, text):  
        return text.strip().lower()  
  
    def get_evaluation_metrics(self):  
        # For CompleteTheSentence, we use exact match accuracy  
        def exact_match(predictions, references):  
            correct = 0  
            total = 0  
            for pred, ref in zip(predictions, references):  
                # lower case both pred and ref
                if pred is not None and ref is not None and pred.lower() == ref.lower():  
                    correct += 1  
                total += 1  
            accuracy = correct / total if total > 0 else 0  
            return {'exact_match': accuracy}  
        # reference is contained in the output
        def reference_contained(predictions, references):
            correct = 0
            total = 0
            for pred, ref in zip(predictions, references):
                # lower case both pred and ref
                if pred is not None and ref is not None and ref.lower() in pred.lower():
                    correct += 1
                total += 1
            accuracy = correct / total if total > 0 else 0
            return {'reference_contained': accuracy}
        return {'exact_match': exact_match, 'reference_contained': reference_contained}  

def get_validation_split(dataset_name):
    validation_splits = {
        'gsm8k': 'test',
        'mlqa': 'validation_matched',
        'squad': 'validation',
        'mmlu': 'test',
        'xnli': 'validation',
        'fill_the_blank': 'test',
        'complete_the_sentence': 'test',
        # Add other datasets here
    }
    return validation_splits.get(dataset_name.lower())
  