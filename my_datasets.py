# datasets.py  

import torch  
from torch.utils.data import Dataset  
from datasets import load_dataset  
from transformers import AutoTokenizer  
from tqdm import tqdm  
  
class BaseDataset(Dataset):  
    def __init__(self, dataset_name, tokenizer_name, max_input_length, max_output_length, split, subset_size=None):  
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
        return ''.join(answer.split()).lower()  
  
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
            return {'accuracy': accuracy}  
        return {'exact_match': exact_match}  
  
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
        data = [  
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
            {"sentence": "To be, or not to be: that is the ___.", "answer": "question"},  
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
        ]  
        return data[:self.subset_size] 
  
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
        data = [  
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
        ]  
        return data[:self.subset_size]
  
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
  