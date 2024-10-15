import torch  
import torch.nn as nn  
import torch.optim as optim  
from torch.utils.data import Dataset, DataLoader  
from torch.optim.lr_scheduler import OneCycleLR  
from datasets import load_dataset  
from transformers import AutoTokenizer  
from model_comm_v2 import DualModelTransformer  
import random  
import numpy as np  
from tqdm import tqdm  
import logging  
import os  
from torch.utils.tensorboard import SummaryWriter  
import warnings  
warnings.filterwarnings("ignore")  
  
# Set up logging  
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  
logger = logging.getLogger(__name__)  
  
import torch  
import pandas as pd  
from tabulate import tabulate  
  
def print_gpu_memory_usage():  
    """  
    Retrieves and prints the memory usage for all available GPUs in a formatted table.  
    """  
    if not torch.cuda.is_available():  
        print("CUDA is not available. No GPU detected.")  
        return  
    # Get the number of available GPUs  
    num_gpus = torch.cuda.device_count()  
  
    # Prepare data for the table  
    data = []  
    for i in range(num_gpus):  
        torch.cuda.set_device(i)  
        total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # Convert to GB  
        allocated_memory = torch.cuda.memory_allocated(i) / (1024**3)  
        cached_memory = torch.cuda.memory_reserved(i) / (1024**3)  
        free_memory = total_memory - allocated_memory - cached_memory  
  
        data.append([  
            f"GPU {i}",  
            f"{total_memory:.2f}",  
            f"{allocated_memory:.2f}",  
            f"{cached_memory:.2f}",  
            f"{free_memory:.2f}"  
        ])  
    # Create a pandas DataFrame  
    df = pd.DataFrame(data, columns=["GPU", "Total (GB)", "Allocated (GB)", "Cached (GB)", "Free (GB)"])  
    # Print the table  
    print("\nGPU Memory Usage:")  
    print(tabulate(df, headers='keys', tablefmt='pretty', showindex=False))  
  
# Configuration  
config = {  
    "large_model_name": "microsoft/Phi-3-small-8k-instruct",  
    "small_model_name": "microsoft/Phi-3-mini-4k-instruct",  
    "stop_tokens": [".", "!", "?"],  
    "small_model_dim": 3072,  
    "large_model_dim": 4096,  
    "learning_rate": 2e-5,  
    "batch_size": 1,  
    "num_epochs": 3,  
    "warmup_steps": 100,  
    "max_grad_norm": 1.0,  
    "train_subset_size": 1000,  # Set to None to use full dataset  
    "test_subset_size": 100,    # Set to None to use full dataset  
    "weight_decay": 0.01,  
    "gradient_accumulation_steps": 4,  
    "num_workers": 4,  
    "max_length": 512,  
    "main_device": "cuda:0",  
    "second_device": "cuda:1",  
    "third_device": "cuda:2",  
}  
  
# Set random seeds for reproducibility  
def set_seed(seed):  
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
  
set_seed(42)  
  
# Custom Dataset class  
class GSM8KDataset(Dataset):  
    def __init__(self, dataset, tokenizer, max_length):  
        self.dataset = dataset  
        self.tokenizer = tokenizer  
        self.max_length = max_length  
        self.cached_data = self.preprocess_data()  
  
    def preprocess_data(self):  
        cached_data = []  
        for item in tqdm(self.dataset, desc="Preprocessing data"):  
            question = item['question']  
            answer = item['answer']  
            prompt = f"Solve the following math problem step by step. Show your work and provide the final answer after '#### '.\n\nQuestion: {question}\n\nStep by Step working and Answer:"  
            inputs = self.tokenizer(prompt, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")  
            labels = self.tokenizer(answer, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")  
            cached_data.append({  
                'input_ids': inputs['input_ids'].squeeze(),  
                'attention_mask': inputs['attention_mask'].squeeze(),  
                'labels': labels['input_ids'].squeeze()  
            })  
        return cached_data  
  
    def __len__(self):  
        return len(self.cached_data)  
  
    def __getitem__(self, idx):  
        return self.cached_data[idx]  
  
# Data processing function  
def process_data(config):  
    dataset = load_dataset("gsm8k", "main")  
    train_dataset = dataset["train"]  
    test_dataset = dataset["test"]  
  
    if config["train_subset_size"]:  
        train_dataset = train_dataset.select(range(config["train_subset_size"]))  
    if config["test_subset_size"]:  
        test_dataset = test_dataset.select(range(config["test_subset_size"]))  
  
    tokenizer = AutoTokenizer.from_pretrained(config["small_model_name"], trust_remote_code=True)  
    tokenizer.padding_side = 'left'
    train_dataset = GSM8KDataset(train_dataset, tokenizer, config["max_length"])  
    test_dataset = GSM8KDataset(test_dataset, tokenizer, config["max_length"])  
  
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])  
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])  
  
    return train_loader, test_loader, tokenizer  
  
# Model definition function  
def create_model(config):  
    return DualModelTransformer(  
        config["large_model_name"],  
        config["small_model_name"],  
        config["stop_tokens"],  
        config["small_model_dim"],  
        config["large_model_dim"],  
        config["max_length"],  
        config["main_device"],  
        config["second_device"],  
        config["third_device"]  
    )  
  
# Training function  
def train_epoch(model, train_loader, optimizer, scheduler, config):  
    model.train()  
    total_loss = 0  
    progress_bar = tqdm(train_loader, desc="Training")  
    for step, batch in enumerate(progress_bar):  
        batch = {k: v.to(config["main_device"]) for k, v in batch.items()}  
        outputs = model(batch['input_ids'], batch['attention_mask'], labels=batch['labels'])  
        loss = outputs.loss  
        total_loss += loss.item()  
          
        loss = loss / config["gradient_accumulation_steps"]  
        loss.backward()  
  
        if (step + 1) % config["gradient_accumulation_steps"] == 0:  
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])  
            optimizer.step()  
            scheduler.step()  
            optimizer.zero_grad()  
  
        progress_bar.set_postfix({'loss': loss.item() * config["gradient_accumulation_steps"]})  
  
    return total_loss / len(train_loader)  
  
# Evaluation function  
def extract_answer(text):  
    pos = text.find('####')  
    if pos == -1:  
        return None  
    answer = text[pos+4:].strip()  
    return ''.join(answer.split()).lower()  
  
def evaluate(model, test_loader, tokenizer, config):  
    model.eval()  
    total_loss = 0  
    correct = 0  
    total = 0  
    progress_bar = tqdm(test_loader, desc="Evaluating")  
  
    with torch.no_grad():  
        for batch in progress_bar:  
            batch = {k: v.to(config["main_device"]) for k, v in batch.items()}  
            outputs = model(batch['input_ids'], batch['attention_mask'], labels=batch['labels'])  
            loss = outputs.loss  
            total_loss += loss.item()  
  
            # Generate text  
            generated = model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])  
  
            # Convert generated tensors to text  
            generated_texts = tokenizer.batch_decode(generated, skip_special_tokens=True)  
            label_texts = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)  
  
            # Extract answers from generated and label texts  
            predicted_answers = [extract_answer(text) for text in generated_texts]  
            actual_answers = [extract_answer(text) for text in label_texts]  
  
            # Compare answers  
            for pred, actual in zip(predicted_answers, actual_answers):  
                if pred is not None and actual is not None and pred == actual:  
                    correct += 1  
                total += 1  
  
            progress_bar.set_postfix({'loss': loss.item()})  
  
    accuracy = correct / total if total > 0 else 0  
    return total_loss / len(test_loader), accuracy  
  
# Main training and evaluation pipeline  
def main():  
    train_loader, test_loader, tokenizer = process_data(config)  
    model = create_model(config)  
    
    print("Memory Usage after loading model")  
    print_gpu_memory_usage()  
  
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config["learning_rate"], weight_decay=config["weight_decay"])  
    scheduler = OneCycleLR(  
        optimizer,  
        max_lr=config["learning_rate"],  
        steps_per_epoch=len(train_loader) // config["gradient_accumulation_steps"],  
        epochs=config["num_epochs"],  
        pct_start=0.1  
    )  
    print("Memory Usage after loading optimizer")  
    print_gpu_memory_usage()  
  
    writer = SummaryWriter(log_dir='runs/gsm8k_training')  
    print("Memory Usage after loading SummaryWriter")  
    print_gpu_memory_usage()  
  
    best_accuracy = 0  
    for epoch in range(config["num_epochs"]):  
        logger.info(f"Epoch {epoch+1}/{config['num_epochs']}:")  
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, config)  
        test_loss, test_accuracy = evaluate(model, test_loader, tokenizer, config)  
        logger.info(f"Train Loss: {train_loss:.4f}")  
        logger.info(f"Test Loss: {test_loss:.4f}")  
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")  
  
        writer.add_scalar('Loss/train', train_loss, epoch)  
        writer.add_scalar('Loss/test', test_loss, epoch)  
        writer.add_scalar('Accuracy/test', test_accuracy, epoch)  
  
        if test_accuracy > best_accuracy and epoch > 0:  
            best_accuracy = test_accuracy  
            torch.save(model.state_dict(), "best_dual_model.pth")  
  
    writer.close()  
    logger.info(f"Best Test Accuracy: {best_accuracy:.4f}")  
  
if __name__ == "__main__":  
    main()  
