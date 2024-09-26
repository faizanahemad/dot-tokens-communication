import os 
os.environ["TOKENIZERS_PARALLELISM"] = "true" 
import torch  
import torch.nn as nn  
import torch.optim as optim  
from torch.utils.data import Dataset, DataLoader  
from torch.optim.lr_scheduler import OneCycleLR  
from datasets import load_dataset  
from transformers import AutoTokenizer  
from model_fsdp import DualModelTransformer, setup_fsdp, save_model, load_model  
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullStateDictConfig,
    StateDictType,
)

import random  
import numpy as np  
from tqdm import tqdm  
import logging  
from torch.utils.tensorboard import SummaryWriter  
import warnings  
import torch.distributed as dist  
from torch.nn.parallel import DistributedDataParallel as DDP  
from torch.utils.data.distributed import DistributedSampler  
  
warnings.filterwarnings("ignore")  
  
# Set up logging  
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  
logger = logging.getLogger(__name__)  
  
# Configuration  
config = {  
    "large_model_name": "EleutherAI/pythia-1b-deduped",  
    "small_model_name": "EleutherAI/pythia-410m",  
    "stop_tokens": ["<|endoftext|>", ".", "!", "?"],  
    "small_model_dim": 1024,  
    "large_model_dim": 2048,  
    "learning_rate": 1e-3,  
    "batch_size": 8,  
    "num_epochs": 10,  
    "warmup_steps": 10,  
    "max_grad_norm": 1.0,  
    "train_subset_size": 32,  # Set to None to use full dataset  
    "test_subset_size": 32,    # Set to None to use full dataset  
    "weight_decay": 0.001,  
    "gradient_accumulation_steps": 1,  
    "num_workers": 4,  
    "max_length": 256, 
    "scheduler": None, # "OneCycleLR"
}  
  
# Set random seeds for reproducibility  
def set_seed(seed):  
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
  
set_seed(42)  

# Custom Dataset class  
class GSM8KDataset(torch.utils.data.Dataset):  
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
  
            # Construct the prompt  
            prompt = f"Solve the following math problem step by step. Show your work and provide the final answer after '#### '.\n\nQuestion: {question}\n\nStep by Step working and Answer:"  
  
            # Tokenize the prompt with left padding  
            self.tokenizer.padding_side = 'left'  
            encoded_prompt = self.tokenizer(  
                prompt,  
                max_length=self.max_length,  
                padding='max_length',  
                truncation=True,  
                return_tensors='pt',  
                add_special_tokens=True  # Ensure [BOS] token is added if applicable  
            )  
  
            input_ids = encoded_prompt['input_ids'].squeeze()  
            attention_mask = encoded_prompt['attention_mask'].squeeze()  
  
            # Tokenize the answer with right padding  
            self.tokenizer.padding_side = 'right'  
            encoded_answer = self.tokenizer(  
                answer,  
                max_length=self.max_length,  
                padding='max_length',  
                truncation=True,  
                return_tensors='pt',  
                add_special_tokens=True  # Ensure [EOS] token is added if applicable  
            )  
  
            labels = encoded_answer['input_ids'].squeeze()  
            # Create labels' attention mask if needed  
            labels_attention_mask = encoded_answer['attention_mask'].squeeze()  
  
            cached_data.append({  
                'input_ids': input_ids,  
                'attention_mask': attention_mask,  
                'labels': labels,  
                'labels_attention_mask': labels_attention_mask  
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
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  
    train_dataset = GSM8KDataset(train_dataset, tokenizer, config["max_length"])  
    test_dataset = GSM8KDataset(test_dataset, tokenizer, config["max_length"])  
  
    train_sampler = DistributedSampler(train_dataset, shuffle=True)  
    test_sampler = DistributedSampler(test_dataset, shuffle=False)  
  
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], sampler=train_sampler, num_workers=config["num_workers"])  
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], sampler=test_sampler, num_workers=config["num_workers"])  
  
    return train_loader, test_loader, tokenizer  
  
# Model definition function  
def create_model(config, fsdp_config):  
    return DualModelTransformer(  
        config["large_model_name"],  
        config["small_model_name"],  
        config["stop_tokens"],  
        config["small_model_dim"],  
        config["large_model_dim"],  
        config["max_length"],  
        fsdp_config,
        enable_checkpointing=True,
        
    )  
  
# Training function  
def train_epoch(model, epoch, train_loader, optimizer, scheduler, config):  
    model.train()  
    total_loss = 0  
    train_loader.sampler.set_epoch(epoch)  
    progress_bar = tqdm(train_loader, desc="Training", disable=not dist.get_rank() == 0)  
  
    for step, batch in enumerate(progress_bar):  
        batch = {k: v.to(torch.cuda.current_device()) for k, v in batch.items()}  
        outputs = model(batch['input_ids'], batch['attention_mask'], labels=batch['labels'], labels_attention_mask=batch['labels_attention_mask'])  
        loss = outputs  
  
        loss = loss / config["gradient_accumulation_steps"]  
        loss.backward()  
  
        if (step + 1) % config["gradient_accumulation_steps"] == 0:  
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])  
            optimizer.step()  
            scheduler.step()  
            optimizer.zero_grad()  
  
        total_loss += loss.item() * config["gradient_accumulation_steps"]  
  
        if dist.get_rank() == 0:  
            progress_bar.set_postfix({'loss': loss.item() * config["gradient_accumulation_steps"]})  
  
    return total_loss / len(train_loader)  
  
# Evaluation function  
def extract_answer(text):  
    pos = text.find('####')  
    if pos == -1:  
        return None  
    answer = text[pos+4:].strip().replace("<|endoftext|>", "")  
    return ''.join(answer.split()).lower()  
  
def evaluate(model, test_loader, tokenizer, config, print_generations=False):  
    model.eval()  
    total_loss = 0  
    correct = 0  
    total = 0  
    progress_bar = tqdm(test_loader, desc="Evaluating", disable=not dist.get_rank() == 0)  
  
    with torch.no_grad():  
        for batch in progress_bar:  
            batch = {k: v.to(torch.cuda.current_device()) for k, v in batch.items()}  
            outputs = model(batch['input_ids'], batch['attention_mask'], labels=batch['labels'], labels_attention_mask=batch['labels_attention_mask'])  
            loss = outputs  
            total_loss += loss.item()  
  
            # Generate text  
            generated = model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])  
  
            # Debug information  
            # print(f"Generated type: {type(generated)}, generated: {generated}")  
            # print(f"Generated shape: {generated.shape if isinstance(generated, torch.Tensor) else 'Not a tensor'}")  
  
            # Convert generated tensors to text  
            if isinstance(generated, torch.Tensor):  
                generated_texts = tokenizer.batch_decode(generated, skip_special_tokens=True) 
            elif isinstance(generated, (list, tuple)) and isinstance(generated[0], torch.Tensor):
                generated_texts = [tokenizer.decode(text, skip_special_tokens=True) for text in generated]
            elif isinstance(generated, (list, tuple)) and isinstance(generated[0], str):
                generated_texts = generated
            elif isinstance(generated, str):
                generated_texts = [generated]
            else:  
                raise ValueError(f"Invalid type for generated in evaluate: {type(generated)}")
  
            label_texts = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)  
  
            # Extract answers from generated and label texts  
            predicted_answers = [extract_answer(text) for text in generated_texts]  
            actual_answers = [extract_answer(text) for text in label_texts]  
  
            # if print_generations:
            #     print("Generated:")
            #     print(generated_texts)
            #     print("Actual:")
            #     print(actual_answers)
  
            # Compare answers  
            for pred, actual in zip(predicted_answers, actual_answers):  
                if pred is not None and actual is not None and pred == actual:  
                    correct += 1  
                total += 1  
  
            if dist.get_rank() == 0:  
                progress_bar.set_postfix({'loss': loss.item()})  
  
    # Gather results from all processes  
    correct = torch.tensor(correct, device=torch.cuda.current_device())  
    total = torch.tensor(total, device=torch.cuda.current_device())  
    dist.all_reduce(correct)  
    dist.all_reduce(total)  
  
    accuracy = correct.item() / total.item() if total.item() > 0 else 0  
    return total_loss / len(test_loader), accuracy  

  
# Main training and evaluation pipeline  
def main():  
    # torch.cuda.set_device(int(os.environ['LOCAL_RANK']))  
    # dist.init_process_group(backend='nccl', init_method='env://')  
    # Setup FSDP  
    fsdp_config = setup_fsdp()  
  
    train_loader, test_loader, tokenizer = process_data(config)  
    model = create_model(config, fsdp_config)  
  
    # Wrap the entire model with FSDP  
    fsdp_config['auto_wrap_policy'] = None
    model = FSDP(model, **fsdp_config,)  
  
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config["learning_rate"], weight_decay=config["weight_decay"])  
    if config["scheduler"] == "OneCycleLR":
        scheduler = OneCycleLR(  
            optimizer,  
            max_lr=config["learning_rate"],  
            steps_per_epoch=len(train_loader) // config["gradient_accumulation_steps"],  
            epochs=config["num_epochs"],  
            pct_start=0.2  
        )  
    elif config["scheduler"] == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["num_epochs"], eta_min=0, last_epoch=-1, verbose=False)
    elif config["scheduler"] == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif config["scheduler"] == "MultiStepLR":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.1)
    # add elif for linear warmup and then constant LR scheduler
    elif config["scheduler"] == "WarmupScheduler":
        def warmup_lambda(epoch):
            if epoch < config["warmup_steps"]:
                return float(epoch) / float(max(1, config["warmup_steps"]))
            return 1.0
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
    else:
        # add scheduler which doesn't change the learning rate
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1)
  
    if dist.get_rank() == 0:  
        writer = SummaryWriter(log_dir='runs/gsm8k_training')  
  
    # test_loss, test_accuracy = evaluate(model, test_loader, tokenizer, config)  
    # save_model(model, "initial_model.pth")  
    best_accuracy = 0  
    for epoch in range(config["num_epochs"]):  
        train_loss = train_epoch(model, epoch, train_loader, optimizer, scheduler, config)  
        test_loss, test_accuracy = evaluate(model, test_loader, tokenizer, config)  
        train_loss, train_accuracy = evaluate(model, train_loader, tokenizer, config)
  
        if dist.get_rank() == 0:  
            logger.info(f"Epoch {epoch+1}/{config['num_epochs']}:")  
            logger.info(f"Train Loss: {train_loss:.4f}")  
            logger.info(f"Test Loss: {test_loss:.4f}")  
            logger.info(f"Test Accuracy: {test_accuracy:.4f}")  
  
            writer.add_scalar('Loss/train', train_loss, epoch)  
            writer.add_scalar('Loss/test', test_loss, epoch)  
            writer.add_scalar('Accuracy/test', test_accuracy, epoch)  
            
            logger.info(f"Train Loss: {train_loss:.4f}")  
            logger.info(f"Train Accuracy: {train_accuracy:.4f}") 
            writer.add_scalar('Loss/train', train_loss, epoch)  
            writer.add_scalar('Accuracy/train', train_accuracy, epoch)
  
            if test_accuracy > best_accuracy and epoch > 0:  
                best_accuracy = test_accuracy  
                save_model(model, "best_dual_model.pth")  
                
            # if last epoch, save the model
            if epoch == config["num_epochs"] - 1:
                # calcullate train_loss and train accuracy
                # train_loss, train_accuracy = evaluate(model, train_loader, tokenizer, config, print_generations=True)
                logger.info(f"Train Loss: {train_loss:.4f}")  
                logger.info(f"Train Accuracy: {train_accuracy:.4f}") 
                writer.add_scalar('Loss/train', train_loss, epoch)  
                writer.add_scalar('Accuracy/train', train_accuracy, epoch)
                # save the model
                save_model(model, "final_dual_model.pth")
  
    if dist.get_rank() == 0:  
        writer.close()  
        logger.info(f"Best Test Accuracy: {best_accuracy:.4f}")  
  
    dist.destroy_process_group()  
  
if __name__ == "__main__":  
    main()  
