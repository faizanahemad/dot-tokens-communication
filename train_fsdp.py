import os

from model_strong_baselines import OneModelTransformer
from my_datasets import get_validation_split  
os.environ["TOKENIZERS_PARALLELISM"] = "true"  
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["HF_TOKEN"] = "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"
import torch  
import torch.nn as nn  
import torch.optim as optim  
from torch.utils.data import Dataset, DataLoader  
from torch.optim.lr_scheduler import OneCycleLR  
from transformers import AutoTokenizer  
from model_fsdp import DualModelTransformer, setup_fsdp, save_model, load_model  
from model_lora import LoRAModelTransformer, calculate_lora_parameters, lora_setup_fsdp
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
from utils import set_seed, create_model  
from my_datasets import get_dataset_class  # Import the dataset module  
  
warnings.filterwarnings("ignore")  
  
# Set up logging  
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  
logger = logging.getLogger(__name__)  


config = {  
    "large_model_name": "unsloth/Meta-Llama-3.1-8B-Instruct",  # "EleutherAI/pythia-1b-deduped",  
    "small_model_name": "meta-llama/Llama-3.2-1B-Instruct",  # "EleutherAI/pythia-410m",  
    "stop_tokens": [], # [".", "!", "?"],  
    "small_model_dim": 2048,  
    "large_model_dim": 4096,  
    "learning_rate": 2e-5,  
    "batch_size": 16,  
    "num_epochs": 5,  
    "warmup_steps": 10,  
    "max_grad_norm": 1.0,  
    "train_subset_size": None,  # Set to None to use full dataset  
    "test_subset_size": 16*16,    # Set to None to use full dataset  
    "weight_decay": 0.001,  
    "gradient_accumulation_steps": 1,  
    "num_workers": 4,  
    "max_input_length": 256,  
    "max_output_length": 128,
    "scheduler": None,  # Options: "OneCycleLR", "CosineAnnealingLR", "StepLR", "MultiStepLR", "WarmupScheduler"  
    "dataset_name": "gsm8k",  # Specify the dataset to use  # complete_the_sentence # fill_the_blank
    "seed": 42,  
    "model_cls": LoRAModelTransformer,
    
}  
  
set_seed(config["seed"])  
  
# Data processing function  
def process_data(config):  
    dataset_class = get_dataset_class(config["dataset_name"])  
    if dataset_class is None:  
        raise ValueError(f"Dataset {config['dataset_name']} is not supported.")  
  
    tokenizer_name = config["small_model_name"]  
  
    train_dataset = dataset_class(  
        dataset_name=config["dataset_name"],  
        tokenizer_name=tokenizer_name,  
        max_input_length=config["max_input_length"],  
        max_output_length=config["max_output_length"],  
        split='train',  
        subset_size=config["train_subset_size"]  
    )  
  
    test_dataset = dataset_class(  
        dataset_name=config["dataset_name"],  
        tokenizer_name=tokenizer_name,  
        max_input_length=config["max_input_length"],  
        max_output_length=config["max_output_length"],  
        split=get_validation_split(config["dataset_name"]),  
        subset_size=config["test_subset_size"]  
    )  
  
    tokenizer = train_dataset.tokenizer  # Use the tokenizer from the dataset class  
  
    if torch.distributed.is_initialized():
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        test_sampler = torch.utils.data.SequentialSampler(test_dataset)
  
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], sampler=train_sampler, num_workers=config["num_workers"])  
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], sampler=test_sampler, num_workers=config["num_workers"])  
  
    return train_loader, test_loader, tokenizer, train_dataset, test_dataset  
  
# Training function  
def train_epoch(model, epoch, train_loader, optimizer, scheduler, config):  
    model.train()  
    total_loss = 0  
    if torch.distributed.is_initialized():
        train_loader.sampler.set_epoch(epoch)  
    progress_bar = tqdm(train_loader, desc="Training", disable=torch.distributed.is_initialized() and not dist.get_rank() == 0)  
  
    for step, batch in enumerate(progress_bar):  
        batch = {k: v.to(torch.cuda.current_device()) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}  
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
  
        if torch.distributed.is_initialized() and dist.get_rank() == 0:  
            progress_bar.set_postfix({'loss': loss.item() * config["gradient_accumulation_steps"]})  
  
    return total_loss / len(train_loader)  
  
# Evaluation function  
def evaluate(model, data_loader, tokenizer, dataset, config, mode="baseline"):  
    model.eval()  
    total_loss = 0  
    progress_bar = tqdm(data_loader, desc="Evaluating", disable= torch.distributed.is_initialized() and not dist.get_rank() == 0)  
    metric_functions = dataset.get_evaluation_metrics()  
    # Initialize metric accumulators  
    metric_accumulators = {name: 0.0 for name in metric_functions.keys()}  
    total_samples = 0  

    with torch.no_grad():  
        for batch in progress_bar:  
            batch = {k: v.to(torch.cuda.current_device()) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}  
            outputs = model(batch['input_ids'], batch['attention_mask'], labels=batch['labels'], labels_attention_mask=batch['labels_attention_mask'])  
            loss = outputs  
            total_loss += loss.item()  
  
            # Generate text  
            # print(batch['input_ids'].shape)
            generated = model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], max_length=config["max_output_length"], mode=mode)  
  
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
            predicted_answers = [dataset.extract_answer(text) for text in generated_texts]  
            actual_answers = [dataset.extract_answer(ref.lower()) for ref in batch.get('reference_answer', label_texts)]  
  
            # Evaluate metrics  
            for name, func in metric_functions.items():  
                results = func(predicted_answers, actual_answers)  
                # print(f"Metric: {name}, Results: {results}, Sample predictions: {predicted_answers[:2]}, Sample actuals: {actual_answers[:2]}")
                metric_accumulators[name] += results[name] * len(predicted_answers)  
  
            total_samples += len(predicted_answers)  
  
            if torch.distributed.is_initialized() and dist.get_rank() == 0:  
                progress_bar.set_postfix({'loss': loss.item()})  
  
    # Gather results from all processes  
    total_samples_tensor = torch.tensor(total_samples, device=torch.cuda.current_device())  
    if torch.distributed.is_initialized():
        dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)  
    total_samples = total_samples_tensor.item()  
  
    for name in metric_accumulators:  
        metric_tensor = torch.tensor(metric_accumulators[name], device=torch.cuda.current_device())  
        if torch.distributed.is_initialized():
            dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)  
        metric_accumulators[name] = metric_tensor.item() / total_samples if total_samples > 0 else 0.0  
  
    return total_loss / len(data_loader), metric_accumulators  
  
# Main training and evaluation pipeline  
def main():  
    # Setup FSDP  
    
    
    # Wrap the entire model with FSDP  
    if config["model_cls"] == LoRAModelTransformer:
        fsdp_config = setup_fsdp()  
        lora_fsdp_config = lora_setup_fsdp()
        # use_orig_params=True,  
        fsdp_config['use_orig_params'] = True
        fsdp_config['auto_wrap_policy'] = None
        model = create_model(config, model_cls=config["model_cls"], fsdp_config=fsdp_config, lora_fsdp_config=lora_fsdp_config) 
        fsdp_config['auto_wrap_policy'] = None  
        model = FSDP(model, **fsdp_config)  
         
        
    else:
        fsdp_config = setup_fsdp()  
        fsdp_config['auto_wrap_policy'] = None  
        model = create_model(config, model_cls=config["model_cls"], fsdp_config=fsdp_config)  
        model = FSDP(model, **fsdp_config)  
    # fsdp_config["use_orig_params"] = True
    
    
  
    train_loader, test_loader, tokenizer, train_dataset, test_dataset = process_data(config)  
    
    local_rank = os.environ['LOCAL_RANK']
    
    # load model from checkpoint
    # Load model from checkpoint
    # if dist.get_rank() == 0:
    #     checkpoint = torch.load('final_dual_model_gsm8k.pth', map_location='cpu')
    #     model.load_state_dict(checkpoint)
    torch.distributed.barrier()  
    
    # test_loss, test_metrics = evaluate(model, test_loader, tokenizer, test_dataset, config, mode="baseline")  
    # train_loss_eval, train_metrics = evaluate(model, train_loader, tokenizer, train_dataset, config, mode="baseline")  
    # print(f"Baseline Test Loss: {test_loss:.4f}, Baseline Test Metrics: {test_metrics}, Baseline Train Loss: {train_loss_eval:.4f}, Baseline Train Metrics: {train_metrics}")
    # print(f"Baseline Test Loss: {test_loss:.4f}, Baseline Test Metrics: {test_metrics}")
    
    # test_loss, test_metrics = evaluate(model, test_loader, tokenizer, test_dataset, config, mode="large-baseline")  
    # train_loss_eval, train_metrics = evaluate(model, train_loader, tokenizer, train_dataset, config, mode="large-baseline")  
    # print(f"Large Baseline Test Loss: {test_loss:.4f}, Large Baseline Test Metrics: {test_metrics}, Large Baseline Train Loss: {train_loss_eval:.4f}, Large Baseline Train Metrics: {train_metrics}")
    # print(f"Large Baseline Test Loss: {test_loss:.4f}, Large Baseline Test Metrics: {test_metrics}")
    
  
    
    
  
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config["learning_rate"], weight_decay=config["weight_decay"])  
    if dist.get_rank() == 0:  
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {trainable_params}, in millions: {(trainable_params / 1e6):.2f}M")
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
    elif config["scheduler"] == "WarmupScheduler":  
        def warmup_lambda(epoch):  
            if epoch < config["warmup_steps"]:  
                return float(epoch) / float(max(1, config["warmup_steps"]))  
            return 1.0  
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)  
    else:  
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1)  
  
    if dist.get_rank() == 0:  
        writer = SummaryWriter(log_dir=f'runs/{config["dataset_name"]}_training')  
  
    best_metric = 0  
    
    for epoch in range(config["num_epochs"]):  
        train_loss = train_epoch(model, epoch, train_loader, optimizer, scheduler, config)  
        test_loss, test_metrics = evaluate(model, test_loader, tokenizer, test_dataset, config, mode="test")  
        # train_loss_eval, train_metrics = evaluate(model, train_loader, tokenizer, train_dataset, config, mode="train")  
        torch.distributed.barrier()  
  
        if dist.get_rank() == 0:  
            logger.info(f"Epoch {epoch+1}/{config['num_epochs']}:")  
            # logger.info(f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Metrics: {test_metrics}, Train Metrics: {train_metrics}")
            logger.info(f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Metrics: {test_metrics}")
  
            writer.add_scalar('Loss/train', train_loss, epoch)  
            writer.add_scalar('Loss/test', test_loss, epoch)  
            for metric_name, value in test_metrics.items():  
                writer.add_scalar(f'{metric_name}/test', value, epoch)  
  
            
            
        torch.distributed.barrier()  
        logger.info(f"Process {dist.get_rank()} finished epoch {epoch+1}")  
        if dist.get_rank() == 0:  
            logger.info(f"Writing to tensorboard")
            # for metric_name, value in train_metrics.items():  
                # writer.add_scalar(f'{metric_name}/train', value, epoch)  
                
            logger.info(f"Done Writing to tensorboard")
  
            # Save the best model based on the primary metric  
        primary_metric = 'accuracy' if 'accuracy' in test_metrics else list(test_metrics.keys())[0]  
        if test_metrics[primary_metric] > best_metric and epoch > 0:  
            best_metric = test_metrics[primary_metric]  
            logger.info(f"Saving best model for rank: {dist.get_rank()}")
            save_model(model, f"best_dual_model_{config['dataset_name']}_{config['model_cls'].__name__}.pth")  
        torch.distributed.barrier()  
        logger.info(f"Proceeding to save final model on rank: {dist.get_rank()}")
        
        if epoch == config["num_epochs"] - 1:  
            logger.info(f"Saving final model")
            save_model(model, f"final_dual_model_{config['dataset_name']}_{config['model_cls'].__name__}.pth")  
        torch.distributed.barrier()  
        logger.info(f"Done saving final model")
  
    if dist.get_rank() == 0:  
        writer.close()  
        logger.info(f"Best Test {primary_metric.capitalize()}: {best_metric:.4f}")  
  
    dist.destroy_process_group()  
    
def main_non_fsdp():  
    # Setup FSDP  
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Wrap the entire model with FSDP  
    if config["model_cls"] == LoRAModelTransformer:
        model = create_model(config, model_cls=config["model_cls"], fsdp_config=None) 
         
        
    else:
        model = create_model(config, model_cls=config["model_cls"], fsdp_config=None)  
    # fsdp_config["use_orig_params"] = True
    model.to(device)
    
    
  
    train_loader, test_loader, tokenizer, train_dataset, test_dataset = process_data(config)  
    
    # test_loss, test_metrics = evaluate(model, test_loader, tokenizer, test_dataset, config, mode="baseline")  
    # train_loss_eval, train_metrics = evaluate(model, train_loader, tokenizer, train_dataset, config, mode="baseline")  
    # print(f"Baseline Test Loss: {test_loss:.4f}, Baseline Test Metrics: {test_metrics}, Baseline Train Loss: {train_loss_eval:.4f}, Baseline Train Metrics: {train_metrics}")
    # print(f"Baseline Test Loss: {test_loss:.4f}, Baseline Test Metrics: {test_metrics}")
    
    # test_loss, test_metrics = evaluate(model, test_loader, tokenizer, test_dataset, config, mode="large-baseline")  
    # train_loss_eval, train_metrics = evaluate(model, train_loader, tokenizer, train_dataset, config, mode="large-baseline")  
    # print(f"Large Baseline Test Loss: {test_loss:.4f}, Large Baseline Test Metrics: {test_metrics}, Large Baseline Train Loss: {train_loss_eval:.4f}, Large Baseline Train Metrics: {train_metrics}")
    # print(f"Large Baseline Test Loss: {test_loss:.4f}, Large Baseline Test Metrics: {test_metrics}")
    
  
    
    
  
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config["learning_rate"], weight_decay=config["weight_decay"])  
     
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {trainable_params}, in millions: {(trainable_params / 1e6):.2f}M")
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
    elif config["scheduler"] == "WarmupScheduler":  
        def warmup_lambda(epoch):  
            if epoch < config["warmup_steps"]:  
                return float(epoch) / float(max(1, config["warmup_steps"]))  
            return 1.0  
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)  
    else:  
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1)  
  
    writer = SummaryWriter(log_dir=f'runs/{config["dataset_name"]}_training')  
  
    best_metric = 0  
    
    for epoch in range(config["num_epochs"]):  
        train_loss = train_epoch(model, epoch, train_loader, optimizer, scheduler, config)  
        test_loss, test_metrics = evaluate(model, test_loader, tokenizer, test_dataset, config, mode="test")  
        # train_loss_eval, train_metrics = evaluate(model, train_loader, tokenizer, train_dataset, config, mode="train")  
         
  
        
        logger.info(f"Epoch {epoch+1}/{config['num_epochs']}:")  
        # logger.info(f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Metrics: {test_metrics}, Train Metrics: {train_metrics}")
        logger.info(f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Metrics: {test_metrics}")

        writer.add_scalar('Loss/train', train_loss, epoch)  
        writer.add_scalar('Loss/test', test_loss, epoch)  
        for metric_name, value in test_metrics.items():  
            writer.add_scalar(f'{metric_name}/test', value, epoch)  
  
            
            
        if torch.distributed.is_initialized():
            logger.info(f"Process {dist.get_rank()} finished epoch {epoch+1}")  
        
        # Save the best model based on the primary metric  
        primary_metric = 'accuracy' if 'accuracy' in test_metrics else list(test_metrics.keys())[0]  
        if test_metrics[primary_metric] > best_metric and epoch > 0:  
            best_metric = test_metrics[primary_metric]  
            
            save_model(model, f"best_dual_model_{config['dataset_name']}_{config['model_cls'].__name__}.pth")  
        
        if torch.distributed.is_initialized():
            logger.info(f"Proceeding to save final model on rank: {dist.get_rank()}")
        
        if epoch == config["num_epochs"] - 1:  
            logger.info(f"Saving final model")
            save_model(model, f"final_dual_model_{config['dataset_name']}_{config['model_cls'].__name__}.pth")  
        if torch.distributed.is_initialized():
            torch.distributed.barrier()  
        logger.info(f"Done saving final model")
  
    
    writer.close()  
    logger.info(f"Best Test {primary_metric.capitalize()}: {best_metric:.4f}")  
  
    

  
if __name__ == "__main__":  
    main()  