import os  
import torch  
import torch.nn as nn  
from torch.utils.data import DataLoader  
from transformers import AutoTokenizer  
from model_fsdp import DualModelTransformer  
from model_fsdp_better_supervision import DualModelTransformerBetterSupervision
from model_fsdp_distrib import DualModelTransformerDistrib
from model_strong_baselines import OneModelTransformer
from my_datasets import get_dataset_class, get_max_output_length, get_validation_split  
from utils import set_seed, create_model  
from tqdm import tqdm  
import logging  
from train_fsdp import evaluate, config
from model_lora import LoRAModelTransformer
  
# Set up logging  
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  
logger = logging.getLogger(__name__)  
  
# Configuration  
config_test = {  
    "large_model_name": "meta-llama/Llama-3.2-3B",  
    "small_model_name": "meta-llama/Llama-3.2-1B",   
    "batch_size": 16,  
    "test_subset_size": 512,  
    "max_input_length": 512,  
    "dataset_name":  "amanrangapur/Fin-Fact", # "EleutherAI/truthful_qa_mc"  # "TIGER-Lab/MMLU-Pro" # "lighteval/MATH-Hard" # "tau/commonsense_qa" # "amanrangapur/Fin-Fact" # "FinanceMTEB/financial_phrasebank" # 
    "model_cls": DualModelTransformerDistrib,
}  
config_test["max_output_length"] = get_max_output_length(config_test["dataset_name"])
config_test["baselines"] = True
config["additional_save_keywords"] = "base_model"
# saved_model_path = f'final_dual_model_gsm8k_{config["model_cls"].__name__}.pth'
# saved_model_path = f"saved_models/epoch_0_model_pretraining_{config["model_cls"].__name__}.pth"
# saved_model_path = f"saved_models/final_model_{config['dataset_name'].replace('/', '_')}_{config['model_cls'].__name__}.pth"
saved_model_path = f"saved_models/final_model_pretraining_{config['model_cls'].__name__}_{config['additional_save_keywords']}.pth"

config.update(config_test)
  
set_seed(config["seed"])  
  
def process_data(config):  
    dataset_class = get_dataset_class(config["dataset_name"])  
    if dataset_class is None:  
        raise ValueError(f"Dataset {config['dataset_name']} is not supported.")  
  
    tokenizer_name = config["small_model_name"]  
    test_dataset = dataset_class(  
        dataset_name=config["dataset_name"],  
        tokenizer_name=tokenizer_name,  
        max_input_length=config["max_input_length"],  
        max_output_length=config["max_output_length"],  
        split=get_validation_split(config["dataset_name"]),  
        subset_size=config["test_subset_size"]  
    )  
    assert test_dataset is not None
  
    tokenizer = test_dataset.tokenizer  
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])  
    return test_loader, tokenizer, test_dataset  
  
def main():  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    logger.info(f"Using device: {device}")  
    
    test_loader, tokenizer, test_dataset = process_data(config) 
  
    # Create model without FSDP  
    model = config["model_cls"](  
        large_model_name=config["large_model_name"],  
        small_model_name=config["small_model_name"],  
        stop_tokens=config["stop_tokens"],  
        small_model_dim=config["small_model_dim"],  
        large_model_dim=config["large_model_dim"],  
        max_length=config["max_input_length"],  
        fsdp_config=None,  # No FSDP config  
        enable_checkpointing=True,  
        enable_flash_attention=True  
    )  
  
    # Load the saved model  
    checkpoint = torch.load(saved_model_path, map_location=device)  
    model.load_state_dict(checkpoint)  
    model.to(device)  
 
    if config["baselines"]:
  
        logger.info("Starting evaluation...")  
        test_loss, test_metrics = evaluate(model, test_loader, tokenizer, test_dataset, config, mode="baseline")  
        logger.info(f"Baseline Loss: {test_loss:.4f}, Baseline Metrics: {test_metrics}")  
        
        try:    
            logger.info("Starting evaluation...")  
            test_loss, test_metrics = evaluate(model, test_loader, tokenizer, test_dataset, config, mode="large-baseline")  
            logger.info(f"Large-baseline Loss: {test_loss:.4f}, Large-baseline Metrics: {test_metrics}")  
        except Exception as e:
            logger.info(f"Error in large-baseline evaluation: {e}")
    
    test_loss, test_metrics = evaluate(model, test_loader, tokenizer, test_dataset, config, mode="test")  
    logger.info(f"Test Loss (test mode): {test_loss:.4f}, Test Metrics: {test_metrics}")  
  
if __name__ == "__main__":  
    main()  
