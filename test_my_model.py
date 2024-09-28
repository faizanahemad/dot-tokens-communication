import os  
import torch  
import torch.nn as nn  
from torch.utils.data import DataLoader  
from transformers import AutoTokenizer  
from model_fsdp import DualModelTransformer  
from my_datasets import get_dataset_class, get_validation_split  
from utils import set_seed, create_model  
from tqdm import tqdm  
import logging  
  
# Set up logging  
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  
logger = logging.getLogger(__name__)  
  
# Configuration  
config = {  
    "large_model_name": "meta-llama/Llama-3.2-3B-Instruct",  
    "small_model_name": "meta-llama/Llama-3.2-1B-Instruct",  
    "stop_tokens": [],  
    "small_model_dim": 2048,  
    "large_model_dim": 3072,  
    "batch_size": 8,  
    "test_subset_size": 16*8,  
    "num_workers": 4,  
    "max_input_length": 256,  
    "max_output_length": 128,  
    "dataset_name": "gsm8k",  
    "seed": 42,  
}  
  
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
  
    tokenizer = test_dataset.tokenizer  
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])  
    return test_loader, tokenizer, test_dataset  
  
def evaluate(model, data_loader, tokenizer, dataset, config, device, mode="baseline"):  
    model.eval()  
    total_loss = 0  
    progress_bar = tqdm(data_loader, desc="Evaluating")  
    metric_functions = dataset.get_evaluation_metrics()  
    metric_accumulators = {name: 0.0 for name in metric_functions.keys()}  
    total_samples = 0  
  
    with torch.no_grad():  
        for batch in progress_bar:  
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}  
            outputs = model(batch['input_ids'], batch['attention_mask'], labels=batch['labels'], labels_attention_mask=batch['labels_attention_mask'])  
            loss = outputs  
            total_loss += loss.item()  
  
            generated = model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], max_length=config["max_output_length"], mode=mode)  
  
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
            predicted_answers = [dataset.extract_answer(text) for text in generated_texts]  
            actual_answers = [dataset.extract_answer(ref.lower()) for ref in batch.get('reference_answer', label_texts)]  
  
            for name, func in metric_functions.items():  
                # print(predicted_answers, actual_answers)
                results = func(predicted_answers, actual_answers)  
                metric_accumulators[name] += results[name] * len(predicted_answers)  
  
            total_samples += len(predicted_answers)  
            progress_bar.set_postfix({'loss': loss.item()})  
  
    for name in metric_accumulators:  
        metric_accumulators[name] /= total_samples if total_samples > 0 else 0.0  
  
    return total_loss / len(data_loader), metric_accumulators  
  
def main():  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    logger.info(f"Using device: {device}")  
  
    # Create model without FSDP  
    model = DualModelTransformer(  
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
    checkpoint = torch.load('final_dual_model_gsm8k.pth', map_location=device)  
    model.load_state_dict(checkpoint)  
    model.to(device)  
  
    test_loader, tokenizer, test_dataset = process_data(config)  
  
    logger.info("Starting evaluation...")  
    test_loss, test_metrics = evaluate(model, test_loader, tokenizer, test_dataset, config, device, mode="baseline")  
    logger.info(f"Baseline Loss: {test_loss:.4f}, Baseline Metrics: {test_metrics}")  
    
    logger.info("Starting evaluation...")  
    test_loss, test_metrics = evaluate(model, test_loader, tokenizer, test_dataset, config, device, mode="large-baseline")  
    logger.info(f"Large-baseline Loss: {test_loss:.4f}, Large-baseline Metrics: {test_metrics}")  
  
    test_loss, test_metrics = evaluate(model, test_loader, tokenizer, test_dataset, config, device, mode="test")  
    logger.info(f"Test Loss (test mode): {test_loss:.4f}, Test Metrics: {test_metrics}")  
  
if __name__ == "__main__":  
    main()  
