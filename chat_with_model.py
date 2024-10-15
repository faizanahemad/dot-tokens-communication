import os
import sys
import argparse
import logging
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
from collections import deque
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration dictionary
config = {
    "large_model_name": "meta-llama/Llama-3.2-3B",
    "small_model_name": "meta-llama/Llama-3.2-1B",
    
    "small_model_dim": 2048,  
    "large_model_dim": 3072,  
    "model_cls": DualModelTransformer,
    "max_output_length": None,  # Will be set later
    "seed": 42,
    "max_input_length": 1024,
    "stop_tokens": ["</end", "</finished>", "User:"], 
}
# Set the maximum output length based on the dataset (if required)  
config["max_output_length"] = 256  
config["additional_save_keywords"] = "base_model"
saved_model_path = f"saved_models/final_model_pretraining_{config['model_cls'].__name__}_{config['additional_save_keywords']}.pth"
  
# Set the random seed for reproducibility  
set_seed(config["seed"])  
  
def main():  
    # Parse command-line arguments  
    parser = argparse.ArgumentParser(description="Interactive script for model inference.")  
    parser.add_argument('--chat', action='store_true', help='Enable chat mode.')  
    parser.add_argument('--model-path', type=str, default=saved_model_path, help='Path to the saved model.')  
    args = parser.parse_args()  
      
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    logger.info(f"Using device: {device}")  
      
    # Load tokenizer  
    tokenizer_name = config["small_model_name"]  
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)  
      
    # Create the model instance  
    model = create_model(config, model_cls=config["model_cls"], fsdp_config=None)  
    model.to(device)
    model.eval()
    # Load the saved model  
    saved_model_path = args.model_path  
    if os.path.exists(saved_model_path):  
        checkpoint = torch.load(saved_model_path, map_location=device)  
        model.load_state_dict(checkpoint)  
        model.to(device)  
        model.eval()  
        logger.info(f"Loaded model from {saved_model_path}")  
    else:  
        logger.error(f"Saved model not found at {saved_model_path}")  
        sys.exit(1)  
      
    # Check if chat mode is enabled  
    if args.chat:  
        run_chat_mode(model, tokenizer, config)  
    else:  
        run_single_input_mode(model, tokenizer, config)  
  
def run_single_input_mode(model, tokenizer, config):  
    """  
    Function to handle single input mode.  
    """  
    # Prompt the user for input  
    user_input = input("Enter your input: ")  
    device = next(model.parameters()).device
    # Tokenize user input  
    input_ids = tokenizer.encode(user_input, return_tensors='pt').to(device)  
    # Ensure input length does not exceed max input length  
    input_ids = input_ids[:, :config["max_input_length"]]  
    attention_mask = torch.ones_like(input_ids).to(device)  
    # Generate response  
    with torch.no_grad():  
        response = model.generate(  
            input_ids=input_ids,  
            attention_mask=attention_mask,  
            max_length=config["max_output_length"],  
            mode='test'  # Adjust the mode if necessary  
        )  
    # Decode and print the response  
    # response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)  
    print("Model response:")  
    print(response)  
  
def run_chat_mode(model, tokenizer, config):  
    """  
    Function to handle chat mode.  
    """  
    # Initialize the chat history queue with a maximum length of 4  
    chat_history = deque(maxlen=4)  
    print("Entering chat mode. Type 'exit' or 'quit' to stop.")  
    while True:  
        user_input = input("You: ")  
        if user_input.lower() in ['exit', 'quit']:  
            print("Exiting chat mode.")  
            break  
        # Add user input to chat history  
        chat_history.append({"role": "user", "content": user_input})  
        # Construct the prompt with chat history  
        prompt = ""  
        for turn in chat_history:  
            if turn["role"] == "user":  
                prompt += "User: " + turn["content"] + "\n"  
            else:  
                prompt += "Model: " + turn["content"] + "\n"  
        # Tokenize the prompt  
        device = next(model.parameters()).device
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)  
        # Truncate the input if it exceeds max_input_length  
        input_ids = input_ids[:, -config["max_input_length"]:]  
        attention_mask = torch.ones_like(input_ids).to(device)  
        # Generate response  
        with torch.no_grad():  
            response = model.generate(  
                input_ids=input_ids,  
                attention_mask=attention_mask,  
                max_length=config["max_output_length"],  
                mode='test'  # Adjust the mode if necessary  
            )  
        # Decode and print the response  
        # response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)  
        print("Model:", response.strip())  
        # Add model response to chat history  
        chat_history.append({"role": "model", "content": response.strip()})  
  
if __name__ == "__main__":  
    main()  
