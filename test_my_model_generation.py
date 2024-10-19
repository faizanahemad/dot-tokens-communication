from scipy.datasets import ascent
import torch  
import os  
# Set TOKENIZERS_PARALLELISM to true in environment variables  
os.environ["TOKENIZERS_PARALLELISM"] = "true"  
os.environ["OMP_NUM_THREADS"] = "2"  
os.environ["HF_TOKEN"] = "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"  
import torch.nn as nn  
import copy  
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM  
from typing import List, Optional, Union, Tuple  
import torch.nn.functional as F  
from model_fsdp_better_query import DualModelTransformerBetterQuery
from utils import set_seed, create_model
from train_fsdp import evaluate, config
  
# Set the device to GPU if available, else CPU  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  


# Configuration
config_test = {  
    "large_model_name": "meta-llama/Llama-3.2-3B",  
    "small_model_name": "meta-llama/Llama-3.2-1B-Instruct",   
    "batch_size": 16,  
    "test_subset_size": 512,  
    "max_input_length": 512,  
    "model_cls": DualModelTransformerBetterQuery,
}  

config_test["max_output_length"] = 100
config_test["baselines"] = True
config["additional_save_keywords"] = "instruct_base_model"

saved_model_path = f"saved_models/final_model_pretraining_{config['model_cls'].__name__}_{config['additional_save_keywords']}.pth"

set_seed(config["seed"])
config.update(config_test)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
model = config["model_cls"](
    large_model_name=config["large_model_name"],
    small_model_name=config["small_model_name"],
    stop_tokens=config["stop_tokens"],
    small_model_dim=config["small_model_dim"],
    large_model_dim=config["large_model_dim"],
    max_length=config["max_input_length"],
    fsdp_config=None,
    enable_checkpointing=True,
    enable_flash_attention=True
)

checkpoint = torch.load(saved_model_path, map_location=device)
model.load_state_dict(checkpoint)
  
# Load the tokenizer and model  
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B-Instruct')  
tokenizer.padding_side = 'left'  
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
print("Padding side:", tokenizer.padding_side)  
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.to(device)  
model.eval()
embedding_layer = model.embedding_layer
  
# Function to generate text from input  
def generate_text(input_text, max_length=50):  
    # Tokenize the input text and move to the appropriate device  
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)  
    print(input_ids.shape, input_ids)
      
    # Generate text  
    output_ids = model.small_model.generate(  
        input_ids,  
        max_length=max_length,  
        num_return_sequences=1,  
        early_stopping=True  
    )  
      
    # Decode the generated tokens  
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)  
    return generated_text  
  
# Sample input text  
input_text = "Tell me a joke."  
  
# Generate text for the single input  
generated_single = generate_text(input_text)  
print("Generated Text for Single Input:")  
print(generated_single)  

    
def sampling(logits, sampling_method, temperature, top_k=50, top_p=0.95):
    # Ensure logits is 2D: [batch_size, vocab_size]
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    
    if sampling_method == "greedy":
        next_token = torch.argmax(logits, dim=-1)
    elif sampling_method == "sample":
        probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
    elif sampling_method == "top_k":
        top_k = min(top_k, logits.size(-1))  # Safety check
        top_k_values, _ = torch.topk(logits, top_k, dim=-1)
        indices_to_remove = logits < top_k_values[..., -1, None]
        logits[indices_to_remove] = float('-inf')
        probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
    elif sampling_method == "top_p":
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits / temperature, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Use scatter to set logits to -inf
        logits = logits.scatter(1, sorted_indices, sorted_logits)
        logits[sorted_indices_to_remove] = float('-inf')
        
        probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
    else:
        raise ValueError("Invalid sampling method. Choose 'greedy', 'sample', 'top_k', or 'top_p'.")
    
    return next_token
    
def simple_baseline( 
    model,
    input_prompt: str,
    max_length,
    temperature: float = 1.0,  
    sampling_method: str = "greedy",
) -> str:  
    
    
    
    
    model.eval()  
    device = model.device
    
    encoded = tokenizer(input_prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_length)  
    input_ids = encoded['input_ids'].to(model.device)  
    attention_mask = encoded['attention_mask'].to(model.device)  
    print(input_ids)
    with torch.no_grad():  
        generated_ids = input_ids.clone()  
        newly_generated_ids = None  
        past_key_values = None  # Initialize past_key_values as None  

        # Initialize position_ids  
        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long, device=device).unsqueeze(0).repeat(input_ids.size(0), 1)  
        current_attention_mask = attention_mask.clone()  

        for idx in range(max_length):  
            if idx == 0:  
                # First iteration: Use the full input embeddings  
                input_embeds = embedding_layer(input_ids) 
                # print(input_embeds[:, -4, -4:]) 
            else:  
                # Subsequent iterations: Use last generated token's embedding  
                last_token_id = generated_ids[:, -1:]  
                input_embeds = embedding_layer(last_token_id)  
                # Update position_ids  
                position_ids = torch.cat([position_ids, position_ids[:, -1:] + 1], dim=1)  
                # Update attention mask  
                current_attention_mask = torch.cat(  
                    [current_attention_mask, torch.ones((current_attention_mask.size(0), 1), device=device)],  
                    dim=1  
                )  
            model_output = model(  
                inputs_embeds=input_embeds,  
                attention_mask=current_attention_mask,  
                position_ids=position_ids[:, -input_embeds.size(1):],  
                past_key_values=past_key_values,  
                use_cache=True  
            )  

            logits = model_output.logits[:, -1, :]  
            past_key_values = model_output.past_key_values  

            # Sampling  
            next_token = sampling(logits, sampling_method, temperature)

            # Append generated token  
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(1)], dim=-1)  
            if newly_generated_ids is None:  
                newly_generated_ids = next_token.unsqueeze(1)  
                # print(next_token.shape, newly_generated_ids.shape)
            else:  
                newly_generated_ids = torch.cat([newly_generated_ids, next_token.unsqueeze(1)], dim=-1) 
                
    final_output = tokenizer.decode(newly_generated_ids[0], skip_special_tokens=True)  
    # final_output = newly_generated_ids
    return final_output  

print("Simple Baseline")
print(simple_baseline(model.small_model, input_text, 50))

print("My Model")
print(model.generate(input_prompt=input_text, max_length=50, mode="baseline"))

print("My Model")
print(model.generate(input_prompt=input_text, max_length=50, mode="ours"))

print("My Model")
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
attention_mask = torch.ones_like(input_ids)
print(model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=50, mode="ours"))