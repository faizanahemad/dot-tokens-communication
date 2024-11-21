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
  
# Set the device to GPU if available, else CPU  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
  
# Load the tokenizer and model  
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B-Instruct')  
tokenizer.padding_side = 'left'  
print("Padding side:", tokenizer.padding_side)  
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-1B-Instruct')  
# print model parameters 
# print(model.model.embed_tokens.weight, model.model.layers[0].self_attn.q_proj.weight)
model.to(device)  
  
# Function to generate text from input  
def generate_text(input_text, max_length=50):  
    # Tokenize the input text and move to the appropriate device  
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)  
    print(input_ids.shape, input_ids)
      
    # Generate text  
    output_ids = model.generate(  
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
  
# Sample batch input texts  
batch_input_texts = [  
    "The future of artificial intelligence is",  
    "In the realm of quantum physics,",  
    "Machine learning algorithms can be utilized to"  
]  
  
# Function to generate text for a batch of inputs  
def generate_text_batch(input_texts, max_length=50):  
    # Tokenize the batch of input texts with padding and move to the device  
    inputs = tokenizer(input_texts, return_tensors='pt', padding=True).to(device)  
      
    # Generate text  
    output_ids = model.generate(  
        input_ids=inputs['input_ids'],  
        attention_mask=inputs['attention_mask'],  
        max_length=max_length,  
        num_return_sequences=1,  
        early_stopping=True  
    )  
      
    # Decode the generated tokens  
    generated_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)  
    return generated_texts  
  
# Generate text for the batch input  
# generated_batch = generate_text_batch(batch_input_texts)  
# print("\nGenerated Texts for Batch Input:")  
# for i, gen_text in enumerate(generated_batch):  
#     print(f"\nInput {i+1}: {batch_input_texts[i]}")  
#     print(f"Generated Text {i+1}: {gen_text}")  
    
    
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
    input_prompt: str,
    max_length,
    temperature: float = 1.0,  
    sampling_method: str = "greedy",
) -> str:  
    
    embedding_layer = copy.deepcopy(model.model.embed_tokens)
    
    model.eval()  
    device = model.device
    
    encoded = tokenizer(input_prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_length)  
    input_ids = encoded['input_ids'].to(model.device)  
    attention_mask = encoded['attention_mask'].to(model.device)  

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
print(simple_baseline("Human: Tell me a joke.\nAssistant: ", 50))