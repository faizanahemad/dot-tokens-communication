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
model.to(device)  
  
# Function to generate text from input  
def generate_text(input_text, max_length=50):  
    # Tokenize the input text and move to the appropriate device  
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)  
      
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
generated_batch = generate_text_batch(batch_input_texts)  
print("\nGenerated Texts for Batch Input:")  
for i, gen_text in enumerate(generated_batch):  
    print(f"\nInput {i+1}: {batch_input_texts[i]}")  
    print(f"Generated Text {i+1}: {gen_text}")  
