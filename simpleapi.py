from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time
from typing import List



app = Flask(__name__)

sampling = None # greedy, topp, topk


# main_model = "togethercomputer/Llama-2-7B-32K-Instruct"
main_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

main_model = "NovoCode/Phi-2-DPO"

main_model = "cognitivecomputations/dolphin-2_6-phi-2"
main_model = "microsoft/phi-2"
main_model = "g-ronimo/phi-2-OpenHermes-2.5"
main_model = "marcel/phi-2-openhermes-30k"
main_model = "amu/dpo-phi2"


# Load models and tokenizers
tokenizer = AutoTokenizer.from_pretrained(main_model)
model = AutoModelForCausalLM.from_pretrained(main_model, device_map='auto', torch_dtype=torch.bfloat16)
# quantization_config=quantization_config
# torch_dtype=torch.bfloat16
model.eval()

def apply_temperature_scaling(logits, temperature=1.0):
    # Scale the logits by the temperature
    scaled_logits = logits / temperature
    return scaled_logits

def convert_logits_to_probabilities(logits, temperature=1.0):
    # Apply temperature scaling
    scaled_logits = apply_temperature_scaling(logits, temperature)
    # Apply softmax to get probabilities
    probabilities = torch.softmax(scaled_logits, dim=-1)
    return probabilities


def simple_probability_sampling(probs):
    return torch.multinomial(probs, num_samples=1)

# Greedy Sampling
def greedy_sampling(probs):
    return torch.argmax(probs.squeeze(), dim=-1)

# Top-K Sampling
def top_k_sampling(probs, k=10):
    topk_probs, topk_indices = torch.topk(probs, k, dim=-1)
    reweighted_probs = topk_probs / torch.sum(topk_probs, dim=-1, keepdim=True)
    sampled_index = torch.multinomial(reweighted_probs, num_samples=1)
    return topk_indices.gather(-1, sampled_index)

# Top-P (Nucleus) Sampling
def top_p_sampling(probs, p=0.9):
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    # Remove tokens with cumulative prob above p
    sorted_indices_to_remove = cumulative_probs > p
    # Shift the indices to the right to keep also the first token above p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    sorted_probs[sorted_indices_to_remove] = 0
    reweighted_probs = sorted_probs / torch.sum(sorted_probs, dim=-1, keepdim=True)
    # Sample from the re-normalized subset
    sampled_index = torch.multinomial(reweighted_probs, num_samples=1)
    return sorted_indices.gather(-1, sampled_index)

from typing import List

def combine_logits_and_generate_text(input_text: str, max_tokens: int, stop_tokens: List[str], temperature: float, sampling: str = None):
    generated_text = ""
    # Encode the initial input text and move to the correct device
    input_ids = tokenizer.encode(input_text, return_tensors="pt").cuda()
    
    # Initialize past_key_values (cache) to None for efficient generation
    past_key_values = None

    for _ in range(max_tokens):
        with torch.no_grad():
            # Ensure use_cache=True if your model supports it for caching mechanism
            outputs = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
            main_logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values
        
        # Apply temperature scaling and convert logits to probabilities
        # Added a guard against temperature being too low
        probs = convert_logits_to_probabilities(main_logits, temperature=max(temperature, 1e-9))
        
        # Determine the sampling strategy
        if sampling == "greedy" or temperature == 0:
            next_token_id = torch.argmax(probs, dim=-1, keepdim=True)
        elif sampling == "topk":
            next_token_id = top_k_sampling(probs)
        elif sampling == "topp":
            next_token_id = top_p_sampling(probs)
        else:  # Default to simple probability sampling if none specified
            next_token_id = simple_probability_sampling(probs)
        
        # Convert the token ID to the format expected by the model
        input_ids = next_token_id #.unsqueeze(0)

        # Decode the generated token ID to text and append to the generated text
        next_token = tokenizer.decode(next_token_id[0], skip_special_tokens=True)
        generated_text += next_token
        
        # Check for stop tokens in the generated text
        if any(stop_token in generated_text for stop_token in stop_tokens):
            break

    return generated_text


@app.route('/chat/completions', methods=['POST'])
def chat_completions():
    data = request.json
    messages = data.get('messages', [])
    max_tokens = data.get('max_tokens', 50)
    temperature = data.get('temperature', 1.0)
    stop = data.get('stop', [])
    if not isinstance(stop, list):
        stop = [stop]  # Ensure stop is a list for uniform processing

    if messages:
        
        input_text = messages[-1]['content']
        if len(messages)>1 and messages[-2]['role']=="system":
            input_text = "<|system|>\n" + messages[-2]['content'] + "\n<|user|>\n" + input_text + "\n<|assistant|>\n"
        # else:
        #     input_text = "<|system|>\n" + "You are a helpful assistant. Please respond to the user request." + "\n<|user|>\n" + input_text + "\n<|assistant|>\n"
        generated_text = combine_logits_and_generate_text(input_text, max_tokens, stop, temperature)
    else:
        return jsonify({"error": "No messages provided"}), 400

    response = {
        "id": "chatcmpl-unique-id",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "custom-gpt2-combination",
        "usage": {
            "prompt_tokens": len(tokenizer.encode(input_text)),
            "completion_tokens": len(tokenizer.encode(generated_text)) - len(tokenizer.encode(input_text)),
            "total_tokens": len(tokenizer.encode(generated_text)),
        },
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": generated_text.replace(input_text, "").strip()
                },
                "logprobs": None,
                "finish_reason": "length",
                "index": 0
            }
        ]
    }
    return jsonify(response)

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    else:
        port = 5000
    app.run(debug=False, threaded=True, port=port)
