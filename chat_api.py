import os
from more_itertools import run_length
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from flask import Flask, render_template_string, request, jsonify
import time
from model_fsdp_better_query import DualModelTransformerBetterQuery
from utils import set_seed, create_model
from train_fsdp import evaluate, config
from transformers import AutoTokenizer, AutoModelForCausalLM


app = Flask(__name__)

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

# print(model.small_model.model.embed_tokens.weight, model.small_model.model.layers[0].self_attn.q_proj.weight)
model.small_model = AutoModelForCausalLM.from_pretrained(config["small_model_name"], trust_remote_code=True, torch_dtype=torch.bfloat16)
model.large_model = AutoModelForCausalLM.from_pretrained(config["large_model_name"], trust_remote_code=True, torch_dtype=torch.bfloat16)
model.to(device)
model.eval()

# print(model.small_model.model.embed_tokens.weight)
# print(model.embedding_layer.weight)


tokenizer = AutoTokenizer.from_pretrained(config["small_model_name"], trust_remote_code=True)
tokenizer.padding_side = 'left'  

def generate_text(input_text, max_tokens, stop, temperature, mode="baseline"):
    input_ids = tokenizer.encode(input_text, return_tensors="pt", add_special_tokens=True).to(device)
    # print(input_ids.shape, input_ids)
    attention_mask = torch.ones_like(input_ids)
    
    generated, generation_probs, top_logprobs_dict = model.generate(
        input_prompt=input_text,
        max_length=max_tokens,
        temperature=temperature,
        sampling_method="greedy",
        mode=mode
    )
    
    generated_text =generated[0] if isinstance(generated, (tuple,list)) else generated
    
    generated_text = generated_text.replace(input_text, "").strip()
    
    for stop_sequence in stop:
        if stop_sequence in generated_text:
            generated_text = generated_text[:generated_text.index(stop_sequence)]
    
    return generated_text, generation_probs, top_logprobs_dict


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Text Generation UI</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"></script>
    <style>
        body { padding-top: 50px; }
        #output { white-space: pre-wrap; }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="mb-4">Text Generation UI</h1>
        <div class="row">
            <div class="col-md-6 mb-3">
                <select id="apiSelector" class="form-select">
                    <option value="completions">Completion API</option>
                    <option value="chat/completions">Chat Completion API</option>
                </select>
            </div>
            <div class="col-md-6 mb-3">
                <select id="mode" class="form-select">
                    <option value="baseline">Baseline</option>
                    <option value="large-baseline">Large Baseline</option>
                    <option value="ours">Ours</option>
                </select>
            </div>
        </div>
        <div class="mb-3">
            <textarea id="input" class="form-control" rows="5" placeholder="Enter your prompt here"></textarea>
        </div>
        <div class="row">
            <div class="col-md-6 mb-3">
                <label for="maxTokens" class="form-label">Max Tokens</label>
                <input type="number" id="maxTokens" class="form-control" value="50">
            </div>
            <div class="col-md-6 mb-3">
                <label for="temperature" class="form-label">Temperature</label>
                <input type="number" id="temperature" class="form-control" value="1.0" step="0.1" min="0" max="2">
            </div>
        </div>
        <button onclick="generateText()" class="btn btn-primary mb-3">Generate</button>
        <h2>Output:</h2>
        <div id="output" class="border p-3 bg-light"></div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        async function generateText() {
            const input = document.getElementById('input').value;
            const maxTokens = parseInt(document.getElementById('maxTokens').value);
            const temperature = parseFloat(document.getElementById('temperature').value);
            const mode = document.getElementById('mode').value;
            const apiEndpoint = document.getElementById('apiSelector').value;

            const payload = apiEndpoint === 'completions'
                ? { prompt: input, max_tokens: maxTokens, temperature, mode }
                : { messages: [{ role: 'user', content: input }], max_tokens: maxTokens, temperature, mode };

            try {
                document.getElementById('output').textContent = 'Generating...';
                const response = await axios.post(`/${apiEndpoint}`, payload);
                const output = apiEndpoint === 'completions'
                    ? response.data.choices[0].text
                    : response.data.choices[0].message.content;
                document.getElementById('output').textContent = output;
            } catch (error) {
                console.error('Error:', error);
                document.getElementById('output').textContent = 'Error generating text. Please try again.';
            }
        }
    </script>
</body>
</html>
"""

mode_initial = "ours"

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/completions', methods=['POST'])
def completions():
    data = request.json
    prompt = data.get('prompt', '')
    max_tokens = data.get('max_tokens', 50)
    temperature = data.get('temperature', 1.0)
    stop = data.get('stop', ['<|eot_id|>', '</s>', '<|endoftext|>', '<|user|>', '<|im_end|>', 'Human:', 'User:', '[SEP]', '<|endofassistant|>'])
    mode = data.get('mode', mode_initial)
    
    if not isinstance(stop, list):
        stop = [stop]

    if isinstance(prompt, list):
        generated_texts = [generate_text(p, max_tokens, stop, temperature, mode) for p in prompt]
    else:
        generated_texts = [generate_text(prompt, max_tokens, stop, temperature, mode)]
        
    # print(generated_texts)

    response = {
        "id": f"cmpl-{time.time()}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": f"custom-{config['model_cls'].__name__}",
        "choices": [
            {
                "text": generated_text,
                "index": 0,
                "logprobs": {"token_logprobs": generation_prob, "top_logprobs": top_logprobs_dict},
                "finish_reason": "length"
            } for generated_text, generation_prob, top_logprobs_dict in generated_texts
        ],
        "usage": {
            "prompt_tokens": len(tokenizer.encode(prompt)),
            "completion_tokens": [len(tokenizer.encode(generated_text)) - len(tokenizer.encode(prompt)) for generated_text, generation_prob, top_logprobs_dict in generated_texts],
            "total_tokens": [len(tokenizer.encode(generated_text)) for generated_text, generation_prob, top_logprobs_dict in generated_texts],
        }
    }
    return jsonify(response)


if __name__ == '__main__':
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    app.run(debug=False, threaded=True, port=port)
    
    
# lm_eval --model local-completions --tasks gsm8k --model_args model=meta-llama/Llama-3.2-3B-Instruct,base_url=http://localhost:5000/completions,num_concurrent=1,max_retries=1,tokenized_requests=False