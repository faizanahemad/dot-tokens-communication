# evaluation_script.py  
  
import os  
os.environ["HF_TOKEN"] = "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"
import torch  
import copy
import torch.nn.functional as F  
from transformers import AutoModelForCausalLM, AutoTokenizer  
from torch.utils.data import DataLoader  
from torch.utils.data.distributed import DistributedSampler  
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  
import torch.distributed as dist  
from tqdm import tqdm  
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
from my_datasets import get_dataset_class, get_validation_split  # Import the dataset module  
  
# Configuration dictionary  
config = {  
    "dataset_name": "fill_the_blank",  # or "complete_the_sentence"  
    "small_model_name": "meta-llama/Llama-3.2-3B-Instruct",  
    "max_input_length": 64,  
    "max_output_length": 16,  
    "train_subset_size": 8,  
    "test_subset_size": 8,  
    "batch_size": 8,  
    "num_workers": 4,  
}  
  
# Set the device to GPU if available, else CPU  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
  
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
  
    train_sampler = DistributedSampler(train_dataset, shuffle=True)  
    test_sampler = DistributedSampler(test_dataset, shuffle=False)  
  
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], sampler=train_sampler, num_workers=config["num_workers"])  
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], sampler=test_sampler, num_workers=config["num_workers"])  
  
    return train_loader, test_loader, tokenizer, train_dataset, test_dataset  

def _get_embedding_layer(model):  
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):  
        return model.model.embed_tokens  
    elif hasattr(model, 'get_input_embeddings'):  
        return model.get_input_embeddings()  
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):  
        return model.transformer.wte  
    elif hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):  
        return model.model.embed_tokens  
    elif hasattr(model, 'embeddings') and hasattr(model.embeddings, 'word_embeddings'):  
        return model.embeddings.word_embeddings  
    else:  
        raise AttributeError(f"Unable to find embedding layer for {type(model).__name__}")  
        
def generate(input_ids, attention_mask, model, tokenizer, max_length):  
    """  
    Custom generate function that returns only the newly generated token IDs.  
    """  
    model.eval()  
    generated_sequences = []  
  
    with torch.no_grad():  
        # Iterate over each sequence in the batch  
        for idx in range(input_ids.size(0)):  
            # Get the initial input sequence and attention mask for this instance  
            input_id = input_ids[idx].unsqueeze(0)  
            attn_mask = attention_mask[idx].unsqueeze(0)  
            # The sequence of newly generated token IDs (excluding the input_ids)  
            newly_generated_ids = []  
  
            # Initialize past_key_values for faster generation  
            past_key_values = None  
  
            # Initialize the generated sequence with the input_ids  
            generated = input_id.clone()  
            current_attn_mask = attn_mask.clone()  
  
            for _ in range(max_length):  
                outputs = model(  
                    input_ids=generated[:, -1:],  # Use only the last token for incremental decoding  
                    attention_mask=current_attn_mask,  
                    past_key_values=past_key_values,  
                    use_cache=True  
                )  
                logits = outputs.logits  
                past_key_values = outputs.past_key_values  
  
                # Get the last token's logits and apply softmax  
                next_token_logits = logits[:, -1, :]  
                # Greedy decoding: select the token with the highest probability  
                next_token = torch.argmax(next_token_logits, dim=-1)  
  
                # Append the predicted token to the sequence  
                generated = torch.cat((generated, next_token.unsqueeze(1)), dim=1)  
                newly_generated_ids.append(next_token.item())  
  
                # Update attention mask  
                current_attn_mask = torch.cat(  
                    (current_attn_mask, torch.ones((1, 1), dtype=current_attn_mask.dtype, device=current_attn_mask.device)),  
                    dim=1  
                )  
  
                # Stop if EOS token is generated  
                if next_token.item() == tokenizer.eos_token_id:  
                    break  
  
            # Convert the list of newly generated token IDs to a tensor  
            newly_generated_tensor = torch.tensor(newly_generated_ids, device=input_ids.device, dtype=torch.long)  
            generated_sequences.append(newly_generated_tensor)  
  
    # Pad sequences to the maximum length  
    max_seq_length = max([seq.size(0) for seq in generated_sequences])  
    padded_sequences = torch.full(  
        (len(generated_sequences), max_seq_length),  
        tokenizer.pad_token_id,  
        dtype=torch.long,  
        device=input_ids.device  
    )  
    for i, seq in enumerate(generated_sequences):  
        padded_sequences[i, :seq.size(0)] = seq  
  
    return padded_sequences  

def generate_new(input_ids, attention_mask, model, tokenizer, max_length):  
    """  
    Custom generate function that returns only the newly generated token IDs.  
    """  
    model.eval()  
    generated_sequences = []  
  
    with torch.no_grad():  
        # Iterate over each sequence in the batch  
        for idx in range(input_ids.size(0)):  
            # Extract the input sequence and attention mask for the current instance  
            input_id = input_ids[idx].unsqueeze(0)  # Shape: [1, seq_len]  
            attn_mask = attention_mask[idx].unsqueeze(0)  # Shape: [1, seq_len]  
  
            # List to store newly generated token IDs  
            newly_generated_ids = []  
            # Initialize past_key_values to None  
            past_key_values = None  
            # Initialize generated sequence with the input_ids  
            generated = input_id.clone()  
            # Clone the attention mask  
            current_attn_mask = attn_mask.clone()  
  
            for step in range(max_length):  
                if past_key_values is None:  
                    # First step: use the full input_ids and attention_mask  
                    outputs = model(  
                        input_ids=generated,  
                        attention_mask=current_attn_mask,  
                        use_cache=True  
                    )  
                else:  
                    # Subsequent steps: use only the last generated token  
                    next_input_ids = next_token.unsqueeze(0)  # Shape: [1, 1]  
                    next_attn_mask = torch.ones_like(next_input_ids)  # Shape: [1, 1]  
  
                    outputs = model(  
                        input_ids=next_input_ids,  
                        attention_mask=next_attn_mask,  
                        past_key_values=past_key_values,  
                        use_cache=True  
                    )  
  
                logits = outputs.logits  
                past_key_values = outputs.past_key_values  
  
                # Get the logits for the next token  
                next_token_logits = logits[:, -1, :]  
                # Greedy decoding: select the token with the highest probability  
                next_token = torch.argmax(next_token_logits, dim=-1)  # Shape: [1]  
  
                # Append the predicted token to the generated sequence  
                generated = torch.cat((generated, next_token.unsqueeze(1)), dim=1)  
                newly_generated_ids.append(next_token.item())  
  
                # Update attention mask  
                current_attn_mask = torch.cat(  
                    (current_attn_mask, torch.ones((1, 1), dtype=current_attn_mask.dtype, device=current_attn_mask.device)),  
                    dim=1  
                )  
  
                # Stop generation if EOS token is generated  
                if next_token.item() == tokenizer.eos_token_id:  
                    break  
  
            # Convert newly generated token IDs to a tensor  
            newly_generated_tensor = torch.tensor(newly_generated_ids, device=input_ids.device, dtype=torch.long)  
            generated_sequences.append(newly_generated_tensor)  
  
    # Pad sequences to the maximum length for batch consistency  
    max_seq_length = max(seq.size(0) for seq in generated_sequences)  
    padded_sequences = torch.full(  
        (len(generated_sequences), max_seq_length),  
        tokenizer.pad_token_id,  
        dtype=torch.long,  
        device=input_ids.device  
    )  
  
    for i, seq in enumerate(generated_sequences):  
        padded_sequences[i, :seq.size(0)] = seq  
  
    return padded_sequences  


def _get_embedding_layer(model):  
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):  
        return model.model.embed_tokens  
    elif hasattr(model, 'get_input_embeddings'):  
        return model.get_input_embeddings()  
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):  
        return model.transformer.wte  
    elif hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):  
        return model.model.embed_tokens  
    elif hasattr(model, 'embeddings') and hasattr(model.embeddings, 'word_embeddings'):  
        return model.embeddings.word_embeddings  
    else:  
        raise AttributeError(f"Unable to find embedding layer for {type(model).__name__}")  
  
def generate_new_v2(  
    input_ids: torch.Tensor,  
    attention_mask: torch.Tensor,  
    model, embedding_layer, tokenizer, max_length,
    temperature: float = 1.0,  
    sampling_method: str = "greedy"  
) -> str:  
    """  
    Generates text using the embedding layer and ensures positional embeddings are correctly handled.  
    Outputs only the newly generated text.  
  
    Args:  
        input_ids (torch.Tensor): Tensor of input token IDs.  
        attention_mask (torch.Tensor): Attention mask for the input.  
        max_length (int): Maximum number of tokens to generate.  
        temperature (float): Sampling temperature for controlling randomness.  
        sampling_method (str): Method of sampling ('greedy' or 'sample').  
  
    Returns:  
        str: Newly generated text as a string.  
    """  
    
    model.eval()  
    device = input_ids.device  
  
    with torch.no_grad():  
        generated_ids = input_ids.clone()  
        newly_generated_ids = None  
        past_key_values = None  # Initialize past_key_values as None  
  
        # Initialize position_ids  
        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long, device=device).unsqueeze(0)  
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
            # if idx == 2:
            #     print(input_embeds[..., -4:]) 
  
            # Generate next token  
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
            if sampling_method == "greedy":  
                next_token = torch.argmax(logits, dim=-1)  
                if idx == 1:
                    pass
                    # print(logits[4:6, -128:-96])
                    # print(next_token)
            elif sampling_method == "sample":  
                probs = torch.nn.functional.softmax(logits / temperature, dim=-1)  
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)  
            else:  
                raise ValueError("Invalid sampling method. Choose 'greedy' or 'sample'.")  
  
            # Append generated token  
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(1)], dim=-1)  
            if newly_generated_ids is None:  
                newly_generated_ids = next_token.unsqueeze(1)  
            else:  
                newly_generated_ids = torch.cat([newly_generated_ids, next_token.unsqueeze(1)], dim=-1)  
  
            # Decode current output  
            newly_generated_text = tokenizer.decode(newly_generated_ids[0], skip_special_tokens=True)  
            # Debugging information (optional)  
            # print(f"idx: {idx}, Newly generated text: '{newly_generated_text}', Next token ID: {next_token.item()}, EOS token ID: {self.small_tokenizer.eos_token_id}")  
  
            # Check for EOS token  
            # if next_token.item() == tokenizer.eos_token_id:  
            #     break  
  
    # Decode final output  
    # final_output = tokenizer.decode(newly_generated_ids[0], skip_special_tokens=True)  
    final_output = newly_generated_ids
    return final_output  


def generate_old(input_ids, attention_mask, model, tokenizer, max_length):  
    """  
    Custom generate function to replace model.generate  
    """  
    model.eval()  
    generated_sequences = []  
  
    with torch.no_grad():  
        # Get the encoder's outputs  
        past_key_values = None  
  
        # Iterate over each sequence in the batch  
        for idx in range(input_ids.size(0)):  
            input_id = input_ids[idx].unsqueeze(0)  
            attn_mask = attention_mask[idx].unsqueeze(0)  
            generated = input_id.clone()  
  
            for _ in range(max_length):  
                outputs = model(input_ids=generated, attention_mask=attn_mask, past_key_values=past_key_values, use_cache=True)  
                logits = outputs.logits  
  
                # Get the last token's logits and apply softmax  
                next_token_logits = logits[:, -1, :]  
                probs = F.softmax(next_token_logits, dim=-1)  
  
                # Greedy decoding: select the token with the highest probability  
                next_token = torch.argmax(probs, dim=-1).unsqueeze(0)  
  
                # Append the predicted token to the sequence  
                generated = torch.cat((generated, next_token), dim=1)  
  
                # Update attention mask  
                attn_mask = torch.cat((attn_mask, torch.ones((1, 1), dtype=attn_mask.dtype, device=attn_mask.device)), dim=1)  
  
                # Stop if EOS token is generated  
                if next_token.item() == tokenizer.eos_token_id:  
                    break  
  
            generated_sequences.append(generated.squeeze(0))  
  
        # Pad sequences to the maximum length  
        max_seq_length = max([seq.size(0) for seq in generated_sequences])  
        padded_sequences = torch.full((len(generated_sequences), max_seq_length), tokenizer.pad_token_id, dtype=torch.long, device=device)  
        for i, seq in enumerate(generated_sequences):  
            padded_sequences[i, :seq.size(0)] = seq  
  
    return padded_sequences  
  
def evaluate(model, embedding_layer, data_loader, tokenizer, dataset, config, print_generations=False):  
    model.eval()  
    total_loss = 0  
    progress_bar = tqdm(data_loader, desc="Evaluating", disable=not (dist.get_rank() == 0 if dist.is_initialized() else True))  
    metric_functions = dataset.get_evaluation_metrics()  
  
    # Initialize metric accumulators  
    metric_accumulators = {name: 0.0 for name in metric_functions.keys()}  
    total_samples = 0  
  
    with torch.no_grad():  
        for batch in progress_bar:  
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}  
            # print(batch['input_ids'][:, -32:])
  
  
            # Generate text using the custom generate function  
            generated = generate_new_v2(  
                input_ids=batch['input_ids'],  
                attention_mask=batch['attention_mask'],  
                model=model,  
                embedding_layer=embedding_layer,
                tokenizer=tokenizer,  
                max_length=config["max_output_length"]  
            )  
  
            # Decode generated sequences  
            generated_texts = tokenizer.batch_decode(generated, skip_special_tokens=True)  
            batch_size = batch['input_ids'].shape[0]
            
            input_text = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
            for ix in range(batch_size):
                generated_texts[ix] = generated_texts[ix].replace(input_text[ix], "").replace(tokenizer.eos_token, "").replace(tokenizer.pad_token, "").strip()
                # print(generated_texts[ix])

            
            # print(generated_texts)
  
            # Decode labels  
            label_texts = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)  
  
            # Extract answers from generated and label texts  
            predicted_answers = [dataset.extract_answer(text) for text in generated_texts]  
            actual_answers = [ref.lower() for ref in batch.get('reference_answer', label_texts)]  
  
            # Evaluate metrics  
            for name, func in metric_functions.items():  
                results = func(predicted_answers, actual_answers)  
                metric_accumulators[name] += results[name] * len(predicted_answers)  
  
            total_samples += len(predicted_answers)  
  
            if dist.get_rank() == 0 if dist.is_initialized() else True:  
                pass
  
    # Gather results from all processes if distributed  
    if dist.is_initialized():  
        
        total_samples_tensor = torch.tensor(total_samples, device=device)  
        dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)  
        total_samples = total_samples_tensor.item()  
        print("total_samples", total_samples)
  
        for name in metric_accumulators:  
            metric_tensor = torch.tensor(metric_accumulators[name], device=device)  
            dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)  
            metric_accumulators[name] = metric_tensor.item() / total_samples if total_samples > 0 else 0.0  
    else:  
        for name in metric_accumulators:  
            metric_accumulators[name] = metric_accumulators[name] / total_samples if total_samples > 0 else 0.0  
  
    return total_loss / len(data_loader), metric_accumulators  
  
def main():  
    # Initialize distributed processing if needed  
    if 'WORLD_SIZE' in os.environ:  
        dist.init_process_group('nccl')  
        local_rank = int(os.environ['LOCAL_RANK'])  
        torch.cuda.set_device(local_rank)  
    else:  
        local_rank = 0  
  
    # Load tokenizer and model  
    tokenizer = AutoTokenizer.from_pretrained(config["small_model_name"])  
    tokenizer.padding_side = 'left'  
    if tokenizer.pad_token is None:  
        tokenizer.pad_token = tokenizer.eos_token  
  
    model = AutoModelForCausalLM.from_pretrained(config["small_model_name"], trust_remote_code=True, torch_dtype=torch.bfloat16)  
    
    model.eval()
    
    embedding_layer = copy.deepcopy(_get_embedding_layer(model))
    embedding_layer.eval()
    embedding_layer.to(device)
    model.to(device)  
  
    # Wrap the model with FSDP if distributed  
    if dist.is_initialized():  
        fsdp_params = dict(  
            sharding_strategy=ShardingStrategy.FULL_SHARD,  
            cpu_offload=CPUOffload(offload_params=False),  
            auto_wrap_policy=None,  
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  
            mixed_precision=MixedPrecision(param_dtype=torch.float16, reduce_dtype=torch.float16, buffer_dtype=torch.float16),  
            device_id=torch.cuda.current_device(),  
            sync_module_states=True,  
        )  
        model = FSDP(model, **fsdp_params)  
        embedding_layer = FSDP(embedding_layer, **fsdp_params)
    # Process data  
    train_loader, test_loader, tokenizer, train_dataset, test_dataset = process_data(config)  
  
    # Evaluate the model  
    avg_loss, metrics = evaluate(model, embedding_layer, test_loader, tokenizer, test_dataset, config)  
  
    if local_rank == 0:  
        print(f"Average Loss: {avg_loss}")  
        for metric_name, value in metrics.items():  
            print(f"{metric_name.capitalize()}: {value:.4f}")  
  
    # Clean up distributed processing  
    if dist.is_initialized():  
        dist.destroy_process_group()  
  
if __name__ == "__main__":  
    main()  
