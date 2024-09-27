# evaluation_script.py  
  
import os  
import torch  
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
    "train_subset_size": None,  
    "test_subset_size": None,  
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
  
def generate(input_ids, attention_mask, model, tokenizer, max_length):  
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
  
def evaluate(model, data_loader, tokenizer, dataset, config, print_generations=False):  
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
  
  
            # Generate text using the custom generate function  
            generated = generate(  
                input_ids=batch['input_ids'],  
                attention_mask=batch['attention_mask'],  
                model=model,  
                tokenizer=tokenizer,  
                max_length=config["max_output_length"]  
            )  
  
            # Decode generated sequences  
            generated_texts = tokenizer.batch_decode(generated, skip_special_tokens=True)  
  
            # Decode labels  
            label_texts = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)  
  
            # Extract answers from generated and label texts  
            predicted_answers = [dataset.extract_answer(text) for text in generated_texts]  
            actual_answers = [ref.lower() for ref in batch.get('reference_answer', label_texts)]  
  
            # Evaluate metrics  
            for name, func in metric_functions.items():  
                results = func(predicted_answers, actual_answers)  
                metric_accumulators[name] += results.get('accuracy', 0.0) * len(predicted_answers)  
  
            total_samples += len(predicted_answers)  
  
            if dist.get_rank() == 0 if dist.is_initialized() else True:  
                pass
  
    # Gather results from all processes if distributed  
    if dist.is_initialized():  
        total_samples_tensor = torch.tensor(total_samples, device=device)  
        dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)  
        total_samples = total_samples_tensor.item()  
  
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
  
    model = AutoModelForCausalLM.from_pretrained(config["small_model_name"])  
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
  
    # Process data  
    train_loader, test_loader, tokenizer, train_dataset, test_dataset = process_data(config)  
  
    # Evaluate the model  
    avg_loss, metrics = evaluate(model, test_loader, tokenizer, test_dataset, config)  
  
    if local_rank == 0:  
        print(f"Average Loss: {avg_loss}")  
        for metric_name, value in metrics.items():  
            print(f"{metric_name.capitalize()}: {value:.4f}")  
  
    # Clean up distributed processing  
    if dist.is_initialized():  
        dist.destroy_process_group()  
  
if __name__ == "__main__":  
    main()  
