import copy
import os  
import torch  
import torch.nn as nn  
import torch.optim as optim  
import torch.distributed as dist  
from typing import Optional, List, Union  
from peft import LoraConfig, get_peft_model, LoraModel  
from peft.tuners.lora import Linear as LoRALinear  
from transformers import AutoModelForCausalLM, AutoTokenizer  
from torch.distributed.fsdp import (  
    FullyShardedDataParallel as FSDP,  
    StateDictType,  
    FullStateDictConfig,  
    CPUOffload,  
    MixedPrecision,  
    ShardingStrategy,  
    BackwardPrefetch  
)  
from torch.utils.tensorboard import SummaryWriter  
from peft import LoraConfig, get_peft_model, LoraModel  

  
class LoRAModelTransformer(nn.Module):  
    """  
    A transformer-based model that combines a small language model with LoRA integration.  
  
    This class implements a novel approach to language generation by using a small language model  
    as the primary generator and incorporating low-rank adaptation (LoRA) for parameter-efficient fine-tuning.  
    It supports various generation modes and can be used for both inference and training.  
  
    Attributes:  
        small_model (AutoModelForCausalLM): The small language model with LoRA applied.  
        small_tokenizer (AutoTokenizer): Tokenizer for the small model.  
        stop_tokens (List[str]): List of tokens to stop generation.  
        small_model_dim (int): Dimension of the small model's hidden states.  
        large_model_dim (int): Dimension of the large model's hidden states.  
        max_length (int): Maximum length for generation.  
    """  
  
    def __init__(self,  
                 large_model_name: str,   
                 small_model_name: str,  
                 stop_tokens: Optional[List[str]],  
                 small_model_dim: int,  
                 large_model_dim: int, 
                 max_length: int = 512, 
                 fsdp_config: dict = None,
                 enable_checkpointing: bool = True,
                 enable_flash_attention: bool = True, 
                 **kwargs):  
        """Initialize the OneModelTransformer."""  
        super().__init__()  
        self.small_model_name = small_model_name  
        self.small_tokenizer = AutoTokenizer.from_pretrained(small_model_name)  
        self.small_tokenizer.padding_side = 'left'  
        
  
        if self.small_tokenizer.pad_token is None:  
            self.small_tokenizer.pad_token = self.small_tokenizer.eos_token  
  
        # Load the small model  
        self.small_model = AutoModelForCausalLM.from_pretrained(small_model_name)  
        
        # self.small_model.eval()
        
        self.small_model_dim = small_model_dim  
        self.large_model_dim = large_model_dim  
        # for param in self.small_model.parameters():  
        #     param.requires_grad = False  
        self.embedding_layer = copy.deepcopy(self._get_embedding_layer(self.small_model))
        for param in self.embedding_layer.parameters():
            param.requires_grad = False
        
  
        self._apply_lora()
        lora_param_count = calculate_lora_parameters(self.small_model)  
        print(f"Total number of LoRA parameters: {lora_param_count}, in millions: {(lora_param_count / 1e6):.2f}M") 
        for name, param in self.small_model.named_parameters():  
            param.requires_grad = False
            if 'lora_' in name:  
                param.requires_grad = True  
   
        
        # Enable gradient checkpointing  
        if enable_checkpointing:  
            self.small_model.enable_input_require_grads()
            if hasattr(self.small_model, 'gradient_checkpointing_enable'):  
                self.small_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})  
            else:  
                self.small_model.config.gradient_checkpointing = True  
        # fsdp_config_lora = setup_fsdp_for_lora()
        # fsdp_small_model = setup_fsdp_for_small_model()
        # self.small_model = self._wrap_lora_modules_with_fsdp(self.small_model, fsdp_config_lora)
        # self.small_model = FSDP(self.small_model, **fsdp_small_model)
        
        # for name, param in self.small_model.named_parameters():  
        #     if param.requires_grad:  
        #         print(f"Parameter {name} requires grad and has shape {param.shape}")  

        saved_model_path = kwargs.get("saved_model_path")
        device = next(iter(self.parameters())).device
        if saved_model_path:
            checkpoint = torch.load(saved_model_path, map_location=device)  
            self.load_state_dict(checkpoint)  
        
        self.small_model.base_model.enable_input_require_grads()
        if fsdp_config is not None:
            self.embedding_layer = FSDP(self.embedding_layer, **fsdp_config)
            lora_fsdp_config = kwargs.get("lora_fsdp_config", fsdp_config)
            # self.small_model = FSDP(self.small_model, **lora_fsdp_config)  
            self.small_model = self._wrap_lora_modules_with_fsdp(self.small_model, lora_fsdp_config)
        # for name, param in self.small_model.named_parameters():  
        #     if param.requires_grad:  
        #         print(f"Parameter {name} requires grad and has shape {param.shape}")  
        
        self.stop_tokens = stop_tokens if stop_tokens is not None else []  
        self.max_length = max_length  
        
        trainable_params = []  
        for name, param in self.small_model.named_parameters():  
            if param.requires_grad:  
                trainable_params.append(name)  
        trainable_param_count = sum(p.numel() for p in self.small_model.parameters() if p.requires_grad)
        total_param_count = sum(p.numel() for p in self.small_model.parameters())
        print(f"Trainable parameters in small_model: {trainable_param_count:,} ({trainable_param_count / 1e6:.2f}M)")
        print(f"Total parameters in small_model: {total_param_count:,} ({total_param_count / 1e6:.2f}M)")
        print(f"Percentage of trainable parameters: {(trainable_param_count / total_param_count) * 100:.2f}%")
        
    def _wrap_lora_modules_with_fsdp(self, module, fsdp_config_lora):  
        """Wrap LoRA modules with FSDP using the provided configuration."""  
        # Recursively wrap LoRA modules  
        return self._apply_fsdp_to_lora_modules(module, fsdp_config_lora)  
  
    def _apply_fsdp_to_lora_modules(self, module, fsdp_config_lora):  
        """Recursively apply FSDP wrapping to LoRA modules."""  
        from peft.tuners.lora import Linear  
        for name, child in module.named_children():  
            if isinstance(child, (LoraModel, LoRALinear, Linear)):  
                wrapped_child = FSDP(child, **fsdp_config_lora)  
                setattr(module, name, wrapped_child)  
            else:  
                self._apply_fsdp_to_lora_modules(child, fsdp_config_lora)  
        return module 
  
    def _apply_lora(self):  
        """Apply LoRA to the small_model."""  
        # Derive lora_r and lora_alpha from existing parameters  
        # For demonstration, we'll set lora_r such that the total LoRA parameters approximate large_model_dim x small_model_dim  
  
        # Estimate the number of LoRA parameters  
        from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
        target_param_count = self.large_model_dim * self.small_model_dim  
  
        # Number of parameters per LoRA layer  
        lora_layers = 0  
        for name, module in self.small_model.named_modules():  
            if isinstance(module, nn.Linear):  
                lora_layers += 1  
  
        # Calculate lora_r  
        lora_r = max(1, target_param_count // (2 * lora_layers * self.small_model_dim))  
  
        # Set lora_alpha  
        lora_alpha = lora_r * 2  # Common practice is to set lora_alpha to lora_r * scaling factor  
  
        # Apply LoRA configuration  
        lora_config = LoraConfig(  
            r=lora_r,  
            lora_alpha=lora_alpha,  
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "out_proj", "fc1", "fc2"],  
            lora_dropout=0.,  
            inference_mode=False,
            bias='none',  
            task_type=TaskType.CAUSAL_LM  # For causal language modeling  
        )  
        self.small_model = get_peft_model(self.small_model, lora_config)  
        self.small_model.print_trainable_parameters()
        print(f"Applied LoRA with r={lora_r}, alpha={lora_alpha}")  
  
    def _get_embedding_layer(self, model):  
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
  
    def _get_last_hidden_state(self, model_output):  
        if hasattr(model_output, 'hidden_states') and model_output.hidden_states is not None:  
            return model_output.hidden_states[-1] + model_output.hidden_states[-2] + model_output.hidden_states[-3] + model_output.hidden_states[-4]  
        else:  
            raise AttributeError(f"Unable to extract last hidden state from model output: {type(model_output).__name__}")  
  
    def _get_logits(self, model_output):  
        if hasattr(model_output, 'logits'):  
            return model_output.logits  
        elif hasattr(model_output, 'last_hidden_state'):  
            return model_output.last_hidden_state @ self._get_embedding_layer(self.small_model).weight.T  
        elif isinstance(model_output, torch.Tensor):  
            return model_output @ self._get_embedding_layer(self.small_model).weight.T  
        else:  
            raise AttributeError(f"Unable to extract logits from model output: {type(model_output).__name__}")  
  
    def forward(  
        self,  
        input_ids: Optional[torch.Tensor] = None,  
        attention_mask: Optional[torch.Tensor] = None,  
        labels: Optional[torch.Tensor] = None,  
        labels_attention_mask: Optional[torch.Tensor] = None,  
        input_prompt: Optional[Union[str, List[str]]] = None,  
        expected_output: Optional[Union[str, List[str]]] = None,  
        temperature: float = 1.0,  
        sampling_method: str = "greedy"  
    ) -> torch.Tensor:  
        """  
        Forward method for training the model.  
  
        Args:  
            input_ids (torch.Tensor, optional): Tensor of input token IDs (input_prompt).  
            attention_mask (torch.Tensor, optional): Attention mask for the input.  
            labels (torch.Tensor, optional): Tensor of labels for computing the loss.  
            labels_attention_mask (torch.Tensor, optional): Attention mask for the labels.  
            input_prompt (str or List[str], optional): Input prompt(s) as string(s), if input_ids is not provided.  
            expected_output (str or List[str], optional): Expected output(s) as string(s), if labels are not provided.  
  
        Returns:  
            torch.Tensor: The computed loss.  
        """  
        # Ensure either input_ids or input_prompt is provided  
        if input_ids is None and input_prompt is None:  
            raise ValueError("Either input_ids or input_prompt must be provided")  
  
        # Ensure either labels or expected_output is provided  
        if labels is None and expected_output is None:  
            raise ValueError("Either labels or expected_output must be provided")  
  
        # Tokenize input_prompt if input_ids not provided  
        if input_ids is None:  
            if isinstance(input_prompt, str):  
                input_prompt = [input_prompt]  
  
            # Set padding_side to 'left' for input prompts  
            self.small_tokenizer.padding_side = 'left'  
            encoded_prompt = self.small_tokenizer(  
                input_prompt,  
                return_tensors="pt",  
                padding='max_length',  
                truncation=True,  
                max_length=self.max_length,  
                add_special_tokens=True  
            )  
            input_prompt_ids = encoded_prompt['input_ids'].to(self.small_model.device)  
            attention_mask_prompt = encoded_prompt['attention_mask'].to(self.small_model.device)  
        else:  
            # Assume input_ids includes input_prompt  
            input_prompt_ids = input_ids  
            attention_mask_prompt = attention_mask  
  
        batch_size = input_prompt_ids.size(0)  
        seq_len_prompt = input_prompt_ids.size(1)  
  
        # Tokenize expected_output to get labels if labels not provided  
        if labels is None:  
            if isinstance(expected_output, str):  
                expected_output = [expected_output]  
  
            # Set padding_side to 'right' for labels  
            self.small_tokenizer.padding_side = 'right'  
            encoded_output = self.small_tokenizer(  
                expected_output,  
                return_tensors="pt",  
                padding='max_length',  
                truncation=True,  
                max_length=self.max_length,  
                add_special_tokens=True  
            )  
            labels_ids = encoded_output['input_ids'].to(self.small_model.device)  
            labels_attention_mask = encoded_output['attention_mask'].to(self.small_model.device)  
        else:  
            labels_ids = labels  
            labels_attention_mask = labels_attention_mask  
  
        seq_len_labels = labels_ids.size(1)  
  
        # Prepare input_ids by concatenating input_prompt_ids and labels_ids  
        input_ids = torch.cat([input_prompt_ids, labels_ids], dim=1)  # [batch_size, seq_len_prompt + seq_len_labels]  
        attention_mask = torch.cat([attention_mask_prompt, labels_attention_mask], dim=1)  # [batch_size, seq_len_total]  
        seq_len_total = input_ids.size(1)  
  
        # Create loss mask to compute loss only on labels (expected_output)  
        # Exclude positions corresponding to the input_prompt  
        loss_mask = torch.zeros_like(input_ids, dtype=torch.bool)  
        loss_mask[:, seq_len_prompt:] = True  # Start computing loss from the labels  
        loss_mask[:, :seq_len_prompt] = False  # Exclude the input_prompt from loss computation
        # Exclude padding tokens from loss computation
        loss_mask = loss_mask & (input_ids != self.small_tokenizer.pad_token_id)
        # sum the loss_mask
        # print(f"Loss mask: {loss_mask.sum()}")
        # Pass through the small model with LoRA  
        # print(self.small_model)
        outputs = self.small_model(  
            input_ids=input_ids,  
            attention_mask=attention_mask,  
            output_hidden_states=False  
        )  
  
        logits = self._get_logits(outputs)  # [batch_size, seq_len_total, vocab_size]  
  
        # Shift logits and input_ids for language modeling loss  
        logits = logits[:, :-1, :].contiguous()  
        target_ids = input_ids[:, 1:].contiguous()  
        loss_mask = loss_mask[:, 1:].contiguous()  
  
        # Flatten logits, target_ids, and loss_mask  
        logits_flat = logits.view(-1, logits.size(-1))  
        target_ids_flat = target_ids.view(-1)  
        loss_mask_flat = loss_mask.view(-1)  
  
        # Apply loss mask  
        logits_selected = logits_flat[loss_mask_flat]  
        target_ids_selected = target_ids_flat[loss_mask_flat]  
  
        # Compute loss  
        loss_fct = nn.CrossEntropyLoss(ignore_index=self.small_tokenizer.pad_token_id)  
        loss = loss_fct(logits_selected, target_ids_selected)  
        # print(f"Loss: {loss}")
        # print(f"Logits selected requires_grad: {logits_selected.requires_grad}")  
        # print(f"Target IDs selected requires_grad: {target_ids_selected.requires_grad}")  
        # print(f"Loss requires_grad before backward: {loss.requires_grad}")  
        return loss  
    
    def generate_text(  
        self,  
        input_ids: torch.Tensor,  
        attention_mask: torch.Tensor,  
        max_length: int = 100,  
        temperature: float = 1.0,  
        sampling_method: str = "greedy",  
        alpha: float = 1.0  # Scaling parameter for blending embeddings  
    ) -> str:  
        """  
        Generates text by appending the knowledge vector to the input embeddings, without recomputing the KV cache.  
        Args:  
            input_ids (torch.Tensor): Tensor of input token IDs.  
            attention_mask (torch.Tensor): Attention mask for the input.  
            max_length (int): Maximum number of tokens to generate.  
            temperature (float): Sampling temperature for controlling randomness.  
            sampling_method (str): Method of sampling ('greedy' or 'sample').  
            alpha (float): Scaling factor to blend the knowledge vector.  
        Returns:  
            str: Generated text as a string.  
        """  
        with torch.no_grad():  
            # Obtain the knowledge vector from the large model  
            # Get embeddings of the input tokens  
            model = self.small_model
            embedding_layer = self.embedding_layer
            model.eval()  
            device = input_ids.device  
            input_embeds = embedding_layer(input_ids)  
            position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long, device=device).unsqueeze(0).repeat(input_ids.size(0), 1)  
    
            # Pass through the small model to get the last hidden state  
            small_output = self.small_model(  
                inputs_embeds=input_embeds,  
                attention_mask=attention_mask,  
                position_ids=position_ids,
                output_hidden_states=True,  
                use_cache=False  # Not using cache here  
            )  
            generated_ids = input_ids.clone()  
            newly_generated_ids = None
            current_attention_mask = attention_mask.clone()  
            past_key_values = None  # Initialize past_key_values as None  
    
            for idx in range(max_length):  
              
                if idx == 0:  
                    # First iteration: Append knowledge vector to embeddings  
                    input_embeds = embedding_layer(generated_ids)  
                    combined_input =input_embeds
                    position_ids = torch.arange(0, combined_input.size(1), dtype=torch.long, device=device).unsqueeze(0).repeat(combined_input.size(0), 1)  
                    # Update attention mask  
                    
                    # Since we've changed the inputs, we need to recompute past_key_values  
                    past_key_values = None  
                else:  
                    # Subsequent iterations: Use last generated token's embedding  
                    combined_input = embedding_layer(generated_ids[:, -1:])  
                    position_ids = torch.cat([position_ids, position_ids[:, -1:] + 1], dim=1)  
                    current_attention_mask = torch.cat(  
                        [current_attention_mask, torch.ones((current_attention_mask.size(0), 1), device=current_attention_mask.device)],  
                        dim=1  
                    )
                # Generate next token  
                # print(f"idx: {idx}, Combined input shape: {combined_input.shape}, current_attention_mask shape: {current_attention_mask.shape}")
                model_output = self.small_model(  
                    inputs_embeds=combined_input,  
                    attention_mask=current_attention_mask,  
                    position_ids=position_ids[:, -combined_input.size(1):],  
                    past_key_values=past_key_values,  
                    use_cache=True  
                )  
                logits = self._get_logits(model_output)[:, -1, :]  
                # Update past_key_values  
                past_key_values = model_output.past_key_values  
    
                # Sampling  
                if sampling_method == "greedy":  
                    next_token = torch.argmax(logits, dim=-1)  
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
                    
                
        final_output = newly_generated_ids # self.small_tokenizer.decode(newly_generated_ids[0], skip_special_tokens=True)  
        return final_output  

    
    def generate(  
        self,  
        input_ids: Optional[torch.Tensor] = None,  
        attention_mask: Optional[torch.Tensor] = None,  
        input_prompt: Optional[Union[str, List[str]]] = None,  
        max_length: Optional[int] = None,  
        temperature: float = 1.0,  
        sampling_method: str = "greedy",
        mode: str = "baseline"
    ) -> Union[str, List[str]]:  
        self.small_model.eval()
        # print(input_ids.shape)
        if max_length is None:  
            max_length = self.max_length  
  
        if input_ids is None and input_prompt is None:  
            raise ValueError("Either input_ids or input_prompt must be provided")  
  
        if input_ids is None:  
            if isinstance(input_prompt, str):  
                input_prompt = [input_prompt]  
            encoded = self.small_tokenizer(input_prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_length)  
            input_ids = encoded['input_ids'].to(self.small_model.device)  
            attention_mask = encoded['attention_mask'].to(self.small_model.device)  
            
        # inverse the input_ids to text
        input_text = self.small_tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        batch_size = input_ids.shape[0]  
        generated_texts = []  
        
        generated = self.generate_text(  # generate_text # simple_generate_new_v2
            input_ids=input_ids,  
            attention_mask=attention_mask,  
            max_length=max_length,
            temperature=temperature,  
            sampling_method=sampling_method  
        )  
        # print(batch_size)
        # print(generated.shape)
        generated_texts = self.small_tokenizer.batch_decode(generated, skip_special_tokens=True)  
        
        assert len(generated_texts) == batch_size
        for ix in range(batch_size):
            generated_texts[ix] = generated_texts[ix].replace(input_text[ix], "").replace(self.small_tokenizer.eos_token, "").replace(self.small_tokenizer.pad_token, "").strip()
        return generated_texts[0] if len(generated_texts) == 1 else generated_texts  
  
  
  
# Save model function  
def save_model(model, path):  
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)  
    torch.distributed.barrier()  
    if dist.get_rank() == 0:  
        print("Entering state_dict_type context manager for saving model.")  
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):  
        print(f"Rank {dist.get_rank()} is calling model.state_dict()")  
        state_dict = model.state_dict()  
        print(f"Rank {dist.get_rank()} has completed model.state_dict()")  
    torch.distributed.barrier()  
    if dist.get_rank() == 0:  
        print("Saving model on rank 0.")  
        torch.save(state_dict, path)  
        print("Model saved successfully.")  
    torch.distributed.barrier()  
    if dist.get_rank() == 0:  
        print("Completed save_model function.")  
  
# Load model function  
def load_model(model, path):  
    load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)  
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, load_policy):  
        if dist.get_rank() == 0:  
            state_dict = torch.load(path)  
        else:  
            state_dict = None  
        dist.barrier()  
        model.load_state_dict(state_dict)  
    return model  

    
def calculate_lora_parameters(model):  
    total_params = 0  
    for name, param in model.named_parameters():  
        if 'lora_' in name:  
            total_params += param.numel()  
    return total_params  
 


def lora_setup_fsdp():    
    from torch.nn import Embedding
     
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))    
    
    mp_policy = MixedPrecision(    
        param_dtype=torch.bfloat16,    
        reduce_dtype=torch.bfloat16,    
        buffer_dtype=torch.bfloat16,    
    )    
    
    # Custom auto_wrap_policy to handle LoRA layers    
    def lora_auto_wrap_policy(module, *args, **kwargs):    
        from peft.tuners.lora import LoraLayer 
        from peft.tuners.lora import Linear    
        if isinstance(module, Embedding):  
            return False  
        if isinstance(module, LoraModel) or isinstance(module, Linear) or isinstance(module, LoraLayer) or isinstance(module, LoRALinear):    
            return True    
        return False    
    
    fsdp_config = dict(    
        auto_wrap_policy=lora_auto_wrap_policy,    
        mixed_precision=mp_policy,    
        sharding_strategy=ShardingStrategy.FULL_SHARD,    
        device_id=torch.cuda.current_device(),    
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,    
        cpu_offload=CPUOffload(offload_params=False),  
        use_orig_params=True,  
    )    
    
    return fsdp_config






