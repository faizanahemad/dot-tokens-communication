import torch  
import os
import torch.nn as nn  
import copy
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM  
from typing import List, Optional, Union, Tuple  
import torch.nn.functional as F  
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
import torch.distributed as dist  
  
class DualModelTransformer(nn.Module):  
    def __init__(  
        self,  
        large_model_name: str,  
        small_model_name: str,  
        stop_tokens: List[str],  
        small_model_dim: int,  
        large_model_dim: int,  
        max_length: int = 512,  
        fsdp_config: dict = None,
        enable_checkpointing: bool = True,
        enable_flash_attention: bool = True,
    ):  
        super().__init__()  
  
        # Initialize models  
        self.large_model = AutoModelForCausalLM.from_pretrained(large_model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)  
        self.small_model = AutoModelForCausalLM.from_pretrained(small_model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)  
  
        # Set no_grad to both models and their parameters  
        self.large_model.eval()  
        self.small_model.eval()  
        for param in self.large_model.parameters():  
            param.requires_grad = False  
        for param in self.small_model.parameters():  
            param.requires_grad = False  
  
        # Initialize FFNs with improved activation and initialization  
        self.ffn_small_to_large = nn.Sequential(  
            nn.Linear(small_model_dim, large_model_dim, dtype=torch.bfloat16),  
            nn.GELU(),  
            nn.Linear(large_model_dim, large_model_dim, dtype=torch.bfloat16)  
        )  
        self.ffn_large_to_small = nn.Sequential(  
            nn.Linear(large_model_dim, small_model_dim, dtype=torch.bfloat16),  
            nn.GELU(),  
            nn.Linear(small_model_dim, small_model_dim, dtype=torch.bfloat16)  
        )  
  
        # Initialize FFN parameters  
        self._init_weights(self.ffn_small_to_large)  
        self._init_weights(self.ffn_large_to_small)  

        self.embedding_layer = copy.deepcopy(self._get_embedding_layer(self.small_model))

        # Enable gradient checkpointing  
        if enable_checkpointing:  
            if hasattr(self.large_model, 'gradient_checkpointing_enable'):  
                self.large_model.gradient_checkpointing_enable()  
            else:  
                self.large_model.config.gradient_checkpointing = True  
            
            if hasattr(self.small_model, 'gradient_checkpointing_enable'):  
                self.small_model.gradient_checkpointing_enable()  
            else:  
                self.small_model.config.gradient_checkpointing = True  

        
  
        # Wrap components with FSDP if config is provided  
        if fsdp_config:  
            self.large_model = FSDP(self.large_model, **fsdp_config)  
            self.small_model = FSDP(self.small_model, **fsdp_config)  
            self.ffn_small_to_large = FSDP(self.ffn_small_to_large, **fsdp_config)  
            self.ffn_large_to_small = FSDP(self.ffn_large_to_small, **fsdp_config)  
            self.embedding_layer = FSDP(self.embedding_layer, **fsdp_config)
  
        # Initialize tokenizers  
        self.large_tokenizer = AutoTokenizer.from_pretrained(large_model_name, trust_remote_code=True)  
        self.small_tokenizer = AutoTokenizer.from_pretrained(small_model_name, trust_remote_code=True)  

        # add tokenizer.pad_token = tokenizer.eos_token
        self.small_tokenizer.pad_token = self.small_tokenizer.eos_token
        self.large_tokenizer.pad_token = self.large_tokenizer.eos_token
  
        self.stop_tokens = stop_tokens  
        self.small_model_dim = small_model_dim  
        self.large_model_dim = large_model_dim  
        self.max_length = max_length  
  
    def _init_weights(self, module):  
        if isinstance(module, nn.Linear):  
            torch.nn.init.xavier_uniform_(module.weight)  
            if module.bias is not None:  
                torch.nn.init.zeros_(module.bias)  
  
    def _get_embedding_layer(self, model):  
        if hasattr(model, 'get_input_embeddings'):  
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
  
    def generate_text(  
        self,  
        input_ids: torch.Tensor,  
        attention_mask: torch.Tensor,  
        max_length: int = 100,  
        temperature: float = 1.0,  
        sampling_method: str = "greedy"  
    ) -> str:  
        generated_ids = input_ids.clone()  
        current_attention_mask = attention_mask.clone()  
        embedding_layer = self.embedding_layer
        
            
  
        for idx in range(max_length):  
            with torch.no_grad():  
                # TODO: implement KV cache
                small_output = self.small_model(generated_ids, attention_mask=current_attention_mask, output_hidden_states=True)  
                small_last_hidden = self._get_last_hidden_state(small_output)[:, -1, :]  
                large_input = self.ffn_small_to_large(small_last_hidden).unsqueeze(1)  
                large_output = self.large_model(inputs_embeds=large_input, output_hidden_states=True)  
                large_last_hidden = self._get_last_hidden_state(large_output)[:, -1, :]  
                knowledge_vector = self.ffn_large_to_small(large_last_hidden)  
                input_embeds = embedding_layer(generated_ids)  
                combined_input = torch.cat([input_embeds, knowledge_vector.unsqueeze(1)], dim=1)  
                model_output = self.small_model(inputs_embeds=combined_input, attention_mask=torch.cat([current_attention_mask, torch.ones((current_attention_mask.shape[0], 1), device=current_attention_mask.device)], dim=1))  
                logits = self._get_logits(model_output)[:, -1, :]
  
            if sampling_method == "greedy":  
                next_token = torch.argmax(logits, dim=-1)  
            elif sampling_method == "sample":  
                probs = nn.functional.softmax(logits / temperature, dim=-1)  
                next_token = torch.multinomial(probs, num_samples=1)  
            else:  
                raise ValueError("Invalid sampling method")  
  
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(1)], dim=-1)  
            current_attention_mask = torch.cat([current_attention_mask, torch.ones((current_attention_mask.shape[0], 1), device=current_attention_mask.device)], dim=1)  
  
            if self.small_tokenizer.eos_token_id in next_token:  
                break  
  
        return self.small_tokenizer.decode(generated_ids[0], skip_special_tokens=True)  
  
    def generate(  
        self,  
        input_ids: Optional[torch.Tensor] = None,  
        attention_mask: Optional[torch.Tensor] = None,  
        input_prompt: Optional[Union[str, List[str]]] = None,  
        max_length: Optional[int] = None,  
        temperature: float = 1.0,  
        sampling_method: str = "greedy"  
    ) -> Union[str, List[str]]:  
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
  
        batch_size = input_ids.shape[0]  
        generated_texts = []  
  
        for i in range(batch_size):  
            generated_text = self.generate_text(  
                input_ids[i].unsqueeze(0),  
                attention_mask[i].unsqueeze(0) if attention_mask is not None else None,  
                max_length,  
                temperature,  
                sampling_method  
            )  
            generated_texts.append(generated_text)  
  
        return generated_texts[0] if len(generated_texts) == 1 else generated_texts  
  
    def forward(  
        self,  
        input_ids: Optional[torch.Tensor] = None,  
        attention_mask: Optional[torch.Tensor] = None,  
        labels: Optional[torch.Tensor] = None,  
        input_prompt: Optional[Union[str, List[str]]] = None,  
        expected_output: Optional[Union[str, List[str]]] = None,  
        temperature: float = 1.0,  
        sampling_method: str = "greedy"  
    ) -> Union[str, List[str], torch.Tensor]:  
        if input_ids is None and input_prompt is None:  
            raise ValueError("Either input_ids or input_prompt must be provided")  
  
        if input_ids is None:  
            if isinstance(input_prompt, str):  
                input_prompt = [input_prompt]  
            encoded = self.small_tokenizer(input_prompt, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)  
            input_ids = encoded['input_ids'].to(self.small_model.device)  
            attention_mask = encoded['attention_mask'].to(self.small_model.device)  
  
        if labels is None and expected_output is not None:  
            if isinstance(expected_output, str):  
                expected_output = [expected_output]  
            labels = self.small_tokenizer(expected_output, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)['input_ids'].to(self.small_model.device)  
  
        if labels is None:  
            return self.generate(input_ids, attention_mask, max_length=self.max_length, temperature=temperature, sampling_method=sampling_method)  
  
        embedding_layer = self.embedding_layer
        batch_size, seq_length = input_ids.size()  
  
        all_logits = []  
        for i in range(seq_length - 1):  
            with torch.no_grad():   
                small_output = self.small_model(input_ids[:, :i+1], attention_mask=attention_mask[:, :i+1], output_hidden_states=True)  
                small_last_hidden = self._get_last_hidden_state(small_output)[:, -1, :]  
            
            large_input = self.ffn_small_to_large(small_last_hidden).unsqueeze(1)  
            large_output = self.large_model(inputs_embeds=large_input, output_hidden_states=True)  
            large_last_hidden = self._get_last_hidden_state(large_output)[:, -1, :]  
            knowledge_vector = self.ffn_large_to_small(large_last_hidden)  
  
            input_embeds = embedding_layer(input_ids[:, :i+1])  
            combined_input = torch.cat([input_embeds, knowledge_vector.unsqueeze(1)], dim=1)  
            model_output = self.small_model(inputs_embeds=combined_input, attention_mask=torch.cat([attention_mask[:, :i+1], torch.ones((batch_size, 1), device=attention_mask.device)], dim=1))  
            current_logits = self._get_logits(model_output)[:, -1, :]  
            all_logits.append(current_logits)  
  
        logits = torch.stack(all_logits, dim=1)  
  
        loss_fct = nn.CrossEntropyLoss(ignore_index=self.small_tokenizer.pad_token_id)  
  
        # Shift labels to align with logits  
        shifted_labels = labels[:, 1:seq_length].contiguous()  
  
        # Pad shifted_labels if necessary  
        if shifted_labels.shape[1] < logits.shape[1]:  
            pad_length = logits.shape[1] - shifted_labels.shape[1]  
            shifted_labels = F.pad(shifted_labels, (0, pad_length), value=self.small_tokenizer.pad_token_id)  
  
        # Reshape logits and labels for loss calculation  
        logits_view = logits.view(-1, logits.size(-1))  
        labels_view = shifted_labels.view(-1)  
  
        loss = loss_fct(logits_view, labels_view)  
  
        return loss  
  
def save_model(model, path):  
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)  
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):  
        state_dict = model.state_dict()  
    if dist.get_rank() == 0:  
        torch.save(state_dict, path)  
  
def load_model(model, path):  
    if dist.get_rank() == 0:  
        state_dict = torch.load(path)  
    else:  
        state_dict = None  
    dist.barrier()  
    load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)  
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, load_policy):  
        model.load_state_dict(state_dict)  
  
def setup_fsdp():  
    dist.init_process_group("nccl")  
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))  
  
    mp_policy = MixedPrecision(  
        param_dtype=torch.bfloat16,  
        reduce_dtype=torch.bfloat16,  
        buffer_dtype=torch.bfloat16,  
    )  
  
    fsdp_config = dict(  
        auto_wrap_policy=size_based_auto_wrap_policy,  
        mixed_precision=mp_policy,  
        sharding_strategy=ShardingStrategy.FULL_SHARD,  
        device_id=torch.cuda.current_device(),  
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  
        cpu_offload=CPUOffload(offload_params=False),  
    )  
  
    return fsdp_config  
