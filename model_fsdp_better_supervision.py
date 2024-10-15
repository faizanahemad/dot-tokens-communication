import torch  
import os

from model_fsdp import DualModelTransformer
# Set TOKENIZERS_PARALLELISM to true in environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["HF_TOKEN"] = "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"
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
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  
logger = logging.getLogger(__name__)  

large_loss_weight = 0.2
small_loss_weight = 0.1

class DualModelTransformerBetterSupervision(DualModelTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
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
        # assert (input_ids is not None and attention_mask is not None) or input_prompt is not None
        # assert (labels is not None and labels_attention_mask is not None) or expected_output is not None
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
        
        
        # Prepare input_ids by concatenating input_prompt_ids, placeholder_id, labels_ids    
        pad_token_id = self.small_tokenizer.pad_token_id    
        placeholder_id = torch.full((batch_size, 1), pad_token_id, device=input_prompt_ids.device, dtype=input_prompt_ids.dtype)    
        input_ids = torch.cat([input_prompt_ids, placeholder_id, labels_ids], dim=1)  # [batch_size, seq_len_prompt + 1 + seq_len_labels]  
        attention_mask = torch.cat([attention_mask_prompt, torch.ones((batch_size, 1), device=attention_mask_prompt.device), labels_attention_mask], dim=1)  # [batch_size, seq_len_total]  
        seq_len_total = input_ids.size(1)  
        # print(input_ids.shape, attention_mask.shape)
        
        # Prepare input_ids by concatenating input_prompt_ids and labels_ids  
        # input_ids = torch.cat([input_prompt_ids, labels_ids], dim=1)  # [batch_size, seq_len_prompt + seq_len_labels]  
        # attention_mask = torch.cat([attention_mask_prompt, labels_attention_mask], dim=1)  # [batch_size, seq_len_total]  
        # seq_len_total = input_ids.size(1)  
        # print(input_ids.shape, attention_mask.shape)
    
        # Get embeddings  
        embedding_layer = self.embedding_layer  
        input_embeds = embedding_layer(input_ids)  # [batch_size, seq_len_total, hidden_size]  
        device = input_embeds.device
        
        
    
        # Under no_grad, get knowledge_vector  
        
        small_last_hidden = self.get_query_vector(input_prompt_ids, attention_mask_prompt)
        # Get knowledge_vector from large model  
        large_input = self.ffn_small_to_large(small_last_hidden).unsqueeze(1)  # [batch_size, 1, large_hidden_size] 
        large_position_ids = torch.arange(0, large_input.size(1), dtype=torch.long, device=device).unsqueeze(0).repeat(large_input.size(0), 1)   
        large_output = self.large_model(  
            inputs_embeds=large_input,  
            output_hidden_states=True,
            position_ids=large_position_ids
        )  
        large_last_hidden = self._get_last_hidden_state(large_output)  # [batch_size, large_hidden_size]  
        knowledge_vector = self.ffn_large_to_small(large_last_hidden)  # [batch_size, hidden_size]  
    
        # Insert the knowledge_vector as a new token at the end of the input_prompt_ids embeddings  
        knowledge_vector_position = seq_len_prompt  # Index where knowledge_vector is placed  
        # Create a new embedding for knowledge_vector  
        knowledge_vector_embeds = knowledge_vector.unsqueeze(1)  # [batch_size, 1, hidden_size]  
        # Concatenate embeddings: input_prompt_embeds + knowledge_vector_embeds + labels_embeds  
        # input_embeds = torch.cat([input_embeds[:, :seq_len_prompt, :], knowledge_vector_embeds, input_embeds[:, seq_len_prompt:, :]], dim=1)  
        input_embeds[:, seq_len_prompt:seq_len_prompt+1, :] = knowledge_vector_embeds
        
        large_loss = self.compute_large_loss(input_ids, large_input, seq_len_prompt, attention_mask, labels_ids, device, large_loss_weight)
        
        
        
        
        # Update attention_mask to account for the new token  
        # knowledge_vector_attention_mask = torch.ones((batch_size, 1), device=attention_mask.device)  
        # attention_mask = torch.cat([attention_mask[:, :seq_len_prompt], knowledge_vector_attention_mask, attention_mask[:, seq_len_prompt:]], dim=1)  
        small_loss = self.compute_small_loss(input_embeds, seq_len_prompt, attention_mask, labels_ids, device, small_loss_weight)
        
        position_ids = torch.arange(0, input_embeds.size(1), dtype=torch.long, device=device).unsqueeze(0).repeat(input_embeds.size(0), 1)  
        outputs = self.small_model(  
            inputs_embeds=input_embeds,  
            attention_mask=attention_mask,  
            use_cache=False,  
            output_hidden_states=False,
            position_ids=position_ids
        )  
    
        logits = self._get_logits(outputs)  # [batch_size, seq_len_total, vocab_size]  
    
        # Shift logits and input_ids for language modeling loss  
        # print shape of logits and input_ids
        # print(logits.shape, input_ids.shape)
        logits = logits[:, :-1, :].contiguous()  
        target_ids = input_ids[:, 1:].contiguous()  
        # print(logits.shape, target_ids.shape)
    
        # Create loss mask to compute loss only on labels (expected_output)  
        # Exclude positions corresponding to the input_prompt and knowledge_vector  
        loss_mask = torch.zeros_like(target_ids, dtype=torch.bool)  
        loss_mask[:, seq_len_prompt:] = True  # Start computing loss from the labels  
        loss_mask[:, :seq_len_prompt] = False  # Exclude the input_prompt from loss computation
        # Exclude padding tokens from loss computation
        loss_mask = loss_mask & (target_ids != self.small_tokenizer.pad_token_id)
        
        loss = self.compute_loss(logits, target_ids, loss_mask) 
        return loss + large_loss + small_loss
    
    def compute_large_loss(self, input_ids, large_input, seq_len_prompt, attention_mask, labels_ids, device, large_loss_weight):
        large_embeds = self.embedding_layer_large(input_ids)
        large_embeds[:, seq_len_prompt:seq_len_prompt+1, :] = large_input
        large_embeds = large_embeds[:, seq_len_prompt:, :]
        large_attention_mask = attention_mask[:, seq_len_prompt:]
        large_position_ids = torch.arange(0, large_embeds.size(1), dtype=torch.long, device=device).unsqueeze(0).repeat(large_embeds.size(0), 1)
        large_output_for_loss = self.large_model(
            inputs_embeds=large_embeds,
            attention_mask=large_attention_mask,
            use_cache=False,
            output_hidden_states=False,
            position_ids=large_position_ids
        )
        
        large_logits = self._get_logits(large_output_for_loss)
        large_logits = large_logits[:, :-1, :].contiguous()
        large_target_ids = labels_ids
        large_loss_mask = large_target_ids != self.small_tokenizer.pad_token_id
        # print shape of all these large_logits, large_target_ids, large_loss_mask
        # print(f"large_logits shape: {large_logits.shape}, large_target_ids shape: {large_target_ids.shape}, large_loss_mask shape: {large_loss_mask.shape}")
        large_loss = large_loss_weight * self.compute_loss(large_logits, large_target_ids, large_loss_mask)
        return large_loss
    
    def compute_small_loss(self, input_embeds, seq_len_prompt, attention_mask, labels_ids, device, small_loss_weight):
        position_ids = torch.arange(0, input_embeds.size(1), dtype=torch.long, device=device).unsqueeze(0).repeat(input_embeds.size(0), 1)  
        # Pass through the small model  
        small_model_outputs = self.small_model(  
            inputs_embeds=input_embeds[:, seq_len_prompt:],  
            attention_mask=attention_mask[:, seq_len_prompt:],  
            use_cache=False,  
            output_hidden_states=False,
            position_ids=position_ids[:, seq_len_prompt:]
        ) 
        small_model_logits = self._get_logits(small_model_outputs)
        small_model_logits = small_model_logits[:, :-1, :].contiguous()
        small_model_target_ids = labels_ids
        small_model_loss_mask = small_model_target_ids != self.small_tokenizer.pad_token_id
        small_model_loss = small_loss_weight * self.compute_loss(small_model_logits, small_model_target_ids, small_model_loss_mask)
        return small_model_loss