import torch  
import os

from model_fsdp import DualModelTransformer
from model_fsdp_better_supervision import DualModelTransformerBetterSupervision
# Set TOKENIZERS_PARALLELISM to true in environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["HF_TOKEN"] = "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"
import torch.nn as nn  
import copy
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM  
from typing import List, Optional, Union, Tuple  
from SamplingMixin import SamplingMixin
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

class DualModelTransformerDistrib(DualModelTransformerBetterSupervision, SamplingMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
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
        Generates text by distributing the knowledge vector across non-pad input token embeddings.  You've proposed an insightful approach to mitigate the abrupt change in embedding distribution caused by adding the knowledge vector only to the last token's embedding or as an extra embedding. Your idea is to distribute the knowledge vector across all non-pad input token embeddings by equally dividing it by the number of non-pad tokens and then adding it to each non-pad token's embedding. This should help maintain the overall embedding distribution, allowing the model to perform better without additional training.
    
        Args:  
            input_ids (torch.Tensor): Tensor of input token IDs of shape [batch_size, seq_length].  
            attention_mask (torch.Tensor): Attention mask for the input of shape [batch_size, seq_length].  
            max_length (int): Maximum number of tokens to generate.  
            temperature (float): Sampling temperature for controlling randomness.  
            sampling_method (str): Method of sampling ('greedy' or 'sample').  
            alpha (float): Scaling factor for blending embeddings.  
    
        Returns:  
            str or List[str]: Generated text as a string if batch_size == 1, else a list of strings.  
        """  
        with torch.no_grad():  
            batch_size, seq_length = input_ids.size()  
            device = input_ids.device  
    
            # Initial processing with the small model to get past_key_values  
            small_last_hidden = self.get_query_vector(input_ids, attention_mask)
            # Transform small model's last hidden state to match large model's input dimension  
            large_input = self.ffn_small_to_large(small_last_hidden).unsqueeze(1)  # [batch_size, 1, large_hidden_size]  
            large_position_ids = torch.arange(0, large_input.size(1), dtype=torch.long, device=device).unsqueeze(0).repeat(large_input.size(0), 1)  
            large_output = self.large_model(  
                inputs_embeds=large_input,
                output_hidden_states=True,
                position_ids=large_position_ids,
            )  
    
            # Get last hidden state from the large model  
            large_last_hidden = self._get_last_hidden_state(large_output)  # [batch_size, large_hidden_size]  
    
            # Transform large model's last hidden state back to small model's dimension  
            knowledge_vector = self.ffn_large_to_small(large_last_hidden)  # [batch_size, hidden_size]  
    
            # Distribute the knowledge vector across non-pad token embeddings  
            embedding_layer = self.embedding_layer  
            input_embeds = embedding_layer(input_ids)  # [batch_size, seq_length, hidden_size]  
    
            # Compute the number of non-pad tokens per batch item  
            num_non_pad_tokens = attention_mask.sum(dim=1, keepdim=True)  # [batch_size, 1]  
            num_non_pad_tokens = num_non_pad_tokens.clamp(min=1)  # Avoid division by zero  
    
            # Scale the knowledge vector  
            scaled_knowledge_vector = knowledge_vector / num_non_pad_tokens  # [batch_size, hidden_size]  
            scaled_knowledge_vector = scaled_knowledge_vector.unsqueeze(1)  # [batch_size, 1, hidden_size]  
    
            # Expand attention mask to match the embedding dimensions  
            attention_mask_expanded = attention_mask.unsqueeze(-1)  # [batch_size, seq_length, 1]  
    
            # Compute the additive term  
            additive_term = scaled_knowledge_vector * attention_mask_expanded * alpha  # [batch_size, seq_length, hidden_size]  
    
            # Modify the input embeddings  
            input_embeds = input_embeds + additive_term  # [batch_size, seq_length, hidden_size]  
            position_ids = torch.arange(0, input_embeds.size(1), dtype=torch.long, device=device).unsqueeze(0).repeat(input_embeds.size(0), 1)
    
            # Re-run the small model with modified embeddings to update past_key_values  
            
             
            # Update past_key_values after modifying the embeddings  
            past_key_values = None
    
            # Start the generation loop  
            generated_ids = input_ids.clone()  
            attention_mask = attention_mask.clone()  
            newly_generated_ids = None
    
            for idx in range(max_length):  
              
                if idx == 0:  
                    combined_input = input_embeds
                    position_ids = torch.arange(0, combined_input.size(1), dtype=torch.long, device=device).unsqueeze(0).repeat(combined_input.size(0), 1)
                    current_attention_mask = attention_mask
                else:  
                    # Subsequent iterations: Use last generated token's embedding  
                    combined_input = embedding_layer(generated_ids[:, -1:])  
                    position_ids = torch.cat([position_ids, position_ids[:, -1:] + 1], dim=1)  
                    current_attention_mask = torch.cat(  
                        [current_attention_mask, torch.ones((current_attention_mask.size(0), 1), device=current_attention_mask.device)],  
                        dim=1  
                    )
                # Generate next token  
                # print(f"idx: {idx}, Combined input shape: {combined_input.shape}, current_attention_mask shape: {current_attention_mask.shape}, position_ids shape: {position_ids[:, -combined_input.size(1):].shape}")
                # for l in past_key_values:
                #     print(f"past_key_values shape: {l[0].shape}, {l[1].shape}")
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
                next_token, probs, top_logprobs_dict = self._sampling(logits, sampling_method, temperature)
        
                # Append generated token  
                generated_ids = torch.cat([generated_ids, next_token.unsqueeze(1)], dim=-1)  
                if newly_generated_ids is None:
                    newly_generated_ids = next_token.unsqueeze(1)
                else:
                    newly_generated_ids = torch.cat([newly_generated_ids, next_token.unsqueeze(1)], dim=-1)
                    
    
        final_output = newly_generated_ids # self.small_tokenizer.decode(newly_generated_ids[0], skip_special_tokens=True)  
        return final_output  
    
    def forward(  
        self,  
        input_ids: Optional[torch.Tensor] = None,  
        attention_mask: Optional[torch.Tensor] = None,  
        labels: Optional[torch.Tensor] = None,  
        labels_attention_mask: Optional[torch.Tensor] = None,  
        input_prompt: Optional[Union[str, List[str]]] = None,  
        expected_output: Optional[Union[str, List[str]]] = None,  
        temperature: float = 1.0,  
        sampling_method: str = "greedy",  
        alpha: float = 1.0  # Scaling parameter for blending embeddings  
    ) -> torch.Tensor:  
        """  
        Forward method for training the model with the knowledge vector distributed across non-pad input token embeddings.  
  
        Args:  
            input_ids (torch.Tensor, optional): Tensor of input token IDs (input_prompt).  
            attention_mask (torch.Tensor, optional): Attention mask for the input.  
            labels (torch.Tensor, optional): Tensor of labels for computing the loss.  
            labels_attention_mask (torch.Tensor, optional): Attention mask for the labels.  
            input_prompt (str or List[str], optional): Input prompt(s) as string(s), if input_ids is not provided.  
            expected_output (str or List[str], optional): Expected output(s) as string(s), if labels are not provided.  
            alpha (float): Scaling factor for blending embeddings.  
  
        Returns:  
            torch.Tensor: The computed total loss.  
        """  
        # Step 1: Prepare Inputs and Tokenization  
        if input_ids is None and input_prompt is None:  
            raise ValueError("Either input_ids or input_prompt must be provided")  
  
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
                padding='longest',  
                truncation=True,  
                max_length=self.max_length,  
                add_special_tokens=True  
            )  
            input_prompt_ids = encoded_prompt['input_ids'].to(self.small_model.device)  
            attention_mask_prompt = encoded_prompt['attention_mask'].to(self.small_model.device)  
        else:  
            input_prompt_ids = input_ids  
            attention_mask_prompt = attention_mask  
  
        batch_size = input_prompt_ids.size(0)  
        seq_len_prompt = input_prompt_ids.size(1)  
  
        # Tokenize expected_output if labels not provided  
        if labels is None:  
            if isinstance(expected_output, str):  
                expected_output = [expected_output]  
  
            # Set padding_side to 'right' for labels  
            self.small_tokenizer.padding_side = 'right'  
            encoded_output = self.small_tokenizer(  
                expected_output,  
                return_tensors="pt",  
                padding='longest',  
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
  
        # Concatenate input_prompt_ids and labels_ids  
        input_ids = torch.cat([input_prompt_ids, labels_ids], dim=1)  # [batch_size, seq_len_prompt + seq_len_labels]  
        attention_mask = torch.cat([attention_mask_prompt, labels_attention_mask], dim=1)  # [batch_size, seq_len_total]  
        seq_len_total = input_ids.size(1)  
  
        # Step 2: Obtain the Knowledge Vector  
        device = input_ids.device  
        embedding_layer = self.embedding_layer  
  
        # Get embeddings of the input_prompt_ids  
        input_prompt_embeds = embedding_layer(input_prompt_ids)  # [batch_size, seq_len_prompt, hidden_size]  
  
        # Pass through the small model to get the last hidden state  
        # We're only interested in the input_prompt_ids for obtaining the knowledge vector  
        small_last_hidden = self.get_query_vector(input_prompt_ids, attention_mask_prompt)
        # Transform and pass through the large model  
        large_input = self.ffn_small_to_large(small_last_hidden).unsqueeze(1)  # [batch_size, 1, large_model_dim]  
        large_position_ids = torch.arange(0, large_input.size(1), dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)  
        large_output = self.large_model(  
            inputs_embeds=large_input,  
            position_ids=large_position_ids,  
            output_hidden_states=True  
        )  
        large_last_hidden = self._get_last_hidden_state(large_output)  # [batch_size, large_model_dim]  

        # Transform back to small model's dimension  
        knowledge_vector = self.ffn_large_to_small(large_last_hidden)  # [batch_size, hidden_size] 
        # print shape of all these input_ids, large_input, seq_len_prompt, attention_mask, labels_ids, device, large_loss_weight
        # print(f"input_ids shape: {input_ids.shape}, large_input shape: {large_input.shape}, seq_len_prompt: {seq_len_prompt}, attention_mask shape: {attention_mask.shape}, labels_ids shape: {labels_ids.shape}")
        large_loss = self.compute_large_loss(input_ids, large_input, seq_len_prompt, attention_mask, labels_ids, device, large_loss_weight)
  
        # Step 3: Distribute the Knowledge Vector Across Non-Pad Embeddings  
        # Compute the number of non-pad tokens in input_prompt_ids  
        num_non_pad_tokens = attention_mask_prompt.sum(dim=1, keepdim=True)  # [batch_size, 1]  
        num_non_pad_tokens = num_non_pad_tokens.clamp(min=1)  # Avoid division by zero  
  
        # Scale the knowledge vector  
        scaled_knowledge_vector = (knowledge_vector / num_non_pad_tokens) * alpha  # [batch_size, hidden_size]  
        scaled_knowledge_vector = scaled_knowledge_vector.unsqueeze(1)  # [batch_size, 1, hidden_size]  
  
        # Expand attention mask to match embeddings  
        attention_mask_prompt_expanded = attention_mask_prompt.unsqueeze(-1)  # [batch_size, seq_len_prompt, 1]  
  
        # Compute the additive term for non-pad positions  
        additive_term = scaled_knowledge_vector * attention_mask_prompt_expanded  # [batch_size, seq_len_prompt, hidden_size]  
  
        # Modify the input_prompt embeddings  
        input_prompt_embeds = input_prompt_embeds + additive_term  # [batch_size, seq_len_prompt, hidden_size]  
  
        # Get embeddings for labels_ids  
        labels_embeds = embedding_layer(labels_ids)  # [batch_size, seq_len_labels, hidden_size]  
  
        # Concatenate modified input_prompt_embeds with labels_embeds  
        input_embeds = torch.cat([input_prompt_embeds, labels_embeds], dim=1)  # [batch_size, seq_len_total, hidden_size]  
  
        # Update attention_mask and position_ids  
        position_ids = torch.arange(0, seq_len_total, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)  
  
        # Step 4: Pass Modified Embeddings Through the Small Model  
        small_model_outputs = self.small_model(  
            inputs_embeds=input_embeds,  
            attention_mask=attention_mask,  
            position_ids=position_ids,  
            output_hidden_states=False,  
            use_cache=False  
        )  
  
        logits = self._get_logits(small_model_outputs)  # [batch_size, seq_len_total, vocab_size]  
  
        # Step 5: Compute the Main Loss  
        # Shift logits and targets for language modeling loss  
        logits = logits[:, :-1, :].contiguous()  
        target_ids = input_ids[:, 1:].contiguous()  
  
        # Create loss mask to compute loss only on labels (expected_output)  
        loss_mask = torch.zeros_like(target_ids, dtype=torch.bool)  
        loss_mask[:, seq_len_prompt-1:] = True  # Start computing loss from the position after input_prompt  
        # Exclude padding tokens from loss computation  
        loss_mask = loss_mask & (target_ids != self.small_tokenizer.pad_token_id)  
  
        # Compute the main loss  
        loss = self.compute_loss(logits, target_ids, loss_mask)  
  
        # Step 6: Compute Additional Losses (if applicable)  
        # Compute small_model_loss and large_loss as in the existing forward method  
        # Example placeholders (you may adjust the computation based on your implementation)  
  
        # Small model loss on labels (optional)  
        # You may choose to compute additional losses if needed  
        small_model_loss = torch.tensor(0.0, device=device)  
  
        # Step 7: Return the Total Loss  
        total_loss = loss + small_model_loss + large_loss  
  
        return total_loss  
    
    def compute_large_loss(self, input_ids, large_input, seq_len_prompt, attention_mask, labels_ids, device, large_loss_weight):
        large_embeds = self.embedding_layer_large(input_ids)
        # large_embeds[:, seq_len_prompt:seq_len_prompt+1, :] = large_input
        large_embeds = torch.cat([large_input, large_embeds[:, seq_len_prompt:, :]], dim=1)
        large_attention_mask = attention_mask[:, seq_len_prompt:]
        large_attention_mask = torch.cat([torch.ones((large_attention_mask.size(0), 1), device=large_attention_mask.device), large_attention_mask], dim=1)
        large_position_ids = torch.arange(0, large_embeds.size(1), dtype=torch.long, device=device).unsqueeze(0).repeat(large_embeds.size(0), 1)
        # print shape of all these large_embeds, large_attention_mask, large_position_ids
        # print(f"large_embeds shape: {large_embeds.shape}, large_attention_mask shape: {large_attention_mask.shape}, large_position_ids shape: {large_position_ids.shape}")
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
        
        # large_target_ids = large_target_ids[:, :-1].contiguous()
        # large_loss_mask = large_loss_mask[:, :-1].contiguous()
        # print shape of all these large_logits, large_target_ids, large_loss_mask
        # print(f"large_logits shape: {large_logits.shape}, large_target_ids shape: {large_target_ids.shape}, large_loss_mask shape: {large_loss_mask.shape}")
        large_loss = large_loss_weight * self.compute_loss(large_logits, large_target_ids, large_loss_mask)
        return large_loss

  