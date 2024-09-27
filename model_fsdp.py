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
            nn.Linear(large_model_dim, large_model_dim, dtype=torch.bfloat16),
            # nn.LayerNorm(large_model_dim, dtype=torch.bfloat16)
        )  
        self.ffn_large_to_small = nn.Sequential(  
            nn.Linear(large_model_dim, small_model_dim, dtype=torch.bfloat16),  
            nn.GELU(),  
            nn.Linear(small_model_dim, small_model_dim, dtype=torch.bfloat16),
            # nn.LayerNorm(small_model_dim, dtype=torch.bfloat16)
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
        self.small_tokenizer.padding_side = 'left'  
        self.large_tokenizer.padding_side = 'left'
        if self.small_tokenizer.pad_token is None:  
            self.small_tokenizer.pad_token = self.small_tokenizer.eos_token  
        if self.large_tokenizer.pad_token is None:  
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
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
  
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
        
    def generate_text_v3(  
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
    
            # Initial processing with the small model to get past_key_values  
            small_output = self.small_model(  
                input_ids=input_ids,  
                attention_mask=attention_mask,  
                use_cache=True  
            )  
            past_key_values = small_output.past_key_values  
    
            # Obtain the knowledge vector from the large model  
            small_last_hidden = self._get_last_hidden_state(small_output)[:, -1, :]  # [batch_size, hidden_size]  
    
            # Transform small model's last hidden state to match large model's input dimension  
            large_input = self.ffn_small_to_large(small_last_hidden).unsqueeze(1)  # [batch_size, 1, large_hidden_size]  
            large_output = self.large_model(  
                inputs_embeds=large_input  
            )  
    
            # Get last hidden state from the large model  
            large_last_hidden = self._get_last_hidden_state(large_output)[:, -1, :]  # [batch_size, large_hidden_size]  
    
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
    
            # Re-run the small model with modified embeddings to update past_key_values  
            small_output = self.small_model(  
                inputs_embeds=input_embeds,  
                attention_mask=attention_mask,  
                use_cache=True  
            )  
            # Update past_key_values after modifying the embeddings  
            past_key_values = small_output.past_key_values  
    
            # Start the generation loop  
            generated_ids = input_ids.clone()  
            attention_mask = attention_mask.clone()  
    
            for idx in range(max_length):  
                # Generate next token using past_key_values  
                output = self.small_model(  
                    input_ids=generated_ids[:, -1:],  # Only the last generated token ID  
                    past_key_values=past_key_values,  
                    use_cache=True  
                )  
                logits = self._get_logits(output)[:, -1, :]  # [batch_size, vocab_size]  
                # Update past_key_values for the next iteration  
                past_key_values = output.past_key_values  
    
                # Sampling  
                if sampling_method == 'greedy':  
                    next_token = torch.argmax(logits, dim=-1)  # [batch_size]  
                elif sampling_method == 'sample':  
                    probs = torch.nn.functional.softmax(logits / temperature, dim=-1)  
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(1)  # [batch_size]  
                else:  
                    raise ValueError("Invalid sampling method. Choose 'greedy' or 'sample'.")  
    
                # Append the generated token to the sequence  
                generated_ids = torch.cat([generated_ids, next_token.unsqueeze(1)], dim=-1)  # [batch_size, seq_length + idx + 1]  
    
                # Update attention_mask for new tokens  
                attention_mask = torch.cat(  
                    [attention_mask, torch.ones(batch_size, 1, device=attention_mask.device, dtype=attention_mask.dtype)],  
                    dim=1  
                )  # [batch_size, seq_length + idx + 1]  
    
                # Decode the current sequence to check for stop conditions  
                current_output = self.small_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)  
    
                # Check for stop tokens in each batch element  
                stop_generation = [False] * batch_size  
                for i in range(batch_size):  
                    if (  
                        next_token[i].item() == self.small_tokenizer.eos_token_id or  
                        any(stop_token in current_output[i] for stop_token in self.stop_tokens)  
                    ):  
                        stop_generation[i] = True  
    
                if all(stop_generation):  
                    break  
    
            # Decode the final generated sequences  
            final_output = self.small_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)  
            # For single input, return the first generated text  
            return final_output[0] if batch_size == 1 else final_output  

  
    def generate_text_v2(  
        self,  
        input_ids: torch.Tensor,  
        attention_mask: torch.Tensor,  
        max_length: int = 100,  
        temperature: float = 1.0,  
        sampling_method: str = "greedy",  
        alpha: float = 1.0  # Scaling parameter for blending embeddings  
    ) -> str:  
        """  
        Generates text by modifying the embedding of the last token with the knowledge vector, requiring KV cache update.  
    
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
            # Initial processing with the small model to get past_key_values  
            small_output = self.small_model(  
                input_ids=input_ids,  
                attention_mask=attention_mask,  
                use_cache=True  
            )  
            past_key_values = small_output.past_key_values  
    
            # Obtain the knowledge vector from the large model  
            small_last_hidden = self._get_last_hidden_state(small_output)[:, -1, :]  
            # Transform small model's last hidden state to match large model's input dimension  
            large_input = self.ffn_small_to_large(small_last_hidden).unsqueeze(1)  
            large_output = self.large_model(  
                inputs_embeds=large_input  
            )  
            # Get last hidden state from the large model  
            large_last_hidden = self._get_last_hidden_state(large_output)[:, -1, :]  
            # Transform large model's last hidden state back to small model's dimension  
            knowledge_vector = self.ffn_large_to_small(large_last_hidden)  
    
            # Modify the embedding of the last token with the knowledge vector  
            embedding_layer = self.embedding_layer  
            input_embeds = embedding_layer(input_ids)  
            # Apply alpha scaling to blend the knowledge vector  
            input_embeds[:, -1, :] = input_embeds[:, -1, :] * (1 - alpha) + knowledge_vector * alpha  
    
            # Re-run the small model with modified embeddings to update past_key_values  
            small_output = self.small_model(  
                inputs_embeds=input_embeds,  
                attention_mask=attention_mask,  
                use_cache=True  
            )  
            # Update past_key_values after modifying the embeddings  
            past_key_values = small_output.past_key_values  
    
            # Start the generation loop  
            generated_ids = input_ids.clone()  
    
            for idx in range(max_length):  
                with torch.no_grad():  
                    # Generate next token using past_key_values  
                    output = self.small_model(  
                        input_ids=generated_ids[:, -1:],  # Only the last generated token ID  
                        past_key_values=past_key_values,  
                        use_cache=True  
                    )  
                    logits = self._get_logits(output)[:, -1, :]  
                    # Update past_key_values for the next iteration  
                    past_key_values = output.past_key_values  
    
                # Sampling  
                if sampling_method == 'greedy':  
                    next_token = torch.argmax(logits, dim=-1)  
                elif sampling_method == 'sample':  
                    probs = torch.nn.functional.softmax(logits / temperature, dim=-1)  
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(1)  
                else:  
                    raise ValueError("Invalid sampling method. Choose 'greedy' or 'sample'.")  
    
                # Append the generated token to the sequence  
                generated_ids = torch.cat([generated_ids, next_token.unsqueeze(1)], dim=-1)  
    
                # Decode the current sequence to check for stop conditions  
                current_output = self.small_tokenizer.decode(generated_ids[0], skip_special_tokens=True)  
    
                # Check for EOS token or stop tokens  
                if (  
                    next_token.item() == self.small_tokenizer.eos_token_id or  
                    any(stop_token in current_output for stop_token in self.stop_tokens)  
                ):  
                    break  
    
            # Decode the final generated sequence  
            final_output = self.small_tokenizer.decode(generated_ids[0], skip_special_tokens=True)  
            return final_output  

    
    
    def simple_generate_new_v2(  self,
        input_ids: torch.Tensor,  
        attention_mask: torch.Tensor,  
        max_length,
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
        tokenizer = self.small_tokenizer
        model = self.small_model
        embedding_layer = self.embedding_layer
        model.eval()  
        device = input_ids.device  
    
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
                    # print(next_token.shape, newly_generated_ids.shape)
                else:  
                    newly_generated_ids = torch.cat([newly_generated_ids, next_token.unsqueeze(1)], dim=-1) 
                    # print(next_token.shape, newly_generated_ids.shape)
    
                # Decode current output  
                # newly_generated_text = tokenizer.decode(newly_generated_ids[0], skip_special_tokens=True)  
                # Debugging information (optional)  
                # print(f"idx: {idx}, Newly generated text: '{newly_generated_text}', Next token ID: {next_token.item()}, EOS token ID: {self.small_tokenizer.eos_token_id}")  
    
                # Check for EOS token  
                # if next_token.item() == tokenizer.eos_token_id:  
                #     break  
    
        # Decode final output  
        # final_output = tokenizer.decode(newly_generated_ids[0], skip_special_tokens=True)  
        final_output = newly_generated_ids
        return final_output  
    
    
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
            small_last_hidden = self._get_last_hidden_state(small_output)[:, -1, :]  
    
            # Transform and pass through the large model  
            large_input = self.ffn_small_to_large(small_last_hidden).unsqueeze(1)  
            large_position_ids = torch.arange(0, large_input.size(1), dtype=torch.long, device=device).unsqueeze(0).repeat(large_input.size(0), 1)  
            large_output = self.large_model(  
                inputs_embeds=large_input,  
                output_hidden_states=True,
                position_ids=large_position_ids,
            )  
            large_last_hidden = self._get_last_hidden_state(large_output)[:, -1, :]  
            knowledge_vector = self.ffn_large_to_small(large_last_hidden)  
    
            # Scale the knowledge vector  
            knowledge_vector_scaled = alpha * knowledge_vector.unsqueeze(1)  
    
            # Initialize variables for generation  
            generated_ids = input_ids.clone()  
            newly_generated_ids = None
            current_attention_mask = attention_mask.clone()  
            past_key_values = None  # Initialize past_key_values as None  
    
            for idx in range(max_length):  
              
                if idx == 0:  
                    # First iteration: Append knowledge vector to embeddings  
                    input_embeds = embedding_layer(generated_ids)  
                    combined_input = torch.cat([input_embeds, knowledge_vector_scaled], dim=1)  
                    position_ids = torch.arange(0, combined_input.size(1), dtype=torch.long, device=device).unsqueeze(0).repeat(combined_input.size(0), 1)  
                    # Update attention mask  
                    current_attention_mask = torch.cat(  
                        [current_attention_mask, torch.ones((current_attention_mask.size(0), 1), device=current_attention_mask.device)],  
                        dim=1  
                    )  
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
                    
                # print(f"Newly generated ids: {newly_generated_ids.shape}, generated_ids: {generated_ids.shape}")
        
                # Update attention mask  
                # if idx > 0:  # Already updated in the first iteration  
                #     current_attention_mask = torch.cat(  
                #         [current_attention_mask, torch.ones((current_attention_mask.size(0), 1), device=current_attention_mask.device)],  
                #         dim=1  
                #     )  
        
                # Decode current output  
                # current_output = self.small_tokenizer.decode(generated_ids[0], skip_special_tokens=True)  
                # newly_generated_text = self.small_tokenizer.decode(newly_generated_ids[0], skip_special_tokens=True)
                # print(f"Newly generated text: {newly_generated_text}, Current output: {current_output}, generated_ids: {generated_ids}, Next token: {next_token.item()}, EOS token: {self.small_tokenizer.eos_token_id}")
                # print(f"Newly generated text: '{newly_generated_text}', newly_generated_ids: {newly_generated_ids[0]}, Next token: {next_token.item()}, EOS token: {self.small_tokenizer.eos_token_id}")
        
                # Check for EOS token or stop tokens  
                # if (  
                #     (next_token.item() == self.small_tokenizer.eos_token_id and newly_generated_text.strip() != "" and len(newly_generated_text.strip()) < max_length) or  
                #     any(stop_token in current_output for stop_token in self.stop_tokens)  
                # ):  
                #     break  
    
        # Decode final output  
        # print(f"Newly generated ids: {newly_generated_ids.shape}, generated_ids: {generated_ids.shape}, newly_generated_text: {newly_generated_text}")
        final_output = newly_generated_ids # self.small_tokenizer.decode(newly_generated_ids[0], skip_special_tokens=True)  
        return final_output  
  

  
    def generate(  
        self,  
        input_ids: Optional[torch.Tensor] = None,  
        attention_mask: Optional[torch.Tensor] = None,  
        input_prompt: Optional[Union[str, List[str]]] = None,  
        max_length: Optional[int] = None,  
        temperature: float = 1.0,  
        sampling_method: str = "greedy"  
    ) -> Union[str, List[str]]:  
        self.small_model.eval()
        self.large_model.eval()
        self.ffn_small_to_large.eval()
        self.ffn_large_to_small.eval()
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
        # print(input_ids[:, -32:])
        # print(input_ids.shape)
        generated = self.generate_text(  
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
            # print(generated_texts[ix])
  
        # for i in range(batch_size):  
        #     generated_text = self.simple_generate_new_v2(  
        #         input_ids[i].unsqueeze(0),  
        #         attention_mask[i].unsqueeze(0) if attention_mask is not None else None,  
        #         max_length,  
        #         temperature,  
        #         sampling_method  
        #     )
        #     # if generated text is tensor, then decode it
        #     if isinstance(generated_text, torch.Tensor):
        #         generated_text = self.small_tokenizer.batch_decode(generated_text, skip_special_tokens=True)  
        #         generated_text = generated_text[0]
        #     generated_text = generated_text.replace(input_text[i], "").replace(self.small_tokenizer.eos_token, "").replace(self.small_tokenizer.pad_token, "").strip()
        #     generated_texts.append(generated_text)  
        #     # print(f"Input: {input_text[i]}\tGenerated: {generated_text}\n")
  
        return generated_texts[0] if len(generated_texts) == 1 else generated_texts  
  
    def forward_v2(  
        self,  
        input_ids: Optional[torch.Tensor] = None,  
        attention_mask: Optional[torch.Tensor] = None,  
        labels: Optional[torch.Tensor] = None,  
        labels_attention_mask: Optional[torch.Tensor] = None,  
        input_prompt: Optional[Union[str, List[str]]] = None,  
        expected_output: Optional[Union[str, List[str]]] = None,  
        temperature: float = 1.0,  
        sampling_method: str = "greedy",  
        alpha: float = 1.0  # Scaling factor for blending embeddings  
    ) -> torch.Tensor:  
        """  
        Forward method corresponding to generate_text_v2, where the knowledge vector is added to the last token embedding.  
    
        Args:  
            input_ids (torch.Tensor, optional): Tensor of input token IDs (input_prompt).  
            attention_mask (torch.Tensor, optional): Attention mask for the input.  
            labels (torch.Tensor, optional): Tensor of labels for computing the loss.  
            labels_attention_mask (torch.Tensor, optional): Attention mask for the labels.  
            input_prompt (str or List[str], optional): Input prompt(s) as string(s), if input_ids is not provided.  
            expected_output (str or List[str], optional): Expected output(s) as string(s), if labels are not provided.  
            alpha (float, optional): Scaling factor for blending embeddings.  
    
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
        # Tokenize expected_output to get labels  
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
        # Get embeddings  
        embedding_layer = self.embedding_layer  
        input_embeds_prompt = embedding_layer(input_prompt_ids)  # [batch_size, seq_len_prompt, hidden_size]  
        # Under no_grad, get knowledge_vector  
        with torch.no_grad():  
            # Pass input_prompt_ids through small_model to get small_last_hidden  
            small_output = self.small_model(  
                input_ids=input_prompt_ids,  
                attention_mask=attention_mask_prompt,  
                output_hidden_states=True,  
                use_cache=False  
            )  
            small_last_hidden = self._get_last_hidden_state(small_output)[:, -1, :]  # [batch_size, hidden_size]  
            # Get knowledge_vector from large model  
            large_input = self.ffn_small_to_large(small_last_hidden).unsqueeze(1)  # [batch_size, 1, large_hidden_size]  
            large_output = self.large_model(  
                inputs_embeds=large_input,  
                output_hidden_states=True  
            )  
            large_last_hidden = self._get_last_hidden_state(large_output)[:, -1, :]  # [batch_size, large_hidden_size]  
        knowledge_vector = self.ffn_large_to_small(large_last_hidden)  # [batch_size, hidden_size]  
        # Modify the embedding of the last token with the knowledge vector  
        input_embeds_prompt[:, -1, :] = input_embeds_prompt[:, -1, :] * (1 - alpha) + knowledge_vector * alpha  
        # Get embeddings for expected_output  
        input_embeds_output = embedding_layer(labels_ids)  # [batch_size, seq_len_labels, hidden_size]  
        # Combine embeddings and input_ids  
        input_embeds = torch.cat([input_embeds_prompt, input_embeds_output], dim=1)  # [batch_size, seq_len_total, hidden_size]  
        input_ids = torch.cat([input_prompt_ids, labels_ids], dim=1)  # [batch_size, seq_len_total]  
        attention_mask = torch.cat([attention_mask_prompt, labels_attention_mask], dim=1)  # [batch_size, seq_len_total]  
        seq_len_total = input_ids.size(1)  
        # Pass through the small model  
        outputs = self.small_model(  
            inputs_embeds=input_embeds,  
            attention_mask=attention_mask,  
            use_cache=False,  
            output_hidden_states=False  
        )  
        logits = self._get_logits(outputs)  # [batch_size, seq_len_total, vocab_size]  
        # Shift logits and input_ids for language modeling loss  
        logits = logits[:, :-1, :].contiguous()  
        target_ids = input_ids[:, 1:].contiguous()  
        # Create loss mask to compute loss only on labels (expected_output)  
        # Positions corresponding to labels (after the prompt)  
        loss_mask = torch.zeros_like(target_ids, dtype=torch.bool)  
        loss_mask[:, seq_len_prompt - 1:] = True  # Start computing loss from the last token of prompt (modified) and onwards  
        # Flatten tensors  
        logits_flat = logits.view(-1, logits.size(-1))  
        target_ids_flat = target_ids.view(-1)  
        loss_mask_flat = loss_mask.view(-1)  
        # Select entries for loss computation  
        logits_selected = logits_flat[loss_mask_flat]  
        target_ids_selected = target_ids_flat[loss_mask_flat]  
        # Compute loss  
        loss_fct = nn.CrossEntropyLoss(ignore_index=self.small_tokenizer.pad_token_id)  
        loss = loss_fct(logits_selected, target_ids_selected)  
        return loss  

    
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
    
        # Under no_grad, get knowledge_vector  
        with torch.no_grad():  
            # Pass input_prompt_ids through small_model to get small_last_hidden  
            small_output = self.small_model(  
                input_ids=input_prompt_ids,  
                attention_mask=attention_mask_prompt,  
                output_hidden_states=True,  
                use_cache=False  
            )  
            small_last_hidden = self._get_last_hidden_state(small_output)[:, -1, :]  # [batch_size, hidden_size]  
    
        # Get knowledge_vector from large model  
        large_input = self.ffn_small_to_large(small_last_hidden).unsqueeze(1)  # [batch_size, 1, large_hidden_size]  
        large_output = self.large_model(  
            inputs_embeds=large_input,  
            output_hidden_states=True  
        )  
        large_last_hidden = self._get_last_hidden_state(large_output)[:, -1, :]  # [batch_size, large_hidden_size]  
        knowledge_vector = self.ffn_large_to_small(large_last_hidden)  # [batch_size, hidden_size]  
    
        # Insert the knowledge_vector as a new token at the end of the input_prompt_ids embeddings  
        knowledge_vector_position = seq_len_prompt  # Index where knowledge_vector is placed  
        # Create a new embedding for knowledge_vector  
        knowledge_vector_embeds = knowledge_vector.unsqueeze(1)  # [batch_size, 1, hidden_size]  
        # Concatenate embeddings: input_prompt_embeds + knowledge_vector_embeds + labels_embeds  
        # input_embeds = torch.cat([input_embeds[:, :seq_len_prompt, :], knowledge_vector_embeds, input_embeds[:, seq_len_prompt:, :]], dim=1)  
        input_embeds[:, seq_len_prompt:seq_len_prompt+1, :] = knowledge_vector_embeds
        
        # Update attention_mask to account for the new token  
        # knowledge_vector_attention_mask = torch.ones((batch_size, 1), device=attention_mask.device)  
        # attention_mask = torch.cat([attention_mask[:, :seq_len_prompt], knowledge_vector_attention_mask, attention_mask[:, seq_len_prompt:]], dim=1)  
    
        seq_len_total = input_embeds.size(1)  
    
        # Pass through the small model  
        outputs = self.small_model(  
            inputs_embeds=input_embeds,  
            attention_mask=attention_mask,  
            use_cache=False,  
            output_hidden_states=False  
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
    
        return loss  

def save_model_temp(model, path):  
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)  
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):  
        state_dict = model.state_dict()  
    if dist.get_rank() == 0:  
        torch.save(state_dict, path)  
        

def save_model(model, path):  
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)  
    torch.distributed.barrier()  
    if dist.get_rank() == 0:  
        logger.info("Entering state_dict_type context manager for saving model.")  
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):  
        logger.info(f"Rank {dist.get_rank()} is calling model.state_dict()")  
        state_dict = model.state_dict()  
        logger.info(f"Rank {dist.get_rank()} has completed model.state_dict()")  
    torch.distributed.barrier()  
    if dist.get_rank() == 0:  
        logger.info("Saving model on rank 0.")  
        torch.save(state_dict, path)  
        logger.info("Model saved successfully.")  
    torch.distributed.barrier()  
    if dist.get_rank() == 0:  
        logger.info("Completed save_model function.")  

  
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
