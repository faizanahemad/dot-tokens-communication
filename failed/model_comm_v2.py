import torch  
import torch.nn as nn  
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM  
from typing import List, Optional, Union, Tuple  
import torch.nn.functional as F  
from transformers.utils import is_flash_attn_2_available
  
class DualModelTransformer(nn.Module):  
    def __init__(  
        self,  
        large_model_name: str,  
        small_model_name: str,  
        stop_tokens: List[str],  
        small_model_dim: int,  
        large_model_dim: int,  
        max_length: int = 512,  
        main_device: str = "cuda:0",  
        second_device: str = "cuda:1",  
        third_device: str = "cuda:2"  
    ):  
        super().__init__()  
        self.main_device = main_device  
        self.second_device = second_device  
        self.third_device = third_device  
  
        # Initialize models on their respective devices  
        self.large_model = AutoModelForCausalLM.from_pretrained(large_model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map=self.third_device, 
        # attn_implementation="flash_attention_2", 
        use_cache = False,
        )
        self.small_model = AutoModelForCausalLM.from_pretrained(small_model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map=self.second_device, 
        # attn_implementation="flash_attention_2", 
        use_cache = False,
        )
          
        # Set models to eval mode and freeze parameters  
        self.large_model.eval()  
        self.small_model.eval()  
        for param in self.large_model.parameters():  
            param.requires_grad = False  
        for param in self.small_model.parameters():  
            param.requires_grad = False  
  
        # Initialize tokenizers  
        self.large_tokenizer = AutoTokenizer.from_pretrained(large_model_name, trust_remote_code=True)  
        self.small_tokenizer = AutoTokenizer.from_pretrained(small_model_name, trust_remote_code=True)  
        # do tokenizer.padding_side  = 'left'
        self.small_tokenizer.padding_side = 'left'
        self.large_tokenizer.padding_side = 'left'
        # Initialize FFNs on the main device  
        self.ffn_small_to_large = nn.Sequential(  
            nn.Linear(small_model_dim, large_model_dim, dtype=torch.bfloat16),  
            nn.GELU(),  
            nn.Linear(large_model_dim, large_model_dim, dtype=torch.bfloat16)  
        ).to(self.main_device)  
        self.ffn_large_to_small = nn.Sequential(  
            nn.Linear(large_model_dim, small_model_dim, dtype=torch.bfloat16),  
            nn.GELU(),  
            nn.Linear(small_model_dim, small_model_dim, dtype=torch.bfloat16)  
        ).to(self.main_device)  
  
        # Initialize FFN parameters  
        self._init_weights(self.ffn_small_to_large)  
        self._init_weights(self.ffn_large_to_small)  
  
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
        generated_ids = input_ids.clone().to(self.second_device)  
        current_attention_mask = attention_mask.clone().to(self.second_device)  
        embedding_layer = self._get_embedding_layer(self.small_model)  
  
        for _ in range(max_length):  
            with torch.no_grad():  
                small_output = self.small_model(generated_ids, attention_mask=current_attention_mask, output_hidden_states=True)  
                small_last_hidden = self._get_last_hidden_state(small_output)[:, -1, :].to(self.main_device)  
                large_input = self.ffn_small_to_large(small_last_hidden).unsqueeze(1).to(self.third_device)  
                large_output = self.large_model(inputs_embeds=large_input, output_hidden_states=True)  
                large_last_hidden = self._get_last_hidden_state(large_output)[:, -1, :].to(self.main_device)  
                knowledge_vector = self.ffn_large_to_small(large_last_hidden).to(self.second_device)  
                input_embeds = embedding_layer(generated_ids)  
                combined_input = torch.cat([input_embeds, knowledge_vector.unsqueeze(1)], dim=1)  
                input_embeds = combined_input
                model_output = self.small_model(inputs_embeds=input_embeds, attention_mask=torch.cat([current_attention_mask, torch.ones((current_attention_mask.shape[0], 1), device=self.second_device)], dim=1))  
                logits = self._get_logits(model_output)[:, -1, :]  
  
            if sampling_method == "greedy":  
                next_token = torch.argmax(logits, dim=-1)  
            elif sampling_method == "sample":  
                probs = nn.functional.softmax(logits / temperature, dim=-1)  
                next_token = torch.multinomial(probs, num_samples=1)  
            else:  
                raise ValueError("Invalid sampling method")  
  
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(1)], dim=-1)  
            current_attention_mask = torch.cat([current_attention_mask, torch.ones((current_attention_mask.shape[0], 1), device=self.second_device)], dim=1)  
  
            if self.small_tokenizer.eos_token_id in next_token:  
                break  
  
        return self.small_tokenizer.decode(generated_ids[0].cpu(), skip_special_tokens=True)  
  
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
            input_ids = encoded['input_ids'].to(self.second_device)  
            attention_mask = encoded['attention_mask'].to(self.second_device)  
  
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
            input_ids = encoded['input_ids'].to(self.second_device)  
            attention_mask = encoded['attention_mask'].to(self.second_device)  
  
        if labels is None and expected_output is not None:  
            if isinstance(expected_output, str):  
                expected_output = [expected_output]  
            labels = self.small_tokenizer(expected_output, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)['input_ids'].to(self.second_device)  
  
        if labels is None:  
            return self.generate(input_ids, attention_mask, max_length=self.max_length, temperature=temperature, sampling_method=sampling_method)  
  
        embedding_layer = self._get_embedding_layer(self.small_model)  
        batch_size, seq_length = input_ids.size()  
  
        all_logits = []  
        for i in range(seq_length - 1):  
            input_ids = input_ids.to(self.second_device)
            attention_mask = attention_mask.to(self.second_device)
            with torch.no_grad():
                small_output = self.small_model(input_ids[:, :i+1], attention_mask=attention_mask[:, :i+1], output_hidden_states=True)  
                small_last_hidden = self._get_last_hidden_state(small_output)[:, -1, :].to(self.main_device)  
            large_input = self.ffn_small_to_large(small_last_hidden).unsqueeze(1).to(self.third_device)  
            large_output = self.large_model(inputs_embeds=large_input, output_hidden_states=True)  
            large_last_hidden = self._get_last_hidden_state(large_output)[:, -1, :].to(self.main_device)  
            knowledge_vector = self.ffn_large_to_small(large_last_hidden).to(self.second_device)  
  
            input_embeds = embedding_layer(input_ids[:, :i+1])  
            combined_input = torch.cat([input_embeds, knowledge_vector.unsqueeze(1)], dim=1)  
            model_output = self.small_model(inputs_embeds=combined_input, attention_mask=torch.cat([attention_mask[:, :i+1], torch.ones((batch_size, 1), device=self.second_device)], dim=1))  
            current_logits = self._get_logits(model_output)[:, -1, :]  
            all_logits.append(current_logits)  
  
        logits = torch.stack(all_logits, dim=1)  
  
        loss_fct = nn.CrossEntropyLoss(ignore_index=self.small_tokenizer.pad_token_id)  
  
        shifted_labels = labels[:, 1:seq_length].contiguous()  
  
        if shifted_labels.shape[1] < logits.shape[1]:  
            pad_length = logits.shape[1] - shifted_labels.shape[1]  
            shifted_labels = F.pad(shifted_labels, (0, pad_length), value=self.small_tokenizer.pad_token_id)  
  
        logits_view = logits.view(-1, logits.size(-1))  
        labels_view = shifted_labels.view(-1)  
  
        loss = loss_fct(logits_view, labels_view)  
  
        return loss  
  
    def save_pretrained(self, path: str):  
        torch.save({  
            'ffn_small_to_large': self.ffn_small_to_large.state_dict(),  
            'ffn_large_to_small': self.ffn_large_to_small.state_dict(),  
            'main_device': self.main_device,  
            'second_device': self.second_device,  
            'third_device': self.third_device,  
        }, path)  
  
    @classmethod  
    def from_pretrained(  
        cls,  
        path: str,  
        large_model_name: str,  
        small_model_name: str,  
        stop_tokens: List[str],  
        small_model_dim: int,  
        large_model_dim: int,  
        main_device: str = "cuda:0",  
        second_device: str = "cuda:1",  
        third_device: str = "cuda:2"  
    ):  
        state_dict = torch.load(path)  
        model = cls(large_model_name, small_model_name, stop_tokens, small_model_dim, large_model_dim,  
                    main_device=state_dict.get('main_device', main_device),  
                    second_device=state_dict.get('second_device', second_device),  
                    third_device=state_dict.get('third_device', third_device))  
        model.ffn_small_to_large.load_state_dict(state_dict['ffn_small_to_large'])  
        model.ffn_large_to_small.load_state_dict(state_dict['ffn_large_to_small'])  
        return model  
  
if __name__ == "__main__":  
    # Example usage  
    large_model_name = "microsoft/Phi-3-medium-4k-instruct"  
    small_model_name = "microsoft/Phi-3-mini-4k-instruct"  
    stop_tokens = [".", "!", "?"]  
    small_model_dim = 3072  
    large_model_dim = 5120  
      
    model = DualModelTransformer(  
        large_model_name,  
        small_model_name,  
        stop_tokens,  
        small_model_dim,  
        large_model_dim,  
        main_device="cuda:0",  
        second_device="cuda:1",  
        third_device="cuda:2"  
    )  
  
    # Test 1: Single sentence input  
    print("Test 1: Single sentence input")  
    input_prompt = "Once upon a time"  
    generated_text = model.generate(input_prompt=input_prompt)  
    print(f"Generated text (single sentence): {generated_text}")  
  
    print("\nTest 2: Multi-sentence input")  
    input_prompts = ["The quick brown fox", "In a galaxy far, far away"]  
    generated_texts = model.generate(input_prompt=input_prompts)  
    print(f"Generated texts (multi-sentence):")  
    for prompt, text in zip(input_prompts, generated_texts):  
        print(f"Prompt: {prompt}")  
        print(f"Generated: {text}\n")  
  
    print("Test 3: Multi-sentence tokenized input")  
    tokenized_input = model.small_tokenizer(input_prompts, return_tensors="pt", padding=True, truncation=True)  
    input_ids = tokenized_input['input_ids'].to(model.second_device)  
    attention_mask = tokenized_input['attention_mask'].to(model.second_device)  
    generated_texts = model.generate(input_ids=input_ids, attention_mask=attention_mask)  
    print(f"Generated texts (multi-sentence tokenized):")  
    for prompt, text in zip(input_prompts, generated_texts):  
        print(f"Prompt: {prompt}")  
        print(f"Generated: {text}\n")  
  
    print("Test 4: Training compatibility")  
    input_prompt = "The capital of France is"  
    expected_output = "Paris."  
    print(f"Input prompt: {input_prompt}")  
    print(f"Expected output: {expected_output}")  
    loss = model(input_prompt=input_prompt, expected_output=expected_output)  
    print(f"Training loss: {loss.item()}")  
  
    print("\nTest 5: Batch training compatibility")  
    input_prompts = ["The capital of France is", "The largest planet in our solar system is"]  
    expected_outputs = ["Paris.", "Jupiter."]  
    print(f"Input prompts: {input_prompts}")  
    print(f"Expected outputs: {expected_outputs}")  
    loss = model(input_prompt=input_prompts, expected_output=expected_outputs)  
    print(f"Batch training loss: {loss.item()}")  
  
    # Save the model  
    model.save_pretrained("dual_model_transformer.pth")  
  
    # Load the model  
    loaded_model = DualModelTransformer.from_pretrained(  
        "dual_model_transformer.pth",  
        large_model_name,  
        small_model_name,  
        stop_tokens,  
        small_model_dim,  
        large_model_dim  
    )  
  
    print("\nTest 6: Loaded model generation")  
    generated_text = loaded_model.generate(input_prompt="Once upon a time")  
    print(f"Generated text (loaded model): {generated_text}")  
