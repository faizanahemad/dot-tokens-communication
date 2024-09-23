import torch  
import torch.nn as nn  
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM  
from typing import List, Optional, Union  
from accelerate import Accelerator

  
class DualModelTransformer(nn.Module):  
    def __init__(  
        self,  
        large_model_name: str,  
        small_model_name: str,  
        stop_tokens: List[str],  
        small_model_dim: int,  
        large_model_dim: int  
    ):  
        super().__init__()  
        # Initialize models  
        self.large_model = AutoModelForCausalLM.from_pretrained(large_model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)  
        self.small_model = AutoModelForCausalLM.from_pretrained(small_model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)  

        # set no_grad to both models and their parameters
        self.large_model.eval()
        self.small_model.eval()
        for param in self.large_model.parameters():
            param.requires_grad = False
        for param in self.small_model.parameters():
            param.requires_grad = False
  
        # Initialize tokenizers  
        self.large_tokenizer = AutoTokenizer.from_pretrained(large_model_name)  
        self.small_tokenizer = AutoTokenizer.from_pretrained(small_model_name)  
  
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
        # # transfer these to the GPU 0
        # self.ffn_small_to_large = self.ffn_small_to_large.to(torch.device("cuda:0"))
        # self.ffn_large_to_small = self.ffn_large_to_small.to(torch.device("cuda:0"))
        # Initialize FFN parameters  
        self._init_weights(self.ffn_small_to_large)  
        self._init_weights(self.ffn_large_to_small)  
  
        self.stop_tokens = stop_tokens  
        self.small_model_dim = small_model_dim  
        self.large_model_dim = large_model_dim  
  
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

    def _get_all_hidden_states(self, model_output):
        if hasattr(model_output, 'hidden_states') and model_output.hidden_states is not None:
            return model_output.hidden_states
        else:
            raise AttributeError(f"Unable to find hidden states in model output: {type(model_output).__name__}")
  
    def _get_last_hidden_state(self, model_output):  
        if hasattr(model_output, 'hidden_states') and model_output.hidden_states is not None:  
            return model_output.hidden_states[-1]  + model_output.hidden_states[-2]  + model_output.hidden_states[-3]  + model_output.hidden_states[-4]  
        # elif hasattr(model_output, 'last_hidden_state'):  
        #     return model_output.last_hidden_state  
        # elif isinstance(model_output, torch.Tensor):  
        #     return model_output  
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
        input_prompt: str,  
        max_length: int = 100,  
        temperature: float = 1.0,  
        sampling_method: str = "greedy"  
    ) -> str:  
        input_ids = self.small_tokenizer.encode(input_prompt, return_tensors="pt").to(self.small_model.device)  
        generated_text = input_prompt  
        embedding_layer = self._get_embedding_layer(self.small_model)  
  
        for _ in range(max_length):  
            with torch.no_grad():  
                small_output = self.small_model(input_ids, output_hidden_states=True)  
                small_last_hidden = self._get_last_hidden_state(small_output)[:, -1, :]  
                large_input = self.ffn_small_to_large(small_last_hidden).unsqueeze(1)  
                large_output = self.large_model(inputs_embeds=large_input, output_hidden_states=True)  
                large_last_hidden = self._get_last_hidden_state(large_output)[:, -1, :]  
                knowledge_vector = self.ffn_large_to_small(large_last_hidden)  
                input_embeds = embedding_layer(input_ids)  
                combined_input = torch.cat([input_embeds, knowledge_vector.unsqueeze(1)], dim=1)  
                model_output = self.small_model(inputs_embeds=combined_input)  
                logits = self._get_logits(model_output)[:, -1, :]  
  
            if sampling_method == "greedy":  
                next_token = torch.argmax(logits, dim=-1)  
            elif sampling_method == "sample":  
                probs = nn.functional.softmax(logits / temperature, dim=-1)  
                next_token = torch.multinomial(probs, num_samples=1)  
            else:  
                raise ValueError("Invalid sampling method")  
  
            next_token_str = self.small_tokenizer.decode(next_token)  
            generated_text += next_token_str  
  
            if any(stop_token in generated_text for stop_token in self.stop_tokens):  
                break  
  
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)  
  
        return generated_text  
  
    def calculate_loss(self, logits: torch.Tensor, expected_output: str) -> torch.Tensor:  
        expected_ids = self.small_tokenizer.encode(expected_output, return_tensors="pt").to(logits.device)  
        loss_fct = nn.CrossEntropyLoss()  
        return loss_fct(logits.view(-1, logits.size(-1)), expected_ids.view(-1))  
  
    def forward(  
        self,  
        input_prompt: Union[str, torch.Tensor],  
        expected_output: Optional[str] = None,  
        temperature: float = 1.0,  
        sampling_method: str = "greedy"  
    ) -> Union[str, torch.Tensor]:  
        if isinstance(input_prompt, str):  
            input_ids = self.small_tokenizer.encode(input_prompt, return_tensors="pt").to(self.small_model.device)  
        else:  
            input_ids = input_prompt  
  
        if expected_output is None:  
            return self.generate_text(self.small_tokenizer.decode(input_ids[0]), temperature=temperature, sampling_method=sampling_method)  
  
        expected_ids = self.small_tokenizer.encode(expected_output, return_tensors="pt").to(self.small_model.device)  
        embedding_layer = self._get_embedding_layer(self.small_model)  
        logits = []  
  
        for i in range(expected_ids.size(1)):  
            with torch.no_grad():  
                small_output = self.small_model(input_ids, output_hidden_states=True)  
                small_last_hidden = self._get_last_hidden_state(small_output)[:, -1, :]  
                large_input = self.ffn_small_to_large(small_last_hidden).unsqueeze(1)  
                large_output = self.large_model(inputs_embeds=large_input, output_hidden_states=True)  
                large_last_hidden = self._get_last_hidden_state(large_output)[:, -1, :]  
                knowledge_vector = self.ffn_large_to_small(large_last_hidden)  
  
            input_embeds = embedding_layer(input_ids)  
            combined_input = torch.cat([input_embeds, knowledge_vector.unsqueeze(1)], dim=1)  
            model_output = self.small_model(inputs_embeds=combined_input)  
            current_logits = self._get_logits(model_output)[:, -1, :]  
            logits.append(current_logits)  
  
            input_ids = torch.cat([input_ids, expected_ids[:, i].unsqueeze(0)], dim=-1)  
  
        logits = torch.stack(logits, dim=1)  
        loss = self.calculate_loss(logits, expected_output)  
        return loss  
  
    def save_pretrained(self, path: str):  
        torch.save({  
            'ffn_small_to_large': self.ffn_small_to_large.state_dict(),  
            'ffn_large_to_small': self.ffn_large_to_small.state_dict(),  
        }, path)  
  
    @classmethod  
    def from_pretrained(  
        cls,  
        path: str,  
        large_model_name: str,  
        small_model_name: str,  
        stop_tokens: List[str],  
        small_model_dim: int,  
        large_model_dim: int  
    ):  
        model = cls(large_model_name, small_model_name, stop_tokens, small_model_dim, large_model_dim)  
        state_dict = torch.load(path)  
        model.ffn_small_to_large.load_state_dict(state_dict['ffn_small_to_large'])  
        model.ffn_large_to_small.load_state_dict(state_dict['ffn_large_to_small'])  
        return model  
  
if __name__ == "__main__":  
    # Example usage  
    large_model_name = "microsoft/Phi-3-medium-4k-instruct" # "microsoft/Phi-3-small-8k-instruct"  
    small_model_name = "microsoft/Phi-3-mini-4k-instruct"  
    stop_tokens = [".", "!", "?"]  
    small_model_dim = 3072  # GPT-2 small hidden size  = 768
    large_model_dim = 5120  # GPT-2 large hidden size  = 1280
    accelerator = Accelerator()
    model = DualModelTransformer(  
        large_model_name,  
        small_model_name,  
        stop_tokens,  
        small_model_dim,  
        large_model_dim  
    )  
    model = accelerator.prepare(model)
    # Move model to GPU if available  
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    # model = model.to(device)  
  
    # Test text generation  
    input_prompt = "Once upon a time"  
    generated_text = model(input_prompt)  
    print(f"Generated text: {generated_text}")  
  
    # Test loss calculation  
    input_prompt = "The quick brown fox"  
    expected_output = "jumps over the lazy dog."  
    loss = model(input_prompt, expected_output)  
    print(f"Loss: {loss.item()}")  
  
    # Save the model  
    model.save_pretrained("dual_model_transformer.pth")  
  
    
  
    # Test the loaded model  
    generated_text = model(input_prompt)  
    print(f"Generated text (loaded model): {generated_text}")  
