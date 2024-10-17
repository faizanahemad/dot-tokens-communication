import torch

class SamplingMixin:
    def _sampling(self, logits, sampling_method, temperature, top_k=50, top_p=0.95):
        # Ensure logits is 2D: [batch_size, vocab_size]
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        
        if sampling_method == "greedy":
            next_token = torch.argmax(logits, dim=-1)
        elif sampling_method == "sample":
            probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        elif sampling_method == "top_k":
            top_k = min(top_k, logits.size(-1))  # Safety check
            top_k_values, _ = torch.topk(logits, top_k, dim=-1)
            indices_to_remove = logits < top_k_values[..., -1, None]
            logits[indices_to_remove] = float('-inf')
            probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        elif sampling_method == "top_p":
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits / temperature, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Use scatter to set logits to -inf
            logits = logits.scatter(1, sorted_indices, sorted_logits)
            logits[sorted_indices_to_remove] = float('-inf')
            
            probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            raise ValueError("Invalid sampling method. Choose 'greedy', 'sample', 'top_k', or 'top_p'.")
        
        return next_token