import torch
from ai_edge_torch.generative.layers import kv_cache as kv_utils
from qwen2 import Qwen2Model,build_0_5b_model
from transformers import AutoTokenizer

def tokenize_input(question: str, tokenizer: AutoTokenizer) -> torch.Tensor:
    """
    Converts the input text into tokens using the Hugging Face AutoTokenizer.

    Args:
        text: The input text string.
        tokenizer: A tokenizer object from Hugging Face.

    Returns:
        tokens: A tensor of token ids.
    """
    tokenized_input = tokenizer.encode(question, return_tensors="pt")
    return tokenized_input

def detokenize_output(token_ids: torch.Tensor, tokenizer: AutoTokenizer) -> str:
    """
    Converts token ids back to a text string using the Hugging Face AutoTokenizer.

    Args:
        token_ids: A tensor of token ids (1, seq_len).pt
        tokenizer: A tokenizer object from Hugging Face.

    Returns:
        text: The decoded text string.
    """
    decoded_text = tokenizer.decode(token_ids.squeeze(), skip_special_tokens=True)
    return decoded_text

def generate_text(model: Qwen2Model, input_text: str, tokenizer: AutoTokenizer, max_new_tokens: int = 50, kv_cache_max_len: int = 1024, temperature: float = 0.7, top_k: int = 50, top_p: float = 0.9):
    """
    Autoregressive text generation based on input text with temperature, top-k, and top-p sampling.

    Args:
        model: The Qwen2Model object for inference.
        input_text: The initial input prompt text.
        tokenizer: A tokenizer object to convert text to tokens.
        max_new_tokens: Maximum number of new tokens to generate.
        kv_cache_max_len: The maximum cache length for KV caching (default: 1024).
        temperature: Sampling temperature to control randomness.
        top_k: Top-k sampling, consider only the top k tokens.
        top_p: Top-p sampling, consider tokens within cumulative probability p.

    Returns:
        generated_text: The generated text output.
    """
    input_tokens = tokenize_input(input_text, tokenizer)
    input_pos = torch.arange(0, input_tokens.size(1)) 
    kv_cache = kv_utils.KVCache([None] * len(model.transformer_blocks))
    generated_tokens = input_tokens

    for _ in range(max_new_tokens):
        with torch.no_grad():  
            output = model.forward(generated_tokens, input_pos, kv_cache)
        logits = output["logits"][:, -1, :] / temperature
        # Apply top-k filtering
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            top_k_values, top_k_indices = torch.topk(logits, top_k)
            logits[logits < top_k_values[..., -1, None]] = -float('Inf')
        # Apply top-p (nucleus) sampling
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens with cumulative probability above top_p
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0  
        # Mask logits that are not in top-k/top-p
        for i in range(sorted_logits.size(0)):
            logits[i, sorted_indices[i][sorted_indices_to_remove[i]]] = -float('Inf')
        # Sample from the filtered distribution
        probabilities = torch.softmax(logits, dim=-1)
        next_token_id = torch.multinomial(probabilities, 1)
        # Append the predicted token to the generated tokens
        generated_tokens = torch.cat([generated_tokens, next_token_id], dim=1)
        # Update input_pos to match new sequence length
        input_pos = torch.arange(0, generated_tokens.size(1))
        # Stop if we encounter an end-of-sequence token (e.g., <eos>)
        if next_token_id.item() == tokenizer.eos_token_id:
            break
    generated_text = detokenize_output(generated_tokens, tokenizer)
    return generated_text

if __name__ == "__main__":
    checkpoint_path = "Downloads/models/qwen2"
    kv_cache_max_len = 1024

    model = build_0_5b_model(checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
    question = "What is the meaning of life?"
    output_text = generate_text(model, question, tokenizer, max_new_tokens=100, temperature=0.7, top_k=100, top_p=0.9)

    print("Input Text:", question)
    print("Generated Output Text:", output_text)

