from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Dict
import torch
from torch import Tensor


def run_inference(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    input_text: str,
    max_length: int = 50,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.95,
) -> str:
    """
    Use the model for text generation inference.
    Args:
        model (PreTrainedModel): The loaded PyTorch model
        tokenizer (PreTrainedTokenizer): The tokenizer
        input_text (str): The input text
        max_length (int): The maximum number of tokens to generate, default is 50
        temperature (float): Controls the randomness of generation, default is 0.7
        top_k (int): Top-K sampling parameter, default is 50
        top_p (float): Top-P sampling parameter, default is 0.95
    Returns:
        str: The generated text
    Raises:
        RuntimeError: If text generation fails
        ValueError: If input parameters are invalid
    """
    # Input validation
    if not input_text:
        raise ValueError("Input text cannot be empty")
    if max_length < 1:
        raise ValueError("max_length must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be between 0 and 1")

    # Get model's device
    device = next(model.parameters()).device

    # Encode input text
    try:
        inputs: Dict[str, Tensor] = tokenizer(input_text, return_tensors="pt")
        input_ids: Tensor = inputs["input_ids"].to(device)
    except Exception as e:
        raise RuntimeError(f"Tokenization failed: {str(e)}")

    # Ensure pad token is set if needed
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Generate text
    try:
        with torch.no_grad():
            outputs: Tensor = model.generate(
                input_ids,
                max_length=max_length,
                num_return_sequences=1,
                do_sample=True,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id,
                # Add additional safety parameters
                repetition_penalty=1.0,
                length_penalty=1.0,
                early_stopping=True,
            )
    except Exception as e:
        raise RuntimeError(f"Generation failed: {str(e)}")

    # Decode the generated tokens to text
    try:
        generated_text: str = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    except Exception as e:
        raise RuntimeError(f"Decoding failed: {str(e)}")
