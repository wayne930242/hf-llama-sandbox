from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict
import torch


def run_inference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_text: str,
    max_length: int = 50,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.95,
) -> str:
    """
    Use the model for text generation inference.

    Args:
        model (AutoModelForCausalLM): The loaded PyTorch model
        tokenizer (AutoTokenizer): The tokenizer
        input_text (str): The input text
        max_length (int): The maximum number of tokens to generate, default is 50
        temperature (float): Controls the randomness of generation, default is 0.7
        top_k (int): Top-K sampling parameter, default is 50
        top_p (float): Top-P sampling parameter, default is 0.95

    Returns:
        str: The generated text
    """
    inputs: Dict[str, torch.Tensor] = tokenizer(input_text, return_tensors="pt")
    input_ids: torch.Tensor = inputs["input_ids"]

    # If the model is on MPS, ensure the input is also on MPS
    if torch.backends.mps.is_available():
        input_ids = input_ids.to("mps")

    # Use the model to generate output
    with torch.no_grad():
        outputs: torch.Tensor = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )

    # Decode the generated tokens to text
    generated_text: str = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text
