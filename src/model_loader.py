from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple
import torch


def load_model_and_tokenizer(
    model_name: str,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load the pre-trained model and the corresponding tokenizer.

    Args:
        model_name (str): The model name or local path

    Returns:
        Tuple[AutoModelForCausalLM, AutoTokenizer]: The model and tokenizer objects
    """
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_name)
    model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(model_name)

    # If MPS is available, move the model to MPS
    if torch.backends.mps.is_available():
        model = model.to("mps")
        print("模型已移動到 MPS 設備")
    else:
        print("MPS 不可用，使用 CPU")

    return model, tokenizer
