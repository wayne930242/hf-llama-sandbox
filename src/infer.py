import yaml
import argparse
from typing import Dict, Any
from model_loader import load_model_and_tokenizer
from inference import run_inference


def load_config(config_path: str) -> Dict[str, Any]:
    """Load the configuration from the YAML file."""
    with open(config_path, "r", encoding="utf-8") as file:
        config: Dict[str, Any] = yaml.safe_load(file)
    return config


def parse_args() -> str:
    parser = argparse.ArgumentParser(description="執行模型推理")
    parser.add_argument("--i", type=str, required=True, help="輸入文字")
    args = parser.parse_args()
    return args.i


def main() -> None:
    """Execute the inference."""
    # Load the configuration file
    config: Dict[str, Any] = load_config("config/config.yaml")

    # Extract the model name (use local path if available)
    model_name: str = config["model"]["local_path"] or config["model"]["name"]

    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Get the input text
    input_text: str = parse_args()

    # Extract the inference parameters
    max_length: int = config["inference"]["max_length"]
    temperature: float = config["inference"]["temperature"]
    top_k: int = config["inference"]["top_k"]
    top_p: float = config["inference"]["top_p"]

    # Execute the inference
    generated_text: str = run_inference(
        model,
        tokenizer,
        input_text,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    print(f"輸入: {input_text}")
    print(f"生成結果: {generated_text}")


if __name__ == "__main__":
    main()