import yaml
from typing import Dict, Any
from model_loader import load_model_and_tokenizer


def load_config(config_path: str) -> Dict[str, Any]:
    """Load the configuration from the YAML file."""
    with open(config_path, "r", encoding="utf-8") as file:
        config: Dict[str, Any] = yaml.safe_load(file)
    return config


def main() -> None:
    """Download and save the model."""
    # Load the configuration file
    config: Dict[str, Any] = load_config("config/config.yaml")

    # Extract the model name (use local path if available)
    model_name: str = config["model"]["local_path"] or config["model"]["name"]

    # Load the model and tokenizer, this will trigger the download (if the model is not local)
    print(f"Loading or downloading model: {model_name}")
    model, tokenizer = load_model_and_tokenizer(model_name)
    print("Model loaded!")
    return model, tokenizer


if __name__ == "__main__":
    main()
