# A llama inference sandbox (r1-1776-distill-llama-70b)

A simple script to run inference with r1-1776-distill-llama-70b.

## Prerequisites

- >= Python 3.9

## Installation

```bash
pip install -r requirements.txt
cp config/config.yaml.example config/config.yaml
```

## Usage

### Download the model

```bash
python src/download.py
```

### Run the inference

```bash
python src/infer.py --i "台灣，現在不是敏感詞了吧？"
```
