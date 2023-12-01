# rise-ner-task by Peter Samoaa

## Description
This repository contains a Transformer model trained for Named Entity Recognition (NER). The model is designed to identify and classify named entities in text into predefined categories.

## Task Description
The NER task involves labeling sequences of words in a text which are the names of things, such as person names, organizations, locations, expressions of times, quantities, monetary values, percentages, etc.

## Model
Details about the model architecture, training data, and training procedure.

## Results
- Summary of the model's performance.
- Charts or graphs (if applicable).

## Notebooks
- Training Notebook using HF: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hazemsamoaa/rise-ner-task/blob/main/notebooks/RISE-NER-Final-HF-Training.ipynb)
- Inference Notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hazemsamoaa/rise-ner-task/blob/main/notebooks/RISE-NER-Final-HF-Inference.ipynb)
- Additional notebooks or scripts used in the project.
    - Training Notebook using PT: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hazemsamoaa/rise-ner-task/blob/main/notebooks/RISE-NER-Final-Pyorch-Traininig.ipynb)
    - Inference Notebook using PT: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hazemsamoaa/rise-ner-task/blob/main/notebooks/RISE-NER-Final-Pytorch-Inference.ipynb)

## Research Paper
If this work is based on a research paper, include citation details here.

## How to Use
Instructions on how to use this model, including installation steps, code snippets, etc.

Step 1: Instal the requirements!

```bash
pip install transformers
pip install datasets
```

Step 2: Load the model.

```python
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

systems = {
    "torch_sys_a": "petersamoaa/rise-ner-distilbert-base-cased-system-a-v2",
    "torch_sys_b": "petersamoaa/rise-ner-distilbert-base-cased-system-b-v2",
    "hf_sys_a": "petersamoaa/rise-ner-distilbert-base-cased-system-a-v1",
    "hf_sys_b": "petersamoaa/rise-ner-distilbert-base-cased-system-b-v1",
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

system_name_or_path = systems["hf_sys_a"]
tokenizer = AutoTokenizer.from_pretrained(system_name_or_path)
model = AutoModelForTokenClassification.from_pretrained(system_name_or_path).to(device)
```

Step 3: Do the the inference.

```python

label_list = model.config.id2label
text = "Mr.Peter Samoaa lives in New York and works for the United Nations."

model.eval()
with torch.no_grad():
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    p_labels = [label_list[id] for id in predictions[0].cpu().numpy()]
    
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    for token, label in zip(tokens, p_labels):
        print(f"{token}: {label}")
```

```bash
>>> [CLS]: O
Mr: O
.: O
Peter: B-PER
Samoa: I-PER
##a: I-PER
lives: O
in: O
New: B-LOC
York: I-LOC
and: O
works: O
for: O
the: O
United: B-ORG
Nations: I-ORG
.: O
[SEP]: O
```


## Citation
If you use this model in your research, please cite it using the following format:

```bibtex
@misc{ner_task_samoaa_2023,
author = {Samoaa, Peter},
title = {NER Task Model},
year = {2023},
publisher = {Hugging Face Hub},
journal = {Hugging Face Model Hub},
howpublished = {\url{https://huggingface.co/petersamoaa}}
}
```

