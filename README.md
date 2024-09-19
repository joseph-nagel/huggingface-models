# Hugging Face models

This repository is just a playground for models provided by Hugging Face.
It is largely based on the tutorials and examples in the official documentation.
The [timm](https://huggingface.co/docs/timm/index) (PyTorch image models) and the [transformers](https://huggingface.co/docs/transformers/index) library provide large number of pretrained models.
Both of them implement convenient APIs for model loading, training, fine-tuning and inferencing.

## Notebooks

- [timm](notebooks/timm.ipynb)

- [transformers: Image classification](notebooks/transformers_classif.ipynb)

- [transformers: Object detection](notebooks/transformers_detect.ipynb)

- [transformers: Semantic segmentation](notebooks/transformers_segment.ipynb)

- [transformers: Transfer learning](notebooks/transformers_transfer.ipynb)

## Installation

```
pip install -e .
```

## Training

```
python scripts/main.py fit --config config/transfer.yaml
```

