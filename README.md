# Hugging Face models

This repository is just a playground for Hugging Face models.
It is loosely based on the tutorials and examples in the official documentation.

The [timm](https://huggingface.co/docs/timm/index) (PyTorch image models)
and the [transformers](https://huggingface.co/docs/transformers/index)
library implement quite convenient APIs for using pretrained models.
Both of them allow for model loading, training, fine-tuning and inference.

In a number of notebooks, it is shown how to import a model and make predictions.
Beyond that, a dedicated demonstration of transfer learning is provided.
This example uses PyTorch and Lightning on top of the transformers library.

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

