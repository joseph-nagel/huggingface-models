# Hugging Face models

This repository is just a playground for Hugging Face models.
It is loosely based on the tutorials and examples in the official documentation.

The [timm](https://huggingface.co/docs/timm/index) (PyTorch image models)
and the [transformers](https://huggingface.co/docs/transformers/index)
library implement quite convenient APIs for using pretrained models.
Both of them allow for model loading, training, fine-tuning and inference.
Similarly, [diffusers](https://huggingface.co/docs/diffusers/index)
makes a number of pretrained generative diffusion models easily accessible.

In a number of notebooks, it is shown how to import a model and make predictions.
Beyond that, a dedicated demonstration of transfer learning is provided.
This example uses PyTorch and Lightning on top of the transformers library.


## Notebooks

- [diffusers](notebooks/diffusers.ipynb)

- [timm](notebooks/timm.ipynb)

- [transformers: Image classification](notebooks/transformers_cv_classif.ipynb)

- [transformers: Object detection](notebooks/transformers_cv_detect.ipynb)

- [transformers: Semantic segmentation](notebooks/transformers_cv_segment.ipynb)

- [transformers: Transfer learning](notebooks/transformers_cv_transfer.ipynb)

- [transformers: Autoregressive LMs](notebooks/transformers_lm_autoreg.ipynb)

- [transformers: Bidirectional LMs](notebooks/transformers_lm_bidirect.ipynb)


## Installation

```
pip install -e .
```


## Training

```
python scripts/main.py fit --config config/transfer.yaml
```

