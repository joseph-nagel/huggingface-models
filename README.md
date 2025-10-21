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


### Computer vision models

- [transformers: Image classification](notebooks/transformers_cv_classif.ipynb)
- [transformers: Object detection](notebooks/transformers_cv_detect.ipynb)
- [transformers: Finetuning](notebooks/transformers_cv_finetune.ipynb)
- [transformers: PEFT with LoRA](notebooks/transformers_cv_lora.ipynb)
- [transformers: Semantic segmentation](notebooks/transformers_cv_segment.ipynb)


### Language models

- [transformers: Bidirectional BERT-like LMs](notebooks/transformers_lm_bert.ipynb)
- [transformers: Finetuning BERT-like LMs](notebooks/transformers_lm_bert_finetune.ipynb)
- [transformers: Autoregressive GPT-like LMs](notebooks/transformers_lm_gpt.ipynb)
- [transformers: Finetuning GPT-like LMs](notebooks/transformers_lm_gpt_finetune.ipynb)


### Vision language models

- [transformers: CLIP](notebooks/transformers_vlm_clip.ipynb)
- [transformers: CLIPSeg for semantic segmentation](notebooks/transformers_vlm_clipseg.ipynb)
- [transformers: OWL-ViT for object detection](notebooks/transformers_vlm_owlvit.ipynb)


### Other things

- [diffusers](notebooks/diffusers.ipynb)
- [timm](notebooks/timm.ipynb)
- [Learning rate schedules](notebooks/lr_schedules.ipynb)


## Installation

```bash
pip install -e .
```


## Training

```bash
python scripts/main.py fit --config config/finetune.yaml
```

```bash
python scripts/main.py fit --config config/finetune.yaml \
  --model.init_args.model_name="microsoft/resnet-18"
```

```bash
python scripts/main.py fit --config config/lora.yaml \
  --model.init_args.model_name="facebook/dinov2-small-imagenet1k-1-layer"
```
