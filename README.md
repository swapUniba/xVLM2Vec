# Code repository for "xVLM2Vec: Adapting LVLM-Based Embedding Models to Multilinguality Using Self-Knowledge Distillation"

This repository contains all steps that have been performed for xVLM2Vec.

All steps of the pipeline (translation, training and evaluation) have a singularity definition file. To build a container from this definition file do the following:

```
sudo singularity build {container_path}.sif {def_file_path}.def
```

Please note that we have also uploaded a copy of the [trl](https://github.com/huggingface/trl) and [vlm2vec](https://github.com/TIGER-AI-Lab/VLM2Vec) source code that we used, since we modified them. Refer to the original licenses of the respective repositories.

# Translation

Run the [translate_instruct.py](translate_instruct.py) script and then format the dataset using [format_dataset_distillation.py](format_dataset_distillation.py).

You can run the script with either Python or the Accelerate launcher. You can check the [original translation script](scripts/run_translate.sh) that was used for reference.

You should have the translated tasks from the original MMMEB dataset as output in the [data directory](data/) and a "parallel_shuffled.jsonl" file in the working directory.

The latter can be used to train the model.

# Training

First, merge the adapter of the [VLM2Vec-LoRA](https://huggingface.co/TIGER-Lab/VLM2Vec-LoRA) model with the base model. To do so, you can use the [merge_weights.py](merge_weights.py) script.

After that, you can use the [train_distillation_with_trainer.py](train_distillation_with_trainer.py) script to perform the Self-Knowledge Distillation training. The original model was obtained by using the Accelerate launcher with the provided [fsdp config](configs/fsdp_config.yaml) on 4 GPUs.

Change the value of the "use_image_loss" variable to apply the image loss ablation described in the paper.

In [requirements/requirements_train](requirements/requirements_train/) you can find the singularity definition files that we used to train the models.

# Evaluation

To evaluate CLIP or SIGLIP models, use the [base_model_process.py](base_model_process.py) and the [base_model_compute.py](base_model_compute.py) scripts. 

The process script extracts embeddings for the targets, so that the process is not repeated multiple times, while the compute script extracts the embedding for the query and compares it with the candidate embeddings.

The scripts accept 3 arguments that are: model checkpoint, dataset tag and task. 

Model checkpoint can be one of: "clip-ViT-B-32-multilingual-v1", "XLM-Roberta-Large-Vit-B-16Plus", "siglip-base-patch16-256-multilingual"

Dataset tag can be one of: xm, xtd, imagenet-1k-val, flickr30k_entities, maxm_v1

Task can be one of: i2t, t2i, vqa, vg, c

In [requirements/requirements_eval](requirements/requirements_eval/) you can find the singularity definition files that we used to evaluate each model.

To evaluate VLM2Vec models, there is a similar pipeline using the [vlm2vec_process.py](vlm2vec_process.py) and the [vlm2vec_compute.py](vlm2vec_compute.py) scripts.

The only thing that changes is that model checkpoint should now be a path to a model trained using the VLM2Vec or xVLM2Vec pipeline.

Final results will also be logged in the results directory.
