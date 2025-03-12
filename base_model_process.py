import os

import json
import argparse

from PIL import Image as Img
from tqdm import tqdm
from datasets import Dataset, Image

from transformers.trainer_utils import set_seed
from src.eval.base_model import ClipSbertModel, MCLIPModel, SiglipModel, BaseOneModel

checkpoint_to_model_class = {
    "clip-ViT-B-32-multilingual-v1": ClipSbertModel,
    "XLM-Roberta-Large-Vit-B-16Plus": MCLIPModel,
    "siglip-base-patch16-256-multilingual": SiglipModel 
}

task_to_tgt_field = {
    "i2t": "captions",
    "t2i": "images",
    "vqa": "answers",
    "vg": "imgs",
    "c": "classes",
}

ds_max_card = {
    "xm": 1000,
    "xtd": 1000,
    "imagenet-1k-val": 1000,
    "flickr30k_entities": 1000,
    "maxm_v1": 100
}

c_lang_map = {
    "en": "image of {img_class}",
    "it": "immagine di {img_class}",
    "fr": "image de {img_class}",
    "de": "Bild von {img_class}",
    "es": "imagen de {img_class}"
}

PLACEHOLDER_IMG_PATH = "http://images.cocodataset.org/train2017/000000514915.jpg"

tgt_text_field_tasks = set(["i2t", "vqa", "c"])
tgt_img_field_tasks = set(["t2i", "vg"])


def main(checkpoint_path, dataset_suffix, task_type):

    set_seed(42)

    tgt_id_field = task_to_tgt_field[task_type]

    model_name = checkpoint_path.split('/')[-1]
    model = checkpoint_to_model_class[model_name]()

    img_path = "./eval_data"

    max_card = ds_max_card[dataset_suffix]

    for lang in ["en", "it", "es", "fr", "de"]:

        mapping = {}

        if not os.path.isfile(f"./eval_data/{dataset_suffix}_{lang}_{max_card}_formatted_{task_type}.jsonl"):
            print(f"NO DATASET: ./eval_data/{dataset_suffix}_{lang}_{max_card}_formatted_{task_type}.jsonl")
            continue

        with open(f"./eval_data/{dataset_suffix}_{lang}_{max_card}_formatted_{task_type}.jsonl", "r", encoding="utf8") as f:

            i = 0
            for l in f:

                line_data = json.loads(l)

                for x in line_data[tgt_id_field]:

                    if x not in mapping:
                        mapping[x] = i
                        i += 1

        unique_ids = list(mapping.keys())

        if task_type in tgt_img_field_tasks:

            split_ = len(unique_ids[0].split('/')) > 1
            extension_= "." in unique_ids[0]

            targets = []

            for x in unique_ids:

                if split_ and extension_:
                    target_to_add = os.path.join(img_path, x)
                elif split_ and not extension_:
                    target_to_add = os.path.join(img_path, x + ".jpg")
                elif not split_ and extension_:
                    target_to_add = os.path.join(img_path, f"{dataset_suffix}_images", x)
                else:
                    target_to_add = os.path.join(img_path, f"{dataset_suffix}_images", x + ".jpg")
                
                targets.append(target_to_add)

            ds = Dataset.from_dict({"id": unique_ids, "target": targets}).cast_column("target", Image())
        else:

            if task_type != "c":
                ds = Dataset.from_dict({"id": unique_ids, "target": unique_ids})
            else:
                lang_c_instr = c_lang_map[lang]
                ds = Dataset.from_dict({"id": unique_ids, "target": [lang_c_instr.format(img_class=x) for x in unique_ids]})

        def embed_texts(batch_text):
            if isinstance(model, BaseOneModel):
                return model.get_embeddings(batch_text, [Img.open(PLACEHOLDER_IMG_PATH)])[0].to("cpu")
            else:
                return model.get_text_embeddings(batch_text).to("cpu")

        def embed_images(batch_images):
            if isinstance(model, BaseOneModel):
                return model.get_embeddings(["placeholder"], batch_images)[1].to("cpu")
            else:
                return model.get_img_embeddings(batch_images).to("cpu")

        batch = []
        embs_list = []

        for x in tqdm(ds):

            batch.append(x["target"])

            if len(batch) == 64:
                if task_type in tgt_img_field_tasks:
                    embeds = embed_images(batch)
                else:
                    embeds = embed_texts(batch)

                for emb in embeds:
                    embs_list.append(emb.tolist())
                batch = []

        if len(batch) != 0:
            if task_type in tgt_img_field_tasks:
                embeds = embed_images(batch)
            else:
                embeds = embed_texts(batch)

            for emb in embeds:
                embs_list.append(emb.tolist())
            batch = []

            ds = ds.add_column("embeddings", embs_list)
            ds = ds.with_format("torch")
            ds.save_to_disk(f'ds_embed/{model_name}/{dataset_suffix}_{task_type}_{lang}')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint_path')
    parser.add_argument('-d', '--dataset_suffix')
    parser.add_argument('-t', '--task_type')
    args = parser.parse_args()
    
    checkpoint_path = args.checkpoint_path
    dataset_suffix = args.dataset_suffix
    task_type = args.task_type

    main(checkpoint_path, dataset_suffix, task_type)
    