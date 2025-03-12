from PIL import Image as Img
import os

import json
import torch
import argparse

from datasets import load_from_disk

from transformers.trainer_utils import set_seed
from src.eval.base_model import ClipSbertModel, MCLIPModel, SiglipModel, BaseOneModel

checkpoint_to_model_class = {
    "clip-ViT-B-32-multilingual-v1": ClipSbertModel,
    "XLM-Roberta-Large-Vit-B-16Plus": MCLIPModel,
    "siglip-base-patch16-256-multilingual": SiglipModel 
}

task_to_qry_field = {
    "i2t": {"qry_img_field": "img", "qry_text_field": None},
    "t2i": {"qry_img_field": None, "qry_text_field": "text"},
    "vqa": {"qry_img_field": "img", "qry_text_field": "question"},
    "vg": {"qry_img_field": "img", "qry_text_field": "text"},
    "c": {"qry_img_field": "img", "qry_text_field": None},
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

PLACEHOLDER_IMG_PATH = "http://images.cocodataset.org/train2017/000000514915.jpg"

def eval_instance(line_data, dataset_suffix, lang, qry_img_field, qry_text_field, tgt_id_field, task_type, embeds, mapping, model):

    img_path = "./eval_data"

    with torch.no_grad():

        if qry_img_field:

            if line_data[qry_img_field] is None:
                return None

            def embed_images(batch_images):
                if isinstance(model, BaseOneModel):
                    return model.get_embeddings(["placeholder"], batch_images)[1]
                else:
                    return model.get_img_embeddings(batch_images)

            split_ = len(line_data[qry_img_field].split('/')) > 1
            extension_= "." in line_data[qry_img_field]

            if split_ and extension_:
                img_ = os.path.join(img_path, line_data[qry_img_field])
            elif split_ and not extension_:
                img_ = os.path.join(img_path, line_data[qry_img_field] + ".jpg")
            elif not split_ and extension_:
                img_ = os.path.join(img_path, f"{dataset_suffix}_images", line_data[qry_img_field])
            else:
                img_ = os.path.join(img_path, f"{dataset_suffix}_images", line_data[qry_img_field] + ".jpg")

            qry_output = embed_images([Img.open(img_).convert("RGB")]).squeeze().to("cuda:0")

        else:

            if line_data[qry_text_field] is None:
                return None

            def embed_texts(batch_text):
                if isinstance(model, BaseOneModel):
                    return model.get_embeddings(batch_text, [Img.open(PLACEHOLDER_IMG_PATH)])[0]
                else:
                    return model.get_text_embeddings(batch_text)

            qry_output = embed_texts([line_data[qry_text_field]]).squeeze().to("cuda:0")

        img_embeddings = list(map(lambda x: embeds[mapping[x]], line_data[tgt_id_field]))
        img_embeddings = torch.stack(img_embeddings).to('cuda').to(torch.bfloat16)
        sim = torch.nn.functional.cosine_similarity(qry_output, img_embeddings)
    
    return sim


def main(checkpoint_path, dataset_suffix, task_type):

    set_seed(42)

    qry_img_field = task_to_qry_field[task_type]["qry_img_field"]
    qry_text_field = task_to_qry_field[task_type]["qry_text_field"]

    tgt_id_field = task_to_tgt_field[task_type]

    model_name = checkpoint_path.split('/')[-1]
    model = checkpoint_to_model_class[model_name]()

    for lang in ["en", "it", "es", "fr", "de"]:
        
        max_card = ds_max_card[dataset_suffix]
        dataset_name = f"{dataset_suffix}_{lang}_{str(max_card)}_formatted_{task_type}.jsonl"

        if not os.path.isfile(f"eval_data/{dataset_name}"):
            print(f"NO DATASET: eval_data/{dataset_name}")
            continue

        if os.path.isfile(f"results/{model_name}/{dataset_suffix}_{task_type}_{lang}.jsonl"):
            print(f"results/{model_name}/{dataset_suffix}_{task_type}_{lang}.jsonl already exists")
            continue
        else:
            print(f"COMPUTING: results/{model_name}/{dataset_suffix}_{task_type}_{lang}.jsonl")

        p_1 = 0

        emb_path = f"./ds_embed/{model_name}/{dataset_suffix}_{task_type}_{lang}"

        embeds = load_from_disk(emb_path).with_format("torch")
        ids = embeds["id"]
        embeds = embeds["embeddings"]
        mapping_idx = {x: i for i, x in enumerate(ids)}

        os.makedirs(f"results/{model_name}", exist_ok=True)

        with open(f"results/{model_name}/{dataset_suffix}_{task_type}_{lang}.jsonl", 'w', encoding='utf8') as f_out:
            with open(f"eval_data/{dataset_name}", 'r', encoding='utf8') as f:

                i = 0

                for l in f:

                    line_data = json.loads(l)

                    new_result = {}
                    new_result["label"] = line_data[tgt_id_field][0]

                    cos_sim = eval_instance(line_data, dataset_suffix, lang, qry_img_field, qry_text_field, tgt_id_field, task_type, embeds, mapping_idx, model)

                    if cos_sim is None:
                        continue
                    else:
                        cos_sim = cos_sim.squeeze()
                        
                    _, indices = torch.sort(cos_sim, -1, descending=True)

                    if 0 in indices[:1]:
                        new_result["p_1"] = 1
                        new_result["p_5"] = 1 / 5 
                        new_result["p_10"] = 1 / 10
                        new_result["r_1"] = 1
                        new_result["r_5"] = 1
                        new_result["r_10"] = 1
                    elif 0 in indices[:5]:
                        new_result["p_1"] = 0
                        new_result["p_5"] = 1 / 5
                        new_result["p_10"] = 1 / 10
                        new_result["r_1"] = 0
                        new_result["r_5"] = 1
                        new_result["r_10"] = 1
                    elif 0 in indices[:10]:
                        new_result["p_1"] = 0
                        new_result["p_5"] = 0
                        new_result["p_10"] = 1 / 10
                        new_result["r_1"] = 0
                        new_result["r_5"] = 0
                        new_result["r_10"] = 1
                    else:
                        new_result["p_1"] = 0
                        new_result["p_5"] = 0
                        new_result["p_10"] = 0
                        new_result["r_1"] = 0
                        new_result["r_5"] = 0
                        new_result["r_10"] = 0

                    i += 1
                    p_1 += new_result["p_1"]
                    print(p_1/i)

                    indices = [line_data[tgt_id_field][idx] for idx in indices]
                    new_result["predictions"] = indices

                    json.dump(new_result, f_out)
                    f_out.write('\n')

            print(f"{checkpoint_path} || ACCURACY FOR {dataset_name} = {str(p_1/i)}")


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