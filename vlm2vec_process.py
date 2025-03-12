from src.mmeb_src.model import MMEBModel
from src.mmeb_src.arguments import ModelArguments
import os

import json
import torch
import argparse

from tqdm import tqdm
from datasets import Dataset, Image, set_caching_enabled
from transformers import AutoProcessor

i2t_target = {
    "en": "{tgt_field}",
    "it": "{tgt_field}",
    "es": "{tgt_field}",
    "fr": "{tgt_field}",
    "de": "{tgt_field}"
}

t2i_target = {
    "en": "<|image_1|>\nRepresent the given image",
    "fr": "<|image_1|>\nReprésentez l'image donnée",
    "de": "<|image_1|>\nStelle das gegebene Bild dar",
    "it": "<|image_1|>\nRappresenta l'immagine data",
    "es": "<|image_1|>\nRepresenta la imagen dada"
}

vqa_target = {
    "en": "{tgt_field}",
    "it": "{tgt_field}",
    "es": "{tgt_field}",
    "fr": "{tgt_field}",
    "de": "{tgt_field}"
}

vg_target = {
    "en": "<|image_1|>\nRepresent the given cropped image of the object",
    "it": "",
    "es": "",
    "fr": "<|image_1|>\nReprésentez l'image recadrée donnée de l'objet",
    "de": ""
}

c_target = {
    "en": "{tgt_field}",
    "it": "{tgt_field}",
    "es": "{tgt_field}",
    "fr": "{tgt_field}",
    "de": "{tgt_field}"
}

task_to_tgt_field = {
    "i2t": "captions",
    "t2i": "images",
    "vqa": "answers",
    "vg": "imgs",
    "c": "classes",
}

task_target = {
    "i2t": i2t_target,
    "t2i": t2i_target,
    "vqa": vqa_target,
    "vg": vg_target,
    "c": c_target
}

ds_max_card = {
    "xm": 1000,
    "xtd": 1000,
    "imagenet-1k-val": 1000,
    "flickr30k_entities": 1000,
    "maxm_v1": 100
}

tgt_text_field_tasks = set(["i2t", "vqa", "c"])
tgt_img_field_tasks = set(["t2i", "vg"])

def process_inputs(input_ids, pixel_values, image_sizes, device):

    inputs = {}

    input_ids = [x.to(device) for x in input_ids]

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=32000
    )

    inputs["input_ids"] = input_ids.to(device)
    attention_mask = []

    attention_mask = (input_ids != 32000)
    inputs["attention_mask"] = attention_mask.to(device)
    
    if pixel_values[0] is not None:
        inputs["pixel_values"] = torch.stack(pixel_values).to(device)
        inputs["image_sizes"] = torch.stack(image_sizes).to(device)

    return inputs


def main(checkpoint_path, dataset_suffix, task_type):

    set_caching_enabled(False)
    tgt_id_field = task_to_tgt_field[task_type]

    model_path_ds = checkpoint_path.split('/')[-1]

    model_args = ModelArguments(
        model_name='microsoft/Phi-3.5-vision-instruct',
        checkpoint_path=checkpoint_path,
        pooling='last',
        normalize=True,
        lora=False,
    )

    model = MMEBModel.load(model_args)
    model.eval()
    model = model.to('cuda', dtype=torch.bfloat16)

    img_path = "./eval_data"

    processor = AutoProcessor.from_pretrained(
        'microsoft/Phi-3.5-vision-instruct',
        trust_remote_code=True,
        num_crops=4,
    )

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

        for formatting_style in ["plain", "punct", "newline"]:

            if os.path.isdir(f'./ds_embed/{model_path_ds}/{dataset_suffix}_{task_type}_{lang}_{formatting_style}'):
                print(f'ds_embed/{model_path_ds}/{dataset_suffix}_{task_type}_{lang}_{formatting_style} aready exists')
                continue
            else:
                print(f'NOW PROCESSING: ds_embed/{model_path_ds}/{dataset_suffix}_{task_type}_{lang}_{formatting_style}')


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
                ds = Dataset.from_dict({"id": unique_ids, "target": unique_ids})

            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            def process(line_data):

                tgt_text = task_target[task_type][lang]
                    
                last_char = "."

                if formatting_style == "punct":

                    if tgt_text[-1] != last_char:
                        tgt_text = tgt_text.strip() + last_char
                    else:
                        tgt_text = tgt_text

                elif formatting_style == "plain":
                    
                    tgt_text = tgt_text.strip()

                    if tgt_text[-1] == "." or tgt_text[-1] == "?":
                        tgt_text = tgt_text[:-1]
                    else:
                        tgt_text = tgt_text
                
                elif formatting_style == "newline":
                    
                    if task_type in tgt_img_field_tasks:
                        tgt_text = tgt_text.strip()
                        tgt_text += last_char + "\n"
                    else:
                        if tgt_text[-1] != last_char:
                            tgt_text = tgt_text.strip() + last_char
                        else:
                            tgt_text = tgt_text

                if task_type in tgt_text_field_tasks:
                    inputs = processor(tgt_text.format(tgt_field=line_data["target"]), return_tensors="pt")
                    return {"input_ids": inputs.input_ids[0], "pixel_values": None, "image_sizes": None}
                else:
                    inputs = processor(tgt_text, [line_data["target"]], return_tensors="pt")
                    return {"input_ids": inputs.input_ids[0], "pixel_values": inputs.pixel_values[0], "image_sizes": inputs.image_sizes[0]}

            def embedd(batch):
                inputs = process_inputs(batch["input_ids"], batch["pixel_values"], batch["image_sizes"], device)
                with torch.no_grad():
                    tgt_output = model(tgt=inputs, text_only=False)["tgt_reps"].to("cpu")
                return tgt_output

            batch = {"input_ids": [], "pixel_values": [], "image_sizes": []}
            i = 0

            embs_list = []

            for x in tqdm(ds):

                inputs = process(x)

                for key, value in inputs.items():
                    batch[key].append(value)
                
                i += 1

                if i == 4:
                    embeds = embedd(batch)
                    for emb in embeds:
                        embs_list.append(emb.tolist())
                    batch = {"input_ids": [], "pixel_values": [], "image_sizes": []}
                    i = 0

            if i != 0:
                embeds = embedd(batch)
                for emb in embeds:
                    embs_list.append(emb.tolist())
                batch = {"input_ids": [], "pixel_values": [], "image_sizes": []}
                i = 0

            ds = ds.add_column("embeddings", embs_list)
            ds = ds.with_format("torch")
            ds.save_to_disk(f'ds_embed/{model_path_ds}/{dataset_suffix}_{task_type}_{lang}_{formatting_style}')


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
    