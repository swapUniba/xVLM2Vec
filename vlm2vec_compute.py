from src.mmeb_src.model import MMEBModel
from src.mmeb_src.arguments import ModelArguments
from PIL import Image
import os

import json
import torch
import argparse

from datasets import load_from_disk
from transformers import AutoProcessor

i2t_instructions = {
    "en": "<|image_1|>\nFind an image caption describing the given everyday image",
    "it": "<|image_1|>\nTrova una didascalia che descriva l'immagine di tutti i giorni",
    "es": "<|image_1|>\nEncuentra una leyenda que describa la imagen cotidiana dada",
    "fr": "<|image_1|>\nTrouvez une légende décrivant l'image donnée",
    "de": "<|image_1|>\nFinde eine Bildunterschrift, die das gegebene Alltagsbild beschreibt"
}

t2i_instructions = {
    "en": "Find me an everyday image that matches the given caption: {qry_field}",
    "it": "Trovami un'immagine di tutti i giorni che corrisponda alla didascalia data: {qry_field}",
    "es": "Encuentra una imagen cotidiana que coincida con la leyenda dada: {qry_field}",
    "fr": "Trouvez-moi une image de tous les jours qui correspond à la légende donnée: {qry_field}",
    "de": "Finde mir ein alltägliches Bild, das der gegebenen Beschriftung entspricht: {qry_field}"
}

vqa_instructions = {
    "en": "<|image_1|>\nRepresent the given image with the following question: {qry_field}",
    "it": "<|image_1|>\nRappresenta l'immagine data con la seguente domanda: {qry_field}",
    "es": "<|image_1|>\nRepresenta la imagen dada con la siguiente pregunta: {qry_field}",
    "fr": "<|image_1|>\nReprésentez l'image donnée avec la question suivante: {qry_field}",
    "de": "<|image_1|>\nStellen Sie das gegebene Bild mit der folgenden Frage dar: {qry_field}"
}

vg_instructions = {
    "en": "<|image_1|>\nSelect the portion of the image that isolates the object labeled as \"{qry_field}\"",
    "it": "",
    "es": "",
    "fr": "<|image_1|>\nSélectionnez la partie de l'image qui isole l'objet étiqueté comme \"{qry_field}\"",
    "de": ""
}

c_instructions = {
    "en": "<|image_1|>\nRepresent the given image for classification",
    "it": "<|image_1|>\nRappresenta l'immagine data per la classificazione",
    "es": "<|image_1|>\nRepresenta la imagen dada para clasificación",
    "fr": "<|image_1|>\nReprésentez l'image donnée pour la classification",
    "de": "<|image_1|>\nStellen Sie das gegebene Bild für die Klassifizierung dar"
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

task_instructions = {
    "i2t": i2t_instructions,
    "t2i": t2i_instructions,
    "vqa": vqa_instructions,
    "vg": vg_instructions,
    "c": c_instructions
}

ds_max_card = {
    "xm": 1000,
    "xtd": 1000,
    "imagenet-1k-val": 1000,
    "flickr30k_entities": 1000,
    "maxm_v1": 100
}

def eval_instance(line_data, dataset_suffix, lang, qry_img_field, qry_text_field, tgt_id_field, formatting_style, task_type, embeds, mapping, model, processor):

    img_path = "./eval_data"

    with torch.no_grad():

        if qry_text_field and line_data[qry_text_field] is None:
            return None

        qry_text = task_instructions[task_type][lang]

        if qry_text_field:
            qry_text = qry_text.format(qry_field=line_data[qry_text_field])
            
        if task_type != "vqa":
            last_char = "."
        else:
            last_char = "?"

        if formatting_style == "punct":

            if qry_text[-1] != "." and qry_text[-1] != "?":
                qry_text = qry_text.strip() + last_char
            else:
                qry_text = qry_text

        elif formatting_style == "plain":
                    
            qry_text = qry_text.strip()

            if qry_text[-1] == "." or qry_text[-1] == "?":
                qry_text = qry_text[:-1]
            else:
                qry_text = qry_text

        elif formatting_style == "newline":

            if qry_text[-1] != "." and qry_text[-1] != "?":
                qry_text = qry_text.strip() + last_char
            else:
                qry_text = qry_text
            
            qry_text += "\n"

        if qry_img_field:

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

            inputs = processor(qry_text, [Image.open(img_)])
        else:
            inputs = processor(qry_text)

        inputs = {key: value.to('cuda') for key, value in inputs.items()}
        qry_output = model(qry=inputs, text_only=False)["qry_reps"].to(torch.bfloat16)

        img_embeddings = list(map(lambda x: embeds[mapping[x]], line_data[tgt_id_field]))
        img_embeddings = torch.stack(img_embeddings).to('cuda').to(torch.bfloat16)
        sim = model.compute_similarity(qry_output, img_embeddings)
    
    return sim


def main(checkpoint_path, dataset_suffix, task_type):

    qry_img_field = task_to_qry_field[task_type]["qry_img_field"]
    qry_text_field = task_to_qry_field[task_type]["qry_text_field"]

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

    processor = AutoProcessor.from_pretrained(
        'microsoft/Phi-3.5-vision-instruct',
        trust_remote_code=True,
        num_crops=4,
    )

    for lang in ["en", "it", "es", "fr", "de"]:
        
        max_card = ds_max_card[dataset_suffix]
        dataset_name = f"{dataset_suffix}_{lang}_{str(max_card)}_formatted_{task_type}.jsonl"

        if not os.path.isfile(f"eval_data/{dataset_name}"):
            print(f"NO DATASET: eval_data/{dataset_name}")
            continue

        for formatting_style in ["plain", "punct", "newline"]:

            if os.path.isfile(f"results/{model_path_ds}/{dataset_suffix}_{task_type}_{lang}_{formatting_style}.jsonl"):
                print(f"results/{model_path_ds}/{dataset_suffix}_{task_type}_{lang}_{formatting_style}.jsonl already exists")
                continue
            else:
                print(f"COMPUTING: results/{model_path_ds}/{dataset_suffix}_{task_type}_{lang}_{formatting_style}.jsonl")

            p_1 = 0

            emb_path = f"./ds_embed/{model_path_ds}/{dataset_suffix}_{task_type}_{lang}_{formatting_style}"

            embeds = load_from_disk(emb_path).with_format("torch")
            ids = embeds["id"]
            embeds = embeds["embeddings"]
            mapping_idx = {x: i for i, x in enumerate(ids)}

            os.makedirs(f"results/{model_path_ds}", exist_ok=True)

            with open(f"results/{model_path_ds}/{dataset_suffix}_{task_type}_{lang}_{formatting_style}.jsonl", 'w', encoding='utf8') as f_out:
                with open(f"eval_data/{dataset_name}", 'r', encoding='utf8') as f:

                    i = 0

                    for l in f:

                        line_data = json.loads(l)

                        new_result = {}
                        new_result["label"] = line_data[tgt_id_field][0]

                        cos_sim = eval_instance(line_data, dataset_suffix, lang, qry_img_field, qry_text_field, tgt_id_field, formatting_style, task_type, embeds, mapping_idx, model, processor)

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

                print(f"{checkpoint_path} || {formatting_style} || ACCURACY FOR {dataset_name} = {str(p_1/i)}")


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