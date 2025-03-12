import os
import json
import string

from datasets import load_dataset

dataset_columns_map_train = {

    "A-OKVQA": {
        "question": "qry",
        "fields_to_translate": ["pos_text", "neg_text"],
        "fields_to_keep": ["qry_image_path", "pos_image_path", "neg_image_path"],
    },

    "CIFAR_100": {
        "question": "qry",
        "fields_to_translate": ["pos_text"],
        "fields_to_keep": ["qry_image_path", "pos_image_path"],
    },

    "CIRR": {
        "question": "qry",
        "fields_to_translate": ["pos_text"],
        "fields_to_keep": ["qry_image_path", "pos_image_path"],
    },

    "FashionIQ": {
        "question": "qry",
        "fields_to_translate": ["pos_text"],
        "fields_to_keep": ["qry_image_path", "pos_image_path"],
    },

    "ImageNet-A": {
        "question": "qry",
        "fields_to_translate": ["pos_text"],
        "fields_to_keep": ["qry_image_path", "pos_image_path"],
    },

    "ImageNet-R": {
        "question": "qry",
        "fields_to_translate": ["pos_text"],
        "fields_to_keep": ["qry_image_path", "pos_image_path"],
    },

    "ImageNet_1K": {
        "question": "qry",
        "fields_to_translate": ["pos_text"],
        "fields_to_keep": ["qry_image_path", "pos_image_path"],
    },

    "MSCOCO": {
        "question": "qry",
        "fields_to_translate": ["pos_text"],
        "fields_to_keep": ["qry_image_path", "pos_image_path"],
    },

    "MSCOCO_i2t": {
        "question": "qry",
        "fields_to_translate": ["pos_text"],
        "fields_to_keep": ["qry_image_path", "pos_image_path"],
    },

    "MSCOCO_t2i": {
        "question": "qry",
        "fields_to_translate": ["pos_text"],
        "fields_to_keep": ["qry_image_path", "pos_image_path"],
    },

    "N24News": {
        "question": "qry",
        "fields_to_translate": ["pos_text", "neg_text"],
        "fields_to_keep": ["qry_image_path", "pos_image_path", "neg_image_path"],
    },

    "NIGHTS": {
        "question": "qry",
        "fields_to_translate": ["pos_text"],
        "fields_to_keep": ["qry_image_path", "pos_image_path"],
    },

    "OK-VQA": {
        "question": "qry",
        "fields_to_translate": ["pos_text", "neg_text"],
        "fields_to_keep": ["qry_image_path", "pos_image_path", "neg_image_path"],
    },

    "SUN397": {
        "question": "qry",
        "fields_to_translate": ["pos_text"],
        "fields_to_keep": ["qry_image_path", "pos_image_path"],
    },

    "VOC2007": {
        "question": "qry",
        "fields_to_translate": ["pos_text"],
        "fields_to_keep": ["qry_image_path", "pos_image_path"],
    },

    "VisDial": {
        "question": "qry",
        "fields_to_translate": ["pos_text", "neg_text"],
        "fields_to_keep": ["qry_image_path", "pos_image_path", "neg_image_path"],
    },

    "Visual7W": {
        "question": "qry",
        "fields_to_translate": ["pos_text", "neg_text"],
        "fields_to_keep": ["qry_image_path", "pos_image_path", "neg_image_path"]
    },

    "Visual7W-pointing": {
        "question": "qry",
        "fields_to_translate": ["pos_text", "neg_text"],
        "fields_to_keep": ["qry_image_path", "pos_image_path", "neg_image_path"]
    },

    "VisualNews_i2t": {
        "question": "qry",
        "fields_to_translate": ["pos_text"],
        "fields_to_keep": ["qry_image_path", "pos_image_path"],
    },

    "VisualNews_t2i": {
        "question": "qry",
        "fields_to_translate": ["pos_text"],
        "fields_to_keep": ["qry_image_path", "pos_image_path"],
    },

    "WebQA": {
        "question": "qry",
        "fields_to_translate": ["pos_text"],
        "fields_to_keep": ["qry_image_path", "pos_image_path"],
    }
}


if __name__ == "__main__":

    langs = ["it", "es", "fr", "de"]

    with open(f'parallel.jsonl', 'w', encoding='utf8') as f_out:

        for lang in langs:

            seen = False

            for x in os.listdir('data'):

                if 'MMEB' not in x:
                    continue

                if 'eval' in x:
                    continue
                
                dataset_tag = x.split('_')[1]
                dataset_lang = x.split('_')[-1].split('.')[0]

                if dataset_tag == "ImageNet":
                    dataset_tag = "ImageNet_1K"

                if dataset_tag == "CIFAR":
                    dataset_tag = "CIFAR_100"
                
                if dataset_tag == "VisualNews":
                    if seen:
                        dataset_tag = "VisualNews_t2i"
                    else:
                        dataset_tag = "VisualNews_i2t"
                        seen = True

                if dataset_lang != lang:
                    continue

                dataset = load_dataset(f"TIGER-Lab/MMEB-train", dataset_tag, cache_dir='./cache', revision="0c3f4b8")["train"]
                dataset = dataset.add_column("id", list(range(len(dataset))))

                if len(dataset) > 10000:
                    dataset = dataset.select(range(10000))
                
                ds_map = {x["id"]: x for x in dataset}

                del dataset

                with open(os.path.join('data', x), 'r', encoding='utf8') as f:

                    for l in f:

                        line_data = json.loads(l)

                        english_line_data = ds_map[line_data["id"]]

                        ## QUESTION FIELD

                        new_dict = {}
                        new_dict["id"] = line_data["id"]
                        new_dict["text_en"] = english_line_data["qry"]
                        new_dict["text_lang"] = line_data["qry"]

                        if len(new_dict["text_lang"]) == 0:
                            continue

                        if len(new_dict["text_en"]) == 0:
                            continue

                        if new_dict["text_en"][-1] == "\n":
                            replaced_en = new_dict["text_en"][:-1]
                        else:
                            replaced_en = new_dict["text_en"]
                        
                        if replaced_en[-1] not in string.punctuation and new_dict["text_lang"][-1] in string.punctuation:
                            new_dict["text_lang"] = new_dict["text_lang"][:-1]
                        
                        if replaced_en[-1] in string.punctuation and new_dict["text_lang"][-1] != replaced_en[-1]:
                            new_dict["text_lang"] += replaced_en[-1]

                        if new_dict["text_en"][-1] == "\n" and new_dict["text_lang"][-1] != "\n":
                            new_dict["text_lang"] += "\n"
                                
                        new_dict["img_en"] = english_line_data["qry_image_path"]
                        new_dict["img_lang"] = line_data["qry_image_path"]

                        if "<|image_1|>\n" in new_dict["text_en"] and "<|image_1|>\n" not in new_dict["text_lang"]:
                            continue

                        json.dump(new_dict, f_out)
                        f_out.write('\n')

                        ## POS FIELD

                        new_dict = {}
                        new_dict["id"] = line_data["id"]
                        new_dict["text_en"] = english_line_data["pos_text"]
                        new_dict["text_lang"] = line_data["pos_text"]

                        if len(new_dict["text_lang"]) == 0:
                            continue

                        if len(new_dict["text_en"]) == 0:
                            continue

                        if new_dict["text_en"][-1] == "\n":
                            replaced_en = new_dict["text_en"][:-1]
                        else:
                            replaced_en = new_dict["text_en"]
                        
                        if replaced_en[-1] not in string.punctuation and new_dict["text_lang"][-1] in string.punctuation:
                            new_dict["text_lang"] = new_dict["text_lang"][:-1]
                        
                        if replaced_en[-1] in string.punctuation and new_dict["text_lang"][-1] != replaced_en[-1]:
                            new_dict["text_lang"] += replaced_en[-1]

                        if new_dict["text_en"][-1] == "\n" and new_dict["text_lang"][-1] != "\n":
                            new_dict["text_lang"] += "\n"

                        new_dict["img_en"] = english_line_data["pos_image_path"]
                        new_dict["img_lang"] = line_data["pos_image_path"]

                        if "<|image_1|>\n" in new_dict["text_en"] and "<|image_1|>\n" not in new_dict["text_lang"]:
                            continue

                        json.dump(new_dict, f_out)
                        f_out.write('\n')


    import random

    with open(f'parallel.jsonl', 'r', encoding='utf8') as f:
        data = []

        for l in f:
            data.append(json.loads(l))

    random.seed(42)
    random.shuffle(data)

    with open(f'parallel_shuffled.jsonl', 'w', encoding='utf8') as f:

        for x in data:
            json.dump(x, f)
            f.write('\n')

    print(len(data))