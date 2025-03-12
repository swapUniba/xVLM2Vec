import re
import os
import json
import string
import argparse

from tqdm.auto import tqdm
from datetime import timedelta
from collections import defaultdict
from accelerate.logging import get_logger
from accelerate.utils import gather_object
from datasets import load_dataset, Dataset
from transformers.trainer_utils import set_seed
from accelerate import PartialState, InitProcessGroupKwargs

from src.translator.translator import Translator


translator_map = {}
for translator in Translator.__subclasses__():
    translator_map[translator.get_model_name()] = translator


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--translator_name", type=str)
    parser.add_argument("--dataset_tag", type=str)
    parser.add_argument("--dataset_split", type=str)
    parser.add_argument("--src_lang", type=str)
    parser.add_argument("--tgt_lang", type=str)

    return parser.parse_args()


question_answer_map = {
    "it": ["Domanda", "Risposta"],
    "es": ["Pregunta", "Respuesta"],
    "fr": ["Question", "Réponse"],
    "de": ["Frage", "Antwort"]
}

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


def merge_current_results(accelerator, results, dataset_name, question_field):

    # Wait for all processes to finish
    accelerator.wait_for_everyone()
    # Gather the results
    results = gather_object([results])

    results_merged = defaultdict(list)

    for result in results:
        for key in result:
            results_merged[key].extend(result[key])

    if accelerator.is_main_process:

        dataset = Dataset.from_dict(results_merged)
    
        with open(f'data/{dataset_name}.jsonl', 'a', encoding='utf8') as f_out:
            for x in dataset:

                if x[question_field] is None:
                    continue

                json.dump(x, f_out)
                f_out.write('\n')

    accelerator.wait_for_everyone()


def main(translator_name, dataset_tag, dataset_split, src_lang, tgt_lang):

    mini_batch_size = 128
    batch_size = 8 if dataset_tag != "WebQA" else 4

    process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=50000)).to_kwargs()
    accelerator = PartialState(**process_group_kwargs)
    logger = get_logger(__name__)

    translator = translator_map[translator_name](src_lang=src_lang, tgt_lang=tgt_lang, device_map={"": accelerator.process_index})

    logger.info(f"TRANSLATOR: {translator_name}", main_process_only=True)

    if dataset_split == "train":
        dataset_columns_map = dataset_columns_map_train
        dataset = load_dataset(f"TIGER-Lab/MMEB-{dataset_split}", dataset_tag, cache_dir='cache', revision="0c3f4b8")["train"]
    else:
        raise Exception ("not a valid split")

    dataset_name = f"MMEB_{dataset_tag}_{dataset_split}_{tgt_lang}"

    if not "id" in dataset.column_names:
        dataset = dataset.add_column("id", list(range(len(dataset))))

    if len(dataset) > 10000 and dataset_split == "train":
        dataset = dataset.select(range(10000))

    already_translated_ids = []

    if os.path.isfile(f'data/{dataset_name}.jsonl'):
        with open(f'data/{dataset_name}.jsonl', 'r', encoding='utf8') as f:

            for l in f:
                line_data = json.loads(l)
                already_translated_ids.append(line_data["id"])
    
    already_translated_ids = set(already_translated_ids)
    dataset = dataset.filter(lambda x: x["id"] not in already_translated_ids)

    accelerator.wait_for_everyone()

    set_seed(42)

    with accelerator.split_between_processes(dataset) as inp:

        question_str, answer_str = question_answer_map[tgt_lang]

        question_field = dataset_columns_map[dataset_tag]["question"]
        fields_to_translate = dataset_columns_map[dataset_tag]["fields_to_translate"]
        fields_to_translate = [question_field] + fields_to_translate
        fields_to_keep = ["id"] + dataset_columns_map[dataset_tag]["fields_to_keep"]

        results = {"id": []}

        for x in fields_to_translate:
            results[x] = []
        
        for x in fields_to_keep:
            results[x] = []

        logger.info("Setting seed to 42", main_process_only=True)

        progress_bar = tqdm(range(len(inp)), disable=not accelerator.is_main_process, desc=f"Translating")

        batch_inp = []
        instances = 0

        for i, d in enumerate(inp):

            fields_with_image_token = []

            to_translate = "Question: "

            if "<|image_1|>\n" in d[question_field]:
                fields_with_image_token.append(question_field)

            question = d[question_field].replace("<|image_1|>\n", "").replace('\n', '')

            to_translate += question

            if not to_translate[-1] in string.punctuation:
                to_translate += "."

            for j, x in enumerate(fields_to_translate):

                if j == 0:
                    continue
                
                if "<|image_1|>\n" in d[x]:
                    fields_with_image_token.append(x)

                to_translate += f" Answer: " + d[x].replace("<|image_1|>\n", "").replace('\n', '')

                if not to_translate[-1] in string.punctuation:
                    to_translate += "."
            
            batch_inp.append(translator.prepare_text(to_translate))

            instances += 1

            for x in fields_to_keep:
                results[x].append(d[x])
        
            if instances >= batch_size or (i == len(inp) - 1 and instances != 0):

                set_seed(42)

                try:
                    batch_results = translator.translate_text(batch_inp)
                except Exception as e:

                    print(e)

                    for x in fields_to_translate:
                        results[x].extend([None for _ in range(batch_size)])

                    continue
                
                for result in batch_results:

                    result = re.split(f"({question_str}:|{answer_str}:)", result)
                    result = result[1:]
                    result = [x for i, x in enumerate(result) if i%2!=0]

                    try:

                        if len(result) != len(fields_to_translate):
                            raise Exception
                        
                        for sub_result, field in zip(result, fields_to_translate):

                            sub_result = sub_result.replace(answer_str, "").replace(answer_str.lower(), "").strip()

                            if field in fields_with_image_token:
                                results[field].append("<|image_1|>\n" + sub_result)
                            else:
                                results[field].append(sub_result)

                    except Exception as e:

                        print(e)

                        for x in fields_to_translate:
                            results[x].append(None)

                        continue

                batch_inp = []
                instances = 0
                progress_bar.update(batch_size)

            if len(results["id"]) == mini_batch_size:
                merge_current_results(accelerator, results, dataset_name, question_field)

                results = {"id": []}

                for x in fields_to_translate:
                    results[x] = []
                
                for x in fields_to_keep:
                    results[x] = []
                    
        merge_current_results(accelerator, results, dataset_name, question_field)


if __name__ == "__main__":

    args = get_args()
    main(**vars(args))
    