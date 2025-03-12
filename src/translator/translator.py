import re
import torch

from abc import ABC, abstractmethod
from transformers import T5ForConditionalGeneration, T5Tokenizer


class Translator(ABC):

    def __init__(self, model, processor, src_lang: str, tgt_lang: str):

        self.model = model
        self.processor = processor
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def prepare_text(self, text_to_translate):
        return text_to_translate

    @abstractmethod
    def translate_text(self, texts_to_translate):
        raise NotImplementedError

    def fix_translation(self, translated_text):
        return translated_text
    
    @classmethod
    @abstractmethod
    def get_model_name(cls):
        raise NotImplementedError


class MadladTranslator(Translator):

    def __init__(self, src_lang: str, tgt_lang: str, device_map: dict, model_b: int = 3):

        model = T5ForConditionalGeneration.from_pretrained(f"google/madlad400-{model_b}b-mt", device_map=device_map, torch_dtype=torch.bfloat16).eval()
        tokenizer = T5Tokenizer.from_pretrained(f"google/madlad400-{model_b}b-mt")

        super().__init__(model, tokenizer, src_lang, tgt_lang)

    def prepare_text(self, text_to_translate):
        return f"<2{self.tgt_lang}> {text_to_translate}"
    
    def translate_text(self, texts_to_translate):
        inputs = self.processor(texts_to_translate, return_tensors="pt", padding=True).to("cuda") 
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=2048)
        translated_text = self.processor.batch_decode(outputs.detach().cpu(), skip_special_tokens=True)
        return translated_text

    def fix_translation(self, translated_text):
        # sometimes madlad translates text by adding a string such as "#300000 -" at the start.
        translated_text_split = re.split("^#[0-9]+ - ", translated_text)
        return translated_text_split[0] if len(translated_text_split) == 1 else translated_text_split[1]
    
    @classmethod
    def get_model_name(cls):
        return "madlad"
    