from abc import ABC, abstractmethod

import torch
import transformers

from transformers import AutoModel, AutoProcessor


class BaseModel(ABC):

    def __init__(self, model, processor):

        self.model = model
        self.processor = processor


class BaseOneModel(BaseModel):

    def __init__(self, model, processor):

        self.model = model
        self.processor = processor

    @abstractmethod
    def get_embeddings(self, texts, images):
        raise NotImplementedError


class BaseTwoModel(BaseModel):

    def __init__(self, text_model, img_model, text_processor, img_processor):

        super().__init__([text_model, img_model], [text_processor, img_processor])

        self.text_model = self.model[0]
        self.text_processor = self.processor[0]

        self.img_model = self.model[1]
        self.img_processor = self.processor[1]

    @abstractmethod
    def get_text_embeddings(self, texts):
        raise NotImplementedError

    @abstractmethod
    def get_img_embeddings(self, images):
        raise NotImplementedError


class MCLIPModel(BaseTwoModel):

    def __init__(self, device="cuda:0"):

        from multilingual_clip import pt_multilingual_clip

        text_model = pt_multilingual_clip.MultilingualCLIP.from_pretrained('M-CLIP/XLM-Roberta-Large-Vit-B-16Plus', cache_dir="./cache").eval()
        tokenizer = transformers.AutoTokenizer.from_pretrained('M-CLIP/XLM-Roberta-Large-Vit-B-16Plus')
        text_model.to(device)
        
        import open_clip

        img_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-plus-240', pretrained="laion400m_e32", cache_dir="./cache")
        img_model = img_model.to(device).eval()

        super().__init__(text_model, img_model, tokenizer, preprocess)
    
    def get_text_embeddings(self, texts):
        with torch.no_grad():
            txt_tok = self.text_processor(texts, padding=True, return_tensors='pt').to(self.text_model.device)
            embs = self.text_model.transformer(**txt_tok)[0]
            att = txt_tok['attention_mask']
            embs = (embs * att.unsqueeze(2)).sum(dim=1) / att.sum(dim=1)[:, None]
            return self.text_model.LinearTransformation(embs)
    
    def get_img_embeddings(self, images):

        img_embeddings = []
        with torch.no_grad():
            for image in images:
                image = self.img_processor(image).unsqueeze(0).to(self.text_model.device)
                img_embeddings.append(self.img_model.encode_image(image))
        
        return torch.vstack(img_embeddings)


class SiglipModel(BaseOneModel):

    def __init__(self, device="cuda:0"):

        model = AutoModel.from_pretrained("google/siglip-base-patch16-256-multilingual", cache_dir="./cache").eval()
        model.to(device)
        processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-256-multilingual", cache_dir="./cache")

        super().__init__(model, processor)
    

    def get_embeddings(self, texts, images):

        inputs = self.processor(text=texts, images=images, truncation=True, padding="max_length", return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs.text_embeds, outputs.image_embeds


class ClipSbertModel(BaseTwoModel):

    def __init__(self, device = "cuda:0"):

        from sentence_transformers import SentenceTransformer

        img_model = SentenceTransformer('clip-ViT-B-32', cache_folder='./cache').eval().to(device)

        model_name = 'sentence-transformers/clip-ViT-B-32-multilingual-v1'
        text_model = SentenceTransformer(model_name, cache_folder='./cache').to(device)

        super().__init__(text_model, img_model, None, None)
    

    def get_text_embeddings(self, texts):

        with torch.no_grad():
            return torch.from_numpy(self.text_model.encode(texts))
    
    def get_img_embeddings(self, images):
        with torch.no_grad():
            return torch.from_numpy(self.img_model.encode(images))