from transformers import AutoProcessor

from src.mmeb_src.model import MMEBModel
from src.mmeb_src.arguments import ModelArguments


def main():

    model_args = ModelArguments(
        model_name='microsoft/Phi-3.5-vision-instruct',
        pooling='last',
        normalize=True,
        lora=True,
        checkpoint_path='TIGER-Lab/VLM2Vec-LoRA')
    
    processor = AutoProcessor.from_pretrained(
        'microsoft/Phi-3.5-vision-instruct',
        trust_remote_code=True,
        num_crops=4,
    )

    processor.tokenizer.padding_side = "right"
    model = MMEBModel.load(model_args)
    model.encoder._hf_peft_config_loaded = False
    model.encoder.save_pretrained('./models/LVLM2vec_lora_merged', safe_serialization=False)


if __name__ == "__main__":
    main()