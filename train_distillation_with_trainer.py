import json
import torch

from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig

from src.trl_src import DistillationTrainer, DistillationTrainerWithImageLoss, DistillationConfig

from torch.utils.data import Dataset, DataLoader


class ParallelDS(Dataset):

    def __init__(self, parallel_data_file, processor):

        data = []

        with open(parallel_data_file, 'r', encoding='utf8') as f:

            for l in f:
                data.append(json.loads(l))

        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = self.data[idx]["img_en"]

        if image_path:
            image = [Image.open(image_path).resize((1344, 1344))]
        else:
            image = None

        text_en = self.data[idx]["text_en"]
        inputs_en = processor(text_en, image)

        text_lang = self.data[idx]["text_lang"]
        inputs_lang = processor(text_lang, image)

        return inputs_en["input_ids"][0], \
            inputs_lang["input_ids"][0], \
            inputs_en["pixel_values"][0] if image_path else None, \
            inputs_en["image_sizes"][0] if image_path else None, \
            True if image_path else False


def process_inputs(input_ids, pixel_values, image_sizes, has_image):

    inputs = {}

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=32000
    )

    inputs["input_ids"] = input_ids
    attention_mask = []

    attention_mask = (input_ids != 32000)
    inputs["attention_mask"] = attention_mask

    pixel_values_final = []
    image_sizes_final = []

    for i, hi in enumerate(has_image):

        if hi:
            pixel_values_final.append(pixel_values[i])
            image_sizes_final.append(image_sizes[i])
    
    if len(pixel_values_final) > 0:
        inputs["pixel_values"] = torch.stack(pixel_values_final)
    
    if len(image_sizes_final) > 0:
        inputs["image_sizes"] = torch.stack(image_sizes_final)

    return inputs


def collate_fn(data):
    input_ids_en, input_ids_lang, pixel_values, image_sizes, has_image = zip(*data)
    input_eng = process_inputs(input_ids_en, pixel_values, image_sizes, has_image)
    input_lang = process_inputs(input_ids_lang, pixel_values, image_sizes, has_image)

    inputs = {}
    inputs["input_ids_eng"] = input_eng["input_ids"]
    inputs["input_ids_lang"] = input_lang["input_ids"]
    inputs["attention_mask_eng"] = input_eng["attention_mask"]
    inputs["attention_mask_lang"] = input_lang["attention_mask"]

    if "pixel_values" in input_eng:
        inputs["pixel_values"] = input_eng["pixel_values"]
        inputs["image_sizes"] = input_eng["image_sizes"]

    return inputs

use_image_loss = True
epochs = 1
learning_rate = 1e-5

config = AutoConfig.from_pretrained('microsoft/Phi-3.5-vision-instruct', trust_remote_code=True, cache_dir="cache")
config.use_cache = False
config.padding_side = "right"

base_model = AutoModelForCausalLM.from_pretrained(
    "./models/LVLM2vec_lora_merged",
    config=config,
    attn_implementation="flash_attention_2", 
    torch_dtype=torch.bfloat16, 
    trust_remote_code=True, 
    cache_dir="cache",
    local_files_only=True
)
base_model.padding_side = "right"
base_model.train()

teacher_model = AutoModelForCausalLM.from_pretrained(
    "./models/LVLM2vec_lora_merged",
    config=config,
    attn_implementation="flash_attention_2", 
    torch_dtype=torch.bfloat16, 
    trust_remote_code=True, 
    cache_dir="cache",
    local_files_only=True
)
teacher_model.padding_side = "right"

processor = AutoProcessor.from_pretrained(
    'microsoft/Phi-3.5-vision-instruct',
    trust_remote_code=True,
    num_crops=4,
)

train_dataset = ParallelDS("parallel_shuffled.jsonl", processor)
train_loader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=8, shuffle=False, num_workers=2)

output_dir = "./models/xVLM2Vec" if not use_image_loss else "./models/xVLM2Vec_image"
trainer_class = DistillationTrainer if not use_image_loss else DistillationTrainerWithImageLoss

distillation_config = DistillationConfig(
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    logging_steps=1,
    learning_rate=learning_rate,
    save_strategy="epoch",
    save_total_limit=5,
    weight_decay=0.0,
    fp16=False,
    bf16=True,
    tf32=True,
    max_grad_norm=0.3,
    warmup_ratio=0.0,
    lr_scheduler_type="cosine",
    ddp_find_unused_parameters=False,
    save_safetensors=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    group_by_length=True,
    report_to=None
)

trainer = trainer_class(
    model=base_model,
    tokenizer=processor.tokenizer,
    teacher_model=teacher_model,
    args=distillation_config,
    train_dataset=train_loader,
    dataset_kwargs={
        "skip_prepare_dataset": True,
    }
)

trainer.train()

if trainer.is_fsdp_enabled:
    trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

trainer.save_model(output_dir)