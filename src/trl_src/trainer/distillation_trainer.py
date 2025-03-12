# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import warnings
from copy import deepcopy
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from accelerate.utils import is_deepspeed_available
from transformers import AutoModelForCausalLM, PreTrainedModel

from ..import_utils import is_liger_kernel_available
from ..models import PreTrainedModelWrapper
from .distillation_config import DistillationConfig
from .sft_trainer import SFTTrainer
from .utils import disable_dropout_in_model, empty_cache


if is_deepspeed_available():
    import deepspeed

if is_liger_kernel_available():
    from liger_kernel.transformers import AutoLigerKernelForCausalLM


class DistillationTrainer(SFTTrainer):
    _tag_names = ["trl", "distillation"]

    def __init__(
        self,
        teacher_model: Union[PreTrainedModel, nn.Module, str],
        args: Optional[DistillationConfig] = None,
        *sft_args,
        **kwargs,
    ):

        super().__init__(*sft_args, args=args, **kwargs)

        if args.teacher_model_init_kwargs is None:
            teacher_model_init_kwargs = {}

        elif not isinstance(teacher_model, str):
            raise ValueError(
                "You passed teacher_model_init_kwargs to the GKDConfig, but your teacher_model is already instantiated."
            )
        else:
            teacher_model_init_kwargs = args.teacher_model_init_kwargs
            teacher_model_init_kwargs["torch_dtype"] = (
                teacher_model_init_kwargs["torch_dtype"]
                if teacher_model_init_kwargs["torch_dtype"] in ["auto", None]
                else getattr(torch, teacher_model_init_kwargs["torch_dtype"])
            )

        if isinstance(teacher_model, str):
            warnings.warn(
                "You passed a teacher model_id to the GKDTrainer. This will automatically create an "
                "`AutoModelForCausalLM`"
            )
            if args.use_liger:
                teacher_model = AutoLigerKernelForCausalLM.from_pretrained(teacher_model, **teacher_model_init_kwargs)
            else:
                teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model, **teacher_model_init_kwargs)

        if args.disable_dropout:
            disable_dropout_in_model(self.model)

        if self.is_deepspeed_enabled:
            self.teacher_model = self._prepare_deepspeed(teacher_model)
        else:
            self.teacher_model = self.accelerator.prepare_model(teacher_model, device_placement=True, evaluation_mode=True)
    
    def get_train_dataloader(self):
        return self.accelerator.prepare(self.train_dataset)

    def _pooling(self, last_hidden_state, attention_mask, pooling, normalize: bool):
        if pooling == 'last':
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_state.shape[0]
            reps = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
        elif pooling == 'mean':
            reps = torch.stack([last_hidden_state[i, attention_mask[i]].mean(0) for i in range(last_hidden_state.shape[0])])
        else:
            raise NotImplementedError
        if normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps

    def _encode(self, model, inputs, separate, normalize = False):

        hidden_states = model(**inputs, return_dict=True, output_hidden_states=True)
        hidden_states = hidden_states.hidden_states[-1]

        if separate:
        
            text_start_idx = torch.Tensor([(x == 1).nonzero()[-1] for x in inputs["input_ids"]])

            pooled_text = []
            pooled_others = []

            for i, x in enumerate(text_start_idx):

                if normalize:
                    pooled_text.append(torch.nn.functional.normalize(hidden_states[i, x.int() + 1:, :][inputs["attention_mask"][i, x.int() + 1:]].mean(0), p=2, dim=-1))
                else:
                    pooled_text.append(hidden_states[i, x.int() + 1:, :][inputs["attention_mask"][i, x.int() + 1:]].mean(0))

                if normalize:
                    pooled_others.append(torch.nn.functional.normalize(hidden_states[i, :x.int() + 1, :][inputs["attention_mask"][i, :x.int() + 1]].mean(0), p=2, dim=-1))
                else:
                    pooled_others.append(hidden_states[i, :x.int() + 1, :][inputs["attention_mask"][i, :x.int() + 1]].mean(0))
                
            pooled_text = torch.stack(pooled_text)
            pooled_others = torch.stack(pooled_others)
        
        else:
            pooled_others = None
            pooled_text = None
        
        pooled_output = self._pooling(hidden_states, inputs["attention_mask"], "last", normalize)

        return pooled_output, pooled_text, pooled_others

    def compute_loss(self, model, inputs, return_outputs=False):

        mse_loss = nn.MSELoss()

        inputs_eng = {}
        inputs_eng["input_ids"] = inputs["input_ids_eng"].to("cuda")
        inputs_eng["attention_mask"] = inputs["attention_mask_eng"].to("cuda")

        inputs_lang = {}
        inputs_lang["input_ids"] = inputs["input_ids_lang"].to("cuda")
        inputs_lang["attention_mask"] = inputs["attention_mask_lang"].to("cuda")

        if "pixel_values" in inputs:
            pixel_values = inputs["pixel_values"].to("cuda")
            pixel_sizes = inputs["image_sizes"].to("cuda")
            inputs_eng["pixel_values"] = pixel_values
            inputs_eng["image_sizes"] = pixel_sizes
            inputs_lang["pixel_values"] = pixel_values
            inputs_lang["image_sizes"] = pixel_sizes

        student_output_eng, student_text_eng, student_others_eng = self._encode(model, inputs_eng, False, False)
        student_output_lang, student_text_lang, student_others_lang = self._encode(model, inputs_lang, False, False)

        # compute teacher output in eval mode
        self.teacher_model.eval()
        with torch.no_grad():
            teacher_output_eng, teacher_text_eng, teacher_others_eng = self._encode(self.teacher_model, inputs_eng, False, False)
        
        en_output_loss = mse_loss(student_output_eng, teacher_output_eng)
        lang_output_loss = mse_loss(student_output_lang, teacher_output_eng)

        loss = (en_output_loss + lang_output_loss) / 2

        empty_cache()

        return loss
   

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch) -> torch.Tensor:
        """
        Perform a training step for the Generalized Knowledge Distillation (GKD) model.

        This method implements the on-policy learning approach described in the GKD paper.
        With probability `self.lmbda`, it generates new responses using the student model,
        which are then used for training instead of the original inputs.
        """
        loss = super().training_step(model, inputs, num_items_in_batch)
        return loss

    def _prepare_deepspeed(self, model: PreTrainedModelWrapper):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model


class DistillationTrainerWithImageLoss(DistillationTrainer):
    
    def compute_loss(self, model, inputs, return_outputs=False):

        mse_loss = nn.MSELoss()

        inputs_eng = {}
        inputs_eng["input_ids"] = inputs["input_ids_eng"].to("cuda")
        inputs_eng["attention_mask"] = inputs["attention_mask_eng"].to("cuda")

        inputs_lang = {}
        inputs_lang["input_ids"] = inputs["input_ids_lang"].to("cuda")
        inputs_lang["attention_mask"] = inputs["attention_mask_lang"].to("cuda")

        if "pixel_values" in inputs:
            pixel_values = inputs["pixel_values"].to("cuda")
            pixel_sizes = inputs["image_sizes"].to("cuda")
            inputs_eng["pixel_values"] = pixel_values
            inputs_eng["image_sizes"] = pixel_sizes
            inputs_lang["pixel_values"] = pixel_values
            inputs_lang["image_sizes"] = pixel_sizes

        student_output_eng, student_text_eng, student_others_eng = self._encode(model, inputs_eng, True, False)
        student_output_lang, student_text_lang, student_others_lang = self._encode(model, inputs_lang, True, False)

        self.teacher_model.eval()
        with torch.no_grad():
            teacher_output_eng, teacher_text_eng, teacher_others_eng = self._encode(self.teacher_model, inputs_eng, True, False)
        
        en_img_loss = mse_loss(student_others_eng, teacher_others_eng)
        lang_img_loss = mse_loss(student_others_lang, teacher_others_eng)
        en_output_loss = mse_loss(student_output_eng, teacher_output_eng)
        lang_output_loss = mse_loss(student_output_lang, teacher_output_eng)

        loss = (en_output_loss + lang_output_loss + en_img_loss + lang_img_loss) / 4

        empty_cache()

        return loss 