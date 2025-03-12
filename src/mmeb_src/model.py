from typing import Dict
import torch
import torch.distributed as dist
from torch import nn, Tensor
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoConfig

from src.mmeb_src.arguments import ModelArguments


class MMEBModel(nn.Module):
    TRANSFORMER_CLS = AutoModelForCausalLM

    def __init__(self,
                 encoder: PreTrainedModel,
                 pooling: str = 'cls',
                 normalize: bool = False,
                 temperature: float = 1.0,
                 ):
        super().__init__()
        self.config = encoder.config
        self.config.hidden_size = 4096
        self.hidden_size = 4096
        self.encoder = encoder
        self.pooling = pooling
        self.normalize = normalize
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.is_ddp = dist.is_initialized()
        if self.is_ddp:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def encode_input(self, input, separate):

        if separate:
            text_start_idx = torch.Tensor([(x == 1).nonzero()[-1] for x in input["input_ids"]])

        hidden_states = self.encoder(**input, return_dict=True, output_hidden_states=True)
        hidden_states = hidden_states.hidden_states[-1]

        if separate:

            pooled_text = []
            pooled_others = []

            for i, x in enumerate(text_start_idx):
                pooled_text.append(hidden_states[i, x.int() + 1:, :][input["attention_mask"][i, x.int() + 1:]].mean(0))
                pooled_others.append(hidden_states[i, :x.int() + 1, :][input["attention_mask"][i, :x.int() + 1]].mean(0))
            
            pooled_text = torch.stack(pooled_text)
            pooled_others = torch.stack(pooled_others)
        
        else:

            pooled_others = None
            pooled_text = None

        pooled_output = self._pooling(hidden_states, input["attention_mask"], "last")

        return pooled_others, pooled_text, pooled_output

    def _pooling(self, last_hidden_state, attention_mask, pooling):

        if pooling is None:
            pooling = self.pooling

        if pooling == 'last':
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_state.shape[0]
            reps = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
        elif pooling == 'mean':
            reps = torch.stack([last_hidden_state[i, attention_mask[i]].mean(0) for i in range(last_hidden_state.shape[0])])
        else:
            raise NotImplementedError
        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps

    @classmethod
    def build(cls, model_args: ModelArguments, **hf_kwargs):
        # Loading the base model
        config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True, cache_dir="./cache")
        config.use_cache = False
        config.padding_side = "right"
        base_model = cls.TRANSFORMER_CLS.from_pretrained(
            model_args.model_name, **hf_kwargs, config=config, 
            attn_implementation="flash_attention_2", 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True, 
            cache_dir="./cache")
        base_model.padding_side = "right"

        if model_args.lora:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                target_modules=model_args.lora_target_modules.split(','),
                lora_dropout=model_args.lora_dropout,
                init_lora_weights="gaussian",
                use_dora=True,
                inference_mode=False
            )
            lora_model = get_peft_model(base_model, lora_config)
            model = cls(
                encoder=lora_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature
            )
        else:
            model = cls(
                encoder=base_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature
            )
        return model

    @classmethod
    def load(cls, model_args: ModelArguments, **hf_kwargs):
        # Loading the base model
        config = AutoConfig.from_pretrained(model_args.checkpoint_path, trust_remote_code=True, cache_dir="./cache")
        config.use_cache = False
        config.padding_side = "right"
        base_model = cls.TRANSFORMER_CLS.from_pretrained(
            model_args.checkpoint_path, **hf_kwargs, config=config,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            cache_dir="./cache")
        base_model.padding_side = "right"

        # Building the model on top of the base
        if model_args.lora:
            from peft import LoraConfig, PeftModel
            lora_config = LoraConfig.from_pretrained(model_args.checkpoint_path, cache_dir="./cache")
            lora_model = PeftModel.from_pretrained(base_model, model_args.checkpoint_path, config=lora_config, cache_dir="./cache")
            lora_model = lora_model.merge_and_unload()
            model = cls(
                encoder=lora_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize
            )
        else:
            model = cls(
                encoder=base_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize
            )
        return model

    def save(self, output_dir: str):
        self.encoder.save_pretrained(output_dir)

    def forward(self, qry: Dict[str, Tensor] = None, tgt: Dict[str, Tensor] = None, text_only: bool = False):

        if qry:
            _, _, qry_reps = self.encode_input(qry, text_only)
        else:
            qry_reps = None

        if tgt:
            _, _, tgt_reps = self.encode_input(tgt, text_only)
        else:
            tgt_reps = None

        if qry_reps is None or tgt_reps is None:
            return {"qry_reps": qry_reps, "tgt_reps": tgt_reps}

        if self.is_ddp:
            all_qry_reps = self._dist_gather_tensor(qry_reps)
            all_tgt_reps = self._dist_gather_tensor(tgt_reps)
        else:
            all_qry_reps = qry_reps
            all_tgt_reps = tgt_reps

        scores = self.compute_similarity(all_qry_reps, all_tgt_reps)
        scores = scores.view(all_qry_reps.size(0), -1)
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (all_qry_reps.size(0) // all_tgt_reps.size(0))
        loss = self.cross_entropy(scores / self.temperature, target)
        if self.is_ddp:
            loss = loss * self.world_size

        return loss

    def _dist_gather_tensor(self, t: Tensor):
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)
        return all_tensors

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))
