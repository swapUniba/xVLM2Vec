# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

# flake8: noqa

__version__ = "0.11.0"

from typing import TYPE_CHECKING
from .import_utils import _LazyModule, is_diffusers_available, OptionalDependencyNotAvailable


_import_structure = {
    "core": [
        "set_seed",
    ],
    "environment": [
        "TextEnvironment",
        "TextHistory",
    ],
    "extras": [
        "BestOfNSampler",
    ],
    "import_utils": [
        "is_diffusers_available",
        "is_liger_kernel_available",
        "is_llmblender_available",
    ],
    "models": [
        "AutoModelForCausalLMWithValueHead",
        "AutoModelForSeq2SeqLMWithValueHead",
        "PreTrainedModelWrapper",
        "create_reference_model",
        "setup_chat_format",
        "SUPPORTED_ARCHITECTURES",
    ],
    "trainer": [
        "DataCollatorForCompletionOnlyLM",
        "DPOConfig",
        "DPOTrainer",
        "CPOConfig",
        "CPOTrainer",
        "AlignPropConfig",
        "AlignPropTrainer",
        "IterativeSFTTrainer",
        "KTOConfig",
        "KTOTrainer",
        "BCOConfig",
        "BCOTrainer",
        "ModelConfig",
        "NashMDConfig",
        "NashMDTrainer",
        "OnlineDPOConfig",
        "OnlineDPOTrainer",
        "XPOConfig",
        "XPOTrainer",
        "ORPOConfig",
        "ORPOTrainer",
        "PPOConfig",
        "PPOTrainer",
        "PPOv2Config",
        "PPOv2Trainer",
        "RewardConfig",
        "RewardTrainer",
        "RLOOConfig",
        "RLOOTrainer",
        "SFTConfig",
        "SFTTrainer",
        "FDivergenceConstants",
        "FDivergenceType",
        "GKDTrainer",
        "GKDConfig",
        "WinRateCallback",
        "BaseJudge",
        "BaseRankJudge",
        "BasePairwiseJudge",
        "RandomRankJudge",
        "RandomPairwiseJudge",
        "PairRMJudge",
        "HfPairwiseJudge",
        "OpenAIPairwiseJudge",
        "LogCompletionsCallback",
        "DistillationConfig",
        "DistillationTrainer",
        "DistillationTrainerWithImageLoss"
    ],
    "commands": [],
    "commands.cli_utils": ["init_zero_verbose", "SFTScriptArguments", "DPOScriptArguments", "TrlParser"],
    "trainer.callbacks": ["RichProgressCallback", "SyncRefModelCallback"],
    "trainer.utils": ["get_kbit_device_map", "get_peft_config", "get_quantization_config"],
    "multitask_prompt_tuning": [
        "MultitaskPromptEmbedding",
        "MultitaskPromptTuningConfig",
        "MultitaskPromptTuningInit",
    ],
    "data_utils": [
        "apply_chat_template",
        "extract_prompt",
        "is_conversational",
        "maybe_apply_chat_template",
        "maybe_extract_prompt",
        "maybe_unpair_preference_dataset",
        "unpair_preference_dataset",
    ],
}

try:
    if not is_diffusers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["models"].extend(
        [
            "DDPOPipelineOutput",
            "DDPOSchedulerOutput",
            "DDPOStableDiffusionPipeline",
            "DefaultDDPOStableDiffusionPipeline",
        ]
    )
    _import_structure["trainer"].extend(["DDPOConfig", "DDPOTrainer"])

if TYPE_CHECKING:
    from .core import set_seed
    from .environment import TextEnvironment, TextHistory
    from .extras import BestOfNSampler
    from .import_utils import is_diffusers_available, is_liger_kernel_available, is_llmblender_available
    from .models import (
        AutoModelForCausalLMWithValueHead,
        AutoModelForSeq2SeqLMWithValueHead,
        PreTrainedModelWrapper,
        create_reference_model,
        setup_chat_format,
        SUPPORTED_ARCHITECTURES,
    )
    from .trainer import (
        DataCollatorForCompletionOnlyLM,
        DPOConfig,
        DPOTrainer,
        CPOConfig,
        CPOTrainer,
        AlignPropConfig,
        AlignPropTrainer,
        IterativeSFTTrainer,
        KTOConfig,
        KTOTrainer,
        BCOConfig,
        BCOTrainer,
        ModelConfig,
        NashMDConfig,
        NashMDTrainer,
        OnlineDPOConfig,
        OnlineDPOTrainer,
        XPOConfig,
        XPOTrainer,
        ORPOConfig,
        ORPOTrainer,
        PPOConfig,
        PPOTrainer,
        PPOv2Config,
        PPOv2Trainer,
        RewardConfig,
        RewardTrainer,
        RLOOConfig,
        RLOOTrainer,
        SFTConfig,
        SFTTrainer,
        FDivergenceConstants,
        FDivergenceType,
        GKDTrainer,
        GKDConfig,
        WinRateCallback,
        BaseJudge,
        BaseRankJudge,
        BasePairwiseJudge,
        RandomRankJudge,
        RandomPairwiseJudge,
        PairRMJudge,
        HfPairwiseJudge,
        OpenAIPairwiseJudge,
        LogCompletionsCallback,
        DistillationConfig,
        DistillationTrainer,
        DistillationTrainerWithImageLoss
    )
    from .trainer.callbacks import RichProgressCallback, SyncRefModelCallback
    from .trainer.utils import get_kbit_device_map, get_peft_config, get_quantization_config
    from .commands.cli_utils import init_zero_verbose, SFTScriptArguments, DPOScriptArguments, TrlParser
    from .data_utils import (
        apply_chat_template,
        extract_prompt,
        is_conversational,
        maybe_apply_chat_template,
        maybe_extract_prompt,
        maybe_unpair_preference_dataset,
        unpair_preference_dataset,
    )

    try:
        if not is_diffusers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .models import (
            DDPOPipelineOutput,
            DDPOSchedulerOutput,
            DDPOStableDiffusionPipeline,
            DefaultDDPOStableDiffusionPipeline,
        )
        from .trainer import DDPOConfig, DDPOTrainer

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        extra_objects={"__version__": __version__},
    )
