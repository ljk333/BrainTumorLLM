from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
from transformers import PreTrainedModel

from ..data import get_template_and_fix_tokenizer
from ..extras.callbacks import LogCallback
from ..extras.logging import get_logger
from ..hparams import get_infer_args, get_train_args
from ..model import load_model, load_tokenizer
from .pt import run_pt
from .sft import run_sft
from .rm import run_rm
from .ppo import run_ppo
from .dpo import run_dpo
from .kto import run_kto
from .orpo import run_orpo

if TYPE_CHECKING:
    from transformers import TrainerCallback

logger = get_logger(__name__)


# === 实验运行分发函数 ===
def run_exp(args: Optional[Dict[str, Any]] = None, callbacks: Optional[List["TrainerCallback"]] = None) -> None:
    if callbacks is None:
        callbacks = []

    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)
    callbacks.append(LogCallback(training_args.output_dir))

    stage_dispatcher = {
        "sft": lambda: run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks),
        "rm": lambda: run_rm(model_args, data_args, training_args, finetuning_args, callbacks),
        "ppo": lambda: run_ppo(model_args, data_args, training_args, finetuning_args, generating_args, callbacks),
    }

    if finetuning_args.stage not in stage_dispatcher:
        raise ValueError(f"Unknown finetuning stage: {finetuning_args.stage}")

    stage_dispatcher[finetuning_args.stage]()


# === 模型导出函数 ===
def export_model(args: Optional[Dict[str, Any]] = None) -> None:
    model_args, data_args, finetuning_args, _ = get_infer_args(args)

    if not model_args.export_dir:
        raise ValueError("Please specify `export_dir` to save model.")
    if model_args.adapter_name_or_path and model_args.export_quantization_bit:
        raise ValueError("Please merge adapters before quantizing the model.")

    # 加载 tokenizer 和模型
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module.get("processor")
    get_template_and_fix_tokenizer(tokenizer, data_args.template)
    model = load_model(tokenizer, model_args, finetuning_args)

    # 导出前的验证
    if not isinstance(model, PreTrainedModel):
        raise ValueError("Model is not a `PreTrainedModel`, export aborted.")
    if getattr(model, "quantization_method", None) and model_args.adapter_name_or_path:
        raise ValueError("Cannot merge adapters to a quantized model.")

    # 设置导出模型精度
    if getattr(model, "quantization_method", None) is None:
        dtype = getattr(model.config, "torch_dtype", torch.float16)
        setattr(model.config, "torch_dtype", dtype)
        model = model.to(dtype)
    else:
        setattr(model.config, "torch_dtype", torch.float16)

    # 保存模型到本地或推送到 Hub
    save_kwargs = {
        "save_directory": model_args.export_dir,
        "max_shard_size": f"{model_args.export_size}GB",
        "safe_serialization": not model_args.export_legacy_format,
    }
    model.save_pretrained(**save_kwargs)

    if model_args.export_hub_model_id:
        model.push_to_hub(model_args.export_hub_model_id, token=model_args.hf_hub_token, **save_kwargs)

    # 保存 tokenizer 和 image_processor（如有）
    try:
        tokenizer.padding_side = "left"
        tokenizer.init_kwargs["padding_side"] = "left"
        tokenizer.save_pretrained(model_args.export_dir)

        if model_args.export_hub_model_id:
            tokenizer.push_to_hub(model_args.export_hub_model_id, token=model_args.hf_hub_token)

        if model_args.visual_inputs and processor:
            image_processor = getattr(processor, "image_processor", None)
            if image_processor:
                image_processor.save_pretrained(model_args.export_dir)
                if model_args.export_hub_model_id:
                    image_processor.push_to_hub(model_args.export_hub_model_id, token=model_args.hf_hub_token)

    except Exception as e:
        logger.warning(f"Failed to save tokenizer or processor: {e}")
