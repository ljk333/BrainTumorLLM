import sys
from llamafactory.cli import main

# 默认参数字典（你可以修改这里）
default_args = [
    "llamafactory-cli", "train",
    "--model_name_or_path", "Meta-Llama-3-8B-Instruct",
    "--stage", "sft",
    "--do_train",
    "--finetuning_type", "lora",
    "--lora_target", "all",
    "--dataset", "train",
    "--template", "llama3",
    "--cutoff_len", "1024",
    "--overwrite_cache",
    "--preprocessing_num_workers", "16",
    "--output_dir", "sft_216_618",
    "--logging_steps", "10",
    "--save_steps", "500",
    "--plot_loss",
    "--overwrite_output_dir",
    "--per_device_train_batch_size", "1",
    "--gradient_accumulation_steps", "8",
    "--learning_rate", "1e-5",
    "--num_train_epochs", "19",
    "--lr_scheduler_type", "cosine",
    "--warmup_ratio", "0.1",
    "--fp16",
    "--ddp_timeout", "180000000",
    "--val_size", "0.1",
    "--per_device_eval_batch_size", "1",
    "--eval_strategy", "steps",
    "--eval_steps", "100"
]

# 提取你命令行传入的参数（不包括脚本名）
user_args = sys.argv[1:]

# 用命令行参数覆盖默认参数（规则是：如果命令行中有相同的key，就替换掉默认的）
def merge_args(default, user):
    def get_arg_indices(args):
        return {args[i]: i for i in range(len(args)) if args[i].startswith("--")}

    default_dict = get_arg_indices(default)
    user_dict = get_arg_indices(user)

    # 替换默认参数
    for key, idx in user_dict.items():
        if key in default_dict:
            default[default_dict[key]+1] = user[idx+1]  # 替换值
        else:
            default.extend([key, user[idx+1]])  # 添加新参数

    return default

# 构建最终参数列表
sys.argv = merge_args(default_args, user_args)

# 启动训练
main()
