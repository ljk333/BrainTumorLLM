import argparse
from llamafactory.train import run_exp
from llamafactory.arguments import ModelArguments, DataArguments, FinetuningArguments, TrainingArguments


def parse_args():
    parser = argparse.ArgumentParser()

    # 添加你想动态修改的超参数
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=int, default=19)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="sft")
    parser.add_argument("--model_name_or_path", type=str, default="Meta-Llama-3-8B-Instruct")
    parser.add_argument("--eval_steps", type=int, default=100)

    return parser.parse_args()


def main():
    args = parse_args()

    model_args = ModelArguments(
        model_name_or_path=args.model_name_or_path,
    )

    data_args = DataArguments(
        dataset="train",
        template="llama3",
        cutoff_len=1024,
        overwrite_cache=True,
        preprocessing_num_workers=16,
        val_size=0.1,
    )

    finetuning_args = FinetuningArguments(
        finetuning_type="lora",
        lora_target="all",
        stage="sft",
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=500,
        eval_steps=args.eval_steps,
        evaluation_strategy="steps",
        fp16=True,
        ddp_timeout=180000000,
        plot_loss=True,
        do_train=True,
    )

    run_exp(model_args, data_args, training_args, finetuning_args)


if __name__ == "__main__":
    main()
