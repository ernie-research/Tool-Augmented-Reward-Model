import os
import sys
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""
import wandb
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, set_seed
from src.data.reward_dataset import RewardDataCollatorForSeq2Seq
from src.models.reward_model import SmallRewardModel
from src.template.instruction_template import CONTEXT, QUESTION, ANSWER


def train(
    # model/data params
    base_model: str = "",  # the only required argument
    data_path: str = "yahma/alpaca-cleaned",
    output_dir: str = "./lora-alpaca",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 2000,
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
    # add by ll
    deepspeed: str = None,
    gradient_checkpointing: bool = False,
    reward_type: str = 'lm',  # lm / linear / lm_linear
    reward_compute: str = 'last',    # last / mean
    lr_scheduler_type: str = 'linear',  # linear / cosine
    weight_decay: float = 0.0,
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
            f"deepspeed: {deepspeed}\n"
            f"gradient_checkpointing: {gradient_checkpointing}\n"
            f"reward_type: {reward_type}\n"
            f"reward_compute: {reward_compute}\n",
            f"lr_scheduler_type: {lr_scheduler_type}\n",
            f"weight_decay: {weight_decay}\n",
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    # set seed before initializing model
    set_seed(42)

    # initialize config, model and tokenizer
    config = AutoConfig.from_pretrained(base_model)
    config.reward_type = reward_type
    config.num_labels = 1
    
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        config=config,
    )
    model = SmallRewardModel(model)

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        def get_answer_prompt(example):
            context = CONTEXT.format(context=data_point['context']) if 'context' in data_point else None
            question = QUESTION.format(question=data_point['question'])
            answer = ANSWER.format(answer=example['answer'])

            input_prompt = "\n".join([question, answer]) if context is None else "\n".join([context, question, answer])
            tokenized_full_prompt = tokenize(input_prompt)
            return tokenized_full_prompt
        
        # postive + negative
        pos_answer = data_point['pos_answer']
        neg_answer = data_point['neg_answer']
        pos_tokenized_full_prompt = get_answer_prompt(pos_answer)
        neg_tokenized_full_prompt = get_answer_prompt(neg_answer)
        return {key: [pos_tokenized_full_prompt[key], neg_tokenized_full_prompt[key]] for key in pos_tokenized_full_prompt}

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )

    if val_set_size > 0:
        train_data = (
            data["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            data["test"].map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            deepspeed=deepspeed,
            gradient_checkpointing=gradient_checkpointing,
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=0.01,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            lr_scheduler_type=lr_scheduler_type,
            weight_decay=weight_decay,
            fp16=True,
            logging_steps=1,
            optim="adamw_torch",
            evaluation_strategy="epoch" if val_set_size > 0 else "no",
            save_strategy="epoch",
            output_dir=output_dir,
            logging_dir=output_dir,
            save_total_limit=2,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=RewardDataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # model.save_pretrained(output_dir)
    wandb.finish()

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    fire.Fire(train)
