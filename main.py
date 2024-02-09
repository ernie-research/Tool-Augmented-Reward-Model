import re
import os
import sys
from typing import List
import argparse
import torch
import transformers
from datasets import load_dataset

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""
import wandb
from transformers import AutoConfig, LlamaForCausalLM, LlamaTokenizer, set_seed
from src.models.reward_model import RewardModel
from src.data.reward_dataset import RewardDataCollatorForSeq2Seq
from src.template.instruction_template import CONTEXT, QUESTION, ANSWER, TOOL, OBSERVATION, WORK


def main():
    parser = argparse.ArgumentParser()
    # model/data params
    parser.add_argument('--base_model', type=str, default='vicuna-7b-1.3', help='the only required argument')
    parser.add_argument('--data_path', type=str, default='data/',help='path to data')
    parser.add_argument('--output_dir', type=str, default='output/',help='path to output')
    # training hyperparams
    parser.add_argument('--batch_size', type=int, default=128, help='global batch size')
    parser.add_argument('--micro_batch_size', type=int, default=4, help='micro batch size')
    parser.add_argument('--num_epochs', type=int, default=3, help='training epochs')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--cutoff_len', type=int, default=2048, help='cutoff length')
    parser.add_argument('--val_set_size', type=int, default=2000, help='cutoff length')
    # llm hyperparams
    parser.add_argument('--add_eos_token', action="store_true", default=False, help='max iteractions')
    parser.add_argument('--group_by_length', action="store_true", default=False, help='device type')
    # wandb params
    parser.add_argument('--wandb_project', type=str, default="", help='cutoff length')
    parser.add_argument('--wandb_run_name', type=str, default="", help='cutoff length')
    parser.add_argument('--wandb_watch', type=str, default="", help='cutoff length')
    parser.add_argument('--wandb_log_model', type=str, default="", help='cutoff length')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='cutoff length')
    # lora hyperparams
    parser.add_argument("--use_lora", action="store_true", default=False, help="local rank")
    parser.add_argument("--lora_r", type=int, default=8, help="world size")
    parser.add_argument("--lora_alpha", type=int, default=16, help="local rank")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="world size")
    parser.add_argument("--lora_target_modules", type=List[str], default=["q_proj", "v_proj"], help="local rank")
    # add by ll
    parser.add_argument("--deepspeed", type=str, default=None, help="deepspeed")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False, help="gradient_checkpointing")
    parser.add_argument("--lr_scheduler_type", type=str, default='cosine', help="lr_scheduler_type")
    parser.add_argument("--lm", action="store_true", default=False, help="only train the reponse of the positive example, if False, mask all inputs")
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight_decay')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    parser.add_argument("--ranking", action="store_true", default=False, help="whether apply pair-wise ranking loss")
    parser.add_argument("--ranking_way", type=str, default='last', help="mean / last")
    parser.add_argument("--invoke_tool", action="store_true", default=False, help="whether invoke tools, if False, QA")
    parser.add_argument("--no_work", action="store_true", default=False, help="w/o work")
    parser.add_argument("--no_observation", action="store_true", default=False, help="w/o observation")
    parser.add_argument("--add_special_tokens", action="store_true", default=False, help="whether add speical tokens")
    args = parser.parse_args()

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(args)
    assert (
        args.base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = args.batch_size // args.micro_batch_size

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(args.wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(args.wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    if len(args.wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = args.wandb_watch
    if len(args.wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = args.wandb_log_model

    # set seed before initializing model
    set_seed(42)

    # initialize config, model and tokenizer
    config = AutoConfig.from_pretrained(args.base_model)
    config.invoke_tool = args.invoke_tool
    config.ranking = args.ranking
    config.ranking_way = args.ranking_way

    model = RewardModel.from_pretrained(
        args.base_model,
        load_in_8bit=False,
        config=config
    )

    tokenizer = LlamaTokenizer.from_pretrained(args.base_model)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    def _add_special_tokens(tokenizer):
        special_tokens_list = ['<start_tool>', '<start_observation>', '<start_work>']
        tokenizer.add_special_tokens({'additional_special_tokens': special_tokens_list})

        special_tokens2ids = {}
        special_tokens2words_ids = {}
        for token in special_tokens_list:
            special_tokens2ids[token] = tokenizer.convert_tokens_to_ids(token)  # special token
            if '/' not in token:
                words = '<s>' + re.sub('[<|>]', '', token)
                words_ids = tokenizer.encode(words, add_special_tokens=False)
            else:
                words = '</s>' + re.sub('[<|>|/]', '', token)
                words_ids = tokenizer.encode(words, add_special_tokens=False)
            special_tokens2words_ids[token] = words_ids
        special_tokens2ids, special_tokens2words_ids

        return tokenizer, special_tokens_list, special_tokens2ids, special_tokens2words_ids
    
    if args.add_special_tokens:
        tokenizer, special_tokens_list, special_tokens2ids, special_tokens2words_ids = _add_special_tokens(tokenizer)
        # then resize the word embedding and init
        model.init_and_resize_embedding(len(tokenizer))

    if args.use_lora:
        config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=args.lora_target_modules,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                modules_to_save=["v_head", "lm_head", "embed_tokens"]
        )
        model.enable_input_require_grads()
        model = get_peft_model(model, config)

        for name, param in model.named_parameters():
            if int(os.environ.get("LOCAL_RANK", 0)) <= 0:
                print(name, param.requires_grad)
        
        model.print_trainable_parameters()

    def generate_and_tokenize_prompt(data_point):
        def get_answer_prompt(example, key='pos'):
            context = CONTEXT.format(context=data_point['context']) if 'context' in data_point else None
            question = QUESTION.format(question=data_point['question'])
            answer = ANSWER.format(answer=example['answer'])
            if isinstance(example['actions'], str):
                tool = example['actions'].replace('\n\n', '\n').replace('Thought: \n', 'Thought: ')
                # add actions
                starts = [match.start() for match in re.finditer('Observation', tool)]
                n_obs = 0
                for start in starts:
                    start += len('<start_observation> ') * n_obs
                    tool = tool[:start] + '<start_observation> ' + tool[start:]
                    n_obs += 1
                # add tools
                starts = [match.start() for match in re.finditer('Thought:', tool)]
                n_tool = 0
                for start in starts:
                    start += len('<start_tool> ') * n_tool
                    tool = tool[:start] + '<start_tool> ' + tool[start:]
                    n_tool += 1
                observation = None
            else:
                tool = TOOL.format(thought=example['actions']['Thought'], action=example['actions']['Action'], action_input=example['actions']['Action Input'])
                observation = OBSERVATION.format(observation=example['actions']['Observation'])
            work = WORK.format(work=example['score_agent']['explanation'].strip())   if not args.no_work else "" # if no_work, drop the work

            if not args.invoke_tool:
                input_prompt = "\n".join([question, answer])
                tokenized_full_prompt = tokenizer(input_prompt, truncation=True, max_length=args.cutoff_len, padding=False, return_tensors=None)
                tokenized_full_prompt['labels'] = tokenized_full_prompt['input_ids'].copy()
            else:
                input_prompt = "\n".join(["### USER:", question, answer]) if context is None else "\n".join(["### USER:", context, question, answer])
                if observation is not None:
                    output_prompt = "\n".join(["### ASSISTANT:", tool, observation, work]).strip()
                else:
                    output_prompt = "\n".join(["### ASSISTANT:", tool, work]).strip()
                tokenized_full_prompt = tokenizer(input_prompt+'\n'+output_prompt)
                labels = tokenized_full_prompt['input_ids'].copy()
                
                if args.lm and key=='pos':
                    # learn the respone of the positive exmaple
                    input_prompt = "### USER:"+'\n'+question

                # default mask all inputs
                question_prompt_ids = tokenizer(input_prompt+'\n', truncation=True, max_length=args.cutoff_len, padding=False, return_tensors=None)['input_ids']
                question_len = len(question_prompt_ids)
                labels = [-100] * question_len + labels[question_len:]

                if args.no_observation:
                    def find_all_index(arr, num):
                        return [i for i in range(len(arr)) if arr[i]==num]
                    # mask observation
                    if special_tokens2ids['<start_observation>'] in labels:
                        tool_index = find_all_index(labels, special_tokens2ids['<start_tool>'])
                        observation_index = find_all_index(labels, special_tokens2ids['<start_observation>'])
                        work_s = labels.index(special_tokens2ids['<start_work>']) if not args.no_work else len(labels) 
                        tool_index = tool_index[1:] + [work_s]
                        for obs_s, tool_e in zip(observation_index, tool_index):
                            obs_s += 1
                            labels[obs_s:tool_e] = [-100] * (tool_e-obs_s)

                tokenized_full_prompt['labels'] = labels

            if tokenized_full_prompt["input_ids"][-1] != tokenizer.eos_token_id \
                    and len(tokenized_full_prompt["input_ids"]) < args.cutoff_len and args.add_eos_token:
                tokenized_full_prompt["input_ids"].append(tokenizer.eos_token_id)
                tokenized_full_prompt["labels"].append(tokenizer.eos_token_id)
                tokenized_full_prompt["attention_mask"].append(1)
            assert len(tokenized_full_prompt["labels"]) == len(tokenized_full_prompt["input_ids"])
            return tokenized_full_prompt

        # postive + negative
        pos_answer = data_point['pos_answer']
        neg_answer = data_point['neg_answer']
        pos_tokenized_full_prompt = get_answer_prompt(pos_answer, key='pos')
        neg_tokenized_full_prompt = get_answer_prompt(neg_answer, key='neg')
        return {key: [pos_tokenized_full_prompt[key], neg_tokenized_full_prompt[key]] for key in pos_tokenized_full_prompt}

    if args.data_path.endswith(".json") or args.data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=args.data_path)
    else:
        data = load_dataset(args.data_path)

    if args.resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            args.resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            args.resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )

    if args.val_set_size > 0:
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
            deepspeed=args.deepspeed,
            gradient_checkpointing=args.gradient_checkpointing,
            per_device_train_batch_size=args.micro_batch_size,
            per_device_eval_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=0.01,
            num_train_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            lr_scheduler_type=args.lr_scheduler_type,
            weight_decay=args.weight_decay,
            fp16=True,
            logging_steps=1,
            optim="adamw_torch",
            evaluation_strategy="epoch" if args.val_set_size > 0 else "no",
            save_strategy="epoch",
            eval_steps=100 if args.val_set_size > 0 else None,
            save_steps=100,
            output_dir=args.output_dir,
            logging_dir=args.output_dir,
            # save_total_limit=3,
            # load_best_model_at_end=True if args.val_set_size > 0 else False,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=args.group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=args.wandb_run_name if use_wandb else None,
        ),
        data_collator=RewardDataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    if args.use_lora:
        old_state_dict = model.state_dict
        model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(
                self, old_state_dict()
            )
        ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    model.save_pretrained(args.output_dir)
    wandb.finish()

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    main()
