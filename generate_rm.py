
import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoConfig, LlamaTokenizer, AutoTokenizer, AutoModelForSequenceClassification, set_seed
from src.models.reward_model import RewardModel, SmallRewardModel
from src.data.reward_dataset import RewardDataCollatorForSeq2Seq
from src.utils.metrics import accuracy, f1_micro
from src.template.instruction_template import CONTEXT, QUESTION, ANSWER

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='path to dataset')
    parser.add_argument('--model_path', type=str, help='path to model')
    parser.add_argument('--checkpoint_path', type=str, help='path to checkpoint')
    parser.add_argument('--output_path', type=str, help='path to output')
    parser.add_argument('--prompt_template_name', type=str, default='instruction_template', help='prompt_template_name')
    parser.add_argument('--cutoff_len', type=int, default=512, help='cutoff length')
    parser.add_argument('--eval_batch_size', type=int, default=4, help='eval batch size')
    parser.add_argument('--ranking', action='store_true', help='reward type')
    parser.add_argument('--ranking_way', type=str, default='last', help='reward compute type')
    parser.add_argument('--invoke_tool', action='store_true', help='reward type')
    args = parser.parse_args()

    set_seed(42)

    # initialize config, model and tokenizer
    config = AutoConfig.from_pretrained(args.model_path)
    config.ranking = args.ranking
    config.ranking_way = args.ranking_way
    config.invoke_tool = args.invoke_tool

    if 'bert' in args.model_path or 'roberta' in args.model_path or 'deberta' in args.model_path:
        config.num_labels = 1
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_path,
            config=config,
        )
        model = SmallRewardModel(model)
        model_state = torch.load(os.path.join(args.checkpoint_path, 'pytorch_model.bin'), map_location='cpu')
        model.load_state_dict(model_state)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path,)
    else:
        model = RewardModel.from_pretrained(
            args.checkpoint_path,
            config=config
        )
    
        tokenizer = LlamaTokenizer.from_pretrained(args.model_path)

        tokenizer.pad_token_id = (
            0  # unk. we want this to be different from the eos token
        )
        tokenizer.padding_side = "left"  # Allow batched inference

    model.cuda()
    model.eval()
    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=args.cutoff_len,
            padding=False,
            return_tensors=None,
        )

        return result

    def generate_and_tokenize_prompt(data_point):
        def get_answer_prompt(example):
            context = CONTEXT.format(context=data_point['context']) if 'context' in data_point else None
            question = QUESTION.format(question=data_point['question'])
            answer = ANSWER.format(answer=example['answer'])

            if 'bert' in args.model_path or 'roberta' in args.model_path or 'deberta' in args.model_path:
                input_prompt = "\n".join([question, answer]) if context is None else "\n".join([context, question, answer])
            else:
                input_prompt = "\n".join([question, answer]) if context is None else "\n".join([context, question, answer])
            tokenized_full_prompt = tokenize(input_prompt)
            return tokenized_full_prompt
        
        # postive + negative
        pos_answer = data_point['pos_answer']
        neg_answer = data_point['neg_answer']
        pos_tokenized_full_prompt = get_answer_prompt(pos_answer)
        neg_tokenized_full_prompt = get_answer_prompt(neg_answer)
        # label
        pos_score, neg_score = pos_answer['score'], neg_answer['score']
        tokenize_res = {key: [pos_tokenized_full_prompt[key], neg_tokenized_full_prompt[key]] for key in pos_tokenized_full_prompt}
        tokenize_res['label'] = [pos_score, neg_score]
        return tokenize_res

    if args.data_path.endswith(".json") or args.data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=args.data_path)
    else:
        data = load_dataset(args.data_path)
    val_data = (
            data["test"].map(generate_and_tokenize_prompt)
        ) 

    # remove ununsed columns
    val_data = val_data.select_columns(['input_ids', 'attention_mask', 'label'])
    val_dataloader = DataLoader(val_data, batch_size=args.eval_batch_size, shuffle=False, \
                                collate_fn=RewardDataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True))
    
    labels, preds = [], []
    mean_scores, pred_scores = [], []
    with torch.no_grad():
        for batch in tqdm(val_dataloader, total=len(val_dataloader)):
            batch = {key: val.cuda() for key, val in batch.items()}
            bs_label = batch.pop('label')
            bs = bs_label.shape[0] // 2
            pos_generation_output = model(**batch)

            # evaluation
            label = torch.stack(bs_label.split(bs), dim=1)
            cm_label = label[:, 0].gt(label[:, 1])
            labels.extend(cm_label.cpu().detach().tolist())

            pred = pos_generation_output.logits
            pred_scores.extend(pred.cpu().detach().tolist())
            cm_pred = pred[:, 0].gt(pred[:, 1])
            preds.extend(cm_pred.cpu().detach().tolist())

            mean_scores.append(pred.mean().item())
    
    acc = accuracy(labels, preds)
    print("Accuracy: ", acc)
    print("Macro: ", f1_micro(labels, preds))
    print('Mean score: ', sum(mean_scores) / len(mean_scores))

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)
    np.save(os.path.join(args.output_path, f'pred_scores.npy'), pred_scores)

if __name__ == '__main__' :
    main()