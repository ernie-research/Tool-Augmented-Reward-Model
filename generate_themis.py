
import re
import math
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import Dataset, load_dataset
from dataclasses import dataclass
from typing import NamedTuple, Union
from transformers import AutoConfig, LlamaTokenizer, GenerationConfig, set_seed
from src.models.reward_model import RewardModel
from src.tools import calculator, Calendar, get_code_interperter, BaiduTranslator, history_weather, wiki_search, GoogleSerperAPIWrapper
from src.utils.file_utils import load_json_data_by_line, write_json_data_by_line, load_json
from src.utils.metrics import accuracy, f1_micro
from src.data.reward_dataset import RewardDataCollatorForSeq2Seq, RewardDataCollatorForGenerate
import deepspeed
from multiprocessing import Process, Pipe, Pool
from src.template.instruction_template import CONTEXT, QUESTION, ANSWER, TOOL, OBSERVATION, WORK

import os
os.environ["SERPER_API_KEY"] = ""   # INPUT your key of google-serper https://serper.dev/

# cache
HISTORY_DATA ={
    'weather': load_json('data/weather/weather_history.json'),
    'translator': load_json('data/translator/translate_history.json'),
} 
code_eval = get_code_interperter()

# run tools and get observation
@dataclass
class AgentAction:
    """Agent's action to take."""
    tool: str
    tool_input: Union[str, dict]
    log: str

class AgentFinish(NamedTuple):
    """Agent's return value."""
    return_values: dict
    log: str
@dataclass
class AgentWork:
    work: str
    log: str

def parse_actions(text):
    FINAL_ANSWER_ACTION = "Finished"
    WORK_ACTION = "Work:"
    WORK_START_ACTION = "<start_work>"
    includes_answer = FINAL_ANSWER_ACTION in text
    include_work = WORK_ACTION in text or WORK_START_ACTION in text
    regex = (
        r"Action\s*\d*\s*:[\s]*(.*?)[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
    )
    action_match = re.search(regex, text, re.DOTALL)
    if action_match:
        action = action_match.group(1).strip()
        action_input = action_match.group(2)
        tool_input = action_input.strip(" ")
        # ensure if its a well formed SQL query we don't remove any trailing " chars
        if tool_input.startswith("SELECT ") is False:
            tool_input = tool_input.strip('"')

        return AgentAction(action, tool_input, text)
    
    elif include_work:
        work = re.sub('<start_work>|Work:', '', text).strip()
        return AgentWork(work, text)

    elif includes_answer:
        return AgentFinish(
                {"output": text.split(FINAL_ANSWER_ACTION)[-1].strip()}, text
            )
    else:
        return None

def parse_stop_part(generation_tokens):
    # new:
    if '<start_observation>' in generation_tokens:
        return generation_tokens[:generation_tokens.index('<start_observation>')]
    # old: 
    stops = ['\nObservation:', '\n\tObservation:']
    for stop in stops:
        action_math = re.search(stop, generation_tokens)
        if action_math:
            return generation_tokens[:action_math.span()[0]]
    return generation_tokens

# all tools 
def calculation_api(tool_inputs, input_text):
    compute_process = tool_inputs.split(', ')
    re_answer = re.findall('####[ *]\d.*', input_text)
    answer = '' if len(re_answer) == 0 else re.findall('\d+[.]*\d*', re_answer[0])[0]
    if answer == '':
        return 'No valid answer.'
    answer = float(answer)
    error_list = []
    final_answer = None
    try:
        for cpro in compute_process:
            cur_ans = re.sub('<<.*>>', '', cpro)
            equation = re.sub('<<|=.*', '', cpro)
            cal_res = calculator(equation)
            if 'Error' in cal_res:
                error_list.append(cal_res)
                continue
            cal_res = cal_res.item()
            if cur_ans != '' and cal_res != float(cur_ans):
                error_list.append(f'{equation} is not equal to {cur_ans}.')
            final_answer = cal_res
    except Exception as e:
        print(e)
        error_list = ['An unexpected error occurred during the calculation']
    
    if len(error_list) == 0:
        if final_answer == answer:
            return 'Both the calculation and the answer are correct.'
    
        return f'The calculation process is correct, but the resulting answer {final_answer} does not match the predicted answer {answer}.'
    else:
        error_str = 'An unexpected error occurred during the calculation.'
        if final_answer == answer:
            return f'The calculation process is incorrect, but the answer matches the predicted answer. Details: {error_str}'
        
        answer_str = f'The calculated answer {final_answer} does not match the predicted answer {answer}.'
        return f'Both the calculation and the answer are incorrect. Details: {error_str} {answer_str}'

def serper_api(tool_inputs, input_text=None):
    # mini-batch
    serper = GoogleSerperAPIWrapper()
    # observation = serper.mini_batch_run(tool_inputs)
    observation = serper.run(tool_inputs)
    return observation

def code_api(answer, test_list):
    result = ''
    predictions = [[answer]] * len(test_list)
    references = test_list
    code_eval_res, case_res = code_eval.compute(predictions=predictions, references=references)
    pass_1 = code_eval_res['pass@1']
    result += f'The pass rate is {pass_1}. '
    
    n_passed, n_failed = 0, 0
    failed_reason = set()
    for key in case_res:
        if case_res[key][0][1]['passed']:
            n_passed += 1
        else:
            failed = case_res[key][0][1]['result'].replace('failed:', "").strip()
            if len(failed) > 0:
                failed_reason.add(failed)
            n_failed += 1

    failed_reason = " Failed reason: {}".format("; ".join(failed_reason)) if len(failed_reason) > 0 else ''
    if n_passed == 0:
        result += "All test cases failed." + failed_reason
    elif n_failed == 0:
        result += "All test cases passed."
    elif n_passed > 0 and n_failed > 0:
        result += f"{n_passed} test cases passed, and {n_failed} test cases failed." + failed_reason
    else:
        raise ValueError('Error in code interperter: ', n_passed, n_failed)
    return result

def calendar_api(tool_inputs, calendar_type):
   calendar = Calendar()
   if 'week_day' in calendar_type:
       return calendar.week_day(tool_inputs.strip())
   elif 'target_day' in calendar_type:
       date, diff = tool_inputs.split(',')
       return calendar.target_day(date.strip(), int(diff.strip()))
   elif 'day_difference' in calendar_type: 
       date1, date2 = tool_inputs.split(',')
       return calendar.day_difference(date1.strip(), date2.strip())

def translator_api(id, tool_inputs, input_text=None):
    translate_history = HISTORY_DATA['translator']
    if id in translate_history:
        return translate_history[id]['answer']
    translator = BaiduTranslator()
    return translator.get_translation(text=tool_inputs, source_lang='auto', tgt_lang='en')

 # for weather (cheat)

def weather_api(tool_inputs, input_text=None):
    weather_history = HISTORY_DATA['weather']
    if tool_inputs in weather_history:
        return weather_history[tool_inputs]
    try:
        city, date = tool_inputs.split(',')
        observation, output_dict = history_weather(city, date)
        return observation
    except:
        return 'None'

def wikisearch_api(tool_inputs, input_text=None):
    observation = wiki_search(tool_inputs, k=1)
    return observation[0]

TOOL_TO_API = {
        'calculator': calculation_api,
        'code_run': code_api,
        'translate': translator_api,
        'history_weather': weather_api,
        'wiki_search': wikisearch_api,
        'google_serper': serper_api
}

def get_tools_api(tool_name):
    if tool_name in TOOL_TO_API:
        return TOOL_TO_API[tool_name]
    if 'calendar' in tool_name.lower():
        return calendar_api

def collect_labels_preds(generation):
    labels, preds = [], []
    for example in generation:
        pos_score, neg_score = example['example']['pos_answer']['score'], example['example']['neg_answer']['score']
        label = True if pos_score > neg_score else False
        
        score_key = 'rm_score'
        if example['pos_generation'] and 'error' not in example['pos_generation']:
            score_key = 'score' if 'rm_score' not in example['pos_generation'] else 'rm_score'
        pos_pred_score = example['pos_generation'][score_key] if example['pos_generation'] and 'error' not in example['pos_generation']  else None
        neg_pred_score = example['neg_generation'][score_key] if example['neg_generation'] and 'error' not in example['neg_generation'] else None
        if pos_pred_score is not None and neg_pred_score is not None:
            pred = float(pos_pred_score) > float(neg_pred_score)
        else:
            pred = not label

        labels.append(label)
        preds.append(pred)
    return labels, preds

def invoke_run(data, index, size):
    size = math.ceil(len(data) / size)
    start = size * index
    end = (index + 1) * size if (index + 1) * size < len(data) else len(data)

    temp_data = data[start:end]
    res = []
    for example in temp_data:
        action, action_input, input_text = example['tool'], example['tool_input'], example['input_text']
        tool_api = get_tools_api(action)
        try:
            if 'calendar' in action:
                observation = tool_api(action_input, action)    # need tool_name
            elif 'translate' in action:
                id = example['id'][:-4]
                observation = tool_api(id, action_input, action)
            else:
                observation = tool_api(action_input, input_text)
        except Exception as e:
            print(f'Error during {action} tool: {e}')
            observation = 'An error occurred during the tool invoke, so no result was returned.'

        input_text += f'Observation: {observation}\n'
        if 'wiki' in action:    input_text += '<start_work> '
        res.append({'id': example['id'], 'input_text': input_text})
    return res

def mp_invoke_api(mp_invoke_data):
    processor = 5
    res, mp_res = [], []

    pbar = tqdm(total=processor)
    pbar.set_description('Multi Process Invoke')
    update = lambda *args: pbar.update()
    
    p = Pool(processor)
    for i in range(processor):
        res.append(p.apply_async(invoke_run, args=(mp_invoke_data, i, processor), callback=update))
    
    p.close()
    p.join()
    for i in res:
        mp_res.extend(i.get())

    assert len(mp_invoke_data) == len(mp_res)
    return mp_res

def batch_invoke_api(batch_invoke_data):
    batch_invoke_results = []
    webgpt_invoke_data = [example for example in batch_invoke_data if 'serper' in example['tool']]    # mini-batch
    
    if len(webgpt_invoke_data) > 0:
        mini_bsz = 40
        webgpt_invoke_results = []
        for i in tqdm(range(0, len(webgpt_invoke_data), mini_bsz), total=len(webgpt_invoke_data)//mini_bsz, desc='Google Serper Invoke'):
            webgpt_bsz_query = [example['tool_input'] for example in webgpt_invoke_data[i:i+mini_bsz]]
            tool_api = get_tools_api('google_serper')
            webgpt_invoke_results.extend(tool_api(webgpt_bsz_query))
        batch_invoke_results.extend({'id': webgpt_invoke_data[i]['id'], 'input_text': webgpt_invoke_data[i]['input_text']+f'Observation: {webgpt_invoke_results[i]}\n'}
                                     for i in range(len(webgpt_invoke_results)))
    
    return batch_invoke_results

def instance_invoke_api(instance_invoke_data):
    res = []
    for example in instance_invoke_data:
        action, action_input, input_text = example['tool'], example['tool_input'], example['input_text']
        tool_api = get_tools_api(action)
        try:
            if 'code' in action:
                id, suffix = example['id'][:-4], example['id'][-3:]
                observation = tool_api(id2example[id][f'{suffix}_answer']['answer'], id2example[id]['test_list'])
            elif 'serper' in action:
                observation = tool_api(action_input, input_text)
        except Exception as e:
            print(f'Error during {action} tool: {e}')
            observation = 'An error occurred during the tool invoke, so no result was returned.'

        input_text += f'Observation: {observation}\n'
        res.append({'id': example['id'], 'input_text': input_text})
    return res

def invoke_tools(invoke_data):
    invoke_results = []
    mp_invoke, batch_invoke, instance_invoke = [], [], []
    for instance in invoke_data:
        tool_name = instance['tool'].lower()
        if 'calculator' in tool_name or 'calendar' in tool_name or 'wiki' in tool_name or 'weather' in tool_name or 'translate' in tool_name:
            mp_invoke.append(instance)
        elif 'serper' in tool_name:
            # batch_invoke.append(instance)
            instance_invoke.append(instance)
        elif 'code' in tool_name:
            instance_invoke.append(instance)
        else:
            invoke_results.append({'id': instance['id'], 'input_text': instance['input_text']})
    
    if len(mp_invoke) > 0:
        mp_results = mp_invoke_api(mp_invoke)
        invoke_results.extend(mp_results)
    if len(instance_invoke) > 0:
        instance_results = instance_invoke_api(instance_invoke)
        invoke_results.extend(instance_results)
    if len(batch_invoke) > 0:
        batch_results = batch_invoke_api(batch_invoke)
        invoke_results.extend(batch_results)

    return invoke_results
    
def generate_next_step(generate_data):
    # generate_data: [{'id', 'input_text', 'generation_token'}]
    finished, itermediate = [], []
    action_invoke_data = []
    for instance in tqdm(generate_data, desc='Parse data'):
        id, input_text, generation_token = instance['id'], instance['input_text'], instance['generation_token']
        work = None
        parse_generation_token = parse_stop_part(generation_token)
        tool_actions = parse_actions(parse_generation_token)
        if isinstance(tool_actions, AgentWork):
            work = tool_actions.work
            input_text += tool_actions.log
            finished.append({'id': id, 'final_generate_tokens': input_text, 'work': work})
        elif isinstance(tool_actions, AgentFinish):
            # add <start_work>
            input_text += tool_actions.log + '\n<start_work> '
            itermediate.append({'id': id, 'input_text': input_text})
        elif isinstance(tool_actions, AgentAction):
            # <start_work>
            tool, tool_input = tool_actions.tool, tool_actions.tool_input.strip()
            input_text += parse_generation_token + '\n<start_observation> '
            action_invoke_data.append({'id': id, 'tool': tool, 'tool_input': tool_input, 'input_text': input_text})
        else:
            input_text += parse_generation_token
            finished.append({'id': id, 'final_generate_tokens': input_text, 'work': work})

    itermediate.extend(invoke_tools(action_invoke_data))
    return finished, itermediate

def merge_finished_data(finished_data):
    generate_examples_dict = {}
    for example in tqdm(finished_data, desc='Merge pos and neg'):
        id, final_generate_tokens, work = example['id'], example['final_generate_tokens'], example['work']
        id, suffix = id[:-4], id[-3:]   # pos or neg?
        if id not in generate_examples_dict:
            generate_examples_dict[id] = {'pos_generation': None, 'neg_generation': None}
        if 'rm_score' in example:
            generate_examples_dict[id][f'{suffix}_generation'] = {
                'final_generate_tokens': final_generate_tokens,
                'rm_score': example['rm_score'],
                'work': work,
            }
        else:
            generate_examples_dict[id][f'{suffix}_generation'] = {
                'final_generate_tokens': final_generate_tokens,
                'work': work,
            }

    # convert to list
    generate_examples = []
    for id in generate_examples_dict:
        generate_examples.append({'id': id, 'example': id2example[id], 'pos_generation': generate_examples_dict[id]['pos_generation'], 
                                'neg_generation': generate_examples_dict[id]['neg_generation']})

    return generate_examples

def pprint_rank(msg, rank=0):
    if rank <= 0:
        print(msg)

def batch_generate(val_data, tokenizer, model, generation_config, global_rank, args):
    def tokenize(prompt, add_eos_token=False):
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
    
    def generate_and_tokenize_intermediate(data_point):
        id, input_text = data_point['id'], data_point['input_text']
        tokenized_full_prompt = tokenize(input_text)
        return {'id': id, 'input_ids': tokenized_full_prompt['input_ids'], 'attention_mask': tokenized_full_prompt['attention_mask']}
    
    finished_data, intermediate_data, interaction = [], [], 0
    finish_len, intermediate_len = torch.Tensor([0]).cuda(), torch.Tensor([0]).cuda()
    while interaction < args.max_iteractions:
        torch.cuda.empty_cache()
        if interaction == 0:
            pprint_rank(f'Total data: {len(val_data)*2}', global_rank)
            val_dataloader = DataLoader(val_data, batch_size=args.eval_batch_size, shuffle=False, \
                                    collate_fn=RewardDataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True))
        else:
            pprint_rank(f'Total data: {len(intermediate_data)}', global_rank)
            intermediate_data = Dataset.from_list(intermediate_data)
            intermediate_data = intermediate_data.map(generate_and_tokenize_intermediate)
            intermediate_data = intermediate_data.select_columns(['id', 'input_ids', 'attention_mask'])
            val_dataloader = DataLoader(intermediate_data, batch_size=args.eval_batch_size*2, shuffle=False, \
                                    collate_fn=RewardDataCollatorForGenerate(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True))
            intermediate_data = []
        pprint_rank(f'Interaction {interaction}, Total dataloader: {len(val_dataloader)}', global_rank)

        # stop when generate \nObservation:
        # sentinel_token_ids = tokenizer.convert_tokens_to_ids("<start_observation>")
        with torch.no_grad():
            for batch in tqdm(val_dataloader, total=len(val_dataloader)):
                if 'ids' in batch: ids = batch.pop('ids')
                batch = {key: val.cuda() for key, val in batch.items()}
                input_len = batch['input_ids'].size(1)

                generation_output = model.generate(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            generation_config=generation_config,
                            return_dict_in_generate=True,
                            max_new_tokens=args.max_new_token)
                
                input_text = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=False)
                generation_tokens = tokenizer.batch_decode(generation_output.sequences[:, input_len:], skip_special_tokens=False)

                intermediate_data.extend({'id': ids[i], 
                                        'input_text': input_text[i].replace('<unk>', '').replace('<s>', ''), 
                                        'generation_token': generation_tokens[i].replace('<unk>', '').replace('</s>', '')} for i in range(len(generation_tokens)))

            # save to file
            if global_rank <= 0:
                finished, intermediate_data = generate_next_step(intermediate_data)
                finished_data.extend(finished)
                torch.save(finished_data, 'temp/finished_data.pt')
                torch.save(intermediate_data, 'temp/intermediate_data.pt')
            torch.distributed.barrier()

            if not global_rank <= 0:
                finished_data = torch.load('temp/finished_data.pt')
                intermediate_data = torch.load('temp/intermediate_data.pt')
            torch.distributed.barrier()

            interaction += 1
            if len(intermediate_data) == 0:
                break
            pprint_rank(f"=====Intermediate data, Interaction {interaction}=====", global_rank)
            pprint_rank(intermediate_data[-1], global_rank)
            if global_rank <= 0 and len(finished) > 0:
                pprint_rank("=====Finished data=====", global_rank)
                pprint_rank(finished[-1], global_rank)

    return finished_data, intermediate_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='path to dataset')
    parser.add_argument('--model_path', type=str, help='path to model')
    parser.add_argument('--output_path', type=str, help='path to model')
    parser.add_argument('--checkpoint_path', type=str, help='path to checkpoint')
    parser.add_argument('--eval_batch_size', type=int, default=4, help='eval_batch_sizes')
    parser.add_argument('--cutoff_len', type=int, default=2048, help='cutoff length')
    parser.add_argument('--max_new_token', type=int, default=1024, help='cutoff length')
    parser.add_argument('--max_iteractions', type=int, default=3, help='max iteractions')
    parser.add_argument('--device', type=str, default='cuda', help='device type')
    
    parser.add_argument('--ranking', action='store_true', help='reward type')
    parser.add_argument('--ranking_way', type=str, default='last', help='reward compute type')
    parser.add_argument('--invoke_tool', action='store_true', help='whether invoke tools')
    parser.add_argument('--add_special_tokens', action='store_true', default=True, help='do sample')
    parser.add_argument('--add_eos_token', action='store_true', default=True, help='do sample')
    
    parser.add_argument('--do_sample', action='store_true', help='do sample')
    parser.add_argument('--temperature', type=float, default=0.2, help='cutoff length')
    parser.add_argument('--top_p', type=float, default=0.95, help='cutoff length')
    parser.add_argument('--top_k', type=int, default=40, help='cutoff length')
    parser.add_argument('--num_beams', type=int, default=1, help='cutoff length')
    parser.add_argument('--repetition_penalty', type=float, default=1, help='cutoff length')
    parser.add_argument("--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", "0")), help="local rank")
    parser.add_argument("--world_size", type=int, default=int(os.getenv("WORLD_SIZE", "1")), help="world size")

    args = parser.parse_args()

    set_seed(42)
    deepspeed.init_distributed()

    global_rank = args.local_rank

    def _add_special_tokens(tokenizer):
        special_tokens_list = ['<start_tool>', '<start_observation>', '<start_work>']
        tokenizer.add_special_tokens({'additional_special_tokens': special_tokens_list})
        special_tokens2ids = {token: tokenizer.convert_tokens_to_ids(token) for token in special_tokens_list}
        return tokenizer, special_tokens_list
    
    # initialize config, model and tokenizer
    config = AutoConfig.from_pretrained(args.model_path)
    config.ranking = args.ranking
    config.ranking_way = args.ranking_way
    config.invoke_tool = args.invoke_tool

    tokenizer = LlamaTokenizer.from_pretrained(args.model_path)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    if args.add_special_tokens:
        tokenizer, special_tokens_list = _add_special_tokens(tokenizer)
        config.vocab_size = len(tokenizer)

    model = RewardModel.from_pretrained(
        args.checkpoint_path,
        config=config,
    )

    model.eval()

    # # deepspeed
    model = deepspeed.init_inference(
                    model=model,      # Transformers models
                    mp_size=8,        # Number of GPU
                    max_out_tokens=1024,
                    replace_method="auto", # Lets DS autmatically identify the layer to replace
                    replace_with_kernel_inject=False, # replace the model with the kernel injector
                )
    model.profile_model_time()

    generation_config = GenerationConfig(
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            num_beams=args.num_beams,
            repetition_penalty=args.repetition_penalty,
    )

    def tokenize(prompt, add_eos_token=False):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=args.cutoff_len,
            padding=False,
            return_tensors=None,
        )

        if result["input_ids"][-1] != tokenizer.eos_token_id \
                and len(result["input_ids"]) < args.cutoff_len and args.add_eos_token:
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        return result

    def generate_and_tokenize_prompt(data_point):
        def get_answer_prompt(example):
            context = CONTEXT.format(context=data_point['context']) if 'context' in data_point else None
            question = QUESTION.format(question=data_point['question'])
            answer = ANSWER.format(answer=example['answer'])
            
            input_prompt = "\n".join(["### USER:", question, answer]) if context is None else "\n".join(["### USER:", context, question, answer])
            tokenized_full_prompt = tokenizer(input_prompt+'\n### ASSISTANT:\n<start_tool> ')
            return tokenized_full_prompt
        
        # postive + negative
        pos_answer = data_point['pos_answer']
        neg_answer = data_point['neg_answer']
        pos_tokenized_full_prompt = get_answer_prompt(pos_answer)
        neg_tokenized_full_prompt = get_answer_prompt(neg_answer)
        return {key: [pos_tokenized_full_prompt[key], neg_tokenized_full_prompt[key]] for key in pos_tokenized_full_prompt}

    # multi dataset
    torch.cuda.empty_cache()    # for
    if args.data_path.endswith(".json") or args.data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=args.data_path)
    else:
        data = load_dataset(args.data_path)

    # construct id2example
    global id2example
    id2example = {}
    for example in data['test']:
        id2example[example['id']] = example

    val_data = (
            data["test"].map(generate_and_tokenize_prompt)
        ) 

    val_data = val_data.select_columns(['id', 'input_ids', 'attention_mask'])

    finished_data, intermediate_data = batch_generate(val_data, tokenizer, model, generation_config, global_rank, args)
    if len(intermediate_data) > 0:
        for example in intermediate_data:
            id, input_text = example['id'], example['input_text']
            finished_data.append({'id': id, 'final_generate_tokens': input_text, 'work': None})

    #  for lm + linear
    if args.ranking:
        def generate_and_tokenize_finished(data_point):
            id, input_text = data_point['id'], data_point['final_generate_tokens']
            tokenized_full_prompt = tokenize(input_text, add_eos_token=True)    # add </s>
            return {'id': id, 'input_ids': tokenized_full_prompt['input_ids'], 'attention_mask': tokenized_full_prompt['attention_mask']}

        pprint_rank(f'Total finished data: {len(finished_data)}', global_rank)
        batch_finished_data = Dataset.from_list(finished_data)
        batch_finished_data = batch_finished_data.map(generate_and_tokenize_finished)
        batch_finished_data = batch_finished_data.select_columns(['id', 'input_ids', 'attention_mask'])
        val_dataloader = DataLoader(batch_finished_data, batch_size=args.eval_batch_size, shuffle=False, \
                                collate_fn=RewardDataCollatorForGenerate(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True))

        f_id = 0
        with torch.no_grad():
            for batch in tqdm(val_dataloader):
                if 'ids' in batch:  ids = batch.pop('ids')
                batch = {key: val.cuda() for key, val in batch.items()}
                reward_scores = model(
                                input_ids=batch['input_ids'],
                                attention_mask=batch['attention_mask']).rewards
                for score in reward_scores:
                    finished_data[f_id]['rm_score'] = score.cpu().detach().item()
                    f_id += 1

    # merge pos and neg
    generate_examples = merge_finished_data(finished_data)

    # compute metrics
    labels, preds = collect_labels_preds(generate_examples)
    acc = accuracy(labels, preds)
    pre, recall, f1 = f1_micro(labels, preds)
    pprint_rank(f'Accuracy: {acc}', global_rank)
    pprint_rank(f'Precision: {pre}, Recall: {recall}, F1 score: {f1}', global_rank)

    if global_rank <= 0:
        with open('output/results.txt', 'a') as f:
            f.writelines(f'{args.checkpoint_path}, {args.data_path}, acc: {acc}\n')
        
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path, exist_ok=True)
        dataset = args.data_path.split('/')[-1]
        args.output_path = os.path.join(args.checkpoint_path, f'{dataset}_generation.json')

if __name__ == '__main__' :
    main()