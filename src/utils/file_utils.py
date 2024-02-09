import json

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)
    
def load_txt(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip().replace('\n', '') for line in lines]
    return lines
    
def load_json_data_by_line(path):
    with open(path, 'r') as f:
        webgpt_data = []
        lines = f.readlines()
        for line in lines:
            webgpt_data.append(json.loads(line))
    return webgpt_data


def write_json_data_by_line(path, data):
    with open(path, 'w') as f:
        for line in data:
            f.writelines(json.dumps(line) + '\n')


def write_txt_data(path, data):
    with open(path, 'w') as f:
        for line in data:
            f.writelines(line + '\n')

def load_gpt_results(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        return {eval(line)['id']: eval(line)  for line in lines}