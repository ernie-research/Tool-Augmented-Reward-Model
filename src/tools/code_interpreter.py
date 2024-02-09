

# human + mbpp
from evaluate import load
import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1" 

def get_code_interperter():
    code_eval = load("code_eval")
    return code_eval