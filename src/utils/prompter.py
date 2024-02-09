"""
A dedicated helper to manage templates and prompt building.
"""

import json
import os.path as osp
from typing import Union
from ..template.instruction_template import INSTRUCTION, OUTPUT, CONTEXT_INSTRUCTION, OUTPUT_WO_EXPLANATION


class Prompter(object):
    __slots__ = ("template", "action_template", "default_action", "_verbose", "args")

    def __init__(self, template_name: str = "", verbose: bool = False, args=None):
        self._verbose = verbose
        self.args = args
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "vicuna_tool"

        self.template = {'input': INSTRUCTION, 'output': OUTPUT, 'output_wo_explanation': OUTPUT_WO_EXPLANATION, 'context_input': CONTEXT_INSTRUCTION}
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template}"
            )
        self.action_template = 'Thought: {}\nAction: {}\nAction Input: {}\nObservation: {}'
        self.default_action = 'Thought: Do I need to use a tool? No\nFinished.'

    def generate_prompt(
        self,
        question: str,
        context: str=None,
        answers = None,
        actions = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if self.reward_type == 'linear':    #  pair-wise ranking loss
            prompt = 'Question: {}\n'.format(question)
            if context:
                prompt = 'Context: {}\n'.format(context) + prompt
            answer = answers['answer']
            res = prompt + 'Answer: {}'.format(answer)
            return res

        answer = answers['answer']
        res = self.template['input'].format(question=question, answer=answer)
        if context is not None:
            res = self.template['context_input'].format(context=context, question=question, answer=answer)
        
        label = None
        if actions:
            # actions
            if self.wo_explanation:
                label = self.template["output_wo_explanation"].format(actions=actions)
            else:
                label = self.template["output"].format(
                    actions=actions, score=answers['score'], explanation=answers['score_agent']['explanation'],
                )

        if label:
            res = res + '\n' + label
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["### ASSISTANT:"])[1].strip()
