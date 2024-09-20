from typing import Union, Literal
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import OpenAI
from langchain.schema import (
    HumanMessage
)
import os
from openai import OpenAI
import openai
import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer, PreTrainedTokenizerFast

import logging

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_llama2(model_dir="/data/share_weight/Llama-2-7b-chat-hf"):
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        if "transformers" in logger.name.lower():
            logger.setLevel(logging.ERROR)
    model = LlamaForCausalLM.from_pretrained(model_dir, device_map = 'auto')	
    tokenizer = LlamaTokenizer.from_pretrained(model_dir)	
    return model, tokenizer

def get_llm(model_dir="/data/share_weight/Meta-Llama-3-8B-Instruct"):
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        if "transformers" in logger.name.lower():
            logger.setLevel(logging.ERROR)
    model = LlamaForCausalLM.from_pretrained(model_dir, device_map = 'auto')	
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir,legacy=False)	
    return model, tokenizer

class LlamaLLM:
    def __init__(self, model_dir="/data/share_weight/Llama-2-7b-chat-hf", model=None, tokenizer=None, *args, **kwargs):

        if model is None:
            self.model = LlamaForCausalLM.from_pretrained(model_dir, device_map = 'auto')	
            self.tokenizer = LlamaTokenizer.from_pretrained(model_dir,legacy=False)	
        else:
            self.model = model
            self.tokenizer = tokenizer
            
        
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
            
        )
        self.terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
    def __call__(self, prompt: str):
        messages = [
            
            {"role": "user", "content": prompt},
        ]
        return self.pipeline(
                    messages,
                    max_new_tokens=1024,
                    eos_token_id=self.terminators,
                    do_sample=True,
                    temperature=0.9,
                    top_p=0.9,
                )[0]["generated_text"][-1]['content']
        
    def generate_responses(self, prompts):
        return [self(prompt) for prompt in prompts]


class ChatGPT:
    def __init__(self, modelname="gpt-3.5-turbo-1106"):
    
        self.client = OpenAI(
            api_key="xxx",
        )
        self.client.base_url="xxx"
        self.modelname = modelname
    def __call__(self, prompt: str):
        client = self.client
        response = client.chat.completions.create(
            messages=[
            {"role": "user", "content": prompt}
            ],
            model=self.modelname ,
            temperature=0.9,
            max_tokens=1024,
        )
        return response.choices[0].message.content

if __name__ == '__main__':
    
    
    llm = ChatGPT()
    print(llm("You're a seasoned math teacher with multiple years of teaching experience.Please use one sentence to summarize the student's learning ability from the following learning log: [{'question': '三角形ABC a b c=3 5 7 三角形 中最 大角', 'answer': 'True', 'select_reason and predict answer': ''}, {'question': '三角形ABC AB=3BC cosA=frac2sqrt23 cosB=', 'answer': 'True', 'select_reason and predict answer': ''}, {'question': 'abc 分别为 三角形ABC 内角 ABC 对边 a=3 b=4 C=60^circ c=', 'answer': 'True', 'select_reason and predict answer': ''}]"))