from llm import LlamaLLM,ChatGPT
import torch
import json
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
import numpy as np
from Train_KT_Agent import E_DKT

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GlobalPlanner:
    def __init__(self, llm=None, tokenizer=None, dataname="junyi", mem_file_path="../data_o/", KT_model_path="../pretrained_models/KES_junyi/Trained_KES_KT_model.pt", bert_path = "../pretrained_models/bert_base_chinese", chat_model="gpt-3.5-turbo-1106",q_text=True, reflection=True):
        super().__init__()
        self.mem_filepath = mem_file_path+dataname+"_memory.json"

        self.llm = LlamaLLM(model=llm,tokenizer=tokenizer) if llm is not None else ChatGPT(modelname=chat_model)
        self.cold_num = 5
        self.memory = {
            'recommend_reflection':'', 
            'history_log': {'question_id':[], 'answer_bi': []}, 
            'history_log_text': [], 
            'student_learning_ability':'',
            'student_learning_preference':''
        }
        self.teacher_setting=f"You're a seasoned math teacher with multiple years of teaching experience."
        self.bert = BertModel.from_pretrained(bert_path).to(device)
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.dataname = dataname
        self.QTEXT = q_text
        self.REFLECTION = reflection
        if dataname == "assist" or dataname == "junyi" or dataname =="TextLog":
            self.k_num = {
                'assist':111,
                'junyi': 41,
                'TextLog':698
            }
            with open(f"../data_o/{dataname}/k2text.json",'r') as f:
                self.k2text = json.load(f)
            with open(f"../data_o/{dataname}/k2q.json",'r') as f:
                self.k2q = json.load(f)
            with open(f"../data_o/{dataname}/q2k.json",'r') as f:
                self.q2k = json.load(f)
            with open(f"../data_o/{dataname}/q2text.json",'r') as f:
                self.q2text = json.load(f)
            self.model = torch.load(f"../pretrained_models/KES_{dataname}/Trained_KES_KT_model.pt").to(device)
            self.model.eval()
        elif dataname == "kss":
            self.k_num = 10
            self.k2text = {str(i): f"第{i}个知识点" for i in range(10)}
            self.k2q = {str(i): [i] for i in range(10)}
            self.q2k = {str(i): i for i in range(10)}
            self.q2text = {str(i): f"第{i}个题目" for i in range(10)}
            self.model = torch.load(f"../pretrained_models/KES_kss/Trained_KSS_KT_model.pt").to(device)
            self.model.eval()
        else:
            print('only junyi / assist / TextLog  supported')
            raise NotImplementedError

    
    def clear_memory(self, cold_num,):
        self.memory = {
            'recommend_reflection':'', 
            'history_log': {'question_id':self.memory['history_log']['question_id'][:cold_num], 'answer_bi': self.memory['history_log']['answer_bi'][:cold_num]}, 
            'history_log_text': self.memory['history_log_text'][:cold_num], 
            'student_learning_ability':'',
            'student_learning_preference':''
        }
    
    def get_knowledge_state_by_KT(self, goal_id):
        question =self.memory['history_log']['question_id']

        answer = torch.tensor([self.memory['history_log']['answer_bi']]).to(device)
       
        print(f'used to test learning state: length:{len(question), len(answer)}, question {question}, answer {answer}')

        question_learning_state = self.model(torch.tensor([question]).to(device), answer).to('cpu').squeeze(0)[-1]
        
        question_list = self.k2q[goal_id]

        learning_state = np.mean([question_learning_state[int(i)].item() for i in question_list])

        return question_learning_state, learning_state
        

    def update_history_log(self, question_id, answer_bi, question_content="", answer_content="", select_reason="",predict_answer=""):
        try:
            self.memory['history_log']['question_id'].append(int(question_id))
            self.memory['history_log']['answer_bi'].append(int(answer_bi))
            item = {}
            if self.QTEXT:
                item['question'] = question_content
                item['answer'] = answer_content
            if self.REFLECTION:
                item['select_reason'] = select_reason
                item['predict_answer'] = predict_answer
            self.memory['history_log_text'].append(item)
        except:
            print(f'ERROR IN HERE: question_id = {question_id}')
        
    def reflection(self):
        if len(self.memory['history_log_text']) > self.cold_num:
            self.memory['student_learning_ability'] = self.llm(self.teacher_setting+"Please use one sentence to summarize the student's learning ability from the following learning log: "+str(self.memory['history_log_text'][-3:]))
            self.memory['student_learning_preference'] = self.llm(self.teacher_setting+"Please use one sentence summarize the student's learning preference from the following learning log: "+str(self.memory['history_log_text'][-3:]))
            self.memory['recommend_reflection'] = self.llm(self.teacher_setting+"Please use one sentence to reflect the method of  recommand questions: "+str(self.memory['history_log_text'][-3:]))

    def get_memory(self):
        return self.memory

    def save_memory(self):
        with open(self.mem_filepath, 'w') as f:
            json.dump(self.memory, f)
        return self.mem_filepath

    def get_history_text(self):
        if self.QTEXT:
            return [('question:'+str(self.memory['history_log_text'][i]['question']), 'leaner answer results:'+self.memory['history_log_text'][i]['answer']) for i in range(len(self.memory['history_log_text']))]
        else:
            return [('question:'+str(self.memory['history_log']['question_id'][i]), 'leaner answer results:'+str(self.memory['history_log']['answer_bi'][i])) for i in range(len(self.memory['history_log']))]
        
