from llm import LlamaLLM, ChatGPT
import torch
import json
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
import numpy as np
from Train_KT_Agent import E_DKT
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LocalPlanner:
    def __init__(self, llm=None, tokenizer=None,dataname="junyi", main_data_path="../data_o",chat_model=None,q_text=True, reflection=True):
        super().__init__()
        self.llm = LlamaLLM(model=llm,tokenizer=tokenizer) if llm is not None else ChatGPT(modelname=chat_model)
        self.main_data_path = main_data_path
        self.teacher_setting=f"You're a seasoned math teacher with ten years of teaching experience."
        self.learning_steps = 1
        self.dataname = dataname
       
        if self.dataname == "assist" or self.dataname == "junyi" or dataname == "xunfei":
            with open(self.main_data_path+f"/{dataname}/K_Directed.txt", "r") as file:
                lines = file.readlines()
            with open(self.main_data_path+f"/{dataname}/k2text.json", 'r') as f:
                self.k2text = json.load(f)
            with open(self.main_data_path+f"/{dataname}/k2q.json", 'r') as f:
                self.k2q = json.load(f)
            with open(self.main_data_path+f"/{dataname}/q2k.json", 'r') as f:
                self.q2k = json.load(f)
            self.q2text = json.load(open(self.main_data_path+f"/{dataname}/q2text.json", 'r'))
            
        else:
            print("only junyi / assist / xunfei are supported")
            raise NotImplementedError
        
        self.k_graph = [tuple(map(int, line.strip().split())) for line in lines]
        self.QTEXT = q_text
        self.REFLECTION = reflection
        self.edutools = False

    def find_one_hop_predecessors(self, node):
        one_hop_predecessors = set([edge[0] for edge in self.k_graph if edge[1] == node])
        return one_hop_predecessors

    def given_advise(self, history_text, history_qid, student_profile, learning_knowledge_id,recommend_reflection, state, goal_state):
        
        goal_state = round(goal_state,2)
        learning_goal = self.k2text[str(learning_knowledge_id)]
        if self.edutools == True:
            one_hop_predecessors = self.find_one_hop_predecessors(learning_knowledge_id)

            threhold = 0.6
        
            if self.QTEXT:
                candidate_questions_about_neighbor_knowledge = [{'question_id':q, 'question_text':self.q2text[str(q)]} for k in one_hop_predecessors for q in self.k2q[k] if q not in history_qid and state[int(q)]<threhold]
                candidates = [{'question_id':q, 'question_text':self.q2text[str(q)]} for q in self.k2q[str(learning_knowledge_id)] if q not in history_qid and state[int(q)]<threhold and state[int(q)] > 0.3]
            else:
                learning_goal = learning_knowledge_id
                candidate_questions_about_neighbor_knowledge = [{'question_id':q } for k in one_hop_predecessors for q in self.k2q[k] if q not in history_qid and state[int(q)]<threhold ]
                candidates = [{'question_id':q} for q in self.k2q[str(learning_knowledge_id)] if q not in history_qid and state[int(q)]<threhold and state[int(q)] > 0.3 ]
                

            if len(candidates) == 0:
                candidates = candidate_questions_about_neighbor_knowledge
                    
                if len(candidates) == 0:
                    if self.QTEXT:
                        candidates = [{'question_id':q, 'question_text':self.q2text[str(q)]} for q in self.k2q[str(learning_knowledge_id)] if state[int(q)] < 0.8 ] + [{'question_id':q, 'question_text':self.q2text[str(q)]} for k in one_hop_predecessors for q in self.k2q[k] if q not in history_qid and state[int(q)] < threhold]
                    else:
                        candidates = [{'question_id':q } for q in self.k2q[str(learning_knowledge_id)] if state[int(q)] < 0.8 ] + [{'question_id':q } for k in one_hop_predecessors for q in self.k2q[k] if q not in history_qid and state[int(q)] < threhold ]
                else:
                    return -1
        else:
            candidates_qid = random.sample(self.q2text.keys(), 10)
            candidates = [{'question_id':q, 'question_text':self.q2text[str(q)]} for q in candidates_qid]
       
        if len(candidates) > 10:
            candidates = random.sample(candidates, 10) 
            
        if self.REFLECTION:
            advise_prompt=f"Given the following history text: {history_text[-5:]} and the recommand reflection {recommend_reflection}, the student profile: {student_profile}, the knowledge learning goal: {learning_goal}. Here are the candidate question list: {candidates}. Please provide the most suitable question from above list, that can help the student to achieve the learning goal efficiently. For example, the output format should be :['question_id': 'xxx', 'recommand_reason': 'recommand reason details', 'predict_answer': 'True' or 'False'], except this format, please do not output anything."
        else:
            advise_prompt=f"Given the following history text: {history_text[-5:]}, the student profile: {student_profile}, the knowledge learning goal: {learning_goal}. Here are the candidate question list: {candidates}. Please provide the most suitable question from above list, that can help the student to achieve the learning goal efficiently. For example, the output format should be :['question_id': 'xxx'], except this format, please do not output anything."
            
        advise=self.llm(self.teacher_setting+advise_prompt)
        
        return advise, self.teacher_setting+advise_prompt
    
    def judge_answer(self, question, answer_content, ground_truth):
        judge_prompt=f"The question is [{question}]. Given the answer content: {answer_content}, the ground truth: {ground_truth}, please provide the judgement about the correctness of the answer. For example, the output format should be:['judge_result': 'True' or 'False']. "
        judge=self.llm(self.teacher_setting+judge_prompt)
        return judge

