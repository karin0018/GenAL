from llm import LlamaLLM, get_llm, get_llama2, ChatGPT
import json
from tqdm import tqdm
from agent_global import GlobalPlanner
from agent_local import LocalPlanner
import random
import numpy as np
from Train_KT_Agent import DKT
import re
import warnings
import sys
import argparse

warnings.filterwarnings("ignore")

def gen(dataname, model,max_step, reflection=True,q_text=True, llm=None, tokenizer = None):
    
    COLD_NUM=30 
    MASK_NUM=10 
    GOAL_NUM=10 
    MAX_STEP=max_step
    CHAT_MODEL=model
    dataname= dataname
    modelpath = "../pretrained_models/KES_"+dataname
    datapath = f"../data_o/{dataname}"
    judge_llm_answer = False
    E_all_e = 0
    E_all_s = 0
    E_all_sup = 0
    with open(modelpath+"/validate.json", "r") as f:
        validate_learner = json.load(f)
        questions = validate_learner["ques"][:100]
        answers = validate_learner["ans"][:100]
    
    if dataname =="kss":
        k2text = {str(i): f"第{i}个知识点" for i in range(10)}
        k2q = {str(i): i for i in range(10)}
        q2k = {str(i): i for i in range(10)}
    else:
        with open(datapath+"/k2text.json", "r") as f:
            k2text = json.load(f)
        with open(datapath+"/q2k.json", "r") as f:
            q2k = json.load(f)
        with open(datapath+"/k2q.json", "r") as f:
            k2q = json.load(f)
            
    all_stu_lr_state4everygoal = {}

    goal_id_list = list(random.sample(range(len(k2text)), GOAL_NUM))
    students = list(random.sample(range(len(questions)), GOAL_NUM))
        
        
    with open(f"{modelpath}/{dataname}_{model}_{max_step}_reflection{reflection}_qtext{q_text}_wo.txt","w") as f:
        try:
            for stu in tqdm(students):
            
                initial_ques_log = questions[stu][:COLD_NUM]
                initial_ans_log = answers[stu][:COLD_NUM]
                all_stu_lr_state4everygoal[stu] = []
                
                global_planner = GlobalPlanner(llm, tokenizer,dataname=dataname,chat_model=CHAT_MODEL,q_text=q_text, reflection=reflection)
                local_planner = LocalPlanner(llm, tokenizer,dataname,chat_model=CHAT_MODEL,q_text=q_text, reflection=reflection)
                print("=="*50) 
                print(f"stu:{stu}, initial_ques_log:{initial_ques_log}, initial_ans_log:{initial_ans_log}")
                f.write(f"stu:{stu}, initial_ques_log:{initial_ques_log}, initial_ans_log:{initial_ans_log}")
                    
                
                for i in range(COLD_NUM):
                    global_planner.update_history_log(initial_ques_log[i], initial_ans_log[i], local_planner.q2text[str(initial_ques_log[i])], "correct" if int(initial_ans_log[i]) == 1 else "wrong")
                    
                print('initial done!')
                
                for goal_id in goal_id_list:
                    goal_id = str(goal_id)
                    
                    learning_state = []
                    global_planner.clear_memory(COLD_NUM)
                    
                    print("=="*20)
                    print(f"stu:{stu}, goal_id:{goal_id}, goal_text:{local_planner.k2text[str(goal_id)]}")
                    f.write(f"stu:{stu}, goal_id:{goal_id}, goal_text:{local_planner.k2text[str(goal_id)]}")
                    
                    for step in range(MAX_STEP):
                        
                        all_learning_state, E_s_learning_state = global_planner.get_knowledge_state_by_KT(goal_id)
                        
                        print(f"step = {step}, maxstep = {MAX_STEP}, E_t = {E_s_learning_state}")
                        f.write("calculate E_s = " + str(E_s_learning_state))
                        
                        learning_state.append(E_s_learning_state)
                        
                        if reflection:
                            global_planner.reflection()
                            
                        memory = global_planner.get_memory()
                 
                        history_text = global_planner.get_history_text()
                        
                        student_profile = {'student_learning_ability': memory['student_learning_ability'], 'student_learning_preference': memory['student_learning_preference'], 'recommend_reflection': memory['recommend_reflection']}
                        
                        print(f"get student profile, {student_profile}")
                        
                        f.write(f"get student profile, {student_profile}")
                        
                        
                        advise, prompt = local_planner.given_advise(history_text, memory['history_log']['question_id'], student_profile, goal_id, memory['recommend_reflection'], all_learning_state, E_s_learning_state)
                        
                        if advise == -1:
                            all_learning_state, E_t = global_planner.get_knowledge_state_by_KT(goal_id)
                            learning_state.append(E_t)
                            break
                        
                        print('get prompt:')
                        print(prompt)
                        print(advise)
                        
                        max_iter=5
                        iter_n=0
                        break_flag = 0
                        while(1):
                            try:
                                question_id = re.search(r"'question_id': (\d+)", advise).group(1) or re.search(r"'question_id': '(\d+)'", advise).group(1)
                                recommand_reason = re.search(r"'recommand_reason': '([^']*)'", advise).group(1) or re.search(r"'recommand_reason': ([^']*)", advise).group(1)
                                predict_answer = re.search(r"'predict_answer': '([^']*)'", advise).group(1) or re.search(r"'predict_answer': ([^']*)", advise).group(1)
                                break
                            except:
                                try:
                                    question_id = re.search(r"(\d+)", advise).group(1) or re.search(r"'(\d+)'", advise).group(1)
                                    recommand_reason = advise
                                    predict_answer = ""
                                    break
                                except:
                                    if iter_n == max_iter:
                                        break_flag = 1
                                        break
                                    
                                    local_planner.llm = LlamaLLM(model=llm,tokenizer=tokenizer) if llm is not None else ChatGPT(CHAT_MODEL)
                                    print("Error: No match found")
                                    
                                    advise, prompt = local_planner.given_advise(history_text,memory['history_log']['question_id'], student_profile, goal_id,memory['recommend_reflection'], all_learning_state, E_s_learning_state)
                                    

                        print('get advise:')
                        print(advise)
                        f.write(f"prompt:{prompt}\nadvise: {advise}\n")
                        print('done')
                        if break_flag:
                            break    

                        all_learning_state, kn_learning_state = global_planner.get_knowledge_state_by_KT(goal_id)
                        answer_bi = 1 if all_learning_state[int(question_id)] > 0.5 else 0
                        
                        global_planner.update_history_log(question_id, answer_bi, local_planner.q2text[str(question_id)], str(answer_bi), recommand_reason, predict_answer)
                        
                        if step == MAX_STEP-1:
                            all_learning_state, E_t = global_planner.get_knowledge_state_by_KT(goal_id)
                            learning_state.append(E_t)
                        
                        print('next ... ')
                        f.write('next ... \n')

                    E_p = abs( max(learning_state) -learning_state[0]) / (1-learning_state[0])
                    E_all_s += learning_state[0]
                    E_all_e += max(learning_state)
                    E_all_sup += 1
                    print(f'goal_id = {goal_id}, stu = {stu}, E_p = {E_p}')
                    f.write(f'goal_id = {goal_id}, stu = {stu}, E_p = {E_p}\n')
                    f.write(str(global_planner.memory['history_log']))
                    
                    all_stu_lr_state4everygoal[stu].append(E_p)

        finally:
            E_all_p =  (E_all_e - E_all_s) / (E_all_sup - E_all_s)
            print(f'E_all_p = {E_all_p}')
            f.write(f'E_all_p = {E_all_p}')
        return  all_stu_lr_state4everygoal, E_all_p

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    

    parser.add_argument("--dataname", type=str, help="Specify the dataname",default="junyi")
    parser.add_argument("--model", type=str, help="Specify the model",default="llama2")
    parser.add_argument("--max_step", type=int, help="Specify the max_step",default=5)

    args = parser.parse_args()
    if args.model in ["llama2","llama3"]:
        llm, tokenizer = get_llm() if args.model=="llama3" else get_llama2()
    else:
        llm, tokenizer = None, None
    gen(args.dataname, args.model, args.max_step, llm, tokenizer)
