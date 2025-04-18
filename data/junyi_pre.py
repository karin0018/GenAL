import json 
from tqdm import tqdm 


def pre_junyi():
    k2text = {}
    k2q = {}
    q2k = {}

    pid2text = {}
    ptext2pid = {}
    pid=0

    with open('junyi_Exercise_table.csv','r') as f:
        lines = f.readlines()
        print(len(lines))
        for i in range(1, len(lines)):
            line = lines[i].strip().split(',')
            ptext, topic, area = line[0], line[-2], line[-1]
            if ptext not in ptext2pid:
                ptext2pid[ptext] = pid
                pid2text[pid] = ptext
                pid += 1
                
            if topic not in k2q:
                k2q[topic] = [ptext]
            else:
                k2q[topic].append(ptext)
            if topic not in k2text:
                k2text[topic] = {'topic': topic, 'area': [area]}
            else:
                k2text[topic]['area'].append(area)
                
    k2q_new = {}
    k2text_new = {}
    for i in range(len(k2q)):
        k2q_new[i] = [ptext2pid[ptext] for ptext in k2q[list(k2q.keys())[i]]]
        k2text[list(k2text.keys())[i]]['area'] = list(set(k2text[list(k2text.keys())[i]]['area']))
        k2text_new[i] = k2text[list(k2text.keys())[i]]

    for k in k2q_new:
        for q in k2q_new[k]:
            q2k[q] = k

    with open('./data/k2q.json','w') as f:
        json.dump(k2q_new, f, ensure_ascii=False, indent=4)
    with open('./data/q2k.json','w') as f:
        json.dump(q2k, f, ensure_ascii=False, indent=4)
    with open('./data/k2text.json','w') as f:
        json.dump(k2text_new, f, ensure_ascii=False, indent=4)
    with open('./data/q2text.json','w') as f:
        json.dump(pid2text, f, ensure_ascii=False, indent=4)

def pre_junyi_log_all(datapath="./junyi/junyi_ProblemLog_original.csv"):
    with open('./data/graph_vertex.json','r') as f:
        ptext2pid = json.load(f)
    with open('./data/q2k.json','r') as f:
        q2k = json.load(f)
    with open(datapath,'r') as f:
        lines = f.readlines()
        log_all = {}
        print(len(lines))
        for i in tqdm(range(1, len(lines))):
            line = lines[i].strip().split(',')
            user_id, excer, time, score = line[0], line[1], line[7], line[10]
            if user_id in log_all:
                log_all[user_id].append({'excer_id': ptext2pid[excer], 'time': time, 'score': 1 if score == "true" else 0})
            else:
                log_all[user_id] = [{'excer_id': ptext2pid[excer], 'time': time, 'score': 1 if score == "true" else 0}]
                
        log_new = []
        log_num = 0
        users = list(log_all.keys())
        for i in tqdm(range(len(users))):
            user_log = {}
            user_log['user_id'] = i
            user_log['log_num'] = len(log_all[users[i]])
            log_num+= len(log_all[users[i]])
            
            logs = sorted(log_all[users[i]], key=lambda x: x['time'])
            user_log['logs'] = []
            for log in logs:
                excer_id,score,skill_id = log['excer_id'], log['score'], q2k[str(log['excer_id'])]
                user_log['logs'].append({'excer_id':excer_id, "score": score,  "knowledge_code":[skill_id]})
            log_new.append(user_log)
        with open("log_data.json",'w') as f:
            json.dump(log_new,f, indent=4)
        print(f"user num = {len(users)}, log num = {log_num}")

def get_log_alldata():
    data_directory_path="./junyi/"

    all_logs = []

    all_logs_num=0

    with open(data_directory_path+"log_data.json",'r') as f:
        logs_math = json.load(f)
        user_num = len(logs_math)
        for item in logs_math:
            qid = []
            answer = []
            all_logs_num+=item["log_num"]
            if item["log_num"] < 50:
                continue
            for log in item['logs']:
                qid.append(log['excer_id'])
                answer.append(log['score'])
                if len(qid)==50:
                    break
            all_logs.append((qid,answer))

        print("avg. log_num=", all_logs_num//user_num, "all num = ", all_logs_num, "user num = ", user_num)
        
    with open(data_directory_path + 'all_logs.txt', 'w') as f:
        for data in tqdm(all_logs):
            f.write(str(len(data[0])) + '\n')  
            for item in data[0]:
                f.write(str(item) + '\t')  
            f.write('\n')
            for item in data[1]:
                f.write(str(item) + '\t')  
            f.write('\n')

pre_junyi_log_all()
get_log_alldata()
pre_junyi()