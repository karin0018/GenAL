import json
from tqdm import tqdm 


def pre_assit(datapath="./skill_builder_data_corrected.csv"):
    with open(datapath, 'rb') as f:
        data = f.read().decode("latin1").splitlines()

        data = data[1:]
        k2text = {}
        log_all = {}
        k2q = {}
        q2text = {}
        for i in tqdm(range(len(data))):
            data[i] = data[i].strip().split(',')
            order_id=data[i][0]
            user_id = data[i][2]
            excer_id = data[i][4]
            original = "Main problem" if data[i][5] == "1" else "Scaffolding problem"
            score = data[i][6]
            attempt_answer_frequence = data[i][7]
            ms_first_response = int(data[i][8])
            skill_id, skill_name = data[i][16], data[i][17]
            if skill_name == "":
                continue
            k2text[skill_id] = skill_name
            if skill_id in k2q: 
                k2q[skill_id].append(excer_id)
            else:
                k2q[skill_id] = [excer_id]

            if excer_id in q2text:
                q2text[excer_id]['average_response_time'] += ms_first_response
                q2text[excer_id]['answer_frequence'] += 1
            else:
                q2text[excer_id] = {'type': original, 'text': skill_name, 'average_response_time':ms_first_response, 'answer_frequence':1}
            
            if user_id in log_all:
                log_all[user_id].append({'excer_id':excer_id, "score":score, "first_response_time":ms_first_response,"order_id":order_id,"original":original, "knowledge_code":skill_id})
            else:
                log_all[user_id] = [{'excer_id':excer_id, "score":score, "first_response_time":ms_first_response,"order_id":order_id,"original":original, "knowledge_code":skill_id}]

        with open("k2txt.json",'w') as f:
            k2text_new = {}
            k2id = {}
            i=0
            for k in k2text:
                if k2text[k] != "":
                    k2text_new[i] = k2text[k]
                    k2id[k] = i
                    i+=1
            json.dump(k2text_new,f, indent=4)
            print('k number = ',i)
            
        
        q2id = {}
        i=0
        for q in q2text:
            q2id[q] = i
            i+=1
        with open("q2id.json",'w') as f:
            json.dump(q2id,f, indent=4)
        print('q number = ',i)
            
        with open("k2q.json",'w') as f:
            k2q_new = {}
            
            for k in k2q:
                try:
                    k2q_new[k2id[k]] = [q2id[q] for q in set(k2q[k])]
                except:
                    print(k,q)
            json.dump(k2q_new,f, indent=4)
        
        with open("q2k.json",'w') as f:
            q2k = {}
            for k in k2q_new:
                for q in k2q_new[k]:
                    q2k[q] = k
            json.dump(q2k,f, indent=4)
            
        with open("q2txt.json",'w') as f:
            for q in q2text:
                q2text[q]['average_response_time'] /= q2text[q]['answer_frequence']
                q2text[q]['average_response_time'] = str(int(q2text[q]['average_response_time']*0.001))+"s"
            q2text_new = {}
            for q in q2text:
                q2text_new[q2id[q]] = q2text[q]
            json.dump(q2text_new,f,indent=4)
            
        log_new = []
        users = list(log_all.keys())
        for i in tqdm(range(len(users))):
            user_log = {}
            user_log['user_id'] = i
            user_log['log_num'] = len(log_all[users[i]])
            logs = sorted(log_all[users[i]], key=lambda x: x['order_id'])
            user_log['logs'] = []
            for log in logs:
                excer_id,score,skill_id = q2id[log['excer_id']], log['score'], k2id[log['knowledge_code']]
                user_log['logs'].append({'excer_id':excer_id, "score":int(score),  "knowledge_code":[skill_id]})
            log_new.append(user_log)
        with open("log_all.json",'w') as f:
            json.dump(log_new,f, indent=4)


def get_log_alldata():
    data_directory_path="./ASSIST/"

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

        print("avg. log_num=", all_logs_num//user_num)
        
    with open(data_directory_path + 'all_logs.txt', 'w') as f:
        for data in tqdm(all_logs):
            f.write(str(len(data[0])) + '\n') 
            for item in data[0]:
                f.write(str(item) + '\t') 
            f.write('\n')
            for item in data[1]:
                f.write(str(item) + '\t')  
            f.write('\n')


pre_assit()
get_log_alldata()