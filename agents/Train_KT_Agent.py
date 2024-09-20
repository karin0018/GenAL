
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from sklearn.metrics import roc_auc_score
import os
import json
from pretrained_bert import get_bert
from transformers import BertModel, BertTokenizer



SEED = 2024
BATCH_SIZE = 12
EPOCH = 30

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class E_DKT(nn.Module):
    def __init__(self, item_num, input_dim, hidden_dim=698, layer_dim=1):
        super(E_DKT, self).__init__()
        self.item_num = item_num
        self.emb = nn.Embedding(2*item_num, input_dim)
        nn.init.xavier_uniform_(self.emb.weight)
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.RNN(input_dim+768, hidden_dim, layer_dim, batch_first=True,nonlinearity='tanh',device=device)
        self.fc = nn.Linear(self.hidden_dim, self.item_num,device=device)
        self.sig = nn.Sigmoid().to(device)
        self.diff = nn.Linear(768, hidden_dim,device=device)

    def forward(self, ques, ans, ques_text, kn_emb):
        x = ques + ans * self.item_num
        x = self.emb(x.long())
        x = torch.cat((x, ques_text), dim=-1)
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)).to(device)
        out,hn = self.rnn(x, h0)
        out, diff = self.sig(out), self.sig(self.diff(ques_text))
        res = self.sig(self.fc((out-diff)*kn_emb))
        res = res.to(device)
        res = res.squeeze(-1)
        return res
    
    def get_difficulty(self, ques_text, kn_emb):
        difficulty = torch.mean(self.sig(self.diff(ques_text))*kn_emb)
        return difficulty
    
    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.fc.apply(clipper)
        
        

class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)

class Data_EDKT(Dataset):
    def __init__(self, file_path, bert,tokenizer, datapath, k_num):
        self.ques = []
        self.ans = []
        self.ques_text = []
        self.input_knowedge_embs = []
        self.k_num = k_num
        self.bert, self.tokenizer = bert.to(device), tokenizer
        data_directory_path=datapath
        question_path = data_directory_path + "/q2text.json"
        q2k = data_directory_path + "/q2k.json"
        with open(q2k, 'r') as f:
            q2k_data = json.load(f)
        
        with open(question_path, 'r') as f:
            question_context = json.load(f)
        
        
        with open(file_path, 'r') as f:
            for line_id, line in tqdm(enumerate(f)):
                i = line_id % 3
                if i == 1:
                    ques = list(line.strip().split('\t'))
                    ques = [int(j) for j in ques]
                    ques_text = [str(question_context[str(j)]) for j in ques] 
                    self.ques.append(ques)
                    self.ques_text.append(ques_text) 
                    input_knowedge_embs = []
                    for q in ques:
                        e_k = q2k_data[str(q)] if type(q2k_data[str(q)]) == list else [q2k_data[str(q)]]

                        input_knowedge_emb = [0.] * self.k_num
                        for k in e_k:
                            input_knowedge_emb[k] = 1
                        input_knowedge_embs.append(input_knowedge_emb)
                    self.input_knowedge_embs.append(input_knowedge_embs)
                elif i == 2:
                    ans = list(line.strip().split('\t'))
                    ans = [int(i) for i in ans]
                    self.ans.append(ans)


    def __getitem__(self, index):
        
        inputs = self.tokenizer(self.ques_text[index], return_tensors='pt', padding=True, truncation=True).to(device)
        outputs = self.bert(**inputs)
        sentence_embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return torch.tensor(self.ques[index]).to(device), torch.tensor(self.ans[index]).to(device), sentence_embeddings.to(device), torch.tensor(self.input_knowedge_embs[index]).to(device)

    def __len__(self):
        return len(self.ques)
       
def train_E_DKT(datapath, modelpath, item_num, knowledge_dim, bert_path = "../pretrained_models/bert_base_chinese"):
    MODEL_PATH=modelpath
    ITEM_NUM=item_num
    INPUT_DIM=128
    HIDDEN_DIM=knowledge_dim
    LAYER_DIM=1
    
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.multiprocessing.set_sharing_strategy('file_system')

    model = E_DKT(item_num=ITEM_NUM,input_dim=INPUT_DIM,hidden_dim=HIDDEN_DIM,layer_dim=LAYER_DIM)
    model.to(device)
    model.float()
    bert = BertModel.from_pretrained(bert_path)
    tokenizer = BertTokenizer.from_pretrained(bert_path)

    dataset = Data_EDKT(datapath+'/all_logs.txt', bert, tokenizer,datapath=datapath, k_num=knowledge_dim)
    train_size = int(len(dataset) * 0.9)
    validate_size = len(dataset) - train_size
    print("train_size, validate_size, ", train_size, validate_size)
    train_dataset, validate_dataset = torch.utils.data.random_split(dataset, [train_size, validate_size])
    data_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validate_loader = DataLoader(dataset=validate_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_func = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_auc = 0
    for epoch in tqdm(range(EPOCH)):
        model.train()
        pred_epoch = torch.Tensor([]).to(device)
        gold_epoch = torch.Tensor([]).to(device)
        for batch in tqdm(data_loader):
            ques, ans, ques_text, kn_emb = batch
            ques = ques[:,:-1]
            ans_tr = ans[:,:-1]
            ques_text = ques_text[:,:-1]
            kn_emb = kn_emb[:,:-1]
            pred= model(ques, ans_tr, ques_text, kn_emb)
            ans = ans[:,1:]
            pred = torch.gather(pred, 2, ques.unsqueeze(-1)).squeeze(-1)
            loss = loss_func(pred.float(), ans.float())
            pred_epoch = torch.cat([pred_epoch, pred], dim=0)
            gold_epoch = torch.cat([gold_epoch, ans], dim=0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            model.apply_clipper()
            
        scheduler.step()
        auc = roc_auc_score(gold_epoch.cpu().detach().numpy(), pred_epoch.cpu().detach().numpy())
        print('train epoch: {}, loss: {}, auc: {}'.format(epoch, loss.item(), auc))

        with torch.no_grad():
            model.eval()
            pred_epoch = torch.Tensor([]).to(device)
            gold_epoch = torch.Tensor([]).to(device)
            for batch in tqdm(validate_loader):
                
                ques, ans, ques_text, kn_emb = batch
                ques = ques.to(device)[:, :-1]
                ans_tr = ans.to(device)[:, :-1]
                
                ques_text = ques_text.to(device)[:,:-1]
                kn_emb = kn_emb.to(device)[:,:-1]
                pred= model(ques, ans_tr, ques_text, kn_emb)
                ans = ans[:, 1:].to(device)
                pred = torch.gather(pred, 2, ques.unsqueeze(-1)).squeeze(-1)
                loss = loss_func(pred.float(), ans.float())
                pred_epoch = torch.cat([pred_epoch, pred], dim=0)
                gold_epoch = torch.cat([gold_epoch, ans], dim=0)
            auc = roc_auc_score(gold_epoch.cpu().detach().numpy(), pred_epoch.cpu().detach().numpy())
            if auc > best_auc:
                best_auc = auc
                os.makedirs(MODEL_PATH, exist_ok=True)
                torch.save(model, os.path.join(MODEL_PATH+f'/Trained_E_DKT_model.pt'))
            print('validate epoch: {}, loss: {}, auc: {}'.format(epoch, loss.item(), auc))

    print('best val auc: {}'.format(best_auc))
    
    with open(os.path.join(MODEL_PATH+'/Trained_E_DKT_details.txt'), 'w') as f:
        f.write('best val auc: {}'.format(best_auc))
        f.write('\n')
        f.write('total epoch: {}'.format(EPOCH))
        f.write('\n')

class Data_KSS(Dataset):
    def __init__(self, file_path):
        self.ques = []
        self.ans = []
        with open(file_path, 'r') as f:
            for line_id, line in tqdm(enumerate(f)):
                i = line_id % 3
                if i == 1:
                    ques = list(line.strip().split('\t'))
                    ques = [int(i) for i in ques]
                    self.ques.append(ques)
                elif i == 2:
                    ans = list(line.strip().split('\t'))
                    ans = [int(i) for i in ans]
                    self.ans.append(ans)


    def __getitem__(self, index):
        return torch.tensor(self.ques[index]), torch.tensor(self.ans[index])


    def __len__(self):
        return len(self.ques)

class DKT(nn.Module):
    def __init__(self, item_num, input_dim, hidden_dim, layer_dim):
        super(DKT, self).__init__()
        self.item_num = item_num
        self.emb = nn.Embedding(2*item_num, input_dim)
        nn.init.xavier_uniform_(self.emb.weight)
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True,nonlinearity='tanh',device=device)
        self.fc = nn.Linear(self.hidden_dim, self.item_num,device=device)
        self.sig = nn.Sigmoid().to(device)

    def forward(self, ques, ans):
        x = ques + ans * self.item_num
        x = self.emb(x.long())
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)).to(device)
        out,hn = self.rnn(x, h0)
        res = self.sig(self.fc(out))
        res = res.to(device)
        res = res.squeeze(-1)
        return res


def train_KSS(datapath, modelpath, item_num):
    print(modelpath)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.multiprocessing.set_sharing_strategy('file_system')

    model = DKT(item_num,128,128,1)
    model.to(device)
    model.float()

    dataset = Data_KSS(datapath+"/all_logs.txt") 
    train_size = int(len(dataset) * 0.9)
    validate_size = len(dataset) - train_size
    print("train_size, validate_size,", train_size, validate_size)
    
    train_dataset, validate_dataset = torch.utils.data.random_split(dataset, [train_size, validate_size])
    
    with open(datapath + '/validate.json', 'w') as f:
        val = {"ques": [], "ans": []}
        for ques, ans in tqdm(validate_dataset):
            
            val["ques"].append(ques.tolist())
            val["ans"].append(ans.tolist())
        json.dump(val, f, indent=4)

    
    data_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validate_loader = DataLoader(dataset=validate_dataset, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_func = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_auc = 0
    for epoch in range(EPOCH):
        model.train()
        pred_epoch = torch.Tensor([]).to(device)
        gold_epoch = torch.Tensor([]).to(device)
        for batch in tqdm(data_loader):
            ques, ans = batch
            ques = ques.to(device)[:,:-1]
            ans_tr = ans.to(device)[:,:-1]
            pred= model(ques, ans_tr)
            ans = ans[:,1:].to(device)
            pred = torch.gather(pred, 2, ques.unsqueeze(-1)).squeeze(-1)
            loss = loss_func(pred.float(), ans.float())
            pred_epoch = torch.cat([pred_epoch, pred], dim=0)
            gold_epoch = torch.cat([gold_epoch, ans], dim=0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        auc = roc_auc_score(gold_epoch.cpu().detach().numpy(), pred_epoch.cpu().detach().numpy())
        

        with torch.no_grad():
            model.eval()
            pred_epoch = torch.Tensor([]).to(device)
            gold_epoch = torch.Tensor([]).to(device)
            for batch in tqdm(validate_loader):
                ques, ans = batch
                ques = ques.to(device)[:, :-1]
                ans_tr = ans.to(device)[:, :-1]
                pred= model(ques, ans_tr)
                ans = ans[:, 1:].to(device)
                pred = torch.gather(pred, 2, ques.unsqueeze(-1)).squeeze(-1)
                loss = loss_func(pred.float(), ans.float())
                pred_epoch = torch.cat([pred_epoch, pred], dim=0)
                gold_epoch = torch.cat([gold_epoch, ans], dim=0)
            auc = roc_auc_score(gold_epoch.cpu().detach().numpy(), pred_epoch.cpu().detach().numpy())
            if auc > best_auc:
                best_auc = auc
                os.makedirs(modelpath, exist_ok=True)
                torch.save(model, modelpath + '/Trained_KES_KT_model.pt')
            print('validate epoch: {}, loss: {}, auc: {}'.format(epoch, loss.item(), auc))

    print('best val auc: {}'.format(best_auc))
    
    with open(modelpath + '/Trained_KSS_KT_details.txt', 'w') as f:
        f.write('best val auc: {}'.format(best_auc))
        f.write('\n')
        f.write('total epoch: {}'.format(EPOCH))
        f.write('\n')



if __name__ == '__main__':
    # train_E_DKT()

    print(device)
