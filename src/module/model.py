import torch
from torch.nn import Module
import torch.nn as nn
from collections import defaultdict
from tqdm import tqdm
from copy import deepcopy


class OlymPredictor(Module):
    def __init__(self, device):
        super().__init__()
        self.a1 = nn.Parameter(torch.tensor(2.0, device=device))  # 国家动量系数
        self.a2 = nn.Parameter(torch.tensor(5.0, device=device))  # 项目动量系数
        self.a3 = nn.Parameter(torch.tensor(2.0, device=device))  # 个人动量系数
        
        self.beta0 = nn.Parameter(torch.tensor(0.2, device=device))  # 国家动量更新分母
        self.beta1 = nn.Parameter(torch.tensor(2.0, device=device))  # 国家动量更新金牌
        self.beta2 = nn.Parameter(torch.tensor(2.0, device=device))  # 国家动量更新银牌
        self.beta3 = nn.Parameter(torch.tensor(2.0, device=device))  # 国家动量更新铜牌
        
        self.gamma0 = nn.Parameter(torch.tensor(0.2, device=device))  # 项目动量更新分母
        self.gamma1 = nn.Parameter(torch.tensor(2.0, device=device))  # 项目动量更新金牌
        self.gamma2 = nn.Parameter(torch.tensor(2.0, device=device))  # 项目动量更新银牌
        self.gamma3 = nn.Parameter(torch.tensor(2.0, device=device))  # 项目动量更新铜牌
        
        self.eta1 = nn.Parameter(torch.tensor(0.3, device=device))  # 个人动量更新金牌
        self.eta2 = nn.Parameter(torch.tensor(0.3, device=device))  # 个人动量更新银牌
        self.eta3 = nn.Parameter(torch.tensor(0.3, device=device))  # 个人动量更新铜牌
        self.eta4 = nn.Parameter(torch.tensor(-0.2, device=device))  # 个人动量更新未获奖
        self.eta5 = nn.Parameter(torch.tensor(0.8, device=device))  # 个人动量更新主办国
        
        self.device = device
        self.max_number = 1000
        
        self.init_state()
    
    def init_state(self):
        self.Cs = defaultdict(lambda: 1)  # 国家动量，key: noc, value: momentum
        self.Ss = defaultdict(lambda: defaultdict(lambda: 1.0))  # 项目动量，key: sport，key: noc，value: momentum
        self.Ps = defaultdict(lambda: defaultdict(lambda: 0.0))  # 个人动量，key: sport, key: athlete, value: momentum
        self.Cs0 = {}  # 个人国家初始动量，key: athlete, value: momentum
        self.Ss0 = defaultdict(lambda: {})  # 个人项目初始动量，key: sport, key: athlete, value: momentum
        
        self.Cts = defaultdict(lambda: [1])  # 国家时间
        self.Sts = defaultdict(lambda: defaultdict(lambda: [1]))  # 项目时间
    
    """
    Naive Functions for calculating momentum, without considering time or country
    """
    def get_M(self, C, S, P):
        return self.a1 * C + self.a2 * S + self.a3 * P
    def get_C_inc(self, gold, silver, bronze, N):
        return (self.beta1 * gold + self.beta2 * silver + self.beta3 * bronze) / (1 + self.beta0 * N)
    def get_C(self, C, C_inc, t):
        alpha = 2 / (t + 1)
        return alpha * C_inc + (1 - alpha) * C
    def get_S_inc(self, gold, silver, bronze, N):
        return (self.gamma1 * gold + self.gamma2 * silver + self.gamma3 * bronze) / (1 + self.gamma0 * N)
    def get_S(self, S, S_inc, t):
        alpha = 2 / (t + 1)
        return alpha * S_inc + (1 - alpha) * S
    def get_P(self, P, gold, silver, bronze, no_medal, host):
        return P + self.eta1 * gold + self.eta2 * silver + self.eta3 * bronze + self.eta4 * no_medal + self.eta5 * host
    def update_Cs(self, medal_dict, medal_total):
        """
        medal_dict: NOC -> [gold, silver, bronze, N]
        medal_total: [total_gold, total_silver, total_bronze]
        """
        for noc, v in medal_dict.items():
            if v is None:
                if noc in self.Cts:
                    self.Cts[noc].append(self.Cts[noc][-1] + 1)
            else:
                self.Cs[noc] = self.get_C(self.Cs[noc], self.get_C_inc(v[0] / medal_total[0], v[1] / medal_total[1], v[2] / medal_total[2], v[3]), self.Cts[noc][-1])
                self.Cts[noc].append(self.Cts[noc][-1] + 1)
    def update_Ss(self, medal_dict, medal_total):
        """
        medal_dict: sport -> NOC -> [gold, silver, bronze, N]
        medal_total: sport -> [total_gold, total_silver, total_bronze]
        """
        for sport, v in medal_dict.items():
            for noc, w in v.items():
                if w is None:
                    if noc in self.Sts[sport]:
                        self.Sts[sport][noc].append(self.Sts[sport][noc][-1] + 1)
                else:
                    for i in range(3):
                        medal_dict[sport][noc][i] /= medal_total[sport][i]
                    self.Ss[sport][noc] = self.get_S(self.Ss[sport][noc], self.get_S_inc(w[0] / medal_total[sport][0], w[1] / medal_total[sport][1], w[2] / medal_total[sport][2], w[3]), self.Sts[sport][noc][-1])
                    self.Sts[sport][noc].append(self.Sts[sport][noc][-1] + 1)
    def update_Ps(self, medal_dict):
        """
        medal_dict: sport -> athlete -> [gold, silver, bronze, no_medal, host]
        """
        for sport, v in medal_dict.items():
            for athlete, w in v.items():
                self.Ps[sport][athlete] = self.get_P(self.Ps[sport][athlete], *w)
        for sport, v in self.Ps.items():
            if len(v) > 1:
                mu = torch.mean(torch.tensor(list(v.values()), device=self.device))
                sigma = torch.std(torch.tensor(list(v.values()), device=self.device))
                for athlete in v:
                    self.Ps[sport][athlete] = (self.Ps[sport][athlete] - mu) / sigma
            
    
    def predict(self, athlete_dict, medal_total):
        """
        athlete_dict: sport -> athlete_names -> num_event
        medal_total: sport -> [total_gold, total_silver, total_bronze]
        return: sport -> athlete -> [gold, silver, bronze]
        """
        medal_expect = {}
        for sport, athletes in tqdm(athlete_dict.items(), desc='Calculating', leave=False):
            Ms = {}
            for athlete in athletes.keys():
                noc = athlete.split('_')[0]
                C = self.Cs0[athlete] if athlete in self.Cs0 else self.Cs[noc]
                S = self.Ss0[sport][athlete] if athlete in self.Ss0[sport] else self.Ss[sport][noc]
                P = self.Ps[sport][athlete]
                Ms[athlete] = torch.exp(self.get_M(C, S, P))
                if Ms[athlete].isnan() or Ms[athlete] > self.max_number:
                    Ms[athlete] = self.max_number
                
            medal_total_gold = medal_total[sport][0]
            medal_total_silver = medal_total[sport][1]
            medal_total_bronze = medal_total[sport][2]
            
            Ms_list = []
            for name, M in Ms.items():
                for _ in range(int(athletes[name].squeeze())):
                    Ms_list.append({'name': name, 'M': M / athletes[name]})
            Ms_list = sorted(Ms_list, key=lambda x: x['M'])
            res = defaultdict(lambda: [0, 0, 0])
            
            while len(Ms_list) > 0:
                total_Ms = sum([x['M'] for x in Ms_list])
                for i in range(len(Ms_list)):
                    Ms_list[i]['M'] = Ms_list[i]['M'] / total_Ms.squeeze()
                cur_Ms = Ms_list.pop()
                gold = min(torch.tensor(1.0, device=self.device), cur_Ms['M'] * medal_total_gold)
                silver = min(torch.tensor(1.0, device=self.device), cur_Ms['M'] * (medal_total_silver + medal_total_gold)) - gold
                total = min(torch.tensor(1.0, device=self.device), cur_Ms['M'] * (medal_total_gold + medal_total_silver + medal_total_bronze))
                bronze = total - gold - silver
                res[cur_Ms['name']][0] += gold
                res[cur_Ms['name']][1] += silver
                res[cur_Ms['name']][2] += bronze
                
                medal_total_gold = medal_total_gold - gold
                medal_total_silver = medal_total_silver - silver
                medal_total_bronze = medal_total_bronze - bronze
                
            medal_expect[sport] = res
        return medal_expect
    
    def forward_once(self, item):        
        self.update_Cs(item['noc_medal_count'], item['total_noc_medals'])
        self.update_Ss(item['sport_medal_count'], item['total_sport_medals'])
        self.update_Ps(item['personal_medal_dict'])
        
        for sport in item['sport_medal_count'].keys():
            for name in item['sport_medal_count'][sport].keys():
                if name not in self.Cs0:
                    self.Cs0[name] = self.Cs[name.split('_')[0]]
                if name not in self.Ss0[sport]:
                    self.Ss0[sport][name] = self.Ss[sport][name.split('_')[0]]

    def save_params(self, path):
        torch.save(self.state_dict(), path)
    
    def load_params(self, path):
        self.load_state_dict(torch.load(path))
    
    def print_params(self):
        for name, param in self.named_parameters():
            print(name, float(param))
        
    def forward(self, items):
        item_past = items[:-1]
        for item in tqdm(item_past, desc='Training', leave=False):
            self.forward_once(item)
        
        item_gd = items[-1]
        pred_athlete_names = defaultdict(lambda: {})
        for sport, athlete_dict in item_gd['personal_medal_dict'].items():
            for athlete, medal in athlete_dict.items():
                pred_athlete_names[sport][athlete] = sum(medal[:-1])
            
        pred = self.predict(pred_athlete_names, item_gd['total_sport_medals'])
        noc_medal_count = self.cal_noc_medal_count(pred)
        loss = self.cal_loss(noc_medal_count, item_gd['noc_medal_count'])
        
        return pred, noc_medal_count, loss
    
    def predict2028(self, items_past, item_gd=None):
        Cs_history = [deepcopy(self.Cs)]
        Ss_history = [deepcopy(self.Ss)]
        Ps_history = [deepcopy(self.Ps)]
        for item in tqdm(items_past, desc='Predicting', leave=False):
            self.forward_once(item)
            Cs_history.append(deepcopy(self.Cs))
            Ss_history.append(deepcopy(self.Ss))
            Ps_history.append(deepcopy(self.Ps))
        
        if item_gd is None:
            item_gd = items_past[-1]
        pred_athlete_names = defaultdict(lambda: {})
        for sport, athlete_dict in item_gd['personal_medal_dict'].items():
            for athlete, medal in athlete_dict.items():
                pred_athlete_names[sport][athlete] = sum(medal[:-1])
            
        pred = self.predict(pred_athlete_names, item_gd['total_sport_medals'])
        noc_medal_count = self.cal_noc_medal_count(pred)
        loss = self.cal_loss(noc_medal_count, item_gd['noc_medal_count'])
        errors = self.cal_error(noc_medal_count, item_gd['noc_medal_count'])
        
        return pred, noc_medal_count, Cs_history, Ss_history, Ps_history, loss, errors
            
    
    def cal_noc_medal_count(self, pred):
        noc_medal_count = defaultdict(lambda: [0, 0, 0])
        for sport, athlete_dict in pred.items():
            for athlete, medal in athlete_dict.items():
                noc = athlete.split('_')[0]
                noc_medal_count[noc][0] += medal[0].squeeze()
                noc_medal_count[noc][1] += medal[1].squeeze()
                noc_medal_count[noc][2] += medal[2].squeeze()
        return noc_medal_count

    def cal_loss(self, pred_medal_count, gd):
        loss = torch.tensor(0.0, device=self.device)
        for noc, medals in pred_medal_count.items():
            for i in range(3):
                loss += (medals[i].squeeze() - gd[noc][i].squeeze()) ** 2
            loss += (sum(medals).squeeze() - sum(gd[noc][:3]).squeeze()) ** 2
        loss /= len(pred_medal_count)
        return loss
    
    def cal_error(self, pred_medal_count, gd):
        es = {"gold": [], "silver": [], "bronze": [], "total": []}
        for noc, medals in pred_medal_count.items():
            es['gold'].append({"gd": gd[noc][0].squeeze(), "pred": medals[0].squeeze()})
            es['silver'].append({"gd": gd[noc][1].squeeze(), "pred": medals[1].squeeze()})
            es['bronze'].append({"gd": gd[noc][2].squeeze(), "pred": medals[2].squeeze()})
            es['total'].append({"gd": sum(gd[noc][:3]).squeeze(), "pred": sum(medals).squeeze()})
        return es

if __name__ == '__main__':
    module = OlymPredictor()