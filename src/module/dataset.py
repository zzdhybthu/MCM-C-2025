from torch.utils.data import Dataset
from utils import *
from collections import defaultdict
import random
import torch

class OlympicDataset(Dataset):
    def __init__(self, random_sample=False, device='cpu'):
        self.summerOly_athletes = read_jsonl('data_format/summerOly_athletes.jsonl')
        self.summerOly_hosts = read_jsonl('data_format/summerOly_hosts.jsonl')
        self.summerOly_medal_counts = read_jsonl('data_format/summerOly_medal_counts.jsonl')
        self.summerOly_programs = read_jsonl('data_format/summerOly_programs.jsonl')
        
        self.years = [1896, 1900, 1904, 1908, 1912, 1920, 1924, 1928, 1932, 1936, 1948, 1952, 1956, 1960, 1964, 1968, 1972, 1976, 1980, 1984, 1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020, 2024]
        self.special_sports = ['Boxing', 'Judo', 'Taekwondo', 'Wrestling', 'Karate']
        self.item_buffer = {}
        self.medal_idx = {'Gold': 0, 'Silver': 1, 'Bronze': 2, 'No medal': 3}
        self.random_sample = random_sample
        self.device = device
        
    def __len__(self):
        return len(self.years) - 1
    
    def __getitem__(self, idx):
        """
        return: [{
            "noc_medal_count": {
                noc: [gold, silver, bronze, N],
            },
            "total_noc_medals": [gold, silver, bronze],
            "sport_medal_count": {
                sport: {
                    noc: [gold, silver, bronze, N]
                }
            },
            "total_sport_medals": {
                sport: [gold, silver, bronze]
            },
            "personal_medal_dict": {
                sport: {
                    name: [gold, silver, bronze, no_medal, is_host]
                }
            }
        }]
        
        "host": host,
        "sports": {
            sport: {
                "total_medals": [gold, silver, bronze],
                "athletes": {
                    name: medal
                }
            }
        },
        """
        
        if self.random_sample:
            # idx = random.randint(1, len(self.years)-1)
            weights = []
            for i in range(1, len(self.years)):
                weights.append(i**1.5)
            idx = random.choices(
                range(1, len(self.years)),
                weights=weights,
                k=1
            )[0]
        else:
            idx = idx % (len(self.years) - 1) + 1
        
        # idx = 7
        
        res = []
        for year in tqdm(self.years[:idx+1], desc='Loading data', leave=False):
            if year in self.item_buffer:
                res.append(self.item_buffer[year])
                continue
            
            host = [host['Host'] for host in self.summerOly_hosts if host['Year'] == year][0]
            
            noc_medal_count = defaultdict(lambda: [0, 0, 0, 0])
            medal_count_that_year = [medal_count for medal_count in self.summerOly_medal_counts if medal_count['Year'] == year]
            for medal_count in medal_count_that_year:
                noc_medal_count[medal_count['NOC']] = [medal_count['Gold'], medal_count['Silver'], medal_count['Bronze'], 0]
                
            # 过渡变量，可优化掉
            sports = {}
            for program in self.summerOly_programs:
                sport_name = program['Sport']
                total_medal = program[str(year)]
                total_medals = [total_medal, total_medal, total_medal]
                if year >= 1956:
                    if sport_name in self.special_sports:
                        total_medals[2] *= 2
                athletes = defaultdict(lambda: [0, 0, 0, 0])
                all_athletes = [athlete for athlete in self.summerOly_athletes if athlete['Sport'] == sport_name and athlete['Year'] == year]
                if len(all_athletes) == 0:
                    continue
                for athlete in all_athletes:
                    medal_idx = self.medal_idx[athlete['Medal']]
                    athletes[athlete['Name']][medal_idx] += 1
                sports[sport_name] = {
                    "total_medals": total_medals,
                    "athletes": athletes
                }
            
            all_athletes = set()
            for v in sports.values():
                for athlete in v['athletes'].keys():
                    all_athletes.add(athlete)
            for athlete in all_athletes:
                noc = athlete.split('_')[0]
                noc_medal_count[noc][-1] += 1
            for noc, v in noc_medal_count.items():
                noc_medal_count[noc][-1] = max(v[-1], sum(v[:-1]))
                
            total_noc_gold = sum([v[0] for v in noc_medal_count.values()])
            total_noc_silver = sum([v[1] for v in noc_medal_count.values()])
            total_noc_bronze = sum([v[2] for v in noc_medal_count.values()])
            total_noc_medals = [total_noc_gold, total_noc_silver, total_noc_bronze]
            
            sport_medal_count = {}
            for sport, v in sports.items():
                sport_medal_count[sport] = defaultdict(lambda: [0, 0, 0, 0])
                for athlete, medal in v['athletes'].items():
                    noc = athlete.split('_')[0]
                    sport_medal_count[sport][noc][0] += medal[0]
                    sport_medal_count[sport][noc][1] += medal[1]
                    sport_medal_count[sport][noc][2] += medal[2]
                    sport_medal_count[sport][noc][3] += 1
            
            total_sport_medals = {k: v['total_medals'] for k, v in sports.items()}
            
            personal_medal_dict = defaultdict(lambda: {})
            for sport, v in sports.items():
                for athlete, medal in v['athletes'].items():
                    is_host = 1 if athlete.split('_')[0] == host else 0
                    personal_medal_dict[sport][athlete] = medal + [is_host]
            
            res.append({
                "noc_medal_count": noc_medal_count,
                "total_noc_medals": total_noc_medals,
                "sport_medal_count": sport_medal_count,
                "total_sport_medals":total_sport_medals,
                "personal_medal_dict": personal_medal_dict
            })
            res[-1] = self.convert_to_float(res[-1])
            self.item_buffer[year] = res[-1]
            
        return res, self.years[idx]
    
    
    def convert_to_float(self, obj):
        if isinstance(obj, dict):
            return {key: self.convert_to_float(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_float(item) for item in obj]
        elif isinstance(obj, str) and obj.isdigit() or isinstance(obj, int) or isinstance(obj, float):
            return torch.tensor(float(obj), device=self.device)
        else:
            return obj


if __name__ == '__main__':
    dataset = OlympicDataset()
    write_json(dataset[0], 'sample.json')
        
        
        