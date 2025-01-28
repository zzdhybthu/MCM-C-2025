import pandas as pd
from utils import *
from collections import defaultdict
from tqdm import tqdm
import re
import os

raw_data_path = 'data/data_raw'
format_data_path = 'data/data_format'
os.makedirs(format_data_path, exist_ok=True)

summerOly_athletes_raw_path = os.path.join(raw_data_path, 'summerOly_athletes.csv')
summerOly_hosts_raw_path = os.path.join(raw_data_path, 'summerOly_hosts.csv')
summerOly_medal_counts_raw_path = os.path.join(raw_data_path, 'summerOly_medal_counts.csv')
summerOly_programs_raw_path = os.path.join(raw_data_path, 'summerOly_programs.csv')

summerOly_athletes_format_path = os.path.join(format_data_path, 'summerOly_athletes.jsonl')
summerOly_hosts_format_path = os.path.join(format_data_path, 'summerOly_hosts.jsonl')
summerOly_medal_counts_format_path = os.path.join(format_data_path, 'summerOly_medal_counts.jsonl')
summerOly_programs_format_path = os.path.join(format_data_path, 'summerOly_programs.jsonl')

team2noc_path = os.path.join(format_data_path, 'team2noc.json')
noc2team_path = os.path.join(format_data_path, 'noc2team.json')

def csv2jsonl():
    summerOly_athletes = pd.read_csv(summerOly_athletes_raw_path)
    summerOly_hosts = pd.read_csv(summerOly_hosts_raw_path)
    summerOly_medal_counts = pd.read_csv(summerOly_medal_counts_raw_path)
    summerOly_programs = pd.read_csv(summerOly_programs_raw_path)

    summerOly_athletes = summerOly_athletes.to_json(orient='records')
    summerOly_hosts = summerOly_hosts.to_json(orient='records')
    summerOly_medal_counts = summerOly_medal_counts.to_json(orient='records')
    summerOly_programs = summerOly_programs.to_json(orient='records')
    
    summerOly_athletes = safe_eval(summerOly_athletes)
    summerOly_hosts = safe_eval(summerOly_hosts)
    summerOly_medal_counts = safe_eval(summerOly_medal_counts)
    summerOly_programs = safe_eval(summerOly_programs)
    
    for athlete in tqdm(summerOly_athletes):
        for key in athlete:
            if isinstance(athlete[key], str):
                athlete[key] = athlete[key].strip()
    for host in tqdm(summerOly_hosts):
        for key in host:
            if isinstance(host[key], str):
                host[key] = host[key].strip()
    for medal_count in tqdm(summerOly_medal_counts):
        for key in medal_count:
            if isinstance(medal_count[key], str):
                medal_count[key] = medal_count[key].strip()
    for program in tqdm(summerOly_programs):
        for key in program:
            if isinstance(program[key], str):
                program[key] = program[key].strip()
                
    write_jsonl(summerOly_athletes, summerOly_athletes_format_path)
    write_jsonl(summerOly_hosts, summerOly_hosts_format_path)
    write_jsonl(summerOly_medal_counts, summerOly_medal_counts_format_path)
    write_jsonl(summerOly_programs, summerOly_programs_format_path)


def team_dict():
    summerOly_athletes = read_jsonl(summerOly_athletes_format_path)
    team2noc = defaultdict(str)
    noc2team = defaultdict(set)
    for athlete in tqdm(summerOly_athletes):
        team2noc[athlete['Team']] = athlete['NOC']
        noc2team[athlete['NOC']].add(athlete['Team'])
    
    addition_dict = {
        'Mixed team': 'ZZX',
        'Russian Empire': 'RUS',
        'Ceylon': 'CEY',
        'United Team of Germany': 'GER',
        'British West Indies': 'BWI',
        'Taiwan': 'TWN',
        'Independent Olympic Participants': 'IOP',
        'Independent Olympic Athletes': 'IOP',
        'FR Yugoslavia': 'SRJ',
        'United Kingdom': 'GBR',
    }
    for k, v in addition_dict.items():
        team2noc[k] = v
        noc2team[v].add(k)
    for noc in noc2team:
        noc2team[noc] = list(noc2team[noc])
    
    summerOly_medal_counts = read_jsonl(summerOly_medal_counts_format_path)
    for i, medal_count in enumerate(summerOly_medal_counts):
        assert medal_count['NOC'] in team2noc, f"{i}, {medal_count['NOC']} not in team_dict"
    write_json(team2noc, team2noc_path)
    write_json(noc2team, noc2team_path)


def clean_programs():
    summerOly_programs = read_jsonl(summerOly_programs_format_path)
    res = {}
    for program in tqdm(summerOly_programs):
        if program['Sport'].startswith('Total'):
            continue
        program['1906'] = program.pop('1906*')
        program.pop('Discipline')
        program.pop('Code')
        program.pop('Sports Governing Body')
        for key, value in program.items():
            if value is None or isinstance(value, str) and value.startswith('Included'):
                program[key] = 0
            elif key.isdigit():
                if isinstance(value, str) and '[s' in value:
                    value = re.sub(r'\[s\d+\]', '', value)
                program[key] = int(value)
        if program['Sport'] not in res:
            res[program['Sport']] = program
        else:
            for key, value in res[program['Sport']].items():
                if key != 'Sport':
                    res[program['Sport']][key] += program[key]
    jsonl_res = []
    for key, value in res.items():
        jsonl_res.append(value)
    write_jsonl(jsonl_res, summerOly_programs_format_path)

def clean_medal_counts():
    summerOly_medal_counts = read_jsonl(summerOly_medal_counts_format_path)
    team2noc = read_json(team2noc_path)
    for medal_count in tqdm(summerOly_medal_counts):
        medal_count['NOC'] = team2noc[medal_count['NOC']]
    write_jsonl(summerOly_medal_counts, summerOly_medal_counts_format_path)

def clean_hosts():
    summerOly_hosts = read_jsonl(summerOly_hosts_format_path)
    team2noc = read_json(team2noc_path)
    res = []
    for host in tqdm(summerOly_hosts):
        try:
            if host['Host'].startswith('Cancelled'):
                continue
            if '(' in host['Host']:
                host['Host'] = host['Host'][:host['Host'].index('(')].strip()
            host['Host'] = host['Host'].split(',')[1].strip()
            host['Host'] = team2noc[host['Host']]
            res.append(host)
        except Exception as e:
            print(host)
            print(e)
    write_jsonl(res, summerOly_hosts_format_path)
    
def clean_athletes():
    summerOly_athletes = read_jsonl(summerOly_athletes_format_path)
    for athlete in tqdm(summerOly_athletes):
        athlete.pop('Sex')
        athlete.pop('Team')
        athlete.pop('City')
        athlete.pop('Event')
        athlete['Name'] = athlete['NOC'] + '_' + athlete['Name'].lower()
    write_jsonl(summerOly_athletes, summerOly_athletes_format_path)

if __name__ == '__main__':
    # 1
    csv2jsonl()

    # 2
    team_dict()
    
    # 3
    clean_programs()
    
    # 4
    clean_medal_counts()
    
    # 5
    clean_hosts()
    
    # 6
    clean_athletes()
    
    # year = 2004
    # summerOly_programs = read_jsonl(summerOly_programs_format_path)
    # medal_2024 = sum([program[str(year)] for program in summerOly_programs])
    # summerOly_medal_counts = read_jsonl(summerOly_medal_counts_format_path)
    # summerOly_medal_counts = [medal_count for medal_count in summerOly_medal_counts if medal_count['Year'] == year]
    # gold_2024 = sum([medal_count['Gold'] for medal_count in summerOly_medal_counts])
    # silver_2024 = sum([medal_count['Silver'] for medal_count in summerOly_medal_counts])
    # bronze_2024 = sum([medal_count['Bronze'] for medal_count in summerOly_medal_counts])
    # if year >= 1956:
    #     special_sports = ['Boxing', 'Judo', 'Taekwondo', 'Wrestling', 'Karate']
    #     for sport in special_sports:
    #         bronze_2024 -= [program for program in summerOly_programs if program['Sport'] == sport][0][str(year)]
    # print(medal_2024, gold_2024, silver_2024, bronze_2024)