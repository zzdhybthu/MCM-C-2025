"""
打印国家擅长的项目
"""

from utils import *
from collections import defaultdict

NOC = 'GER'

file_path = "preds/pred_ckpt-56_tensor([2004])_model.json"

result = read_json(file_path)
pred = result['pred']

medals = defaultdict(lambda: [0, 0, 0])
for sport, v in pred.items():
    for athlete, medal_counts in v.items():
        noc = athlete.split('_')[0]
        if noc == NOC:
            for i, count in enumerate(medal_counts):
                medals[sport][i] += count
for k, v in medals.items():
    v.append(sum(v))

medals_list = []
for sport, medal_count in medals.items():
    medals_list.append({
        "sport": sport,
        "gold": medal_count[0],
        "silver": medal_count[1],
        "bronze": medal_count[2],
        "total": medal_count[3]
    })
medals_list = sorted(medals_list, key=lambda x: x['total'], reverse=True)
print(medals_list)
