"""
绘制2024年奥运会中获得第一枚奖牌的概率最高的10个国家的柱状图
"""

from utils import *
from matplotlib import pyplot as plt
import numpy as np
from vis_format_medal import format_medal

file_path = "preds/pred_ckpt-75_tensor([2000])_model.json"

nc_medal_count_gd_list = format_medal(file_path)

summerOly_medal_counts = read_jsonl('data_format/summerOly_medal_counts.jsonl')
remove_noc = list(set([c['NOC'] for c in summerOly_medal_counts]))
nc_medal_count_gd_list = [r for r in nc_medal_count_gd_list if r['total_gd'] == 0 and r['noc'] not in remove_noc]
noc_list = [r['noc'] for r in nc_medal_count_gd_list]
if 'AIN' in noc_list:
    noc_list.remove('AIN')

result = read_json(file_path)
pred = result['pred']

probs = {}
num_athletes = {}
for noc in noc_list:
    probs[noc] = []
    num_athletes[noc] = 0

for sport, v in pred.items():
    for athlete, medal_counts in v.items():
        noc = athlete.split('_')[0]
        if noc in noc_list:
            probs[noc].append(sum(medal_counts))
            num_athletes[noc] += 1
            if sum(medal_counts) > 0.5:
                print(athlete, medal_counts)

for noc in noc_list:
    prob_no_medal = 1
    for prob in probs[noc]:
        prob_no_medal *= (1 - prob)
    probs[noc] = 1 - prob_no_medal

probs_list = []
for noc, prob in probs.items():
    probs_list.append({
        'noc': noc,
        'prob': prob,
        'num_athletes': num_athletes[noc]
    })

probs_list = sorted(probs_list, key=lambda x: x['prob'], reverse=True)



def plot_first_medal(nocs, probs, num_athletes):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    bar_width = 0.6
    index = np.arange(len(nocs))
    ax1.bar(index, num_athletes, bar_width, label='Number of Athletes in 2024', color='skyblue', alpha=0.7)
    
    # Adjust font sizes for the first axis
    ax1.set_xlabel('NOC', fontsize=18)  # Adjust xlabel font size
    ax1.set_ylabel('Number of Athletes', fontsize=18)  # Adjust ylabel font size
    ax1.set_xticks(index)
    ax1.set_xticklabels(nocs, rotation=45, fontsize=16)  # Adjust xtick font size
    ax1.tick_params(axis='both', labelsize=16)  # Adjust tick label font size
    
    # Second axis (for probability line)
    ax2 = ax1.twinx()
    ax2.plot(index, probs, color='red', marker='^', label='Probability')
    ax2.set_ylabel('Probability', fontsize=18)  # Adjust y-axis label font size
    ax2.tick_params(axis='y', labelsize=16)  # Adjust y-axis tick label font size
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Adjusting the legend font size
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=16)  # Adjust legend font size

    # Title font size
    plt.title('Top 10 NOCs with the Highest Probability of Winning the First Medal', fontsize=18)
    
    plt.tight_layout()
    plt.savefig('figs/first_medal.png')
    plt.close()
    
    
nocs = [item['noc'] for item in probs_list[:10]]
probs = [item['prob'] for item in probs_list[:10]]
num_athletes = [item['num_athletes'] for item in probs_list[:10]]

plot_first_medal(nocs, probs, num_athletes)