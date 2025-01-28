"""
大幅改动奖牌数对奖牌榜的影响
"""

from utils import *
from matplotlib import pyplot as plt
import numpy as np
from vis_format_medal import format_medal

file_before = "preds/pred_ckpt-75_tensor([2000])_model.json"

def main(file_after, sport):
    nc_medal_count_gd_list_before = format_medal(file_before)
    nc_medal_count_gd_list_after = format_medal(file_after)

    def plot_total_medals(nocs, before, after, gd, sport, label='Total'):    
        bar_width = 0.35
        index = np.arange(len(nocs))
        figure_size = (6, 5)
        fig, ax = plt.subplots(figsize=figure_size)
        
        ax.bar(index - bar_width/2, after, bar_width, label=f'Predicted Increase in {label} Medal after Doubling', color='#FFA500')
        ax.bar(index - bar_width/2, before, bar_width, label=f'Predicted {label} Medal for 2028', color='#FFC878')
        ax.bar(index + bar_width/2, gd, bar_width, label='GD 2024', color='skyblue')
        
        ax.set_xticks(index)
        ax.set_xticklabels(nocs, rotation=45, fontsize=10)  # Adjusted font size for xticklabels
        ax.set_xlabel('NOC', fontsize=12)  # Adjusted font size for xlabel
        ax.set_ylabel(f'{label} Medal Count', fontsize=12)  # Adjusted font size for ylabel
        plt.title(f'{label} Medal Count Before and After Doubling Events of {sport}', fontsize=14)  # Adjusted font size for title
        plt.legend(fontsize=10)  # Adjusted font size for legend
        plt.xticks(fontsize=10)  # Adjusted font size for xticks
        plt.yticks(fontsize=10)  # Adjusted font size for yticks
        
        plt.tight_layout()
        plt.savefig(f'figs/increase_{label}_{sport}.png')
        plt.close()
        


    nc_medal_count_gd_dict_before = {noc_medal_count['noc']: noc_medal_count for noc_medal_count in nc_medal_count_gd_list_before}


    nc_medal_count_gd_list_after = sorted(nc_medal_count_gd_list_after, key=lambda x: x['total'], reverse=True)
    nocs = [noc_medal_count['noc'] for noc_medal_count in nc_medal_count_gd_list_after[:10]]
    totals_before = [nc_medal_count_gd_dict_before[noc_medal_count['noc']]['total'] for noc_medal_count in nc_medal_count_gd_list_after[:10]]
    totals_after = [noc_medal_count['total'] for noc_medal_count in nc_medal_count_gd_list_after[:10]]
    gd = [noc_medal_count['total_gd'] for noc_medal_count in nc_medal_count_gd_list_after[:10]]
    plot_total_medals(nocs, totals_before, totals_after, gd, sport, label='Total')


    nc_medal_count_gd_list_after = sorted(nc_medal_count_gd_list_after, key=lambda x: x['gold'], reverse=True)
    nocs = [noc_medal_count['noc'] for noc_medal_count in nc_medal_count_gd_list_after[:10]]
    totals_before = [nc_medal_count_gd_dict_before[noc_medal_count['noc']]['gold'] for noc_medal_count in nc_medal_count_gd_list_after[:10]]
    totals_after = [noc_medal_count['gold'] for noc_medal_count in nc_medal_count_gd_list_after[:10]]
    gd = [noc_medal_count['gold_gd'] for noc_medal_count in nc_medal_count_gd_list_after[:10]]
    plot_total_medals(nocs, totals_before, totals_after, gd, sport, label='Gold')


if __name__ == '__main__':
    file_after = "preds/Shooting_ckpt-75_tensor([2000])_model.json"
    sport = 'Shooting'
    main(file_after, sport)
    file_after = "preds/Judo_ckpt-75_tensor([2000])_model.json"
    sport = 'Judo'
    main(file_after, sport)
    file_after = "preds/Athletics_ckpt-75_tensor([2000])_model.json"
    sport = 'Athletics'
    main(file_after, sport)
