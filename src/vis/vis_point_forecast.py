"""
绘制奖牌预测结果的可视化
"""

from utils import *
from matplotlib import pyplot as plt
import numpy as np
from vis_format_medal import format_medal

file_path = "preds/pred_ckpt-75_tensor([2000])_model.json"

result = read_json(file_path)

nc_medal_count_gd_list = format_medal(file_path)

def plot_total_medals(nocs, predicted_totals, ground_truth_totals):    
    bar_width = 0.35
    index = np.arange(len(nocs))
    fig, ax = plt.subplots()
    ax.bar(index - bar_width/2, predicted_totals, bar_width, label='Predicted')
    ax.bar(index + bar_width/2, ground_truth_totals, bar_width, label='Ground Truth')

    ax.set_xticks(index)
    ax.set_xticklabels(nocs, rotation=45)
    plt.legend()
    plt.show()
    

nc_medal_count_gd_list = sorted(nc_medal_count_gd_list, key=lambda x: x['total'], reverse=True)
nocs = [noc_medal_count['noc'] for noc_medal_count in nc_medal_count_gd_list[:10]]
predicted_totals = [noc_medal_count['total'] for noc_medal_count in nc_medal_count_gd_list[:10]]
ground_truth_totals = [noc_medal_count['total_gd'] for noc_medal_count in nc_medal_count_gd_list[:10]]
plot_total_medals(nocs, predicted_totals, ground_truth_totals)

nc_medal_count_gd_list = sorted(nc_medal_count_gd_list, key=lambda x: x['gold'], reverse=True)
nocs = [noc_medal_count['noc'] for noc_medal_count in nc_medal_count_gd_list[:10]]
predicted_totals = [noc_medal_count['gold'] for noc_medal_count in nc_medal_count_gd_list[:10]]
ground_truth_totals = [noc_medal_count['gold_gd'] for noc_medal_count in nc_medal_count_gd_list[:10]]
plot_total_medals(nocs, predicted_totals, ground_truth_totals)

nc_medal_count_gd_list = sorted(nc_medal_count_gd_list, key=lambda x: x['silver'], reverse=True)
nocs = [noc_medal_count['noc'] for noc_medal_count in nc_medal_count_gd_list[:10]]
predicted_totals = [noc_medal_count['silver'] for noc_medal_count in nc_medal_count_gd_list[:10]]
ground_truth_totals = [noc_medal_count['silver_gd'] for noc_medal_count in nc_medal_count_gd_list[:10]]
plot_total_medals(nocs, predicted_totals, ground_truth_totals)

nc_medal_count_gd_list = sorted(nc_medal_count_gd_list, key=lambda x: x['bronze'], reverse=True)
nocs = [noc_medal_count['noc'] for noc_medal_count in nc_medal_count_gd_list[:10]]
predicted_totals = [noc_medal_count['bronze'] for noc_medal_count in nc_medal_count_gd_list[:10]]
ground_truth_totals = [noc_medal_count['bronze_gd'] for noc_medal_count in nc_medal_count_gd_list[:10]]
plot_total_medals(nocs, predicted_totals, ground_truth_totals)
