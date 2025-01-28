"""
绘制进步和退步最大的国家
"""

from utils import *
from matplotlib import pyplot as plt
import numpy as np
from vis_format_medal import format_medal

file_path = "preds/pred_ckpt-75_tensor([2000])_model.json"

nc_medal_count_gd_list = format_medal(file_path)

def plot_progress(nocs, total_gd, totals, improvements, label='Total'):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    bar_width = 0.4
    index = np.arange(len(nocs))
    ax1.bar(index - bar_width/2, totals, bar_width, label=f'Predicted {label} for 2028', color='orange', alpha=0.7)
    ax1.bar(index + bar_width/2, total_gd, bar_width, label='GD 2024', color='skyblue', alpha=0.7)
    
    # Adjust font sizes for the first axis
    ax1.set_xlabel('NOC', fontsize=16)  # Adjust xlabel font size
    ax1.set_ylabel(f'{label} Medal Count', fontsize=18)  # Adjust ylabel font size
    ax1.set_xticks(index)
    ax1.set_xticklabels(nocs, rotation=45, fontsize=16)  # Adjust xtick font size
    ax1.tick_params(axis='both', labelsize=16)  # Adjust tick label font size
    
    # Second axis (for improvement line)
    ax2 = ax1.twinx()
    ax2.plot(index, improvements, color='red', marker='o', label='Improvement')
    ax2.set_ylabel('Improvement', fontsize=18)  # Adjust y-axis label font size
    ax2.tick_params(axis='y', labelsize=16)  # Adjust y-axis tick label font size
    
    # Adjusting the legend font size
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=16)  # Adjust legend font size
    
    # Title font size
    plt.title(f'Top 10 NOCs with the Largest Improvement ({label} Medals)', fontsize=20)
    
    plt.tight_layout()
    plt.savefig(f'figs/progress_{label}.png')
    plt.close()



def plot_regress(nocs, total_gd, totals, regressions, label='Total'):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    bar_width = 0.4
    index = np.arange(len(nocs))
    ax1.bar(index - bar_width/2, totals, bar_width, label=f'Predicted {label} for 2028', color='orange', alpha=0.7)
    ax1.bar(index + bar_width/2, total_gd, bar_width, label=f'GD 2024', color='skyblue', alpha=0.7)
    
    # Adjust font sizes for the first axis
    ax1.set_xlabel('NOC', fontsize=16)
    ax1.set_ylabel(f'{label} Medal Count', fontsize=18)
    ax1.set_xticks(index)
    ax1.set_xticklabels(nocs, rotation=45, fontsize=16)  # Adjust xtick font size
    ax1.tick_params(axis='both', labelsize=16)  # Adjust tick label font size
    
    # Second axis (for regression line)
    ax2 = ax1.twinx()
    ax2.plot(index, regressions, color='red', marker='o', label='Regression')
    ax2.set_ylabel('Regression', fontsize=18)  # Adjust y-axis label font size
    ax2.tick_params(axis='y', labelsize=16)  # Adjust y-axis tick label font size
    
    # Adjusting the legend font size
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=16)  # Adjust legend font size
    
    # Title font size
    plt.title(f'Top 10 NOCs with the Largest Regression ({label} Medals)', fontsize=20)
    
    plt.tight_layout()
    plt.savefig(f'figs/regress_{label}.png')
    plt.close()

    
nc_medal_count_gd_list.sort(key=lambda x: x['total'] - x['total_gd'], reverse=True)
top_10 = nc_medal_count_gd_list[:10]
nocs = [item['noc'] for item in top_10]
total_gd = [item['total_gd'] for item in top_10]
totals = [item['total'] for item in top_10]
improvements = [item['total'] - item['total_gd'] for item in top_10]
plot_progress(nocs, total_gd, totals, improvements, label='Total')

nc_medal_count_gd_list.sort(key=lambda x: x['gold'] - x['gold_gd'], reverse=True)
top_10 = nc_medal_count_gd_list[:10]
nocs = [item['noc'] for item in top_10]
total_gd = [item['gold_gd'] for item in top_10]
totals = [item['gold'] for item in top_10]
improvements = [item['gold'] - item['gold_gd'] for item in top_10]
plot_progress(nocs, total_gd, totals, improvements, label='Gold')

nc_medal_count_gd_list.sort(key=lambda x: x['total_gd'] - x['total'], reverse=True)
top_10_regress = nc_medal_count_gd_list[:10]
nocs = [item['noc'] for item in top_10_regress]
total_gd = [item['total_gd'] for item in top_10_regress]
totals = [item['total'] for item in top_10_regress]
regressions = [item['total_gd'] - item['total'] for item in top_10_regress]
plot_regress(nocs, total_gd, totals, regressions, label='Total')

nc_medal_count_gd_list.sort(key=lambda x: x['gold_gd'] - x['gold'], reverse=True)
top_10_regress = nc_medal_count_gd_list[:10]
nocs = [item['noc'] for item in top_10_regress]
total_gd = [item['gold_gd'] for item in top_10_regress]
totals = [item['gold'] for item in top_10_regress]
regressions = [item['gold_gd'] - item['gold'] for item in top_10_regress]
plot_regress(nocs, total_gd, totals, regressions, label='Gold')
