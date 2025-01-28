"""
拟合 s2 和 medal count 的关系，计算置信区间，并绘制带置信区间的预测图（前 10 个 NOC）
"""

from utils import *
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import t
from vis_format_medal import format_medal
import math

history_file_path = "preds/pred_error_ckpt-75_tensor([2000])_model.jsonl"
file_path = "preds/pred_ckpt-75_tensor([2000])_model.json"

def cal_e2(errors):
    for e in errors:
        e['e2'] = ((e['gd'] - e['pred']) ** 2) ** 0.5
    return errors

def plot_scatters(gd_values, e2_values):
    plt.scatter(gd_values, e2_values, alpha=0.5)
    plt.xlabel('gd')
    plt.ylabel('e2')
    plt.title('Scatter Plot of gd vs e2')
    plt.grid(True)
    plt.show()

def poly_func(x, *coefficients):
    return sum(coef * x**i for i, coef in enumerate(coefficients))

def poly_regression(gd_values, e2_values, degree, label='Total'):
    initial_guess = [1] * (degree + 1)
    
    params, _ = curve_fit(poly_func, gd_values, e2_values, p0=initial_guess)
    
    poly_eqn = " + ".join(f"{param:.3f}x^{i}" for i, param in enumerate(params))
    print(f"{label} 拟合结果: y = {poly_eqn}")
    
    plt.scatter(gd_values, e2_values, alpha=0.5, label='Data')
    x_fit = np.linspace(min(gd_values), max(gd_values), 100)
    y_fit = poly_func(x_fit, *params)
    plt.plot(x_fit, y_fit, color='red', label=f'Fitted Curve (Degree {degree})')
    plt.xlabel(f'Predicted {label} Medal Count')
    plt.ylabel('e2')
    plt.title(f'Polynomial Fit of Predicted {label} Medal Count vs e2')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'figs/poly_fit_{label}.png')
    plt.close()
    
    return params

def calculate_t_alpha_2(alpha, nu):
    t_alpha_2 = t.ppf(1 - alpha / 2, nu)
    return t_alpha_2


history = read_jsonl(history_file_path)

errors = [r['errors'] for r in history]
gold_errors = []
silver_errors = []
bronze_errors = []
total_errors = []
for e in errors:
    gold_errors.extend(e['gold'])
    silver_errors.extend(e['silver'])
    bronze_errors.extend(e['bronze'])
    total_errors.extend(e['total'])

gold_errors = cal_e2(gold_errors)
silver_errors = cal_e2(silver_errors)
bronze_errors = cal_e2(bronze_errors)
total_errors = cal_e2(total_errors)

params_dict = {}
catagories = ['Gold', 'Silver', 'Bronze', 'total']
for i, errors in enumerate([gold_errors, silver_errors, bronze_errors, total_errors]):
    gd_values = [e['pred'] for e in errors]
    e2_values = [e['e2'] for e in errors]
    params = poly_regression(gd_values, e2_values, 1, label=catagories[i])
    params_dict[catagories[i]] = params
print(params_dict)

n = 29
p = 16
nu = n - p
alpha = 0.1
t_alpha_2 = calculate_t_alpha_2(alpha, nu)

nc_medal_count_gd_list = format_medal(file_path)

for nc_medal_count_gd in nc_medal_count_gd_list:
    gold_e2 = max(0, poly_func(nc_medal_count_gd['gold'], *params_dict['Gold']))
    silver_e2 = max(0, poly_func(nc_medal_count_gd['silver'], *params_dict['Silver']))
    bronze_e2 = max(0, poly_func(nc_medal_count_gd['bronze'], *params_dict['Bronze']))
    total_e2 = max(0, poly_func(nc_medal_count_gd['total'], *params_dict['total']))
    nc_medal_count_gd['gold_interval'] = t_alpha_2 * math.sqrt(n / nu * gold_e2)
    nc_medal_count_gd['silver_interval'] = t_alpha_2 * math.sqrt(n / nu * silver_e2)
    nc_medal_count_gd['bronze_interval'] = t_alpha_2 * math.sqrt(n / nu * bronze_e2)
    nc_medal_count_gd['total_interval'] = t_alpha_2 * math.sqrt(n / nu * total_e2)


def plot_total_medals_with_interval(nocs, total_gd, totals, total_intervals, label='Total'):
    fig, ax = plt.subplots(figsize=(10, 5))

    bar_width = 0.6
    index = np.arange(len(nocs))
    ax.bar(index, total_gd, bar_width, label='GD 2024', color='skyblue', alpha=0.7)

    for i, (total, interval) in enumerate(zip(totals, total_intervals)):
        lower, upper = total - interval, total + interval
        lower = max(lower, 0)
        ax.plot([i, i], [lower, upper], color='red', marker='_', markersize=10, label='90% Confidence Interval for 2028' if i == 0 else "")
        ax.plot(i, total, 'ro', label='Point Estimate for 2028' if i == 0 else "")

    # Adjusting font sizes
    ax.set_xticks(index)
    ax.set_xticklabels(nocs, rotation=45, fontsize=16)  # Adjust xtick font size

    ax.set_xlabel('NOC', fontsize=16)  # Adjust xlabel font size
    ax.set_ylabel(f'{label} Medal Count', fontsize=18)  # Adjust ylabel font size
    ax.set_title(f'Top {len(nocs)} NOCs with the Largest {label} Medal Count', fontsize=20)  # Adjust title font size
    
    # Adjusting ytick and xtick label font sizes
    ax.tick_params(axis='both', labelsize=16)  # Adjust tick label font size

    ax.legend(fontsize=16)  # Adjust legend font size
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(f'figs/total_medals_with_interval_{label}.png')



nc_medal_count_gd_list = sorted(nc_medal_count_gd_list, key=lambda x: x['total'], reverse=True)
nocs = [item['noc'] for item in nc_medal_count_gd_list[:10]]
total_gd = [item['total_gd'] for item in nc_medal_count_gd_list[:10]]
totals = [item['total'] for item in nc_medal_count_gd_list[:10]]
total_intervals = [item['total_interval'] for item in nc_medal_count_gd_list[:10]]
plot_total_medals_with_interval(nocs, total_gd, totals, total_intervals, label='Total')

nc_medal_count_gd_list = sorted(nc_medal_count_gd_list, key=lambda x: x['gold'], reverse=True)
nocs = [item['noc'] for item in nc_medal_count_gd_list[:10]]
total_gd = [item['gold_gd'] for item in nc_medal_count_gd_list[:10]]
totals = [item['gold'] for item in nc_medal_count_gd_list[:10]]
total_intervals = [item['gold_interval'] for item in nc_medal_count_gd_list[:10]]
plot_total_medals_with_interval(nocs, total_gd, totals, total_intervals, label='Gold')

nc_medal_count_gd_list = sorted(nc_medal_count_gd_list, key=lambda x: x['silver'], reverse=True)
nocs = [item['noc'] for item in nc_medal_count_gd_list[:10]]
total_gd = [item['silver_gd'] for item in nc_medal_count_gd_list[:10]]
totals = [item['silver'] for item in nc_medal_count_gd_list[:10]]
total_intervals = [item['silver_interval'] for item in nc_medal_count_gd_list[:10]]
plot_total_medals_with_interval(nocs, total_gd, totals, total_intervals, label='Silver')

nc_medal_count_gd_list = sorted(nc_medal_count_gd_list, key=lambda x: x['bronze'], reverse=True)
nocs = [item['noc'] for item in nc_medal_count_gd_list[:10]]
total_gd = [item['bronze_gd'] for item in nc_medal_count_gd_list[:10]]
totals = [item['bronze'] for item in nc_medal_count_gd_list[:10]]
total_intervals = [item['bronze_interval'] for item in nc_medal_count_gd_list[:10]]
plot_total_medals_with_interval(nocs, total_gd, totals, total_intervals, label='Bronze')
