"""
稍微改变某个项目的奖牌数对该项目的影响
"""

from utils import *

file_before = "preds/pred_ckpt-75_tensor([2000])_model.json"
file_after = "preds/Weightlifting_ckpt-75_tensor([2000])_model.json"

result_before = read_json(file_before)
result_after = read_json(file_after)
pred_before = result_before['pred']
pred_after = result_after['pred']


def get_sport_dict(pred):
    country_sport_medal_dict = {}

    for sport, v in pred.items():
        for athlete, medal_counts in v.items():
            noc = athlete.split('_')[0]
            if noc not in country_sport_medal_dict:
                country_sport_medal_dict[noc] = {}
            if sport not in country_sport_medal_dict[noc]:
                country_sport_medal_dict[noc][sport] = 0
            # country_sport_medal_dict[noc][sport] += sum(medal_counts)
            country_sport_medal_dict[noc][sport] += medal_counts[0]

    return country_sport_medal_dict

country_sport_medal_dict_before = get_sport_dict(pred_before)
country_sport_medal_dict_after = get_sport_dict(pred_after)

print(country_sport_medal_dict_before['CHN']['Weightlifting'])
print(country_sport_medal_dict_after['CHN']['Weightlifting'])
