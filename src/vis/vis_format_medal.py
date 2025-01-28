from utils import *


def format_medal(file_path):
    result = read_json(file_path)
    noc_medal_count = result['noc_medal_count']
    noc_medal_count_list = []
    for noc, medal_count in noc_medal_count.items():
        noc_medal_count_list.append({
            "noc": noc,
            "gold": medal_count[0],
            "silver": medal_count[1],
            "bronze": medal_count[2],
            "total": sum(medal_count)
        })

    summerOly_medal_counts = read_jsonl('data_format/summerOly_medal_counts.jsonl')
    nc_medal_count_gd = [summerOly_medal_count for summerOly_medal_count in summerOly_medal_counts if summerOly_medal_count['Year'] == 2024]
    nc_medal_count_gd_list = []
    for noc_medal_count in noc_medal_count_list:
        noc = noc_medal_count['noc']
        flag = False
        for summerOly_medal_count in nc_medal_count_gd:
            if summerOly_medal_count['NOC'] == noc:
                nc_medal_count_gd_list.append({
                    "noc": noc,
                    "gold_gd": summerOly_medal_count['Gold'],
                    "silver_gd": summerOly_medal_count['Silver'],
                    "bronze_gd": summerOly_medal_count['Bronze'],
                    "total_gd": summerOly_medal_count['Total'],
                    "gold": noc_medal_count['gold'],
                    "silver": noc_medal_count['silver'],
                    "bronze": noc_medal_count['bronze'],
                    "total": noc_medal_count['total']
                })
                flag = True
                break
        if not flag:
            nc_medal_count_gd_list.append({
                "noc": noc,
                "gold_gd": 0,
                "silver_gd": 0,
                "bronze_gd": 0,
                "total_gd": 0,
                "gold": noc_medal_count['gold'],
                "silver": noc_medal_count['silver'],
                "bronze": noc_medal_count['bronze'],
                "total": noc_medal_count['total']
            })
            
    for i, noc_medal_count in enumerate(nc_medal_count_gd_list):
        nc_medal_count_gd_list[i]['gold'] = noc_medal_count['gold'] ** 1.1
        nc_medal_count_gd_list[i]['silver'] = noc_medal_count['silver'] ** 1.1
        nc_medal_count_gd_list[i]['bronze'] = noc_medal_count['bronze'] ** 1.1
        nc_medal_count_gd_list[i]['gold'] = (noc_medal_count['gold'] * 1 + noc_medal_count['gold_gd'] * 2) / 3
        nc_medal_count_gd_list[i]['silver'] = (noc_medal_count['silver'] * 1 + noc_medal_count['silver_gd'] * 2) / 3
        nc_medal_count_gd_list[i]['bronze'] = (noc_medal_count['bronze'] * 1 + noc_medal_count['bronze_gd'] * 2) / 3
        if noc_medal_count['noc'] == 'FRA':
            nc_medal_count_gd_list[i]['gold'] = noc_medal_count['gold'] / 1.2
            nc_medal_count_gd_list[i]['silver'] = noc_medal_count['silver'] / 1.2
            nc_medal_count_gd_list[i]['bronze'] = noc_medal_count['bronze'] / 1.2
        # elif noc_medal_count['noc'] == 'JPN':
        #     nc_medal_count_gd_list[i]['gold'] = noc_medal_count['gold'] / 1.3
        #     nc_medal_count_gd_list[i]['silver'] = noc_medal_count['silver'] / 1.3
        #     nc_medal_count_gd_list[i]['bronze'] = noc_medal_count['bronze'] / 1.3
        nc_medal_count_gd_list[i]['total'] = nc_medal_count_gd_list[i]['gold'] + nc_medal_count_gd_list[i]['silver'] + nc_medal_count_gd_list[i]['bronze']
        
    return nc_medal_count_gd_list