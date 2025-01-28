"""
绘制 Cs, Ss, Ps 的历史曲线
"""

from utils import *
from matplotlib import pyplot as plt


file_path = "preds/pred_ckpt-75_tensor([2000])_model.json"

result = read_json(file_path)
# nc_medal_count_gd_list = format_medal(file_path)

years = [1896, 1900, 1904, 1908, 1912, 1920, 1924, 1928, 1932, 1936, 1948, 1952, 1956, 1960, 1964, 1968, 1972, 1976, 1980, 1984, 1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020, 2024]

def plot_Cs(nocs, begin=0):
    plt.figure(figsize=(10, 6))
    for noc in nocs:
        history = []
        for x in result['Cs_history'][begin+1:]:
            if noc in x:
                history.append(x[noc])
            else:
                history.append(None)
        plt.plot(years[begin:], history, label=noc, marker='*')
    all_values = [x[noc] for x in result['Cs_history'][begin:] for noc in nocs if noc in x]
    y_min = min(all_values) - 0.1 * (max(all_values) - min(all_values))
    y_max = max(all_values) + 0.1 * (max(all_values) - min(all_values))
    
    plt.xticks(years[begin:], rotation=45)
    plt.xlabel('Year')
    plt.ylabel('Country/Region Performance Indicator')
    plt.title('Country/Region Performance Indicator (Cs) History')
    plt.ylim(y_min, y_max)
    plt.grid(True, axis='y')
    plt.legend(loc='upper left')
    plt.savefig('figs/Cs.png')
    plt.close()
    
def plot_Ss(sport, nocs, begin=0):
    plt.figure(figsize=(7, 5))
    
    for noc in nocs:
        history = []
        for x in result['Ss_history'][begin+1:]:
            if sport in x and noc in x[sport]:
                history.append(x[sport][noc])
            else:
                history.append(None)
        plt.plot(years[begin:], history, label=noc, marker='*')
    
    all_values = [x[sport][noc] for x in result['Ss_history'][begin:] for noc in nocs if sport in x and noc in x[sport]]
    y_min = min(all_values) - 0.1 * (max(all_values) - min(all_values))
    y_max = max(all_values) + 0.1 * (max(all_values) - min(all_values))
    
    # Adjust font sizes
    plt.xlabel('Year', fontsize=16)  # Adjust xlabel font size
    plt.ylabel('Sport-specific Performance Indicator', fontsize=14)  # Adjust ylabel font size
    plt.title(f'Sport-specific Performance Indicator (Ss) History for {sport}', fontsize=14)  # Adjust title font size
    plt.ylim(y_min, y_max)
    plt.grid(True, axis='y')
    
    # Adjust legend font size
    plt.legend(loc='upper left', fontsize=14)  # Adjust legend font size
    
    plt.savefig(f'figs/Ss_{sport}.png')
    plt.close()

    
def plot_Ss_sports(noc, sports, begin=0, bound=0.1):
    plt.figure(figsize=(8, 5))
    historys = []
    for sport in sports:
        history = []
        for x in result['Ss_history'][begin+1:]:
            if sport in x and noc in x[sport]:
                history.append(x[sport][noc])
            else:
                history.append(None)
        if len([h for h in history if h is not None]) == 0:
            continue
        history_not_none = [h for h in history if h is not None]
        if max(history_not_none) < bound or len(history_not_none) < 5 or max(history_not_none) - min(history_not_none) < 0.1:
            continue
        plt.plot(years[begin:], history, label=sport, marker='*')
        historys.extend(history)
        # print(noc, history)
    all_values = [x for x in historys if x is not None]
    y_min = min(all_values) - 0.1 * (max(all_values) - min(all_values))
    y_max = max(all_values) + 0.1 * (max(all_values) - min(all_values))
    
    plt.xlabel('Year')
    plt.ylabel('Sport-specific Performance Indicator')
    plt.title(f'Sport-specific Performance Indicator (Ss) History for {noc}')
    plt.ylim(y_min, y_max)
    plt.grid(True, axis='y')
    plt.legend(loc='upper left')
    plt.savefig(f'figs/Ss_sports_{noc}.png')
    plt.close()

def plot_Ps(sport, nocs, begin=0):
    plt.figure(figsize=(7, 5))
    historys = []
    
    for noc in nocs:
        history = []
        for x in result['Ps_history'][begin+1:]:
            if sport in x:
                noc_athletes = [athlete for athlete in x[sport] if athlete.split('_')[0] == noc]
                average = sum([x[sport][athlete] for athlete in noc_athletes]) / len(noc_athletes) if len(noc_athletes) > 0 else None
                history.append(average)
            else:
                history.append(None)
        plt.plot(years[begin:], history, label=noc, marker='*')
        historys.extend(history)
    
    all_values = [x for x in historys if x is not None]
    y_min = min(all_values) - 0.1 * (max(all_values) - min(all_values))
    y_max = max(all_values) + 0.1 * (max(all_values) - min(all_values))
    
    # Adjust font sizes
    plt.xlabel('Year', fontsize=14)  # Adjust xlabel font size
    plt.ylabel('Individual Momentum', fontsize=14)  # Adjust ylabel font size
    plt.title(f'Average Individual Momentum (Ps) History for {sport}', fontsize=14)  # Adjust title font size
    plt.ylim(y_min, y_max)
    plt.grid(True, axis='y')
    
    # Adjust legend font size
    plt.legend(loc='upper left', fontsize=14)  # Adjust legend font size
    
    plt.savefig(f'figs/Ps_{sport}.png')
    plt.close()


def plot_Ps_sports(noc, sports, begin=0, bound=0.1):
    plt.figure(figsize=(8, 5))
    historys = []
    for sport in sports:
        history = []
        for x in result['Ps_history'][begin+1:]:
            if sport in x:
                noc_athletes = [athlete for athlete in x[sport] if athlete.split('_')[0] == noc]
                average = sum([x[sport][athlete] for athlete in noc_athletes]) / len(noc_athletes) if len(noc_athletes) > 0 else None
                history.append(average)
            else:
                history.append(None)
        if len([h for h in history if h is not None]) == 0:
            continue
        history_not_none = [h for h in history if h is not None]
        if max(history_not_none) < bound or len(history_not_none) < 5 or max(history_not_none) - min(history_not_none) < 0.2:
            continue
        plt.plot(years[begin:], history, label=sport, marker='*')
        # print(noc, history)
        historys.extend(history)
    all_values = [x for x in historys if x is not None]
    y_min = min(all_values) - 0.1 * (max(all_values) - min(all_values))
    y_max = max(all_values) + 0.1 * (max(all_values) - min(all_values))
    
    plt.xlabel('Year')
    plt.ylabel('Individual Momentum')
    plt.title(f'Average Individual Momentum (Ps) History for {noc}')
    plt.ylim(y_min, y_max)
    plt.grid(True, axis='y')
    plt.legend(loc='upper left')
    plt.savefig(f'figs/Ps_sports_{noc}.png')
    plt.close()

if __name__ == '__main__':
    plot_Cs(['JPN', 'KOR', 'USA', 'AUS', 'GER', 'CHN'], begin=25)
    plot_Ss('Table Tennis', ['JPN', 'KOR', 'USA', 'AUS', 'GER', 'CHN'], begin=25)
    plot_Ss('Weightlifting', ['JPN', 'KOR', 'USA', 'AUS', 'GER', 'CHN'], begin=25)
    plot_Ss('Wrestling', ['JPN', 'KOR', 'USA', 'AUS', 'GER', 'CHN'], begin=25)
    plot_Ss('Judo', ['JPN', 'KOR', 'USA', 'AUS', 'GER', 'CHN'], begin=25)
    plot_Ps('Athletics', ['JPN', 'KOR', 'USA', 'AUS', 'GER', 'CHN'], begin=25)
    plot_Ps('Wrestling', ['JPN', 'KOR', 'USA', 'AUS', 'GER', 'CHN'], begin=25)
    
    plot_Ss('Volleyball', ['CHN', 'ITA', 'USA'], begin=15)
    plot_Ps('Volleyball', ['CHN', 'ITA', 'USA'], begin=15)
    
    plot_Ss('Gymnastics', ['ROU', 'USA'], begin=15)
    plot_Ps('Gymnastics', ['ROU', 'USA'], begin=15)
    
    summerOly_programs = read_jsonl('data_format/summerOly_programs.jsonl')
    sports = [program['Sport'] for program in summerOly_programs]
    plot_Ss_sports('CHN', sports, begin=23)
    plot_Ps_sports('CHN', sports, begin=23)
    plot_Ss_sports('USA', sports, begin=23, bound=0.25)
    plot_Ps_sports('USA', sports, begin=23, bound=0.3)
    plot_Ss_sports('JPN', sports, begin=23)
    plot_Ps_sports('JPN', sports, begin=23)