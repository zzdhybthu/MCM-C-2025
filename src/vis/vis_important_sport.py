"""
绘制对每个国家最重要的项目
"""

from utils import *
from matplotlib import pyplot as plt
import pandas as pd

file_path = "preds/pred_ckpt-75_tensor([2000])_model.json"

result = read_json(file_path)
pred = result['pred']


def main(total=True):
    country_sport_medal_dict = {}

    for sport, v in pred.items():
        for athlete, medal_counts in v.items():
            noc = athlete.split('_')[0]
            if noc not in country_sport_medal_dict:
                country_sport_medal_dict[noc] = {}
            if sport not in country_sport_medal_dict[noc]:
                country_sport_medal_dict[noc][sport] = 0
            if total:
                country_sport_medal_dict[noc][sport] += sum(medal_counts)
            else:
                country_sport_medal_dict[noc][sport] += medal_counts[0]

    country_important_medal = []
    for noc, sport_medal in country_sport_medal_dict.items():
        important_sport = max(sport_medal, key=sport_medal.get)
        country_important_medal.append({
            "noc": noc,
            "sport": important_sport,
            "medal": sport_medal[important_sport],
        })
    country_important_medal = sorted(country_important_medal, key=lambda x: x['medal'], reverse=True)

    country_important_medal = country_important_medal[:10]

    df = pd.DataFrame(country_important_medal)
    grouped = df.groupby(['noc', 'sport'])['medal'].sum().reset_index()

    pivot_df = grouped.pivot(index='noc', columns='sport', values='medal').fillna(0)
    pivot_df['total'] = pivot_df.sum(axis=1)
    pivot_df = pivot_df.sort_values('total', ascending=True).drop('total', axis=1)

    sports = pivot_df.columns
    colors = plt.cm.tab20(range(len(sports)))

    fig, ax = plt.subplots(figsize=(9, 5))
    pivot_df.plot.barh(
        ax=ax,
        stacked=True,
        color=colors,
        width=0.6
    )
    
    # Adjust font sizes for axis labels and title
    ax.set_xlabel('Medal Count', fontsize=16)  # Adjust xlabel font size
    ax.set_ylabel('NOC', fontsize=16)  # Adjust ylabel font size
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    # Adjust legend
    plt.legend(
        title='Events',
        loc='lower right',
        fontsize=14,  # Adjust legend font size
        title_fontsize=14
    )
    
    # Adjust title font size
    label = 'Total' if total else 'Gold'
    plt.title(f'Top 10 NOCs with the Most Important Sport (By {label} Medal Count)', fontsize=16)
    
    # Adjust layout to avoid clipping of labels
    plt.tight_layout()
    plt.savefig(f'figs/important_sport_{label}.png')
    plt.close()

if __name__ == '__main__':
    main()
    main(False)