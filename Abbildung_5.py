import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib as mpl
from sklearn.linear_model import LinearRegression

csv_path = "2_HICP_OGAP_EINF.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Datei nicht gefunden unter: {csv_path}")

df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()
df['Year'] = df['Year'].astype(int)
finnland_df = df[df['Country'] == 'Finnland'].copy()

def assign_period(year):
    if 2002 <= year <= 2007:
        return '2002-2007'
    elif 2008 <= year <= 2013:
        return '2008-2013'
    elif 2014 <= year <= 2019:
        return '2014-2019'
    elif 2020 <= year <= 2024:
        return '2020-2024'
    return 'Andere'

finnland_df['Period'] = finnland_df['Year'].apply(assign_period)

mpl.rcParams['font.family'] = 'Arial'
sns.set_style("white")

palette = {
    '2002-2007': '#9ecae1',
    '2008-2013': '#6baed6',
    '2014-2019': '#4292c6',
    '2020-2024': '#2171b5'
}
color_hicp = '#6baed6'
color_ogap = '#fd8d3c'

x_min, x_max = -5, 5
y_min, y_max = -2, 8

fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=120, constrained_layout=True)
fig.text(0.04, 0.5, 'HICP-Inflationsrate (%)', va='center', rotation='vertical', fontsize=11)

axes[0].plot(finnland_df['Year'], finnland_df['HICP'], label='HICP-Inflation (%)',
             color=color_hicp, linewidth=2, marker='o')
axes[0].plot(finnland_df['Year'], finnland_df['Output_GAP'], label='Output Gap (%)',
             color=color_ogap, linewidth=2, linestyle='--', marker='s')
axes[0].set_title("Zeitverlauf: Output Gap & HICP – Finnland", fontsize=11, fontweight='bold')
axes[0].set_xlabel("Jahr")

y_buffer = 0.5
ogap_min = finnland_df['Output_GAP'].min()
hicp_max = finnland_df['HICP'].max()
combined_min = min(ogap_min, finnland_df['HICP'].min()) - y_buffer
combined_max = max(hicp_max, finnland_df['Output_GAP'].max()) + y_buffer
axes[0].set_ylim(combined_min, combined_max)
axes[0].set_xlim(finnland_df['Year'].min() - 0.5, finnland_df['Year'].max() + 0.5)

axes[0].set_box_aspect(1)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].grid(True, linestyle=':', linewidth=0.5)
axes[0].legend()

sns.regplot(data=finnland_df, x='Output_GAP', y='HICP', ax=axes[1],
            scatter_kws={'color': color_hicp, 's': 50, 'alpha': 0.7},
            line_kws={'color': 'black', 'linewidth': 2}, ci=None)
axes[1].set_title("Phillips-Kurve – Finnland (2002–2024)", fontsize=11, fontweight='bold')
axes[1].set_xlabel("Output Gap (%)")
axes[1].set_xlim(x_min, x_max)
axes[1].set_ylim(y_min, y_max)
axes[1].set_box_aspect(1)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].grid(True, linestyle=':', linewidth=0.5)
axes[1].axvline(x=0, linestyle=':', color='black', linewidth=1)
axes[1].axhline(y=0, linestyle=':', color='black', linewidth=1)
axes[1].axhline(y=2, linestyle='--', color='green', linewidth=1)

model = LinearRegression()
X = finnland_df[['Output_GAP']]
y = finnland_df['HICP']
model.fit(X, y)
intercept = model.intercept_
axes[1].axhline(y=intercept, linestyle='--', color='red', linewidth=1)

sns.scatterplot(data=finnland_df, x='Output_GAP', y='HICP', hue='Period', ax=axes[2],
                palette=palette, s=60, edgecolor='black', linewidth=0.4)
for period, pdata in finnland_df.groupby('Period'):
    sns.regplot(data=pdata, x='Output_GAP', y='HICP', ax=axes[2],
                scatter=False, line_kws={'color': palette[period], 'linewidth': 2}, ci=None)
axes[2].set_title("Phillips-Kurve – nach Zeitperioden", fontsize=11, fontweight='bold')
axes[2].set_xlabel("Output Gap (%)")
axes[2].set_xlim(x_min, x_max)
axes[2].set_ylim(y_min, y_max)
axes[2].set_box_aspect(1)
axes[2].spines['top'].set_visible(False)
axes[2].spines['right'].set_visible(False)
axes[2].grid(True, linestyle=':', linewidth=0.5)
axes[2].axvline(x=0, linestyle=':', color='black', linewidth=1)
axes[2].axhline(y=0, linestyle=':', color='black', linewidth=1)
axes[2].axhline(y=2, linestyle='--', color='green', linewidth=1)
axes[2].legend(title='Periode')

plt.show()