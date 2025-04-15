import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib as mpl
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress

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
    '2002-2007': '#464646',
    '2008-2013': '#8DEBCC',
    '2014-2019': '#70C1E3',
    '2020-2024': '#8368B5'
}
color_hicp = '#6baed6'
color_ogap = '#A678D1'
rot = '#E33620'
gruen = '#7CF24D'

x_min = finnland_df['Output_GAP'].min() - 0.5
x_max = finnland_df['Output_GAP'].max() + 0.5
y_min = finnland_df['HICP'].min() - 1
y_max = finnland_df['HICP'].max() + 1

fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=120)
fig.subplots_adjust(wspace=0.1)

axes[0].plot(finnland_df['Year'], finnland_df['HICP'], label='HICP-Inflation (%)',
             color=color_hicp, linewidth=2, marker='o')
axes[0].plot(finnland_df['Year'], finnland_df['Output_GAP'], label='Output Gap (%)',
             color=color_ogap, linewidth=2, linestyle='--', marker='s')
axes[0].set_title("Zeitverlauf: Output Gap & HICP-Inflationsrate", fontsize=11)
axes[0].set_xlabel("Jahr")
axes[0].set_ylabel("HICP-Inflationsrate in %", labelpad=1)
combined_min = min(finnland_df['Output_GAP'].min(), finnland_df['HICP'].min()) - 0.5
combined_max = max(finnland_df['Output_GAP'].max(), finnland_df['HICP'].max()) + 0.5
axes[0].set_ylim(combined_min, combined_max)
axes[0].set_xlim(finnland_df['Year'].min() - 0.5, finnland_df['Year'].max() + 0.5)
axes[0].set_box_aspect(1)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].tick_params(axis='both', direction='out', length=3)
axes[0].grid(True, linestyle=':', linewidth=0.5)
axes[0].legend(loc='upper left')

sns.scatterplot(
    data=finnland_df,
    x='Output_GAP',
    y='HICP',
    ax=axes[1],
    color=color_hicp,
    s=60,
    alpha=0.7,
    edgecolor='black',
    linewidth=0.4,
    label="Datenpunkte"
)
model = LinearRegression()
X = finnland_df[['Output_GAP']]
y = finnland_df['HICP']
model.fit(X, y)
slope = model.coef_[0]
intercept = model.intercept_
x_vals = pd.Series([finnland_df['Output_GAP'].min(), finnland_df['Output_GAP'].max()])
y_vals = intercept + slope * x_vals
axes[1].plot(x_vals, y_vals, color='black', linewidth=2, label="Regressionsgerade")

axes[1].set_title("Phillips-Kurve – Finnland (2002–2024)", fontsize=11)
axes[1].set_xlabel("Output Gap in %")
axes[1].set_ylabel("")
axes[1].set_xlim(x_min, x_max)
axes[1].set_ylim(y_min, y_max)
axes[1].set_box_aspect(1)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].tick_params(axis='both', direction='out', length=3)
axes[1].grid(True, linestyle=':', linewidth=0.5)
axes[1].axvline(x=0, linestyle=':', color='black', linewidth=1)
axes[1].axhline(y=0, linestyle=':', color='black', linewidth=1)
axes[1].axhline(y=2, linestyle='--', color=gruen, linewidth=1)
axes[1].axhline(y=intercept, linestyle='--', color=rot, linewidth=1)
axes[1].legend(loc='upper left')

sns.scatterplot(
    data=finnland_df,
    x='Output_GAP',
    y='HICP',
    hue='Period',
    ax=axes[2],
    palette=palette,
    s=60,
    alpha=0.7,
    edgecolor='black',
    linewidth=0.4
)
for period, pdata in finnland_df.groupby('Period'):
    if len(pdata) >= 2:
        x = pdata['Output_GAP']
        y = pdata['HICP']
        slope, intercept, *_ = linregress(x, y)
        x_vals = pd.Series([x.min(), x.max()])
        y_vals = intercept + slope * x_vals
        axes[2].plot(x_vals, y_vals, color=palette[period], linewidth=2)

axes[2].set_title("Phillips-Kurve – nach Zeitperioden", fontsize=11)
axes[2].set_xlabel("Output Gap in %")
axes[2].set_ylabel("")
axes[2].set_xlim(x_min, x_max)
axes[2].set_ylim(y_min, y_max)
axes[2].set_box_aspect(1)
axes[2].spines['top'].set_visible(False)
axes[2].spines['right'].set_visible(False)
axes[2].tick_params(axis='both', direction='out', length=3)
axes[2].grid(True, linestyle=':', linewidth=0.5)
axes[2].axvline(x=0, linestyle=':', color='black', linewidth=1)
axes[2].axhline(y=0, linestyle=':', color='black', linewidth=1)
axes[2].axhline(y=2, linestyle='--', color=gruen, linewidth=1)
axes[2].legend(title='Periode', loc='upper left')  # <-- HIER ist jetzt korrekt platziert

plt.show()