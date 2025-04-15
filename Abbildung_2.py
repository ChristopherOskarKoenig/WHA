import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

plt.rcParams['font.family'] = 'Arial'

df = pd.read_csv("1_HICP_OGAP.csv")
df.columns = df.columns.str.strip()
df['Year'] = df['Year'].astype(int)

periods = {
    '2002–2007': (2002, 2007),
    '2008–2013': (2008, 2013),
    '2014–2019': (2014, 2019),
    '2020–2024': (2020, 2024)
}

colors = {
    '2002–2007': '#464646',
    '2008–2013': '#8DEBCC',
    '2014–2019': '#70C1E3',
    '2020–2024': '#8368B5'
}

fig, ax = plt.subplots(figsize=(12, 8))

for label, (start, end) in periods.items():
    period_df = df[(df['Year'] >= start) & (df['Year'] <= end)]
    x = period_df['Output_GAP']
    y = period_df['HICP']
    ax.scatter(x, y, color=colors[label], alpha=0.35, edgecolor='black', linewidth=0.2)
    slope, intercept, *_ = linregress(x, y)
    x_vals = pd.Series(sorted(x))
    y_vals = intercept + slope * x_vals
    ax.plot(x_vals, y_vals, color=colors[label], linewidth=2.5, label=label)

ax.set_xlabel("")
ax.set_ylabel("HICP-Inflationsrate in %", fontsize=13)
ax.axhline(0, color='black', linewidth=1.2)
ax.axvline(0, color='black', linewidth=1.2)
ax.axhline(2, color='#5CFF4D', linestyle='--', linewidth=1)

ax.text(0, ax.get_ylim()[0] - 1.2, "Output Gap in %", ha='center', va='top', fontsize=13)

ax.legend(loc='upper left', title="Zeiträume", fontsize=11, title_fontsize=12)

for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.5)

ax.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()