import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams['font.family'] = 'Arial'

df = pd.read_csv("2_HICP_OGAP_EINF.csv")
df.columns = df.columns.str.strip()

rot = "#E33620"
gruen = "#7CF24D"
hellblau = "#B3DEF4"

x = df["Output_GAP"]
y = df["HICP"]
slope, intercept, *_ = stats.linregress(x, y)
x_sorted = pd.Series(sorted(x))
y_regression = slope * x_sorted + intercept

plt.figure(figsize=(10, 8))

plt.scatter(
    x, y,
    color=hellblau,
    edgecolor='gray',
    alpha=0.8,
    s=60,
    label="Datenpunkte"
)

plt.plot(
    x_sorted, y_regression,
    color=rot,
    linewidth=2,
    label="Regressionsgerade"
)

plt.axhline(y=intercept, color=rot, linestyle='--', linewidth=1, label='y-Achsenabchnitt')
plt.axhline(y=2.0, color=gruen, linestyle='--', linewidth=1, label='EZB-Inflationsziel')
plt.axhline(0, color='black', linestyle=':', linewidth=1)

plt.axvline(0, color='black', linestyle=':', linewidth=1)

plt.xlabel("")
plt.ylabel("HICP-Inflationsrate in %")

plt.text(0, plt.ylim()[0] - 1.2, "Output Gap in %", ha='center', va='top')

plt.legend(loc='upper left')

plt.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()