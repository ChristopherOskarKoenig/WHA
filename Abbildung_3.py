import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import FormatStrFormatter

plt.rcParams["font.family"] = "Arial"

df = pd.read_csv("3_HICP_OGAP_EINF.csv")

country_colors = {
    "Niederlande": "#2E86C1",
    "Slowakei": "#5DADE2",
    "Belgien": "#A569BD",
    "Österreich": "#58D68D",
    "Kroatien": "#73C6B6",
    "Italien": "#117864",
    "Portugal": "#2980B9",
    "Deutschland": "#85929E",
    "Frankreich": "#BB8FCE",
    "Griechenland": "#76D7C4",
    "Irland": "#566573",
    "Zypern": "#F2F4F4",
    "Slowenien": "#C39BD3",
    "Lettland": "#45B39D",
    "Luxemburg": "#D7DBDD",
    "Spanien": "#154360",
    "Finnland": "#AED6F1",
    "Malta": "#D2B4DE",
    "Estland": "#7FB3D5",
    "Litauen": "#BFC9CA"
}

gruppe_A = ["Niederlande", "Slowakei", "Belgien", "Österreich", "Kroatien"]
gruppe_B = ["Italien", "Portugal", "Deutschland", "Frankreich", "Griechenland",
            "Irland", "Zypern", "Slowenien", "Lettland", "Luxemburg", "Spanien"]
gruppe_C = ["Finnland", "Malta", "Estland", "Litauen"]

gruppen = {
    "Gruppe A – Starke Sensitivität": gruppe_A,
    "Gruppe B – Moderate Sensitivität": gruppe_B,
    "Gruppe C – Schwache oder negative Sensitivität": gruppe_C
}

fig, axs = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

for idx, (ax, (titel, laender)) in enumerate(zip(axs, gruppen.items())):
    intercepts = []
    weights = []

    for land in laender:
        subset = df[df["Country"] == land]
        color = country_colors[land]

        sns.regplot(
            x="Output_GAP", y="HICP", data=subset, ax=ax,
            scatter=True, label=land, color=color,
            ci=None, line_kws={"linewidth": 1.5},
            scatter_kws={"s": 40, "alpha": 0.9}
        )


        if len(subset) >= 5:
            slope, intercept = np.polyfit(subset["Output_GAP"], subset["HICP"], 1)
            intercepts.append(intercept)
            weights.append(len(subset))

    if intercepts:
        weighted_avg_intercept = np.average(intercepts, weights=weights)
        ax.axhline(y=weighted_avg_intercept, color='red', linestyle='dashed', linewidth=0.8)

    ax.set_title(titel, fontsize=14)
    ax.axhline(y=2.0, color='#32CD32', linestyle='dashed', linewidth=0.8)  # dunkles Grau
    ax.axhline(0, color='black', linestyle='dashed', linewidth=0.5)
    ax.axvline(0, color='black', linestyle='dashed', linewidth=0.5)
    ax.set_xlabel("Output Gap in %", fontsize=12)
    ax.legend(loc="upper left", fontsize=9)

    if idx == 0:
        ax.set_ylabel("HICP-Inflationsrate in %", fontsize=12, labelpad=14)
    else:
        ax.set_ylabel("")

    ax.set_yticks(np.arange(-5, 20.1, 2.5))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f %%'))

plt.tight_layout()
plt.show()