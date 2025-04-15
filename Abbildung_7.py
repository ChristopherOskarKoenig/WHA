import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv("2_HICP_OGAP_EINF.csv")
df.columns = df.columns.str.strip()
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df = df.dropna(subset=["Country", "Year", "HICP", "Output_GAP", "Expected_Inflation"])

df_panel = df.set_index(["Country", "Year"])

time_periods = {
    "2002–2007": (2002, 2007),
    "2008–2013": (2008, 2013),
    "2014–2019": (2014, 2019),
    "2020–2024": (2020, 2024)
}

x_min, x_max = df["Output_GAP"].min() - 1, df["Output_GAP"].max() + 1
y_min, y_max = df["Expected_Inflation"].min() - 0.2, df["Expected_Inflation"].max() + 0.2
z_min, z_max = df["HICP"].min() - 0.5, df["HICP"].max() + 0.5

fig = plt.figure(figsize=(16, 12))

for i, (label, (start, end)) in enumerate(time_periods.items()):

    df_period = df_panel.loc[(df_panel.index.get_level_values("Year") >= start) &
                             (df_panel.index.get_level_values("Year") <= end)]

    if df_period.empty:
        continue

    X = df_period[["Output_GAP", "Expected_Inflation"]].values
    Z = df_period["HICP"].values

    model = LinearRegression()
    model.fit(X, Z)

    x_surf, y_surf = np.meshgrid(
        np.linspace(x_min, x_max, 50),
        np.linspace(y_min, y_max, 50)
    )
    z_surf = model.predict(np.c_[x_surf.ravel(), y_surf.ravel()]).reshape(x_surf.shape)

    ax = fig.add_subplot(2, 2, i + 1, projection='3d')
    ax.plot_surface(x_surf, y_surf, z_surf, cmap='viridis', alpha=0.7)

    ax.scatter(df_period["Output_GAP"], df_period["Expected_Inflation"], df_period["HICP"],
               color='blue', s=20, depthshade=False)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    ax.set_xlabel("Output Gap in %")
    ax.set_ylabel("Expected Inflation in %")
    ax.set_zlabel("HICP in %")
    ax.set_title(label, fontsize=11)

plt.tight_layout()

plt.show()