import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

plt.rcParams["font.family"] = "Arial"

df = pd.read_csv("2_HICP_OGAP_EINF.csv")

X = df[["Output_GAP", "Expected_Inflation"]].values
y = df["HICP"].values

model = LinearRegression()
model.fit(X, y)

x_surf, y_surf = np.meshgrid(
    np.linspace(X[:, 0].min(), X[:, 0].max(), 50),
    np.linspace(X[:, 1].min(), X[:, 1].max(), 50)
)
z_surf = model.predict(np.c_[x_surf.ravel(), y_surf.ravel()]).reshape(x_surf.shape)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(x_surf, y_surf, z_surf, cmap='viridis', alpha=0.8)

ax.scatter(X[:, 0], X[:, 1], y, color='blue', s=30, label='Datenpunkte', depthshade=False)

ax.set_xlabel("Output Gap in %")
ax.set_ylabel("Erwartete HICP-Inflationsrate in %")
ax.set_zlabel("HICP-Inflationsrate in %")

fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Modellierte HICP-Inflationsrate')

ax.legend(loc='upper right', bbox_to_anchor=(1.2 , 0.85), frameon=True)

plt.tight_layout()
plt.show()
