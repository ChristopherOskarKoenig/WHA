import pandas as pd
import statsmodels.api as sm

df = pd.read_csv("HICP_OGAP.csv", sep=';')

df.columns = df.columns.str.strip()
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

for col in ['HICP', 'Output_GAP']:
    df[col] = df[col].str.replace(',', '.').astype(float)

slopes = {}

for country in df['Country'].unique():
    data = df[df['Country'] == country]
    X = sm.add_constant(data['Output_GAP'])
    y = data['HICP']
    model = sm.OLS(y, X).fit()
    slopes[country] = round(model.params['Output_GAP'], 5)

for country, slope in sorted(slopes.items(), key=lambda x: x[1], reverse=True):
    print(f"{country}: {slope}")