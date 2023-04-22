import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

df = pd.read_csv('weight-height.csv')
#print(type(df))
#print(df)
print(df.head(10))

#loc - nazwy, labelki
#iloc - indeksy, pozycje
#print(df.iloc[2,2])
#print(df.iloc[ 2:4,:2])
# zmiana jednostek
df.Height *= 2.54
df.Weight /= 2.2
print(df.head(10))

# sprawdzamy rozkład danych
# sns.displot(data=df, x='Weight',hue='Gender')
# plt.show()

# zmiana na dane numeryczne
df = pd.get_dummies(df)
print(df.head(10))
del(df['Gender_Male'])
df = df.rename(columns={'Gender_Female':'Gender'})
print(df)

