import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('otodom.csv')
print(df.columns)
print(df.head(3).to_string())

# print(df.corr())   #korelacja
# sns.heatmap(df.iloc[:,2:].corr(), annot=True)
# plt.show()
#
# sns.displot(df.cena)
# plt.show()

print(df.describe().to_string())

_min = df.describe().loc['min','cena']
q1 = df.describe().loc['25%','cena']
q3 = df.describe().loc['75%','cena']

df1 = df[(df.cena >= q1) & (df.cena <= q3) & (df.rok_budowy < 2024)]
# sns.displot(df1.cena)
# plt.show()
print(df1.describe().to_string())

X = df1.iloc[:, 2:]
y = df1.cena

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X, y)
