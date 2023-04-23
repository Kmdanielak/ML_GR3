import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_decision_regions
from sklearn.linear_model import LogisticRegression

df =pd.read_csv('iris.csv')
species = {
    "Iris-setosa":0, "Iris-versicolor":1, "Iris-virginica":2
}
df["class_value"] = df["class"].map(species)

#X = df[['sepallength', 'sepalwidth']]
X = df[['petallength', 'petalwidth']]
y = df.class_value
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

# model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=0)
# model.fit(X_train, y_train)
# print(model.score(X_test, y_test))
# print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test))))
# plot_decision_regions(X_train.values, y_train.values, model)
# plt.show()
#
# model = RandomForestClassifier(n_estimators=30, max_depth=4, random_state=0)
# model.fit(X_train, y_train)
# print(model.score(X_test, y_test))
# print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test))))
# plot_decision_regions(X_train.values, y_train.values, model)
# plt.show()
#
# model = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=0)
# model.fit(X_train, y_train)
# print(model.score(X_test, y_test))
# print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test))))
# plot_decision_regions(X_train.values, y_train.values, model)
# plt.show()
#
# model = DecisionTreeClassifier(max_depth=4, random_state=0)
# model.fit(X_train, y_train)
# print(model.score(X_test, y_test))
# print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test))))
# plot_decision_regions(X_train.values, y_train.values, model)
# plt.show()

#porownanie z logistyczną regresją
print('Konfrontacja z regresją logistyczną')
df = pd.read_csv('cukrzyca.csv',sep=';', comment='#')
X = df.iloc[: , :-1]
y = df.outcome
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

model = RandomForestClassifier(n_estimators=300, max_depth=17, random_state=0)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(pd.DataFrame( confusion_matrix(y_test, model.predict(X_test)) ))
print(pd.DataFrame(model.feature_importances_, X.columns))
# plot_decision_regions(X.values, y.values, model)
# plt.show()
# print(len(X.values))
# print(len(y.values))
model = LogisticRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test))))
