import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from mlxtend.plotting import plot_decision_regions
from dtreeplt import dtreeplt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import tree

df = pd.read_csv('iris.csv')
# zamienić dane na numeryczne
species = {
    'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2
}
df['class_value'] = df['class'].map(species)

# sns.heatmap(df.iloc[:,:4].corr(), annot=True)
# plt.show()

# X = df[ ['sepallength', 'sepalwidth'] ]
# y = df.class_value
# model = DecisionTreeClassifier(max_depth=5, random_state=0)
# model.fit(X,y)
# plot_decision_regions(X.values, y.values, model)
# plt.show()

X = df[ ['petallength', 'petalwidth'] ]
y = df.class_value
model = DecisionTreeClassifier(max_depth=33, random_state=0)
model.fit(X,y)
# plot_decision_regions(X.values, y.values, model)
# plt.show()
# tree.plot_tree(model.fit(X,y))
# plt.show()
# dtree = dtreeplt(model=model, feature_names=X.columns, target_names=['setosa','versicolor','virginica'])
# dtree.view()
# plt.show()


X = df.iloc[:, :4]
y = df.class_value
model = DecisionTreeClassifier(max_depth=5, random_state=0)
model.fit(X, y)

print(pd.DataFrame(model.feature_importances_, X.columns))
print(pd.DataFrame(model.feature_importances_, X.columns).sum())