### IMPORTS
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from causal_tree import CausalTree
from sklearn.tree import DecisionTreeClassifier
import time

### LOAD DATA
print("Load data")
df = pd.read_csv("testdata.csv")
df = df.sample(frac=0.1, random_state=42)

### SPLIT DATA
print("Split data")
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
all_features = df.columns.values[:12].tolist()

### FIT CAUSAL TREE
X_train = df_train[all_features + ['treatment']].rename(columns={"treatment":"treated"})
y_train = df_train.visit
#tuned_parameters = [{'min_samples_leaf': [2000, 4000, 8000, 16000, 32000]}]
#grid_tree = GridSearchCV(CausalTree(random_state=42), tuned_parameters, cv=5, verbose=10, n_jobs=-1)
#grid_tree.fit(X_train, y_train)
#print("Best parameters set found on development set:")
#print(grid_tree.best_params_)
#causal_tree = grid_tree.best_estimator_

causal_tree = CausalTree(random_state=42, min_samples_leaf=32000)
causal_tree.fit(X_train, y_train)
print("Done")