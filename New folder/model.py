import pandas as pd
import numpy as np
import pickle


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from math import sqrt
from sklearn import tree

data = pd.read_csv("abalone.csv")

le=LabelEncoder()
data['Sex']=le.fit_transform(data['Sex'])

y = data['Rings']
X = data.drop('Rings', axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, shuffle=True)

model1 = DecisionTreeRegressor()

model1.fit(X_train, y_train)


filename='model.pkl'
pickle.dump(model1, open('model.pkl', 'wb'))

