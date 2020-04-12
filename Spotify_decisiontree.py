import pandas as pd  
import numpy as np  
df=pd.read_csv("s.csv")
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
X=df[['danceability']]
Y=df[['key']]
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1)
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
p1 = dt.predict([['1.719']])
p1[0]

