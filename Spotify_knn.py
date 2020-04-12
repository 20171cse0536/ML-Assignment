import pandas as pd  
import numpy as np  
df=pd.read_csv("s.csv")
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split 
X=df[['danceability']]
Y=df[['key']]
X_train,X_test,y_train,y_test= train_test_split(X, Y, test_size = 0.2, random_state=42) 
knn = KNeighborsClassifier(n_neighbors=7) 
knn=KNeighborsClassifier()
knn.fit(X_train,y_train)
p1=knn.predict([['1.519']])
p1[0]
