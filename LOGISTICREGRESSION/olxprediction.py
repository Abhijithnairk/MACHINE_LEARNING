import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv('olx.csv')
df = pd.DataFrame(data)

x = df[["Year","Kilometers"]]
y = df["Price"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

print(len(x_train))
print(len(x_test))

from sklearn.linear_model import LinearRegression

clf = LinearRegression()
clf.fit(x_train,y_train)
clf.predict(x_test)

predictions = clf.predict(x_test)
print(f"prediction is :{predictions}")
print(f"actual price is :{y_test}")

score = clf.score(x_test,y_test)
print(score)
