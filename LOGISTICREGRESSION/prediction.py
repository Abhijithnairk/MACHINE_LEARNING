import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = {
    "SquareFeet": [2000, 1500, 2200, 1200, 1800, 2500, 1900, 2300, 2600, 2100, 2800, 1700, 2400, 2000, 3000, 1600, 2800, 3200, 1800, 2000],
    "HouseAge": [5, 3, 6, 2, 4, 5, 5, 7, 8, 6, 7, 4, 5, 3, 6, 2, 8, 7, 4, 5],
    "Price($)": [300000, 250000, 320000, 200000, 280000, 350000, 270000, 330000, 360000, 310000, 380000, 240000, 320000, 300000, 400000, 230000, 380000, 420000, 260000, 280000]
}

df = pd.DataFrame(data)

# plt.figure(figsize=(8,6))
# plt.scatter(data["SquareFeet"],data["Price($)"],color='red',marker='x')
# plt.title("Realationship between Squarefeet and Selling price")
# plt.xlabel("sqrt")
# plt.ylabel("price")
# plt.show()

## 30% for test sample and 70% for training sample
x = df[["SquareFeet","HouseAge"]]
y = df["Price($)"]

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