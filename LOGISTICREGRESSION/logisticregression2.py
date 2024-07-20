import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = {
    'age' : [30,35,40,45,50,55,60,65,70,75],
    'income' : [50000,55000,60000,65000,70000,75000,80000,85000,90000,95000],
    'buy_car' : [0,0,1,1,1,1,1,1,1,1]
}

df = pd.DataFrame(data)

x = df[["age","income"]]
y = df["buy_car"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

# print(len(x_train))
# print(len(x_test))

model = LogisticRegression()
model.fit(x_train,y_train)

# predicted = model.predict(x_test)
# print(predicted)

age = int(input("Enter the age to predit: "))
income = int(input("Enter income to predict: "))
answer = model.predict([[age,income]])
print(f"prediction for buying car is {answer}")
