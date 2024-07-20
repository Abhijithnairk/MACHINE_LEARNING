import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = {
    'age': [22, 25, 47,  52,  46,  56,  55,  60,  62,  61,  18,  28,  27,  29,  49,  55,  25, 58,  19,  18,  21,  26,  40,  45,  50,  54,  23],
    'bought_insurance': [ 0, 0, 1,  0, 1,1, 0,1, 1, 1, 0, 0,0,0, 1, 1, 1, 1, 0,0,0,0,1,1,1,1, 0]


}

df = pd.DataFrame(data)
# print(df)

# plt.scatter(df["age"],df["bought_insurance"],color='red',marker='+')
# plt.show()

x_train,x_test,y_train,y_test = train_test_split(df[["age"]],df["bought_insurance"],test_size=0.3)
# print(len(x_train))
print(x_test)
# print(x_train)

model = LogisticRegression()
model.fit(x_train,y_train)

predicted = model.predict(x_test)
print(predicted)

# input = int(input("Enter the age to predit: "))
# answer = model.predict([[input]])
# print(f"predicted for age {input} is {answer}")