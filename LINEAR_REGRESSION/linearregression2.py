import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import math

data = {
  "Area": [2600, 3000, 3200, 3600, 4000],
  "bedrooms": [3, 4, None, 3, 5],
  "age": [20, 15, 18, 30, 8],
  "price": [550000, 565000, 610000, 595000, 760000]
}

df = pd.DataFrame(data)
# print(df)

medianbedroom = math.floor(df.bedrooms.median())
# print(medianbedroom)
df.bedrooms = df.bedrooms.fillna(medianbedroom)
print(df)

reg = LinearRegression()
reg.fit(df[["Area","bedrooms","age"]],df["price"])
print("---Enter the values to predict the price---")
Area = int(input("Enter the area: "))
Bedroom = int(input("Enter the bedrooms: "))
Age = int(input("Enter the age: "))

new_price = reg.predict([[Area,Bedroom,Age]])
print(f"---Aproximate price will be---{new_price}")

print(reg.coef_)                             # price, y = m1x1 + m2x2 + m3x3 + c
print(reg.intercept_)                         #   m1 = area ,x1 = reg coef, m2= bedroom , x2= regcoef2, m3 = age, x3= regcoef3 , c= reg intercept
