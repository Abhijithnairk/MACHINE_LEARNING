import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import math

data = pd.read_csv('vegetable_rice_data.csv')

df = pd.DataFrame(data)
# print(df)

meadian_tax = math.floor(df.RoadTax.median())
df.RoadTax = df.RoadTax.fillna(meadian_tax)
median_price = math.floor(df.Price.median())
df.Price = df.Price.fillna(median_price)
print(df)


reg = LinearRegression()
reg.fit(df[["PetrolCharge","RoadTax","KilometersTraveled"]],df["Price"])
print("---Enter the values to predict the price---")
petrolcharge = int(input("Enter petrolcharge: "))
roadtax = int(input("Enter roadtax: "))
km_travelled = int(input("Enter kilometers travelled: "))

new_price = reg.predict([[petrolcharge,roadtax,km_travelled]])
print(f"---Aproximate price will be---{new_price}")
