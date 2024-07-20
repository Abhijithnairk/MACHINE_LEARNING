import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import math

data = pd.read_csv('olx.csv')

df = pd.DataFrame(data)

reg = LinearRegression()
reg.fit(df[["Year","Kilometers"]],df["Price"])

print("---Enter the values to predict the marketprice of Suzuki Access125---")
year = int(input("Enter the year: "))
km_travelled = int(input("Enter kilometers travelled: "))

new_price = reg.predict([[year,km_travelled]])
print(f"---Aproximate price will be---{new_price}")



