import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = {
    'Area': [2600, 3000, 3200, 3600, 4000],
    'Price': [550000, 565000, 610000, 680000, 725000]
}

df = pd.DataFrame(data)
print(df)

plt.scatter(df["Area"],df["Price"],color='red',marker='+')
plt.xlabel("Area")
plt.ylabel("Price $")
# # plt.show()

# fitting linear regression model
reg = LinearRegression()
reg.fit(df[["Area"]],df["Price"])

# plotting
# plt.plot(df["Area"],reg.predict(df[["Area"]]),color='blue')
price = int(input("Enter the area to predict the price: "))
val = reg.predict([[price]])
print("M",reg.coef_)
print("C",reg.intercept_)
print(val)
# plt.show()