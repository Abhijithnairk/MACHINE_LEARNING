import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import math

data = pd.read_csv('patients.csv')

df = pd.DataFrame(data)
print(df)

reg = LinearRegression()
reg.fit(df[["Troponin","Age","BloodPressure","Cholesterol"]],df["ChanceofHeartAttack"])

print("---Enter the values to predict the chance Attack---")

tropin = int(input("Enter the tropin level: "))
age = int(input("Enter the age: "))
bloodpressure = int(input("Enter blood pressure: "))
cholesterol = int(input("Enter cholesterol level: "))

chance_of_attack = reg.predict([[tropin,age,bloodpressure,cholesterol]])

print(f"Chance of a Heart Attack in Percentage is, {chance_of_attack}")