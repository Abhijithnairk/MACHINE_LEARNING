import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import math

data = pd.read_csv('diabetes.csv')

df = pd.DataFrame(data)
print(df)

reg = LinearRegression()
reg.fit(df[["GlucoseLevel","BodyMassIndex","InsulinLevel","Age"]],df["ChanceofDiabetes"])

print("---Enter the values to Check the chance of Diabetes---")

glucoselevel = int(input("Enter glucose level: "))
BMI = int(input("Enter BMI: "))
insulin = int(input("Enter insulin level: "))
age = int(input("Enter age: "))

chance_of_diabetes = reg.predict([[glucoselevel,BMI,insulin,age]])

print(f"Chance of Diabetes in Percentage is: {chance_of_diabetes}")