import pandas as pd 

df = pd.read_csv("DECISIONTREE/cars.csv")
print(df.head())

inputs = df.drop("price_more_than_50k", axis = "columns")
# print(inputs)

targets = df["price_more_than_50k"]

from sklearn.preprocessing import LabelEncoder

le_brand = LabelEncoder()
le_type = LabelEncoder()
le_engine = LabelEncoder()

inputs["brand_n"] = le_brand.fit_transform(inputs["car_brand"])
inputs["type_n"] = le_type.fit_transform(inputs["car_type"])
inputs["engine_n"] = le_engine.fit_transform(inputs["engine_size"])

# print(inputs)

inputs_n = inputs.drop(["car_brand", "car_type", "engine_size"], axis = "columns")

print(inputs_n)

from sklearn import tree

model = tree.DecisionTreeClassifier()
model.fit(inputs_n, targets)

print(model.predict([[2,2,0]]))