import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

# Load dataset
dataset = pd.read_csv("NAIVE_BAYES/carsnaive.csv")

# Extract features and target variable
x = dataset.iloc[:, [0, 1, 2]].values
y = dataset.iloc[:, [-1]].values

# Encode categorical variables
le_car = LabelEncoder()
le_color = LabelEncoder()

x[:, 0] = le_car.fit_transform(x[:, 0])
x[:, 1] = le_color.fit_transform(x[:, 1])

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Naive Bayes model
model = GaussianNB()

# Fit the model
model.fit(x_train, y_train.ravel())

# Prediction
print("-- Prediction --")
car = input("Enter a car (Toyota/Ford/Ferrari/Benz): ")
color = input("Enter color (Cyan/Red/Black): ")
horsepower = int(input("Enter the horsepower: "))

# Encode user inputs
car_encoded = le_car.transform([car])[0]
color_encoded = le_color.transform([color])[0]

# Make prediction
prediction = model.predict([[car_encoded, color_encoded, horsepower]])

if prediction == 0:
    print("The user will not Purchase.")
else:
    print("The user is likely to purchase.")