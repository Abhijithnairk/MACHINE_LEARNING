import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB


data = pd.read_csv("NAIVE_BAYES/heart.csv")
# print(data.keys())
#['Patient', 'Gender', 'Age', 'Cholesterol', 'Troponin', 'Smoker', 'BP','HeartAttack']
#   0           1       2       3               4           5       6      7
x = data.iloc[:, [1,2,3,4,5,6]].values
y = data.iloc[:, [-1]].values
# print(x[0])
# print(x[-1])

le_gender = LabelEncoder()
le_smoker = LabelEncoder()
le_attack = LabelEncoder()

x[:,0] = le_gender.fit_transform(x[:,0])
x[:,4] = le_smoker.fit_transform(x[:,4])
x[:,-1] = le_attack.fit_transform(x[:,-1])
# print(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

model = GaussianNB()

# Fit the model
model.fit(x_train, y_train)

# Prediction
print("-- Prediction --")
print(model.predict(x_test))
print('-----ACTUAL DATA-----')
print(x_test)
gender = input("Enter gender (Male/Female): ")
smoker = input("Enter smoker or not (Yes/No): ")
age = int(input("Enter the age: "))
cholestrol = int(input("Enter cholestrol level: "))
troponin = float(input("Enter troponin level: "))
bp = int(input("Enter bp: "))

gender_encode = le_gender.fit_transform([gender])[0]
smoker_encode = le_smoker.fit_transform([smoker])[0]

prediction = model.predict([[gender_encode,smoker_encode,age,cholestrol,troponin,bp]])

if prediction == 0:
    print("Patient dont have any chance for a attack.")
else:
    print("Beware, Patient have a chance of attack.")
