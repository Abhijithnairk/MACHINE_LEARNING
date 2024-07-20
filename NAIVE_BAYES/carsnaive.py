import pandas as pd

data = pd.read_csv("NAIVE_BAYES/carsnaive.csv")

x = data.iloc[:,[0,1,2]].values
print(x)

y = data.iloc[:,[-1]]
print(y)

from sklearn.preprocessing import LabelEncoder

le_car = LabelEncoder()
le_color = LabelEncoder()

x[:,0] = le_car.fit_transform(x[:,0])
x[:,1] = le_color.fit_transform(x[:,1])
print(x)

from sklearn.model_selection import train_test_split
train_test_split(x,y,test_size=0.2)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(x_train,y_train)

print("-----prediction-----")
# print(model.predict(x_test))

car = input("Enter car name (Toyota/Ford): ").capitalize()
color = input("Enter color (Red/Cyan/Black): ").capitalize()
Horsepower = int(input("Enter power: "))

car_encoded = le_car.transform([car])[0]
color_encoded = le_color.transform([color])[0]
prediction = model.predict([[car_encoded, color_encoded,Horsepower]])

if prediction==0:
    print("The user will not purchase.")
else:
    print("User ready to purchase")