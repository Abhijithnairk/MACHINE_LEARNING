import pandas as pd

dataset = pd.read_csv("NAIVE_BAYES/dataset.csv")

x = dataset.iloc[:,[1,2,3]].values
print(x)

y = dataset.iloc[:,-1].values
print(y)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
x[:,0] = le.fit_transform(x[:,0])
print(x)

from sklearn.model_selection import train_test_split
train_test_split(x,y,test_size=0.2)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

#naive bayes model

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(x_train,y_train)

print("-----prediction-----")
#print(model.predict(x_test))
gender = input("Enter Gender (Male/Female): ")
age = int(input("Enter age: "))
salary = int(input("Enter salry: "))

gender = le.transform([gender])[0]
prediction = model.predict[[gender,age,salary]]

if prediction==0:
    print("The user will not purchase.")
else:
    print("The user like to purchase.")
    
# print("-----actual data-----")
# print(x_test)
