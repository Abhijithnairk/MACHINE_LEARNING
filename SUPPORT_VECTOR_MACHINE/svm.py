import pandas as pd

dataset = pd.read_csv("SUPPORT_VECTOR_MACHINE/Iris.csv")
# print(dataset)
dataset.drop(columns=["id"],inplace=True)
x = dataset.iloc[:,0:4]
# print(x)

y = dataset.iloc[:,-1]
# print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

# print(x_train)
# print(x_test)

from sklearn.svm import SVC
model = SVC(kernel='linear')
model.fit(x_train,y_train)

# prediction = model.predict(x_test)
# print(prediction)

sepal_length = float(input("Enter sepal length: "))
sepal_width = float(input("Enter sepal width: "))
petal_length = float(input("Enter petal length: "))
petal_width = float(input("Enter petal width: "))

prediction = model.predict([[sepal_length,sepal_width,petal_length,petal_width]])
print(f"Corresponding species is: {prediction}")
