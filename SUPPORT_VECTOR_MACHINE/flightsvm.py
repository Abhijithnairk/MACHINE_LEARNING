import pandas as pd

dataset = pd.read_csv("flight.csv")

x = dataset.iloc[:,0:3]
y = dataset.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

from sklearn.svm import SVC
model = SVC(kernel='linear')
model.fit(x_train,y_train)

departure = int(input("Enter departure delay time: "))
arrival = int(input("Enter arrival delay time: "))
distance = int(input("Enter flight distance: "))

prediction = model.predict([[departure,arrival,distance]])
print(f"Chance of flight dalay is {prediction}")