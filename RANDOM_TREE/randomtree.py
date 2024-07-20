from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#load the data of iris
iris = datasets.load_iris()
#target names
print(iris.target_names)
#feature names
print(iris.feature_names)

data = pd.DataFrame({'sepal length':iris.data[:,0],
                   'sepal width':iris.data[:,1],
                   'petal length':iris.data[:,2],
                   'petal width':iris.data[:,3],
                   'species':iris.target
})


print(data)
x = data[['sepal length','sepal width','petal length','petal width']]
y = data['species']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)


clf = RandomForestClassifier(n_estimators=100,criterion='gini')
clf.fit(x_train,y_train)
print("TTS",clf.predict(x_test))

# print("Accuracy",clf.score(x_test,y_test))

sepallength = int(input("Enter sepal length: "))
sepalwidth = int(input("Enter sepalwidth: "))
petallength = int(input("Enter petal length: "))
petalwidth = int(input("Enter petalwidth: "))

val = clf.predict([[sepallength,sepalwidth,petallength,petalwidth]])

if val==1:
    print("setosa")
elif val==2:
    print("vasicolor")
elif val==3:
    print("virginica")