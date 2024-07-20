import matplotlib as plt
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

digits = load_digits()
print(dir(digits))

plt.matshow(digits["images"][67])

print(digits.target[:5])
plt.gray()
model = LogisticRegression(max_iter=1000)

x_train,x_test,y_train,y_test = train_test_split(digits.data,digits.target,test_size=0.1)

print(len(x_train))
print(len(x_test))
print(len(digits.data))

model.fit(x_train,y_train)
predictions = model.predict(x_test)

print("Predictions: ",predictions)
print("*********************")
print(y_test)