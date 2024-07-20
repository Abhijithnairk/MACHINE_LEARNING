import matplotlib.pyplot as plt 
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

digits = load_digits()

print(dir(digits))
for i in range(10):
    # plt.gray()
    plt.matshow(digits["images"][i])
    
print(digits.target[:5])
# plt.gray()
model = LogisticRegression(max_iter=1000)

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.1)

# print(x_train)
# print(len(x_test))
# print(len(digits.data))
# plt.show()

model.fit(x_train, y_train)
predictions = model.predict(x_test)

# print("Predictions", predictions)
# print("****")
# print(y_test)

i = int(input("Enter the number to predict : "))
print(digits.target[i])
plt.show()