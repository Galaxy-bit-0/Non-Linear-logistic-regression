import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv("moonDataset.csv")

X = df[["X1","X2"]].to_numpy() #2D array with () this is a method
y = df["label"].to_numpy() #array 1D 

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

m = X.shape[0]
theta = np.zeros(X_train.shape[1]) #model weight
bias = 0.0

#sigmoid to predict if the class is 1 or 0
def predict(theta,X,bias):
    z = X @ theta + bias
    sigmoid = 1/(1+np.exp(-z))

    return sigmoid

def loss(m,sigmoid,y):

    sigmoid = np.clip(sigmoid,1e-15,1-1e-15)
    cost = -1/m*(np.sum(y*np.log(sigmoid)+(1-y)*np.log(1-sigmoid))) #average cost

    return cost

def gradient(X, y, theta, sigmoid, bias, m, alpha=0.01):

    grad_theta = (1 / m) * (X.T @ (sigmoid - y))
    grad_bias = (1 / m) * np.sum(sigmoid - y)
    theta = theta - alpha * grad_theta
    bias = bias - alpha * grad_bias

    return theta,bias

costs = []
for number_of_iterations in range(100000):
    predictions = predict(theta,X_train, bias)
    cost = loss(m, predictions,y_train)
    costs.append(cost)
    theta, bias = gradient(X_train, y_train, theta,predictions, bias,m, alpha = 0.01)

#Accuracy
n = X_test.shape[0] #number of test samples
test_probabilities = predict(theta,X_test,bias)
test_prediction = (test_probabilities >=0.5).astype(int)
"""accuracy = 1/n*(np.sum(test_prediction == y_test))""" # is same as np.mean()
accuracy = (1/n) * np.sum(test_prediction == y_test)
print(f"accuracy of the models is {accuracy*100:.2f}%")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(costs)
axes[0].set_xlabel("Iteration")
axes[0].set_ylabel("Cost (binary cross-entropy)")
axes[0].set_title("Training cost")
axes[0].grid(alpha=0.3)

pad = 0.5
x1_min, x1_max = X[:, 0].min() - pad, X[:, 0].max() + pad
x2_min, x2_max = X[:, 1].min() - pad, X[:, 1].max() + pad
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 300),
                       np.linspace(x2_min, x2_max, 300))

grid = np.c_[xx1.ravel(), xx2.ravel()]
probs = predict(theta, grid, bias).reshape(xx1.shape)

cf = axes[1].contourf(xx1, xx2, probs, levels=20, cmap="RdBu_r", alpha=0.35)
axes[1].contour(xx1, xx2, probs, levels=[0.5], colors="black", linewidths=2)
fig.colorbar(cf, ax=axes[1], label="P(class = 1)")

axes[1].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="RdBu_r",
                edgecolors="k", s=30, label="train")
axes[1].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="RdBu_r",
                edgecolors="k", marker="^", s=50, label="test")

axes[1].set_xlabel("X1")
axes[1].set_ylabel("X2")
axes[1].set_title(f"Decision boundary (test accuracy {accuracy*100:.1f}%)")
axes[1].legend()

plt.tight_layout()
plt.show()
