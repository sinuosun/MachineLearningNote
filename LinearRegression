import numpy as np

#训练集
X = np.array([
    [1,2,1],
    [2,1,2],
    [3,3,1],
    [4,2,3],
    [5,4,2],
    [6,3,4]   
])
y = np.array([
    8,
    9,
    15,
    17,
    23,
    25
])
#model: y =w1x1+w2x2+w3x3+b
#initialize parameters
w = np.zeros(X.shape[1])
b = 0

#Number of samples and features
n = X.shape[0]
d = X.shape[1]
#learning rate
alpha = 0.01#test parameters
epochs = 1000#梯度下降数
for i in range(epochs):
    y_pred = X @ w + b
    loss = np.mean((y - y_pred)**2)

    dw = -2*np.dot(X.T,(y-y_pred))/n
    db = -2*np.mean(y - y_pred)
    w = w - alpha*dw
    b = b - alpha*db
print(w,b)