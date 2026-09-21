import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4, 5])
y = np.array([2.1,3.9,6.1,8.2,10.1])

w = 0.0
b = 0.0
a = 0.01
epochs = 100
w_history = []
b_history = []
loss_history = []

for i in range(epochs):
    y_pred = w*x + b
    loss = np.mean((y_pred-y)**2)

    w_history.append(w)
    b_history.append(b)
    loss_history.append(loss)

    dw = -2*np.mean(x*(y-y_pred))
    db = -2*np.mean(y-y_pred)
    w = w - a*dw
    b = b- a*db

W,B = np.meshgrid(np.linspace(-1, 5, 50), np.linspace(-2, 2, 50))
Loss = np.zeros_like(W)
for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        y_pred = W[i, j] * x + B[i, j]
        Loss[i, j] = np.mean((y_pred - y) ** 2)

fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121,projection='3d')
ax1.plot_surface(W, B, Loss, cmap='viridis', alpha=0.5)

ax1.plot(w_history, b_history, loss_history, color='r', marker='o', markersize=3, label='Gradient Descent Path')
ax1.set_xlabel('Weight (w)')
ax1.set_ylabel('Bias (b)')
ax1.set_zlabel('Loss')

plt.show()