import numpy as np

def sigmoid(u):
    return 1 / (1 + np.exp(-u))

def step(u):
    return 1 if u >= 0 else 0

# входные данные
x1, x2 = 0.5, 1.5
w1, w2 = 0.8, -0.4
b = 0.1

# взвешенная сумма
u = w1 * x1 + w2 * x2 + b

# линейная активация
y = u

# пороговая функция
y_step = step(u)

#сигмоида
y_sigmoid = sigmoid(u)

print("Взвешенная сумма u =", u)
print("Выход нейрона y =", y)
print("Выход нейрона (пороговая функция):", y_step)

print("Выход нейрона (сигмоида):", y_sigmoid)
