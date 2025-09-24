import numpy as np

# универсальная функция активации
def neuron(x, w, b, activation="linear"):
    """
    x : np.array — входные данные
    w : np.array — веса
    b : float — смещение
    activation : str — тип активации ("linear", "step", "sigmoid", "tanh")
    """
    # вычисление взвешенной суммы
    u = np.dot(w, x) + b
    
    # выбор функции активации
    if activation == "linear":
        return u
    elif activation == "step":
        return 1 if u >= 0 else 0
    elif activation == "sigmoid":
        return 1 / (1 + np.exp(-u))
    elif activation == "tanh":
        return np.tanh(u)
    else:
        raise ValueError("Неизвестная функция активации")

# входные данные (2 входа)
x = np.array([0.5, 1.5])
# веса
w = np.array([0.8, -0.4])
# смещение
b = 0.1

print("Linear:", neuron(x, w, b, "linear"))
print("Step:", neuron(x, w, b, "step"))
print("Sigmoid:", neuron(x, w, b, "sigmoid"))
print("Tanh:", neuron(x, w, b, "tanh"))
