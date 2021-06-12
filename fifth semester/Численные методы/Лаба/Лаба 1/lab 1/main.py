import math
import numpy as np
import matplotlib.pyplot as plt

""" Задание 1.1
def f(x):
    return math.sin(x)


def f1(x):
    return math.cos(x)


def derivative(F, x, h):
    res = (-1.7 * F(x) + 1 / 30 * F(x + 5 * h) + 2.5 * F(x + h) - 5 / 6 * F(x + 2 * h)) / h
    return res



h = 0.0001
n = int(1 / h)
a = -h
b = 2 * math.pi + h
X = np.linspace(a, b, n, endpoint=True)
Y = np.zeros(n)

for i in range(0, n):
    Y[i] = abs(f1(X[i]) - derivative(f, X[i], h))

plt.plot(X, Y, linewidth=0.5)

print(X[np.argmax(Y)])

x = 5.795645843396072

n = 10
X = np.linspace(0, n, n + 1, endpoint=True)
Y = np.zeros(n + 1)
for j in range(0, n + 1):
    h = 1 / 10 ** j
    Y[j] = abs(f1(x) - derivative(f, x, h))
plt.semilogy(X, Y)

plt.grid()
plt.show()
"""

# Задание 1.2 часть 1
# def f(x):
#     return -0.1 * x ** 4 + 1.26 * x ** 2 - x - 0.8
#
#
# X = np.linspace(-1, 3, 1000)
# Y = np.zeros(1000)
# for i in range(0, 1000):
#     Y[i] = f(X[i])
#
# plt.plot(X, Y)
#
# a = 2.5
# b = 3
# x_0 = 1
# n = 0
# while f(x_0) != 0 and x_0 != (a + b) / 2:
#     x_0 = (a + b) / 2
#     if f(a) * f(x_0) > 0:
#         a = x_0
#     elif f(b) * f(x_0) > 0:
#         b = x_0
#     n += 1
#     print(n)
#     print(x_0, f(x_0))
#     print("")
#
#
# plt.grid()
# plt.show()
#
#
#
# def f1(x):
#     return -0.1 * x ** 4 + 1.1 * x ** 2 + 2.2
#
#
# def f2(x):
#     return -0.16 * x ** 2 + x + 3
#
#
# f = f2
#
# n = 1000000
# b = 1
# X = np.linspace(0, b, n, endpoint=True)
# h = (abs(0 - b) / n)
# Y = np.zeros(n)
#
# S = 0
# S += Y[0]
# S += Y[n - 1]
#
# for i in range(0, n):
#     Y[i] = f(X[i])
#
# temp = 0
# for i in range(1, n - 1):
#     temp += f(X[i])
# S += 2 * temp
#
# temp = 0
# for i in range(0, n - 1):
#     temp += f((X[i] + X[i + 1]) / 2)
# S += 4 * temp
# S = S * h / 6
# print(S)
