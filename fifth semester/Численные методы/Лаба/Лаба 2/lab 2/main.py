import math
import numpy as np
import matplotlib.pyplot as plt
import time


def f1(x):
    return 3 * math.e ** (x ** 3) - 2


def f(x, y):
    return 3 * y * x ** 2 + 6 * x ** 2



# h = 0.2
# a = 0
# b = 1.2
# N = int(round((b - a) / h)) + 1
#
# X = np.linspace(a, b, 1000, endpoint=True)
# Y = np.zeros(1000)
#
# for i in range(0, 1000):
#     Y[i] = f1(X[i])
#
# error = Y[-1]
#
# plt.plot(X, Y, linewidth = 2)
#
# X = np.linspace(a, b, N, endpoint=True)
# Y = np.zeros(N)
# Y[0] = 1
# for i in range(0, N - 1):
#     # k1 = h * f(X[i], Y[i])
#     # k2 = h * f(X[i] + h/3, Y[i] + k1/3)
#     # k3 = h * f(X[i] + h*2/3, Y[i] + k2*2/3)
#     # Y[i + 1] = Y[i] + 1/4*(k1 + 3*k3)
#
#     Y[i + 1] = Y[i] + h * f((X[i] + X[i + 1]) / 2, Y[i] + h / 2 * f(X[i], Y[i]))
#     Y[i + 1] = Y[i] + h * f(X[i], Y[i])
#
# plt.plot(X, Y, linewidth=1)
# plt.scatter(X, Y)
#
# print(error)
# error = error - Y[-1]
#
# print(error)
#
# plt.grid()
# plt.show()
############# Решаем методом Эйлера с шагом 0.2







currtime = time.perf_counter()

eps = 10 ** (-11)
p = 3
a = 0
b = 1.2
h = 0.2
N = int(round((b - a) / h)) + 1
t0 = 1

print("Запуск расчёта")

X = np.linspace(a, b, N, endpoint=True)
Y = np.zeros(N)
Y[0] = t0
for i in range(0, N - 1):
    k1 = h * f(X[i], Y[i])
    k2 = h * f(X[i] + h / 3, Y[i] + k1 / 3)
    k3 = h * f(X[i] + h * 2 / 3, Y[i] + k2 * 2 / 3)
    Y[i + 1] = Y[i] + 1 / 4 * (k1 + 3 * k3)

N2h = math.ceil(N / 2)
X2h = np.zeros(N2h)

for i in range(0, N2h):
    X2h[i] = X[2 * i]

Y2h = np.zeros(N2h)

Y2h[0] = t0

h = h*2

for i in range(0, N2h - 1):
    k1 = h * f(X2h[i], Y2h[i])
    k2 = h * f(X2h[i] + h / 3, Y2h[i] + k1 / 3)
    k3 = h * f(X2h[i] + h * 2 / 3, Y2h[i] + k2 * 2 / 3)
    Y2h[i + 1] = Y2h[i] + 1 / 4 * (k1 + 3 * k3)

h = h/2

Err = np.zeros(N2h)
for i in range(0, N2h):
    Err[i] = abs(Y[i * 2] - Y2h[i]) / (2 ** p - 1)

error = Err.max()

Recalculation = 0

while error > eps:
    Recalculation += 1
    Koef = error / eps
    h = h / Koef ** (1 / p)
    N = int(round((b - a) / h)) + 1
    X = np.linspace(a, b, N, endpoint=True)
    Y = np.zeros(N)
    Y[0] = t0
    for i in range(0, N - 1):
        k1 = h * f(X[i], Y[i])
        k2 = h * f(X[i] + h / 3, Y[i] + k1 / 3)
        k3 = h * f(X[i] + h * 2 / 3, Y[i] + k2 * 2 / 3)
        Y[i + 1] = Y[i] + 1 / 4 * (k1 + 3 * k3)

    N2h = math.ceil(N / 2)
    X2h = np.zeros(N2h)

    for i in range(0, N2h):
        X2h[i] = X[2 * i]

    Y2h = np.zeros(N2h)

    Y2h[0] = t0

    h = h*2

    for i in range(0, N2h - 1):
        k1 = h * f(X2h[i], Y2h[i])
        k2 = h * f(X2h[i] + h / 3, Y2h[i] + k1 / 3)
        k3 = h * f(X2h[i] + h * 2 / 3, Y2h[i] + k2 * 2 / 3)
        Y2h[i + 1] = Y2h[i] + 1 / 4 * (k1 + 3 * k3)

    h = h/2

    Err = np.zeros(N2h)
    for i in range(0, N2h):
        Err[i] = abs(Y[i * 2] - Y2h[i]) / (2 ** p - 1)

    error = Err.max()

currtime = time.perf_counter() - currtime

print("Глобальная ошибка: ", error)
print("Шаг разбиения: ", h)
print("Колличество точек разбиения: ", N)
print("Время на рассчёт:", currtime)
print("Кол-во пересчётов: ", Recalculation)
N = Y.size
X = np.linspace(a, b, N, endpoint=True)
plt.plot(X, Y, linewidth=4)

X = np.linspace(a, b, 1000000, endpoint=True)
Y = np.zeros(1000000)

for i in range(0, 1000000):
    Y[i] = f1(X[i])

plt.plot(X, Y, linewidth=1.5)

plt.grid()
plt.show()
###############  Решаем методом Рунге-Кутты с заданой точностью






# h = 0.4
# a = 0
# b = 1.2
# p = 3
# t0 = 1
#
# k = 10
#
# arr_of_err = np.zeros(k)
# arr_of_X = np.linspace(0, k-1, k, endpoint=True)
# for j in range(0, k):
#     h = h/2
#     N = int(round((b - a) / h)) + 1
#     X = np.linspace(a, b, N, endpoint=True)
#     Y = np.zeros(N)
#     Y[0] = 1
#     for i in range(0, N - 1):
#         k1 = h * f(X[i], Y[i])
#         k2 = h * f(X[i] + h/3, Y[i] + k1/3)
#         k3 = h * f(X[i] + h*2/3, Y[i] + k2*2/3)
#         Y[i + 1] = Y[i] + 1/4*(k1 + 3*k3)
#
#         #Y[i + 1] = Y[i] + h * f((X[i] + X[i + 1]) / 2, Y[i] + h / 2 * f(X[i], Y[i]))
#         # Y[i + 1] = Y[i] + h * f(X[i], Y[i])
#
#     N2h = math.ceil(N / 2)
#     X2h = np.zeros(N2h)
#
#     for i in range(0, N2h):
#         X2h[i] = X[2 * i]
#
#     Y2h = np.zeros(N2h)
#
#     Y2h[0] = t0
#
#     h = h*2
#
#     for i in range(0, N2h - 1):
#         k1 = h * f(X2h[i], Y2h[i])
#         k2 = h * f(X2h[i] + h / 3, Y2h[i] + k1 / 3)
#         k3 = h * f(X2h[i] + h * 2 / 3, Y2h[i] + k2 * 2 / 3)
#         Y2h[i + 1] = Y2h[i] + 1 / 4 * (k1 + 3 * k3)
#
#     h = h/2
#
#     Err = np.zeros(N2h)
#     for i in range(0, N2h):
#         Err[i] = abs(Y[i * 2] - Y2h[i]) / (2 ** p - 1)
#
#     arr_of_err[j] = Err.max()
#     print(j, " поделил, h =", h, " ошибка = ", Err.max())
#
#
# plt.semilogy(arr_of_X, arr_of_err)
#
#
#
#
# h = 0.4
# p =2
#
# for j in range(0, k):
#     h = h/2
#     N = int(round((b - a) / h)) + 1
#     X = np.linspace(a, b, N, endpoint=True)
#     Y = np.zeros(N)
#     Y[0] = 1
#     for i in range(0, N - 1):
#         Y[i + 1] = Y[i] + h * f((X[i] + X[i + 1]) / 2, Y[i] + h / 2 * f(X[i], Y[i]))
#
#
#     N2h = math.ceil(N / 2)
#     X2h = np.zeros(N2h)
#
#     for i in range(0, N2h):
#         X2h[i] = X[2 * i]
#
#     Y2h = np.zeros(N2h)
#
#     Y2h[0] = t0
#
#     h = h*2
#
#     for i in range(0, N2h - 1):
#         Y2h[i + 1] = Y2h[i] + h * f((X2h[i] + X2h[i + 1]) / 2, Y2h[i] + h / 2 * f(X2h[i], Y2h[i]))
#
#     h = h/2
#
#     Err = np.zeros(N2h)
#     for i in range(0, N2h):
#         Err[i] = abs(Y[i * 2] - Y2h[i]) / (2 ** p - 1)
#
#     arr_of_err[j] = Err.max()
#     print(j, " поделил, h =", h, " ошибка = ", Err.max())
#
# plt.semilogy(arr_of_X, arr_of_err)
#
# plt.grid()
# plt.show()
# ##########   Выводим графики ошибок, при стремлении h к 0. двух методов





# def f(x, y):
#     return 0.0003 * j * 0.1 * (700 - y) * (230 - y)
#
#
# for j in range(1, 11):
#     currtime = time.perf_counter()
#     #print("Запуск расчёта")
#
#     Recalculation = 0
#
#     eps = 10 ** (-6)
#     p = 3
#     a = 0
#     b = 37
#     h = 0.2
#     N = int(round((b - a) / h)) + 1
#     t0 = 70
#
#     X = np.linspace(a, b, N, endpoint=True)
#     Y = np.zeros(N)
#     Y[0] = t0
#     for i in range(0, N - 1):
#         k1 = h * f(X[i], Y[i])
#         k2 = h * f(X[i] + h / 3, Y[i] + k1 / 3)
#         k3 = h * f(X[i] + h * 2 / 3, Y[i] + k2 * 2 / 3)
#         Y[i + 1] = Y[i] + 1 / 4 * (k1 + 3 * k3)
#
#     N2h = math.ceil(N / 2)
#     X2h = np.zeros(N2h)
#
#     for i in range(0, N2h):
#         X2h[i] = X[2 * i]
#
#     Y2h = np.zeros(N2h)
#
#     Y2h[0] = t0
#
#     h = h * 2
#
#     for i in range(0, N2h - 1):
#         k1 = h * f(X2h[i], Y2h[i])
#         k2 = h * f(X2h[i] + h / 3, Y2h[i] + k1 / 3)
#         k3 = h * f(X2h[i] + h * 2 / 3, Y2h[i] + k2 * 2 / 3)
#         Y2h[i + 1] = Y2h[i] + 1 / 4 * (k1 + 3 * k3)
#
#     h = h / 2
#
#     Err = np.zeros(N2h)
#     for i in range(0, N2h):
#         Err[i] = abs(Y[i * 2] - Y2h[i]) / (2 ** p - 1)
#
#     error = Err.max()
#
#     while error > eps:
#         Recalculation += 1
#         Koef = error / eps
#         h = h / Koef ** (1 / p)
#         N = int(round((b - a) / h)) + 1
#         X = np.linspace(a, b, N, endpoint=True)
#         Y = np.zeros(N)
#         Y[0] = t0
#         for i in range(0, N - 1):
#             k1 = h * f(X[i], Y[i])
#             k2 = h * f(X[i] + h / 3, Y[i] + k1 / 3)
#             k3 = h * f(X[i] + h * 2 / 3, Y[i] + k2 * 2 / 3)
#             Y[i + 1] = Y[i] + 1 / 4 * (k1 + 3 * k3)
#
#         N2h = math.ceil(N / 2)
#         X2h = np.zeros(N2h)
#
#         for i in range(0, N2h):
#             X2h[i] = X[2 * i]
#
#         Y2h = np.zeros(N2h)
#
#         Y2h[0] = t0
#
#         h = h * 2
#
#         for i in range(0, N2h - 1):
#             k1 = h * f(X2h[i], Y2h[i])
#             k2 = h * f(X2h[i] + h / 3, Y2h[i] + k1 / 3)
#             k3 = h * f(X2h[i] + h * 2 / 3, Y2h[i] + k2 * 2 / 3)
#             Y2h[i + 1] = Y2h[i] + 1 / 4 * (k1 + 3 * k3)
#
#         h = h / 2
#
#         Err = np.zeros(N2h)
#         for i in range(0, N2h):
#             Err[i] = abs(Y[i * 2] - Y2h[i]) / (2 ** p - 1)
#
#         error = Err.max()
#
#     currtime = time.perf_counter() - currtime
#
#     # print("Глобальная ошибка: ", error)
#     # print("Шаг разбиения: ", h)
#     # print("Колличество точек разбиения: ", N)
#     # print("Время на рассчёт:", currtime)
#     # print("Кол-во пересчётов: ", Recalculation)
#     N = Y.size
#     X = np.linspace(a, b, N, endpoint=True)
# #    plt.plot(X, Y, linewidth=1, color=str(0.9 - j * 0.085))
#
#     dYdt = np.zeros(N)
#     dYdt[0] = (Y[1] - Y[0]) / h
#     dYdt[-1] = (Y[-1] - Y[-2]) / h
#
#     for i in range(1, N - 1):
#         dYdt[i] = (Y[i + 1] - Y[i - 1]) / (h * 2)
#
#     SdY = 0
#     for i in range(0, N - 1):
#         SdY += (dYdt[i] ** 2 + dYdt[i + 1] ** 2) / 2 * h
#
#     plt.plot(X, dYdt, color=(0 + j*0.09, 0, 1 - j*0.09), linewidth=0.8)
#     print(j, SdY)
#
# plt.grid()
# plt.show()
