# import numpy as np
# import matplotlib.pyplot as plt
#
# eps = 0.01
#
# a, b, c = 1, 4, 2
# ua, ub = 8, 4
#
#
# def k(x):
#     if x >= a and x < c:
#         return 2 * x
#     elif x >= c and x <= b:
#         return 4
#
#
# def q(x):
#     return 1 + x ** 2
#
#
# def u_test(x):
#     if x >= a and x < c:
#         return - (x ** 2) + 2 * x + 7
#     elif x >= c and x <= b:
#         return 0.25 * (x ** 2) - 3 * x + 12
#
#
def f_test(x):
    if x >= a and x < c:
        return -(x ** 4) + 2 * (x ** 3) + 6 * (x ** 2) + 10 * x + 3
    elif c <= x <= b:
        return 0.25 * (x ** 4) - 3 * (x ** 3) + 12.25 * (x ** 2) - 3 * x + 10


def F_test(x, h):
    if x > a and x < c:
        integral = lambda t: -(t ** 5) / 5 + (t ** 4) / 2 + (t ** 3) * 2 + (t ** 2) * 5 + 3 * t
        return (integral(x + h / 2) - integral(x - h / 2)) / h

    elif x >= c and x < b:
        integral = lambda t: (t ** 5) / 20 - (t ** 4) * 3 / 4 + (t ** 3) * 12.25 / 3 - (t ** 2) * 3 / 2 + 10 * t
        return (integral(x + h / 2) - integral(x - h / 2)) / h


def Q(x, h):
    integral = lambda t: t + (t ** 3) / 3
    return (integral(x + h / 2) - integral(x - h / 2)) / h

#
# def K_plus(x, h):
#     return k(x + h / 2)
#
#
# def K_minus(x, h):
#     return k(x - h / 2)
#
# X = np.linspace(a, b, 1000, endpoint=True)
# Y_true = np.zeros(1000)
# for i in range(0, 1000):
#     Y_true[i] = u_test(X[i])
# plt.plot(X, Y_true, linewidth=4.5)
#
# N = 15
# h = (b - a) / (N - 1)
# X = np.linspace(a, b, N, endpoint=True)
#
# A = np.zeros((N, N))
# B = np.zeros(N)
#
# A[0][0] = 1
# A[N - 1][N - 1] = 1
#
# B[0] = ua
# B[N - 1] = ub
#
# for i in range(1, N - 1):
#     A[i][i - 1] = K_minus(X[i], h)
#     A[i][i] = -(K_plus(X[i], h) + K_minus(X[i], h) + Q(X[i], h) * h ** 2)
#     A[i][i + 1] = K_plus(X[i], h)
#     B[i] = - F_test(X[i], h) * h ** 2
#
# Y_calc = np.linalg.solve(A, B)
#
# plt.plot(X, Y_calc, linewidth=2)
# plt.scatter(X, Y_calc, linewidth = 0.5)
#
#
# plt.grid()
# plt.show() #Решения тестового примера


# import numpy as np
# import matplotlib.pyplot as plt
# eps = 0.01
#
# a, b, c = 1, 4, 2
# ua, ub = 8, 4
#
#
# def k(x):
#     if x >= a and x < c:
#         return 2 * x
#     elif x >= c and x <= b:
#         return 4
#
#
# def q(x):
#     return 1 + x ** 2
#
#
# def F_test(x, h):
#     integral = lambda t: (t ** 3) * 12.5 / 3 - (t ** 2) * 45 / 2 + t * 6.75
#     return (integral(x + h / 2) - integral(x - h / 2)) / h
#
#
# def Q(x, h):
#     integral = lambda t: t + (t ** 3) / 3
#     return (integral(x + h / 2) - integral(x - h / 2)) / h
#
#
# def K_plus(x, h):
#     return k(x + h / 2)
#
#
# def K_minus(x, h):
#     return k(x - h / 2)
#
#
# N = 21
# h = (b - a) / (N - 1)
# X = np.linspace(a, b, N, endpoint=True)
#
# A = np.zeros((N, N))
# B = np.zeros(N)
# Err = np.zeros(N)
#
# A[0][0] = 1
# A[N - 1][N - 1] = 1
#
# B[0] = ua
# B[N - 1] = ub
#
# for i in range(1, N - 1):
#     A[i][i - 1] = K_minus(X[i], h)
#     A[i][i] = -(K_plus(X[i], h) + K_minus(X[i], h) + Q(X[i], h) * h ** 2)
#     A[i][i + 1] = K_plus(X[i], h)
#     B[i] = - F_test(X[i], h) * h ** 2
#
# Y_calc = np.linalg.solve(A, B)
#
# N = (N - 1) * 2 + 1
# h = (b - a) / (N - 1)
# X = np.linspace(a, b, N, endpoint=True)
#
# A = np.zeros((N, N))
# B = np.zeros(N)
#
# A[0][0] = 1
# A[N - 1][N - 1] = 1
#
# B[0] = ua
# B[N - 1] = ub
#
# for i in range(1, N - 1):
#     A[i][i - 1] = K_minus(X[i], h)
#     A[i][i] = -(K_plus(X[i], h) + K_minus(X[i], h) + Q(X[i], h) * h ** 2)
#     A[i][i + 1] = K_plus(X[i], h)
#     B[i] = - F_test(X[i], h) * h ** 2
#
# Y_calc05h = np.linalg.solve(A, B)
#
# for i in range(0, round((N - 1) / 2)):
#     Err[i] = (abs(Y_calc[i] - Y_calc05h[i * 2])) / 3
#
# error = np.amax(Err)
#
# while error > eps:
#     Y_calc = Y_calc05h
#
#     Err = np.zeros(N)
#     N = (N - 1) * 2 + 1
#     h = (b - a) / (N - 1)
#     X = np.linspace(a, b, N, endpoint=True)
#
#     A = np.zeros((N, N))
#     B = np.zeros(N)
#
#     A[0][0] = 1
#     A[N - 1][N - 1] = 1
#
#     B[0] = ua
#     B[N - 1] = ub
#
#     for i in range(1, N - 1):
#         A[i][i - 1] = K_minus(X[i], h)
#         A[i][i] = -(K_plus(X[i], h) + K_minus(X[i], h) + Q(X[i], h) * h ** 2)
#         A[i][i + 1] = K_plus(X[i], h)
#         B[i] = - F_test(X[i], h) * h ** 2
#
#     Y_calc05h = np.linalg.solve(A, B)
#
#     for i in range(0, round((N - 1) / 2)+1):
#         Err[i] = (abs(Y_calc[i] - Y_calc05h[i * 2])) / 3
#
#     error = np.amax(Err)
# #plt.plot(X, Y_calc05h, linewidth=2)
# print(N)
# X = np.linspace(a, b, round((N + 1) / 2), endpoint=True)
# plt.plot(X, Err)
# plt.grid()
# plt.show()  # Решение задачи 3.1


import numpy as np
import matplotlib.pyplot as plt

eps = 0.01
C = -2000
x_0 = 2
a, b, c = 1, 4, 2
ua, ub = 8, 4


def k(x):
    if x >= a and x < c:
        return 2 * x
    elif x >= c and x <= b:
        return 4


def q(x):
    return 1 + x ** 2

def F_test(x, h):
    integral = lambda t: (t ** 3) * 12.5 / 3 - (t ** 2) * 45 / 2 + t * 6.75
    return (integral(x + h / 2) - integral(x - h / 2)) / h


def F_zero(x, h):
    return 0


def F_A(x, h):
    if a + (b - a) / 3 <= x <= a + 2 * (b - a) / 3:
        return 0
    else:
        integral = lambda t: (t ** 3) * 12.5 / 3 - (t ** 2) * 45 / 2 + t * 6.75
        return (integral(x + h / 2) - integral(x - h / 2)) / h


def F_B(x, h):
    if a + (b - a) / 3 <= x <= a + 2 * (b - a) / 3:
        integral = lambda t: (t ** 3) * 12.5 / 3 - (t ** 2) * 45 / 2 + t * 6.75
        return (integral(x + h / 2) - integral(x - h / 2)) / h
    else:
        return 0


def F_C(x, h):
    if x_0 - h/2 <= x <= x_0 + h/2:
        return C
    else:
        return 0


def Q(x, h):
    integral = lambda t: t + (t ** 3) / 3
    return (integral(x + h / 2) - integral(x - h / 2)) / h


def K_plus(x, h):
    return k(x + h / 2)


def K_minus(x, h):
    return k(x - h / 2)


N = 61
h = (b - a) / (N - 1)
X = np.linspace(a, b, N, endpoint=True)

A = np.zeros((N, N))
B = np.zeros(N)
Err = np.zeros(N)

A[0][0] = 1
A[N - 1][N - 1] = 1

B[0] = ua
B[N - 1] = ub

for i in range(1, N - 1):
    A[i][i - 1] = K_minus(X[i], h)
    A[i][i] = -(K_plus(X[i], h) + K_minus(X[i], h) + Q(X[i], h) * h ** 2)
    A[i][i + 1] = K_plus(X[i], h)
    B[i] = - F_A(X[i], h) * h ** 2

Y_calc = np.linalg.solve(A, B)

plt.plot(X, Y_calc, linewidth=2, label = "Греем краешки")



A = np.zeros((N, N))
B = np.zeros(N)
Err = np.zeros(N)

A[0][0] = 1
A[N - 1][N - 1] = 1

B[0] = ua
B[N - 1] = ub

for i in range(1, N - 1):
    A[i][i - 1] = K_minus(X[i], h)
    A[i][i] = -(K_plus(X[i], h) + K_minus(X[i], h) + Q(X[i], h) * h ** 2)
    A[i][i + 1] = K_plus(X[i], h)
    B[i] = - F_B(X[i], h) * h ** 2

Y_calc = np.linalg.solve(A, B)

plt.plot(X, Y_calc, linewidth=2, label = "греем серединку")



A = np.zeros((N, N))
B = np.zeros(N)
Err = np.zeros(N)

A[0][0] = 1
A[N - 1][N - 1] = 1

B[0] = ua
B[N - 1] = ub

for i in range(1, N - 1):
    A[i][i - 1] = K_minus(X[i], h)
    A[i][i] = -(K_plus(X[i], h) + K_minus(X[i], h) + Q(X[i], h) * h ** 2)
    A[i][i + 1] = K_plus(X[i], h)
    B[i] = - F_C(X[i], h) * h ** 2

Y_calc = np.linalg.solve(A, B)

plt.plot(X, Y_calc, linewidth=2, label = "греем точечку")



A = np.zeros((N, N))
B = np.zeros(N)
Err = np.zeros(N)

A[0][0] = 1
A[N - 1][N - 1] = 1

B[0] = ua
B[N - 1] = ub

for i in range(1, N - 1):
    A[i][i - 1] = K_minus(X[i], h)
    A[i][i] = -(K_plus(X[i], h) + K_minus(X[i], h) + Q(X[i], h) * h ** 2)
    A[i][i + 1] = K_plus(X[i], h)
    B[i] = - F_test(X[i], h) * h ** 2

Y_calc = np.linalg.solve(A, B)

plt.plot(X, Y_calc, linewidth=2, label = "греем как в первой задаче")


A = np.zeros((N, N))
B = np.zeros(N)
Err = np.zeros(N)

A[0][0] = 1
A[N - 1][N - 1] = 1

B[0] = ua
B[N - 1] = ub

for i in range(1, N - 1):
    A[i][i - 1] = K_minus(X[i], h)
    A[i][i] = -(K_plus(X[i], h) + K_minus(X[i], h) + Q(X[i], h) * h ** 2)
    A[i][i + 1] = K_plus(X[i], h)
    B[i] = - F_zero(X[i], h) * h ** 2

Y_calc = np.linalg.solve(A, B)

plt.plot(X, Y_calc, linewidth=2, label = "не греем")


plt.grid()
plt.legend()
plt.show()
