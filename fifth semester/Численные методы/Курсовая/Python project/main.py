# import matplotlib.pyplot as plt
# import numpy as np
# import math
# from scipy.integrate import solve_ivp
#
# def dvxdt(t, x, y, vx, vy):
#     if x > 0:
#         return -(((2210840 / math.sqrt(x ** 2 + y ** 2)) ** 2) * math.cos(math.atan(y / x))) - (u * q * vx) / (
#                 math.sqrt(vx ** 2 + vy ** 2) * (m_0 - q * t))
#     else:
#         return -(((2210840 / math.sqrt(x ** 2 + y ** 2)) ** 2) * math.cos(-math.pi + math.atan(y / x))) - (
#                 u * q * vx) / (math.sqrt(vx ** 2 + vy ** 2) * (m_0 - q * t))
#
#
# def dvydt(t, x, y, vx, vy):
#     if x > 0:
#         return -(((2210840 / math.sqrt(x ** 2 + y ** 2)) ** 2) * math.sin(math.atan(y / x))) - (u * q * vy) / (
#                 math.sqrt(vx ** 2 + vy ** 2) * (m_0 - q * t))
#     else:
#         return -(((2210840 / math.sqrt(x ** 2 + y ** 2)) ** 2) * math.sin(-math.pi + math.atan(y / x))) - (
#                 u * q * vy) / (math.sqrt(vx ** 2 + vy ** 2) * (m_0 - q * t))
#
#
# for k in range(0, 1):
#     Rl = 1737000  # Радиус луны
#     m_0 = 100  # Начальная масса аппарата
#     m_k = 50
#     q = 0  # Кол-во кг топлива в сек.
#     u = 2000  # Скорость выхлопных газов
#     h = 0.01  # шаг расчёта для задач Коши
#     if q != 0:
#         tau = m_k / q  # Кол-во секунд расчёта траектории, до конца топлива
#     else:
#         tau = 10000
#     N = int(tau / h + 1)
#
#     X = np.zeros(N)
#     Y = np.zeros(N)
#     X[0] = 0  # Начальная Х координата аппарата
#     Y[0] = 2000000 # Начальная Y координата аппарата
#
#     Vx = np.zeros(N)
#     Vy = np.zeros(N)
#
#     Vx[0] = 1563.3  # начальная скорость аппарата по оси X
#     Vy[0] = 0  # начальная скорость аппарата по оси Y
#     for i in range(0, N - 1):
#         print(int(i * h))
#         k1 = h * dvxdt(h * i, X[i], Y[i], Vx[i], Vy[i])
#         k2 = h * dvxdt(h * i + h / 2, X[i], Y[i], Vx[i] + k1 / 2, Vy[i])
#         k3 = h * dvxdt(h * i + h, X[i], Y[i], Vx[i] - k1 + (k2 * 2), Vy[i])
#         Vx[i + 1] = Vx[i] + 1 / 6 * (k1 + 4 * k2 + k3) # Метод Рунге 3-го порядка первый вариант
#         # Vx[i + 1] = Vx[i] + h * dvxdt(h * i, X[i], Y[i], Vx[i], Vy[i])
#
#         k1 = h * dvydt(h * i, X[i], Y[i], Vx[i], Vy[i])
#         k2 = h * dvydt(h * i + h / 2, X[i], Y[i], Vx[i], Vy[i] + k1 / 2)
#         k3 = h * dvydt(h * i + h, X[i], Y[i], Vx[i], Vy[i] - k1 + (k2 * 2))
#         Vy[i + 1] = Vy[i] + 1 / 6 * (k1 + 4 * k2 + k3)
#         # Vy[i + 1] = Vy[i] + h * dvydt(h * i, X[i], Y[i], Vx[i], Vy[i])
#
#         X[i + 1] = X[i] + h * Vx[i + 1]  # Здесь по методу Эйлера, потом переделаю
#         Y[i + 1] = Y[i] + h * Vy[i + 1]
#         if math.sqrt(X[i] ** 2 + Y[i] ** 2) < Rl or math.sqrt(Vx[i] ** 2 + Vy[i] ** 2) < 0.01:
#             X = X[0:i + 1]
#             Y = Y[0:i + 1]
#             Vx = Vx[0:i + 1]
#             Vy = Vy[0:i + 1]
#             N = i + 1
#             Vlast = math.sqrt(Vx[i] ** 2 + Vy[i] ** 2)
#             break
#         if i == N - 2:
#             Vlast = math.sqrt(Vx[i] ** 2 + Vy[i] ** 2)
#
#     print(Vlast, math.sqrt(X[i] ** 2 + Y[i] ** 2) - Rl)
#     powred = 1
#     if Vlast > 100:
#         colored = (1, 0, 0)
#     else:
#         colored = (Vlast / 100, 1 - Vlast / 100, 0)
#     if Vlast < 10:
#         colored = (0, 0, 0)
#     plt.plot(X, Y, linewidth=1.5, color=colored)
#
# Xcircle = np.zeros(360000)
# Ycircle = np.zeros(360000)
#
# degrees = np.linspace(0, 2 * math.pi, 360000, endpoint=True)
#
# for i in range(0, 360000):
#     Xcircle[i] = math.cos(degrees[i]) * Rl
#     Ycircle[i] = math.sin(degrees[i]) * Rl
#
# plt.plot(Xcircle, Ycircle, color='black')
#
# plt.gca().set_aspect('equal', adjustable='box')
# plt.draw()
# plt.grid()
# plt.show()

#
# import matplotlib.pyplot as plt
# import numpy as np
# import math
# from scipy.integrate import odeint
#
# def RightFunc(t, yy):
#     x = yy[0]
#     y = yy[1]
#     vx = yy[2]
#     vy = yy[3]
#     alf = 0
#     if x > 0:
#         alf = math.atan(y / x)
#     elif x < 0:
#         alf = math.atan(y / x) - math.pi
#     else:
#         x == 0
#         if y > 0:
#             alf = math.pi / 2
#         else:
#             alf = -math.pi / 2
#
#     res = [0, 0, 0, 0]
#     res[0] = vx
#     res[1] = vy
#     res[2] = -(((2210840 / math.sqrt(x ** 2 + y ** 2)) ** 2) * math.cos(alf)) - (u * q * vx) / (math.sqrt(vx ** 2 + vy ** 2) * (m_0 - q * t))
#     res[3] = -(((2210840 / math.sqrt(x ** 2 + y ** 2)) ** 2) * math.sin(alf)) - (u * q * vy) / (math.sqrt(vx ** 2 + vy ** 2) * (m_0 - q * t))
#     return res
#
#
# Q = np.zeros(1000)
# U = np.zeros(1000)
# for k in range(1, 1001):
#     print(k)
#     Rl = 1737000  # Радиус луны
#     m_0 = 100  # Начальная масса аппарата
#     m_k = 50  # Масса топлива
#     q = 0.04 + k * 0.000005 # Расход кг топлива в секунду
#     u = 2000  # Скорость выхлопных газов
#     h = 1  # шаг расчёта для задач Коши
#     if q != 0:
#         tau = m_k / q  # Кол-во секунд расчёта траектории, до конца топлива
#     else:
#         tau = 10000
#     N = int(tau / h + 1)
#
#     t_0 = [0, 2000000, 1563.2999561184, 0]
#     T = [0, tau]
#     Tarr = np.linspace(0, tau, N, endpoint=True)
#     sol = odeint(RightFunc, t_0, Tarr, tfirst=True, rtol=0.000000000001)
#
#     X = sol[:, 0]
#     Y = sol[:, 1]
#
#     Xnew = []
#     Ynew = []
#
#     for i in range(0, N):
#         if math.sqrt(X[i] ** 2 + Y[i] ** 2) >= Rl:
#             Xnew.append(X[i])
#             Ynew.append(Y[i])
#         else:
#             break
#     X = Xnew
#     Y = Ynew
#     Q[k-1] = q
#     U[k-1] = q * h * X.__len__()
# plt.plot(Q,U)
#
# #plt.plot(X, Y)
#
# Xcircle = np.zeros(360000)
# Ycircle = np.zeros(360000)
#
# degrees = np.linspace(0, 2 * math.pi, 360000, endpoint=True)
#
# for i in range(0, 360000):
#     Xcircle[i] = math.cos(degrees[i]) * Rl
#     Ycircle[i] = math.sin(degrees[i]) * Rl
#
# #plt.plot(Xcircle, Ycircle, color='black')
#
# #plt.gca().set_aspect('equal', adjustable='box')
# plt.draw()
# plt.grid()
# plt.show()





import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.integrate import odeint

def RightFunc(t, yy):
    x = yy[0]
    y = yy[1]
    vx = yy[2]
    vy = yy[3]
    alf = 0
    if x > 0:
        alf = math.atan(y / x)
    elif x < 0:
        alf = math.atan(y / x) - math.pi
    else:
        x == 0
        if y > 0:
            alf = math.pi / 2
        else:
            alf = -math.pi / 2

    res = [0, 0, 0, 0]
    res[0] = vx
    res[1] = vy
    res[2] = -(((2210840 / math.sqrt(x ** 2 + y ** 2)) ** 2) * math.cos(alf)) - (u * q * vx) / (math.sqrt(vx ** 2 + vy ** 2) * (m_0 - q * t))
    res[3] = -(((2210840 / math.sqrt(x ** 2 + y ** 2)) ** 2) * math.sin(alf)) - (u * q * vy) / (math.sqrt(vx ** 2 + vy ** 2) * (m_0 - q * t))
    return res


Q = np.zeros(1000)
U = np.zeros(1000)
for k in range(1, 1001):
    print(k)
    Rl = 1737000  # Радиус луны
    m_0 = 100  # Начальная масса аппарата
    m_k = 50  # Масса топлива
    q = 0.04 + k * 0.000005 # Расход кг топлива в секунду
    u = 2000  # Скорость выхлопных газов
    h = 1  # шаг расчёта для задач Коши
    if q != 0:
        tau = m_k / q  # Кол-во секунд расчёта траектории, до конца топлива
    else:
        tau = 10000
    N = int(tau / h + 1)

    t_0 = [0, 2000000, 1563.2999561184, 0]
    T = [0, tau]
    Tarr = np.linspace(0, tau, N, endpoint=True)
    sol = odeint(RightFunc, t_0, Tarr, tfirst=True, rtol=0.000000000001)

    X = sol[:, 0]
    Y = sol[:, 1]

    Xnew = []
    Ynew = []

    for i in range(0, N):
        if math.sqrt(X[i] ** 2 + Y[i] ** 2) >= Rl:
            Xnew.append(X[i])
            Ynew.append(Y[i])
        else:
            break
    X = Xnew
    Y = Ynew
    Q[k-1] = q
    U[k-1] = q * h * X.__len__()
plt.plot(Q,U)

#plt.plot(X, Y)

Xcircle = np.zeros(360000)
Ycircle = np.zeros(360000)

degrees = np.linspace(0, 2 * math.pi, 360000, endpoint=True)

for i in range(0, 360000):
    Xcircle[i] = math.cos(degrees[i]) * Rl
    Ycircle[i] = math.sin(degrees[i]) * Rl

#plt.plot(Xcircle, Ycircle, color='black')

#plt.gca().set_aspect('equal', adjustable='box')
plt.draw()
plt.grid()
plt.show()
