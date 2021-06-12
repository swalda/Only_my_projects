import numpy as np
import matplotlib.pyplot as plt
import copy
import math
import time


def NormSystem(X, Y, m):
    n = len(X)
    A = np.zeros((m + 1, m + 1))
    b = np.zeros(m + 1)
    for k in range(0, m + 1, 1):
        sum = 0
        for i in range(0, n, 1):
            sum += Y[i]*X[i]**(k)
        b[k] = sum
        for j in range(0, m + 1, 1):
            sum = 0
            for i in range(0, n, 1):
                sum += X[i]**(k+j)
            A[k][j] = sum
    return A, b

def GausSLAU(A0,B):
    
    m = copy.deepcopy(B)
    A = copy.deepcopy(A0)
    
    lena = len(A0[0])
    for i in range(0, lena):
        #ЭТО ЧАСТЬ АЛГОРИТМА РЕАЛИЗУЕТ ПЕРЕСТАНОВКУ СТРОК 
        for j in range(lena-1, i, -1): # И МЕТОД ЧАСТИЧНОГО ВыБОРА
            if A[j][i] > A[j-1][i]:
                A[[j, j-1]] = A[[j-1, j]]
                m[j],m[j-1] = m[j-1],m[j]

        for j in range (i, lena-1):
            k = A[j+1][i]/A[i][i]
            for f in range(i, lena):
                A[j+1][f] = A[j+1][f] - (A[i][f]*k)
            m[j+1] = m[j+1] - (m[i]*k)
    
    x = np.zeros(lena)
    
    for i in range(lena-1, -1, -1):
        q = 0
        for j in range (lena-1, i, -1):
            q += A[i][j]*x[j]
        x[i] = (m[i] - q)/A[i][i]

    return x

def F(a, x):
    n = len(a)
    sum = 0
    for i in range(0, n, 1):
        sum += a[i]*x**(i)
    return sum

def error(X, Y, a):
    m = len(a)
    n = len(X)
    sum = 0
    for i in range(0, n, 1):
        P = 0
        for j in range(0, m, 1):    
            P += a[j]*X[i]**(j)
        sum += (P - Y[i])**2
        
    sum = math.sqrt(sum/n)
    return sum
"""        
X = np.array([-0.7, -0.375, -0.05, 0.275, 0.6, 0.925, 1.25, 1.575, 1.9, 2.25, 2.55, 2.875, 3.2])
Y = np.array([3.822, -1.498, -2.419, -1.292, 0.828, 1.963, 2.401, 1.877, 2.2, -1.378, -2.395, -1.46, 3.604])
"""
DataList = []
fin = open('Data.txt', 'r')
for line in fin:
    DataList.append(float(line))
Y = np.array(DataList)
X = np.linspace(0, len(Y) - 1, len(Y))


plt.scatter(X, Y, s = 5)
plt.plot(X, Y, linewidth = 0.8)

for i in range(0, -3, -1):
    plt.plot( [0, len(X)], [Y[0:1] + i, Y[0:1] - (len(X)*(1/7)) + i], linewidth = 0.4, color = 'black' )

for i in range(int(math.sqrt(len(X))), int(math.sqrt(len(X))) + 1, 1):

#k = 0
#for i in range(k, k + 10, 1):
    A, b = NormSystem(X, Y, i)
    a = GausSLAU(A, b)
    print (i, a)
    print(i, error(X, Y, a))
    print( )
    x = np.linspace(0,len(Y) - 1, 1000, endpoint = True)
    y = np.zeros(1000)
    
    for j in range(0, 1000, 1):
        y[j] = F(a, x[j])
    
    plt.plot(x,y, label = str(i), linewidth = 1, color = 'red')
    
plt.minorticks_on()
plt.grid(which='major', color = 'k', linewidth = 0.5) 
plt.grid(which='minor', color = 'k', linestyle = ':', linewidth = 0.4) 
plt.legend(fontsize = 4, ncol = 5)    #  количество столбцов)

plt.savefig('Graph.png', dpi = 480)
plt.show()

a = input()