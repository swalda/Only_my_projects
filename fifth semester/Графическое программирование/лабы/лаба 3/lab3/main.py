
from pyglet import app, gl, graphics
from pyglet.window import Window, key
import numpy as np
from sys import exit

d = 12
wx, wy = 2 * d, 2 * d # Параметры области визуализации
width, height = int(20 * wx), int(20 * wy) # Размеры окна вывода

#
mtClr0 = [1, 1, 1, 0] # Цвет материала
light_position0 = [0, 30, 30, 0] # Позиция источника света
lghtClr0 = [0, 1, 1, 0] # Цвет источника света

mtClr = (gl.GLfloat * 4)()
light_position = (gl.GLfloat * 4)()
lghtClr = (gl.GLfloat * 4)()

for i in range(4): mtClr[i] = mtClr0[i]
for i in range(4): light_position[i] = light_position0[i]
for i in range(4): lghtClr[i] = lghtClr0[i]

window = Window(visible=True, width=width, height=height,
                resizable=True, caption='Лаб3')
gl.glClearColor(0.1, 0.1, 0.1, 1.0)
gl.glClear(gl.GL_COLOR_BUFFER_BIT)
gl.glEnable(gl.GL_LIGHTING) # Активизируем использование материалов


@window.event
def on_draw():
    window.clear()
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    gl.glOrtho(-wx, wx, -wy, wy, -20, 20)
    # Проецирование выполнить, повернув изображение на 30° относительно оси X и на 10° относительно оси Y.
    gl.glRotatef(30, 10, 0, 0)

    gl.glShadeModel(gl.GL_SMOOTH) # GL_FLAT - без интерполяции цветов
    gl.glMaterialfv(gl.GL_FRONT, gl.GL_SPECULAR, mtClr)

    gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, light_position)
    gl.glLightfv(gl.GL_LIGHT0, gl.GL_SPECULAR, lghtClr)

    gl.glEnable(gl.GL_LIGHT0) # Включаем в уравнение освещенности источник GL_LIGHT0

    v0 = np.array([0, 0, 0])
    v1 = np.array([-d, -d, -5])
    v2 = np.array([d, -d, -5])
    v3 = np.array([d, d, -5])
    v4 = np.array([-d, d, -5])

    a = v0 - v1
    b = v2 - v1

    a1 = v2 - v0
    b1 = v1 - v0

    a2 = v0 - v2
    b2 = v3 - v2

    n = np.cross(a, b)
    n = n / np.sqrt(np.sum(n ** 2))

    n1 = np.cross(b1, a1)
    n1 = n1 / np.sqrt(np.sum(n1 ** 2))

    n2 = np.cross(b2, a2)
    n2 = n2 / np.sqrt(np.sum(n2 ** 2))

    gl.glColor3f(0, 1, 0)
    gl.glBegin(gl.GL_TRIANGLE_FAN)

    gl.glNormal3f(n[0], n[1], n[2])
    gl.glVertex3f(v0[0], v0[1], v0[2])

    gl.glNormal3f(n1[0], n1[1], n1[2])
    gl.glVertex3f(v1[0], v1[1], v1[2])
    gl.glVertex3f(v2[0], v2[1], v2[2])

    gl.glNormal3f(n2[0], n2[1], n2[2])
    gl.glVertex3f(v3[0], v3[1], v3[2])
    gl.glVertex3f(v4[0], v4[1], v4[2])

    gl.glEnd()

    gl.glLineWidth(3)
    gl.glBegin(gl.GL_LINES)
    gl.glColor3f(0, 1, 0)
    gl.glVertex3f(v1[0], v1[1], v1[2])
    gl.glVertex3f(v1[0] + 5 * n[0], v1[1] + 5 * n[1], v1[2] + 5 * n[2])
    gl.glEnd()

    gl.glBegin(gl.GL_LINES)
    gl.glColor3f(0, 1, 0)
    gl.glVertex3f(v0[0], v0[1], v0[2])
    gl.glVertex3f(v0[0] + 5 * n1[0], v0[1] + 5 * n1[1], v0[2] + 5 * n1[2])
    gl.glEnd()

    gl.glBegin(gl.GL_LINES)
    gl.glColor3f(0, 1, 0)
    gl.glVertex3f(v2[0], v2[1], v2[2])
    gl.glVertex3f(v2[0] + 5 * n2[0], v2[1] + 5 * n2[1], v2[2] + 5 * n2[2])
    gl.glEnd()

    gl.glPointSize(6)
    gl.glEnable(gl.GL_POINT_SMOOTH)
    graphics.draw(3, gl.GL_POINTS,
                  ('v3f', (v1[0], v1[1], v1[2],
                           v0[0], v0[1], v0[2],
                           v2[0], v2[1], v2[2])),
                  ('c3B', (1, 1, 0,
                           1, 1, 0,
                           1, 1, 0)))
    gl.glDisable(gl.GL_POINT_SMOOTH)



@window.event
def on_key_press(symbol, modifiers):
    if symbol == key._1:
        gl.glDisable(gl.GL_LIGHTING)
    elif symbol == key._2:
        gl.glEnable(gl.GL_LIGHTING)


app.run()





