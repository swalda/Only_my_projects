
from pyglet import app, gl, graphics
from pyglet.window import Window, key
import numpy as np


window = Window(visible=True, width=500, height=500,
                resizable=True, caption='ЛР 2. Часть 2')

gl.glClearColor(0.1, 0.1, 0.1, 1.0) # цвет почти черный
gl.glClear(gl.GL_COLOR_BUFFER_BIT) # заливаем окно цветом фона
gl.glEnable(gl.GL_POINT_SMOOTH)
radius = 9
gl.glPointSize(10)
gl.glLineWidth(1)

smooth = False
continuous = False

@window.event
def on_draw():
    global smooth
    global continuous
    window.clear()
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    gl.glOrtho(-18, 18, -15, 15, -1, 1)

    # вывести оси координат штрих-пунктирной линией
    if (not continuous):
        gl.glEnable(gl.GL_LINE_STIPPLE)
        pattern = '0b1111000011110000'
        gl.glLineStipple(2, int(pattern, 2))

    gl.glColor3f(1, 1, 1)
    gl.glBegin(gl.GL_LINES)
    gl.glVertex3f(-240, 0, 0)
    gl.glVertex3f(240, 0, 0)
    gl.glVertex3f(0, -240, 0)
    gl.glVertex3f(0, 240, 0)
    gl.glEnd()
    gl.glDisable(gl.GL_LINE_STIPPLE)

    # вывести, используя GL_LINE_LOOP, правильный девятиугольник
    gl.glColor3f(1, 1, 1)
    gl.glBegin(gl.GL_LINE_LOOP)
    for i in range(0, 9):
        gl.glVertex3f(radius * np.cos(2 * np.pi * i / 9), radius * np.sin(2 * np.pi * i / 9), 0)
    gl.glEnd()

    # в его центре вывести несглаженную точку
    if (smooth):
        gl.glEnable(gl.GL_POINT_SMOOTH)
    else:
        gl.glDisable(gl.GL_POINT_SMOOTH)

    gl.glColor3f(0.3, 0.4, 0.5)
    gl.glBegin(gl.GL_POINTS)
    gl.glVertex3f(0, 0, 0)
    gl.glEnd()

    # в его вершинах вывести сглаженные точки
    gl.glEnable(gl.GL_POINT_SMOOTH)
    gl.glBegin(gl.GL_POINTS)
    for i in range(0, 9):
        gl.glVertex3f(radius * np.cos(2 * np.pi * i / 9), radius * np.sin(2 * np.pi * i / 9), 0)
    gl.glEnd()

    # внутри девятиугольника вывести лицевую сторону треугольника (использовать draw)
    # при выводе треугольника и прямоугольника задать в их вершинах разные цвета
    graphics.draw(3, gl.GL_TRIANGLES,
                  ('v2f', (3, 3,
                           -3, -3,
                           3, -3)),
                  ('c3f', (1, 0, 1,
                           1, 0.5, 1,
                           0, 0, 1)))

    # вывести нелицевую сторону прямоугольника, охватывающего девятиугольник (использовать glBegin / glEnd)
    gl.glBegin(gl.GL_QUADS)  # обход по часовой
    gl.glColor3f(0, 0, 1)
    gl.glVertex3f(radius, radius, 0)
    gl.glColor3f(0, 1, 0)
    gl.glVertex3f(radius, -radius, 0)
    gl.glColor3f(0, 1, 1)
    gl.glVertex3f(-radius, -radius, 0)
    gl.glColor3f(1, 0, 0)
    gl.glVertex3f(-radius, radius, 0)
    gl.glEnd()


# ЧАСТЬ 2 с ключами
# В процедуре on_key_press предусмотреть следующие действия:
# 1  -вывод осей координат сплошной линией;
# 2  -отказ от интеполяции цветов;
# 3  -вывод лицевой стороны в виде точек, а нелицевой в виде ребер;
# 4  -отказ от сглаживания точки;
# 5  -возврат к исходной сцене.

@window.event
def on_key_press(symbol, modifiers):
    mode_f = mode_b = None
    global smooth
    global continuous

    # вывод осей координат сплошной линией
    if symbol == key._1:
        continuous = not continuous

    # отказ от интеполяции цветов
    elif symbol == key._2:
        mode_f = mode_b = gl.GL_FILL
        shade_model = gl.GL_FLAT

    # вывод лицевой стороны в виде точек, а нелицевой в виде ребер
    elif symbol == key._3:
        mode_f = gl.GL_POINT
        mode_b = gl.GL_LINE
        shade_model = gl.GL_SMOOTH

    # отказ от сглаживания точки
    elif symbol == key._4:
        smooth = not smooth

    # возврат к исходной сцене
    elif symbol == key._5:
        mode_f = mode_b = gl.GL_FILL
        shade_model = gl.GL_SMOOTH
        smooth = True
        continuous = False

    if mode_f is not None:
        gl.glPolygonMode(gl.GL_FRONT, mode_f)
        gl.glPolygonMode(gl.GL_BACK, mode_b)
        gl.glShadeModel(shade_model)


app.run()