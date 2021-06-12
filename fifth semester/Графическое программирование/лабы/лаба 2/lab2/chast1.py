import pyglet
from pyglet import app, gl, graphics
from pyglet.window import Window
d, d2 = 10, 10
wx, wy = 2 * d, 2 * d # Параметры области визуализации
width, height = int(10 * wx), int(10 * wy) # Размеры окна вывода
window = Window(visible = True, width = width, height = height,
                resizable = True, caption = 'ЛР 2. Часть 1')
gl.glClearColor(0.1, 0.1, 0.1, 1.0)
gl.glClear(gl.GL_COLOR_BUFFER_BIT)
@window.event
def on_draw():
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    gl.glOrtho(-wx, wx, -wy, wy, -1, 1)
    gl.glPointSize(15)
    gl.glLineWidth(15)

    gl.glBegin(gl.GL_POINTS)
    gl.glColor3f(0, 0, 1)
    gl.glVertex3f(-6, 3, 0)
    gl.glVertex3f(6, 3, 0)
    gl.glVertex3f(-6, -6, 0)

    gl.glEnd()

    gl.glBegin(gl.GL_LINES)
    gl.glColor3f(1, 0, 0)

    gl.glVertex3f(-1.5, 0, 0)
    gl.glVertex3f(4.55, 0, 0)

    gl.glVertex3f(-4.5, 6, 0)
    gl.glVertex3f(4.55, 6, 0)

    gl.glVertex3f(6, -1.5, 0)
    gl.glVertex3f(6, -7.5, 0)

    gl.glVertex3f(-4.5, -9, 0)
    gl.glVertex3f(4.5, -9, 0)
    gl.glEnd()

app.run()

