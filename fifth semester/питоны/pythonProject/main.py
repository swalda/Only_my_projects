import pyglet
from pyglet import app, gl, graphics
from pyglet.window import Window
dx, dy = 50, 50 # Размеры прямоугольника
dx2 = dx / 2
dy2 = dy / 2
wx, wy = 1.5 * dx2, 1.5 * dy2 # Параметры области визуализации
width, height = int(20 * wx), int(20 * wy) # Размеры окна вывода
# Создаем окно визуализации
window = Window(visible = True, width = width, height = height,
                resizable = True, caption = 'Прямоугольник')
gl.glClearColor(0.1, 0.1, 0.1, 1.0) # Задаем почти черный цвет фона
gl.glClear(gl.GL_COLOR_BUFFER_BIT) # Заливка окна цветом фона
@window.event
def on_draw():
    # Проецирование
    gl.glMatrixMode(gl.GL_PROJECTION) # Теперь текущей является матрица проецирования
    gl.glLoadIdentity() # Инициализация матрицы проецирования
    gl.glOrtho(-wx, wx, -wy, wy, -1, 1) # Ортографическое проецирование
    gl.glColor3f(1, 0.4, 1) # Зеленый цвет
    gl.glBegin(gl.GL_QUADS) # Обход против часовой стрелки
    gl.glVertex3f(-dx2, -dy2, 0)
    gl.glVertex3f(dx2, -dy2, 0)
    gl.glVertex3f(dx2, dy2, 0)
    gl.glVertex3f(-dx2, dy2, 0)
    gl.glEnd()
app.run()