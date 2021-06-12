

import numpy as np
from pyglet.gl import *
from pyglet import app
from pyglet import graphics
from pyglet.window import Window, key

d = 12
wx, wy, wz = 3 * d, 3 * d, 3 * d  # Параметры области визуализации
width, height = int(20 * wx), int(20 * wy)  # Размеры окна вывода
tile_x = 1
tile_y = 1
fl = True

def to_c_float_array(data):
    return (GLfloat * len(data))(*data)

# ABDE (нижняя грань) 
A = to_c_float_array([d, d, -d])
B = to_c_float_array([-d, 3/2*d, -d])
C = to_c_float_array([-3/2*d, 0, -d])
D = to_c_float_array([-d, -3/2*d, -d])
E = to_c_float_array([d, -d, -d])
# FGIJ (верхняя грань)
F = to_c_float_array([d, d, 2*d])
G = to_c_float_array([-d, 3/2*d, 2*d])
H = to_c_float_array([-3/2*d, 0, 2*d])
I = to_c_float_array([-d, -3/2*d, 2*d])
J = to_c_float_array([d, -d, 2*d])

p = GL_TEXTURE_2D
def tex_init():
    fn = 'corgi.jpg'
    img = pyglet.image.load(fn)
    i_width = img.width
    i_height = img.height
    img = img.get_data('RGB', i_width * 3)

    r = GL_RGB
    # Задаем параметры текстуры
    p = GL_TEXTURE_2D
    glTexParameterf(p, GL_TEXTURE_WRAP_S, GL_REPEAT)  # GL_CLAMP
    glTexParameterf(p, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameterf(p, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameterf(p, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    # Способ взаимодействия с текущим фрагментом изображения
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)
    # Создаем 2d-текстуру на основе образа img
    glTexImage2D(p, 0, r, i_width, i_height, 0, r, GL_UNSIGNED_BYTE, img)
    glEnable(p)

window = Window(visible = True, width = width, height = height,
                resizable = True, caption = 'Lab4')

glClearColor(0.1, 0.1, 0.1, 1.0)
glClear(GL_COLOR_BUFFER_BIT)
tex_init()
angle = 0

@window.event
def on_draw():
    window.clear()
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glRotatef(270, 1, 0, 0)
    glRotatef(150, 0, 0, 1)
    glRotatef(angle, 0, 1, 0)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(-wx, wx, -wy, wy, -wz, wz)

    # нижняя грань
    graphics.draw(5, GL_POLYGON, ('t3f', (0, 0, 0, 
                                          tile_x, 0, 0, 
                                          tile_x, tile_y, -1, 
                                          0, tile_y, -1,
                                          0, 0, 0)),
                                 ('v3f', (D[0], D[1], D[2],
                                          E[0], E[1], E[2],
                                          A[0], A[1], A[2], 
                                          B[0], B[1], B[2], 
                                          C[0], C[1], C[2])))
    
    # боковая поверхность
    graphics.draw(10, GL_QUAD_STRIP, ('t2f', (0, 0, 0, tile_y, 
                                              tile_x, 0, tile_x, tile_y,
                                              0, 0, 0, tile_y, 
                                              tile_x, 0, tile_x, tile_y,
                                              0, 0, 0, tile_y)), 
                                     ('v3f', (E[0], E[1], E[2], J[0], J[1], J[2],
                                              A[0], A[1], A[2], F[0], F[1], F[2],
                                              B[0], B[1], B[2], G[0], G[1], G[2],
                                              C[0], C[1], C[2], H[0], H[1], H[2], 
                                              D[0], D[1], D[2], I[0], I[1], I[2])))

@window.event
def on_key_press(symbol, modifiers):
    global tile_x, tile_y, angle
    if symbol == key._1:
        tile_x = tile_y = 1
    elif symbol == key._2:
        tile_x = tile_y = 2
    elif symbol == key._3:
        tile_x = 3
        tile_y = 2
    elif symbol == key._4:
        tile_x = tile_y = 3
    elif symbol == key._5:
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
    elif symbol == key._6:
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    elif symbol == key._7:
        angle += 5
    elif symbol == key._8:
        angle -= 5

app.run()





