from sys import exit
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Reshape, Add, Conv2D, Conv1D, MaxPooling2D, MaxPooling1D
import time

show_4 = not True
pathToData = '' # G:/AM/НС/mnist/
img_rows = img_cols = 28
num_classes = 10
epochs = 20

pathToHistory = '' # G:/AM/Лекции/
suff = '.txt'
# Имена файлов, в которые сохраняется история обучения
fn_loss = pathToHistory + 'loss_' + suff
fn_acc = pathToHistory + 'acc_' + suff
fn_val_loss = pathToHistory + 'val_loss_' + suff
fn_val_acc = pathToHistory + 'val_acc_' + suff

def title_number(x):
    for i in range(len(x)):
        if x[i] == 1:
            return i
    return -1

def show_x(x, img_rows, img_cols):
    print(x[0].shape)
    for k in range(1, 5):
        plt.subplot(2, 2, k)
        # Убираем 3-е измерение
        plt.title(title_number(y_test[k]))
        plt.imshow(x[k].reshape(img_rows, img_cols), cmap = 'gray')
        plt.axis('off')
    plt.show()

# Вывод графиков
def one_plot(n, y_lb, loss_acc, val_loss_acc):
    plt.subplot(1, 2, n)
    if n == 1:
        lb, lb2 = 'loss', 'val_loss'
        yMin = 0
        yMax = 1.05 * max(max(loss_acc), max(val_loss_acc))
    else:
        lb, lb2 = 'acc', 'val_acc'
        yMin = min(min(loss_acc), min(val_loss_acc))
        yMax = 1.0
    plt.plot(loss_acc, color = 'r', label = lb, linestyle = '--')
    plt.plot(val_loss_acc, color = 'g', label = lb2)
    plt.ylabel(y_lb)
    plt.xlabel('Эпоха')
    #plt.ylim([0.95 * yMin, yMax])
    plt.legend()

def loadBinData(pathToData, img_rows, img_cols):
    print('Загрузка данных из двоичных файлов...')
    with open(pathToData + 'imagesTrain.bin', 'rb') as read_binary:
        x_train = np.fromfile(read_binary, dtype = np.uint8)
    with open(pathToData + 'labelsTrain.bin', 'rb') as read_binary:
        y_train = np.fromfile(read_binary, dtype = np.uint8)
    with open(pathToData + 'imagesTest.bin', 'rb') as read_binary:
        x_test = np.fromfile(read_binary, dtype = np.uint8)
    with open(pathToData + 'labelsTest.bin', 'rb') as read_binary:
        y_test = np.fromfile(read_binary, dtype = np.uint8)
    # Преобразование целочисленных данных в float32 и нормализация; данные лежат в диапазоне [0.0, 1.0]
    x_train = np.array(x_train, dtype = 'float32') / 255
    x_test = np.array(x_test, dtype = 'float32') / 255
    print(x_test.shape)
    x_train_shape_0 = int(x_train.shape[0] / (img_rows * img_cols))
    x_test_shape_0 = int(x_test.shape[0] / (img_rows * img_cols))
    x_train = x_train.reshape(x_train_shape_0, img_rows, img_cols, 1) # 1 - оттенок серого цвета
    x_test = x_test.reshape(x_test_shape_0, img_rows, img_cols, 1)
    #x_train = x_train.reshape(x_train_shape_0, (img_rows * img_cols * 1)) # избавиться от слоев Reshape И Flatten
    #x_test = x_test.reshape(x_test_shape_0, (img_rows * img_cols * 1))
    # Преобразование в категориальное представление: метки - числа из диапазона [0, 9] в двоичный вектор размера num_classes
    # Так, в случае MNIST метка 5 (соответствует классу 6) будет преобразована в вектор [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
    #print(y_train[0]) # (MNIST) Напечатает: 5
    print('Преобразуем массивы меток в категориальное представление')
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    #print(y_train[0]) # (MNIST) Напечатает: [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
    return x_train, y_train, x_test, y_test

start = time.time()

# Определяем форму входных данных
input_shape = (img_rows, img_cols, 1)
#input_shape = (img_rows * img_cols * 1) # избавиться от слоев Reshape И Flatten

# Создание модели нейронной сети
inp = Input(shape = input_shape) # Входной слой
x = x1 = inp
# Замена Flatten на Reshape
#x = Flatten()(x) # Преобразование 2D в 1D
#x = Reshape((-1,))(x)
x = Conv2D(16, kernel_size = 4, padding = 'same', activation = 'relu')(x)
x = MaxPooling2D(pool_size = 4, strides = 2, padding = 'same')(x)
x = Flatten()(x)
x = Dense(units = 32, activation = 'relu')(x)

#x = Dense(units = 32, activation = 'relu')(x)
#x = Dense(units = 32, activation = 'hard_sigmoid')(x)
#x1 = Dense(units = 32, activation = 'hard_sigmoid')(x1)
#y = Add()([x, x1])
output = Dense(num_classes, activation = 'softmax')(x)
#output = Dense(num_classes, activation = 'softmax')(y)
model = Model(inputs = inp, outputs = output)
model.summary()
#loss = keras.losses.cosine_similarity # 'mse'
#loss = keras.losses.squared_hinge # 'mse'
#optimizer = keras.optimizers.SGD() # 'Adam'
#optimizer = keras.optimizers.Adadelta() # 'Adam'
#model.compile(optimizer = optimizer, loss = loss, metrics = ['accuracy'])
model.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy'])

# Загрузка обучающего и проверочного множества из бинарных файлов
# Загружаются изображения и их метки
x_train, y_train, x_test, y_test = loadBinData(pathToData, img_rows, img_cols)
if show_4:
    show_x(x_test, img_rows, img_cols)
    exit()

# Обучение нейронной сети
history = model.fit(x_train, y_train, batch_size = 128, epochs = epochs,
                        verbose = 2, validation_data = (x_test, y_test))
# Запись истории обучения в текстовые файлы
history = history.history
with open(fn_loss, 'w') as output:
    for val in history['loss']: output.write(str(val) + '\n')
with open(fn_acc, 'w') as output:
    for val in history['accuracy']: output.write(str(val) + '\n')
with open(fn_val_loss, 'w') as output:
    for val in history['val_loss']: output.write(str(val) + '\n')
with open(fn_val_acc, 'w') as output:
    for val in history['val_accuracy']: output.write(str(val) + '\n')

print()
print('sec', time.time() - start)
        
# Вывод графиков обучения
plt.figure(figsize = (9, 4))
plt.subplots_adjust(wspace = 0.5)
one_plot(1, 'Потери', history['loss'], history['val_loss'])
one_plot(2, 'Точность', history['accuracy'], history['val_accuracy'])
plt.suptitle('Потери и точность')
plt.show()



