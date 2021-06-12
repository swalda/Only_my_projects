import numpy as np, time
import matplotlib.pyplot as plt
from sys import exit
#
img_cols = img_rows = 28
pathToData = 'G:\\AM\\2020\\ЛР8_9\\'
# 1 - Монте-Карло
# 2 - метрики
# 3 - ImageDataGenerator
show = 2 # 2 3
#
def load_test_data():
    print('Загрузка данных из двоичных файлов...')
    with open(pathToData + 'imagesTest.bin', 'rb') as read_binary:
        x_test = np.fromfile(read_binary, dtype = np.uint8)
    with open(pathToData + 'labelsTest.bin', 'rb') as read_binary:
        y_test = np.fromfile(read_binary, dtype = np.uint8)
    x_test = np.array(x_test)
    N = img_rows * img_cols
    x_test_shape_0 = int(x_test.shape[0] / N)
    x_test = x_test.reshape(x_test_shape_0, img_rows, img_cols, 1)
    #
    if show == 1:
        # Работа с большими данными
        # На примере вычисления средней загрузки пикселя изображения
        d = 6
        def calc_s(N, data):
            s = 0
            for x in data:
                x = x.reshape(N)
                s += np.sum(x)
            return s / (N * len(data))
        # 1:
        t0 = time.time()
        s = calc_s(N, x_test)
        print('1:', s, 't:', round(time.time() - t0, d))
        # 2:
        t0 = time.time()
        s = 0
        part_size = int(x_test_shape_0 / 20)
        n_parts = 5
        for k in range(n_parts):
            # Генерируем part_size случайных индексов из [0, x_test_shape_0)
            idx = np.random.randint(0, x_test_shape_0, part_size)
            x_test_idx = x_test[idx]
            s_part = calc_s(N, x_test_idx)
            s += s_part
        s /= n_parts
        print('2:', s, 't:', round(time.time() - t0, d))
        # 3:
        from sklearn import model_selection
        t0 = time.time()
        s = 0
        n_parts = 5
        for k in range(n_parts):
            _, x_part, _, y_part = model_selection.train_test_split(x_test, y_test, test_size = 0.01)
            s_part = calc_s(N, x_part)
            s += s_part
        s /= n_parts
        print('3:', s, 't:', round(time.time() - t0, d))
    elif show == 2: # Метрики
        from scipy.spatial import distance
        from skimage.measure import compare_ssim
        i1 = np.random.randint(10000)
        i2 = np.random.randint(10000)
        lb1 = y_test[i1]; lb2 = y_test[i2]
        im1 = x_test[i1].reshape(N)
        im2 = x_test[i2].reshape(N)
        dist = distance.euclidean(im1, im2) # Евклидово расстояние
        dist_cs = abs(distance.cosine(im1, im2)) # Косинусное расстояние
        sim = compare_ssim(im1, im2) # Индекс структурного сходства
        print(dist, dist_cs, sim)
        plt.subplot(1, 2, 1); plt.title(lb1); plt.axis('off')
        plt.imshow(im1.reshape(img_rows, img_cols), cmap = plt.get_cmap('gray'))
        plt.subplot(1, 2, 2); plt.title(lb2); plt.axis('off')
        plt.imshow(im2.reshape(img_rows, img_cols), cmap = plt.get_cmap('gray'))
        plt.show()
    return x_test, y_test
#
# Загрузка данных и формирование обучающих и тестовых выборок
x_test, y_test = load_test_data()
#
if show == 3:
    from keras.preprocessing.image import ImageDataGenerator
    datagen = ImageDataGenerator(
        width_shift_range = 0.1, # Случайный горизонтальный сдвиг изображения
        height_shift_range = 0.1 # Случайный вертикальный сдвиг изображения
        )
    print('Настройка генератора')
    datagen.fit(x_test) # На входе 4-мерный массив 
    print('Получаем сгенерированные образы')
    xy = datagen.flow(x_test, y_test, batch_size = 1) # batch_size = 32
    # xy - keras_preprocessing.image.NumpyArrayIterator
    # xy[i] - tuple (два элемента - массив данных и массив с одной меткой)
    print('Выделяем данные и метки')
    x_gen, y_gen = [], []
    for k in range(len(xy)): # 10'000
        x_gen.append(xy[k][0][0])
        y_gen.append(xy[k][1][0])
    print('Показываем 50 случайных экземпляров')
    k = -1
    for k in range(50):
        i = np.random.randint(len(xy)) # 10'000
        lb = y_gen[i]
        plt.subplot(5, 10, k + 1)
        img = x_gen[i].reshape(img_rows, img_cols)
        plt.imshow(img, cmap = plt.get_cmap('gray'))
        plt.title(lb)
        plt.axis('off')
    plt.subplots_adjust(hspace = 0.1) # wspace
    plt.show()
