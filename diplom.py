# Подключаем необходимые пакеты
# Пакет os нужен для работы с файловой системой
import os
# Пакет pandas нужен для работы с csv файлами, в которых хранится информация о нужном контуре
import pandas as pd
# Пакет для работы с картинками. Позволяет картинки считывать, поворачивать, изменять размер
import cv2
# Пакет для работы с массивами
import numpy as np
# импортируем предобученую модель и необходимый для нее препроцессинг
from keras.applications.mobilenet import MobileNet, preprocess_input
# Генератор картинок, используемый для расширения обучающей выборки. Поворачивает, зумит картинку
from keras.preprocessing.image import ImageDataGenerator
# Встроеные стандартные слои
from keras.layers import Dense, GlobalAveragePooling2D
# Алгоритм оптимизации. Здесь нужен  SGD
from keras.optimizers import Adam
# Класс для создания своей модели
from keras.models import Model


# функция для создания маски: изображения, где отмечена область ценника
def create_mask(shape, contour):
    x, y, w, h = contour
    mask = np.zeros(shape)
    mask[y: y + h, x: x + w] = 1
    return mask


# загрузка файла с метаданными по каждой картинки
metadata = pd.read_csv('metadata.csv')
# получаем список картинок, лежащий в директории 'images/'
images = os.listdir('images/')


# пустые списки для записи туда картинок
X = []
y = []


# цикл, внутри которого идем по списку картинок, считываем картинку, делаем препроцессинг,
# находим метаданные по картинке, по этим метаданным создаем маску изображения, изменяем размер
# исходного изображения и его маски, добавляем в список
for image in images:
    img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
    img = preprocess_input(img.astype(float))
    contour = metadata[metadata.image == image].iloc[0]
    mask = create_mask(img.shape, contour)
    img = cv2.resize(img, (224, 224))
    mask = cv2.resize(mask, (32, 32))
    X.append(img)
    y.append(mask)


# преобразуем списки в массивы для более удобной работы
X = np.array(X)
y = np.array(y)

# разделим размеченную выборку на тренеровачную, валидационную и тестовую
X_train = X[:1300]
y_train = y[:1300]
X_val = X[1300:1600]
y_val = y[1300:1600]
X_test = X[1600:]
y_test = y[1600:]

# зададим константы
train_seed = 777
val_seed = 222
batch_size = 64

# создадим генераторы для непрерывного расширения обучающей выборки
image_generator = ImageDataGenerator(rotation_range=10,
                                     width_shift_range=0.1,
                                     height_shift_range=0.1,
                                     horizontal_flip=True,
                                     vertical_flip=True,
                                     zoom_range=0.25)

mask_generator = ImageDataGenerator(rotation_range=10,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    zoom_range=0.25)


def get_mask_generator(y, seed):
    for batch in mask_generator.flow(y, batch_size=batch_size, seed=seed):
        yield batch[:, :, :, 0].reshape((batch.shape[0], batch.shape[1] * batch.shape[2]))


train_generator = zip(image_generator.flow(X_train, batch_size=batch_size, seed=train_seed),
                      get_mask_generator(y_train, train_seed))

val_generator = zip(image_generator.flow(X_val, batch_size=batch_size, seed=val_seed),
                    get_mask_generator(y_val, val_seed))

# создадим модель, взяв за основу предобучению модель
model_mobilenet = MobileNet(input_shape=(224, 224, 3), include_top=False)
model_output = GlobalAveragePooling2D()(model_mobilenet.output)
model_output = Dense(32 * 32, activation='sigmoid')(model_output)
MODEL = Model(input=model_mobilenet.input, output=model_output)

# скомпилируем модель
MODEL.compile(loss='binary_crossentropy', optimizer=Adam(),
              metrics=['binary_crossentropy', 'accuracy'])

# запустим обучение
MODEL.fit_generator(train_generator, steps_per_epoch=len(X_train) / batch_size,
                    epochs=50, verbose=1,
                    validation_data=val_generator, validation_steps=len(X_val) / batch_size)
