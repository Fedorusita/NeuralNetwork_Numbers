import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist

# Загружаем набор данных MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Нормализуем данные
x_train = x_train / 255.0
x_test = x_test / 255.0

# Преобразуем метки в формат one-hot
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

# Определяем модель
model = keras.Sequential([
    Flatten(input_shape=(28, 28)),  # Преобразуем изображения в одномерный массив
    Dense(128, activation='relu'),   # Полносвязный слой с 128 нейронами и активацией ReLU
    Dense(10, activation='softmax')   # Выходной слой с 10 нейронами для классов цифр (0-9)
])

# Компилируем модель
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Проверяем, существует ли уже обученная модель; если нет, обучаем новую
try:
    model = keras.models.load_model('mnist_model.h5')  # Загружаем модель из файла
    print("Модель загружена.")
except:
    # Обучаем модель, если файл не существует
    model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2)
    # Сохраняем модель после обучения
    model.save('mnist_model.h5')
    print("Модель обучена и сохранена.")

# Оцениваем модель на тестовых данных
test_loss, test_accuracy = model.evaluate(x_test, y_test_cat)
print(f"Тестовая точность: {test_accuracy:.4f}")

# Делаем предсказание на примере изображения
n = 678  # Измените этот индекс для тестирования других изображений
x = np.expand_dims(x_test[n], axis=0)  # Добавляем размерность для предсказания
print(x_test[n])
res = model.predict(x)  # Получаем предсказание от модели
print(res)
print(f"Распознанная цифра: {np.argmax(res)}")  # Находим индекс с максимальным значением

# Отображаем изображение
plt.imshow(x_test[n], cmap=plt.cm.binary)
plt.show()