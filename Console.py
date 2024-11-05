import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Загрузка предобученной модели
model = load_model('mnist_model.h5')

# Создание нового изображения для рисования
width, height = 28, 28
image = Image.new("L", (width, height), 255)  # Создаем черно-белое изображение (L - grayscale)
draw = ImageDraw.Draw(image)


# Функция для обработки рисунка и предсказания цифры
def predict_digit():
    # Нормализуем значения пикселей (0-1)
    img_array = np.array(image) / 255.0  # Нормализация: белый (255) становится 0, черный (0) становится 1

    # Преобразуем в бинарное представление: черный пиксель - 1, белый - 0
    binary_array = np.where(img_array < 0.5, 1.0, 0.0)

    # Визуализируем изображение перед предсказанием
    plt.imshow(binary_array, cmap='gray')
    plt.title("Изображение для предсказания")
    plt.axis('off')
    plt.show()

    # Добавляем новую ось для соответствия форме входа модели
    x = np.expand_dims(binary_array, axis=0)  # Теперь добавляем ось для batch size
    print(binary_array)

    # Делаем предсказание
    res = model.predict(x)

    # Выводим результат
    digit = np.argmax(res)
    result_label.config(text=f"Распознанная цифра: {digit}")


# Функция для рисования на Canvas
def paint(event):
    x = event.x // (canvas_size // width)  # Масштабируем координаты к размеру изображения (28x28)
    y = event.y // (canvas_size // height)

    # Рисуем на Canvas
    canvas.create_oval(event.x - 5, event.y - 5, event.x + 5, event.y + 5, fill='black', outline='black')

    # Рисуем на изображении: черный цвет (0) становится белым (255), а белый цвет (255) становится черным (0)
    draw.rectangle([x, y, x + 1, y + 1], fill=0)


# Функция для очистки Canvas и изображения
def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0, 0, width, height], fill=255)  # Очищаем изображение


# Создаем главное окно
root = tk.Tk()
root.title("Распознавание цифр")

# Создаем Canvas для рисования с размерами 280x280 пикселей (масштабирование)
canvas_size = 280  # Размер окна в пикселях для удобства рисования
canvas = tk.Canvas(root, width=canvas_size, height=canvas_size, bg='white')
canvas.pack()

# Кнопки для предсказания и очистки
predict_button = tk.Button(root, text="Предсказать", command=predict_digit)
predict_button.pack()

clear_button = tk.Button(root, text="Очистить", command=clear_canvas)
clear_button.pack()

# Метка для вывода результата
result_label = tk.Label(root, text="Распознанная цифра: ")
result_label.pack()

# Обработчик событий для рисования на Canvas
canvas.bind("<B1-Motion>", paint)

# Запускаем главный цикл приложения
root.mainloop()
