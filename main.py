from tensorflow import keras
from keras.utils import plot_model
import pickle
import matplotlib.pyplot as plt
import numpy as np

# объявление тренировачной и проверяющей выборки
train_ds = keras.utils.image_dataset_from_directory(
    directory='D:/Datasets/dataset_0.3/train/',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(256, 256))
validation_ds = keras.utils.image_dataset_from_directory(
    directory='D:/Datasets/dataset_0.3/val/',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(256, 256))
test_ds = keras.utils.image_dataset_from_directory(
    directory='D:/Datasets/dataset_0.3/test/',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(256, 256))
#
# model = keras.Sequential([
#     keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(256, 256, 3)),
#     keras.layers.AveragePooling2D((2, 2), strides=2),
#     keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
#     keras.layers.AveragePooling2D((2, 2), strides=2),
#     keras.layers.Flatten(),
#     keras.layers.Dense(128, activation='relu'),
#     keras.layers.Dense(2,  activation='softmax')
# ])
#
# model = keras.Sequential([
#     keras.layers.Conv2D(1, (5, 5), padding='valid', input_shape=(256, 256, 3)),
#     keras.layers.Conv2D(16, (5, 5), padding='valid', activation='relu'),
#     keras.layers.AveragePooling2D((3, 3), strides=2),
#     keras.layers.Conv2D(16, (3, 3), padding='valid', activation='relu'),
#     keras.layers.AveragePooling2D((3, 3), strides=2),
#     keras.layers.Conv2D(16, (3, 3), padding='valid', activation='relu'),
#     keras.layers.AveragePooling2D((3, 3), strides=2),
#     keras.layers.Conv2D(16, (3, 3), padding='valid', activation='relu'),
#     keras.layers.AveragePooling2D((3, 3), strides=2),
#     keras.layers.Flatten(),
#     keras.layers.Dense(128, activation='relu'),
#     keras.layers.Dense(128, activation='relu'),
#     keras.layers.Dense(2,  activation='softmax')
# ])
#
model = keras.Sequential([
    keras.layers.Conv2D(1, (3, 3), padding='valid', input_shape=(256, 256, 3)),
    keras.layers.Conv2D(2, (3, 3), padding='valid'),
    keras.layers.Conv2D(4, (3, 3), padding='valid', activation='relu'),
    keras.layers.AveragePooling2D((2, 2), strides=2, padding='same'),

    keras.layers.Conv2D(16, (3, 3), padding='valid', activation='relu'),
    keras.layers.AveragePooling2D((2, 2), strides=2, padding='valid'),

    keras.layers.Conv2D(16, (3, 3), padding='valid', activation='relu'),
    keras.layers.AveragePooling2D((2, 2), strides=2, padding='valid'),

    keras.layers.Conv2D(16, (3, 3), padding='valid', activation='relu'),
    keras.layers.AveragePooling2D((2, 2), strides=2, padding='valid'),

    keras.layers.Conv2D(16, (3, 3), padding='valid', activation='relu'),
    keras.layers.AveragePooling2D((2, 2), strides=2, padding='valid'),

    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])
#
# загрузка модели и сохранение ее структуры в виде png изображения
# model = keras.models.load_model('model3_03_new/model05.h5')
# print(model.summary())
# plot_model(model, to_file='model.png')
# predictions = model.predict(test_ds)
# print(predictions)
#
# загрузка изображения в обученную нейронную сеть и проверка изображения на наличие скрытой информации
# image = keras.utils.load_img('D:/Datasets/dataset_0.3/test/encode/0.jpg')
# input_arr = keras.utils.img_to_array(image)
# input_arr = np.array([input_arr])  # Convert single image to a batch.
# predictions = model.predict(input_arr)
# print(np.argmax(predictions))
#
# # модель resnet
# model = keras.applications.ResNet50V2(
#     weights=None, input_shape=(256, 256, 3), classes=2)
# print(model.summary())
#
# компиляция модели и создание объекта для сохранения модели каждую эпоху
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy', keras.metrics.AUC(), keras.metrics.Precision(),
                       keras.metrics.Recall()])
checkpoint = keras.callbacks.ModelCheckpoint('model3_03_new/model{epoch:02d}.h5', period=1)

# тренеровка модели
history = model.fit(train_ds, callbacks=[checkpoint], epochs=5, validation_data=validation_ds)

# сохранение истории обучения
np.save('model3_03_new/history.npy', history.history)

# загрузка истории обучения
history = np.load('model3_03_new/history.npy', allow_pickle=True).item()
print(history.keys())
# видуализация обучения и проверка точности значений
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# видуализация обучения и проверка величины потерь
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# доля объектов, названных классификатором положительными и при этом действительно являющимися положительными
plt.plot(history['precision'])
plt.plot(history['val_precision'])
plt.title('model precision')
plt.ylabel('precision')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# какую долю объектов положительного класса из всех объектов положительного класса нашел алгоритм
plt.plot(history['recall'])
plt.plot(history['val_recall'])
plt.title('model recall')
plt.ylabel('recall')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# частота истинных срабатываний и частоту ложных срабатываний
plt.plot(history['auc'])
plt.plot(history['val_auc'])
plt.plot([0, 4], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
plt.title('model auc')
plt.ylabel('auc')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
