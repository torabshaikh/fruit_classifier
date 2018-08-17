from skimage import io,filters,color
import keras,os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Reshape
from keras.optimizers import SGD
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
import matplotlib.pyplot as plt
# from parser import load_data
categories = os.listdir("./training_data")
train_data = []
file_list = []
labels = []
number_of_images = 0

model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
# model.add(Reshape((6, 2)))
model.add(Dense(len(categories), activation='softmax'))
sgd = SGD(lr=0.0025, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
# model.fit(train_data, labels, verbose=2,epochs=2)
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'training_data',
        classes = categories,
        target_size=(100, 100),
        batch_size=32,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        'testing_data',
        classes = categories,
        target_size=(100, 100),
        batch_size=32,
        class_mode='categorical')

#test_data=data_gen.flow_from_directory('testing_data')

history = model.fit_generator(train_generator,steps_per_epoch=100,epochs=1,validation_data = validation_generator)
# model_file = 'two_category.pickle'
# with open(model_file, 'wb') as f:
#     pickle.dump(model,f)
model.save('model.h5')
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
print(history.history)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()