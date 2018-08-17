from skimage import io,filters,color,transform
from keras.models import load_model,model_from_json
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

model = load_model('model.h5')
#image = io.imread("raspberry.jpg")
#image = color.rgb2gray(image)
categories = os.listdir("./training_data")
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
history = model.fit_generator(train_generator,steps_per_epoch=100,epochs=10,validation_data = validation_generator)
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