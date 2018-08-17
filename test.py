from skimage import io,filters,color,transform
from keras import utils as np_utils
from keras.models import load_model
import os
import numpy as np

# file_list = []

model = load_model('model.h5')
#image = io.imread("raspberry.jpg")
#image = color.rgb2gray(image)
categories = os.listdir("./training_data")
train_data = []
labels = []
number_of_images = 0
result = dict()
result_list = list()
file_list = os.listdir("./predict")
print("file_list : ",file_list)
for file_name in file_list:
    image = io.imread("./predict/"+file_name)
    image = transform.resize(image,(100,100))
    #image = color.rgb2gray(image)
    print("./predict/"+file_name)
    number_of_images += 1
    image = image.reshape(1,100,100,3)
    prediction = model.predict_classes(image)
    print("class : ",categories[prediction[0]])
    #print("categories : ",model.predict_proba(image))
    prediction = model.predict(image)
    print('\n Probabilities')
    max_probabilty = prediction[0].argsort()[-10:][::-1]
    for i,value in enumerate(max_probabilty):
        print(categories[value]," :",prediction[0][value])
    input("next")
    #sorted(result.items(), key=lambda x: x[1], reverse=True)
#     result_list.append(result)
# print(result_list)
#         # thres = filters.threshold_otsu(image)
#         train_data.append(image)
#         # thres = filters.threshold_mean(image)
        # image = image > thres
        #fig, ax = filters.isodata(image, figsize=(100, 100), verbose=False)
        #image = filters.gaussian(image,sigma=10)

# image = io.imread("r.jpg")
#         #image = color.rgb2gray(image)
# image = image.reshape(1,100,100,3)
# prediction = model.predict(image)
# print(prediction)
# for i,value in enumerate(prediction[0]):
#     print("Probability of ",categories[i]," = ",value)