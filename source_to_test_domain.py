import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Convolution2D, Flatten, Dropout, MaxPool2D, BatchNormalization, GlobalAveragePooling2D
from keras.layers import Dense, Activation, Input
from keras.optimizers import RMSprop, SGD, Adam
from keras.models import Sequential, Model, load_model
from sklearn.model_selection import train_test_split
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob
image_list = []

#####preparing Test Data
# for filename in glob.glob('E:\DL\project\Test_data\Skin Image Data Set-1\skin_data\melanoma\dermquest/*.jpg'): #assuming jpg
#     im = Image.open(filename)
#     imResize = im.resize((128, 128))#.convert('L')
#     image_list.append(np.array(imResize))
#
# for filename_new in glob.glob('E:\DL\project\Test_data\Skin Image Data Set-2\skin_data\melanoma_not\dermquest/*.jpg'):  # assuming jpg
#     im1 = Image.open(filename_new)
#     imResize1 = im1.resize((128, 128))#.convert('L')
#     image_list.append(np.array(imResize1))
#
# im.show()
# x = np.reshape(image_list, (len(image_list), 128,128,3))
# print(x.shape)
# np.save('data_128by128_CNN_test_dermquest', x)



x = np.load('data_128by128_CNN_test_dermquest.npy')
x_test = x.astype('float32')
x_test /= 255
a = np.ones((76, 1))
print(a.shape)
b = np.zeros((61, 1))
print(b.shape)
y = np.concatenate((a,b),axis=0)
print(y.shape)
np.save('test_data_proj_fin_lab',y)
model_test = load_model('src_mdl_proj_cnn.h5')
for layer in model_test.layers:
    layer.trainable = False
print(model_test.summary())
# model_test.layers.pop()
# model_test.layers.pop()
# model_test.layers.pop()
# model_test.layers.pop()
# model_test.layers.pop()
# model_test.summary()
pre = model_test.predict(x_test)
# print(pre.shape[0])
pre[pre<=0.5] = 0
pre[pre>0.5] = 1
# for d in range(0,len(pre)):
#     a = pre[d, 0]
#     if([a] > 0.5000):
#         pre = 1
#     else:
#         pre = 0
print(np.sum(np.abs(pre-y))/len(pre)*100)
