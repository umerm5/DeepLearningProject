import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Convolution2D, Flatten, Dropout, MaxPool2D, BatchNormalization, GlobalAveragePooling2D
from keras.layers import Dense, Activation, Input
from keras.initializers import random_normal
from keras.optimizers import RMSprop, SGD, Adam
from keras.regularizers import l2
from keras.models import Sequential, Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, Callback, TensorBoard
#from matplotlib import pyplot
from sklearn.model_selection import train_test_split
import pandas as pd
import os
#import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import glob
image_list = []

def create_Model(shape):

    model = Sequential()
    model.add(Convolution2D(32, kernel_size=(3, 3), padding="same", activation="relu", input_shape=shape))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(Convolution2D(64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(Convolution2D(128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(Convolution2D(256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(Convolution2D(512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(Convolution2D(1024, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(Convolution2D(1048, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Flatten())
    model.add(Dense(1,kernel_regularizer=l2(0.5), activation="sigmoid"))

    return model

#
# x = np.load('data_128by128.npy')
# y = np.loadtxt("Target.csv", delimiter=",")
# num = np.count_nonzero(y)
# data = x[0:2*num,:]
# labels = y[0:2*num,]
# count1 = 0
# count2 = 0
# l = 0
# for i in range(len(y)):
#     if(y[i,]== 0 and count1<=num-1):
#         data[l,:] = x[i,:]
#         labels[l,] = y[i,]
#         count1  = count1 + 1
#         l = l+1
#     elif(y[i,]== 1 and count2<=num-1):
#         data[l, :] = x[i, :]
#         labels[l,] = y[i,]
#         count2 = count2 + 1
#         l = l+1

# x = data
# y = labels
# #x = np.load('bal_data.npy')
# #y = np.load('bal_lab.npy')
# datagen = ImageDataGenerator(rotation_range=90)
# datagen.fit(x)
# X_batch, y_batch = datagen.flow(x, y, batch_size=len(x)).next()
# a = np.array(X_batch)
# y_train = np.append(y,y_batch)
# x_train = np.append(x,a)
# x_train = np.reshape(x_train,((len(x)+len(X_batch)),128,128,3))
#
#
# datagen1 = ImageDataGenerator(horizontal_flip=True)
# datagen1.fit(x)
# X_batch1, y_batch1 = datagen1.flow(x, y, batch_size=len(x)).next()
# #print(X_batch1.shape)
# a1 = np.array(X_batch1)
# y_train = np.append(y_train,y_batch1)
# x_train = np.append(x_train,a1)
# x_train = np.reshape(x_train,(len(y_train),128,128,3))
# #plt.imshow(x_train[3].reshape(128, 128,3))
# #plt.imshow(x_train[3].reshape(227,227,3))
# #plt.show()

# n_lab_t_sam = 50
# train_data = np.append(x_train,test_data[0:n_lab_t_sam, :])
# train_data = np.reshape(train_data,((len(x_train)+n_lab_t_sam),128,128,3))
# print(train_data.shape)
# train_labels = np.append(y_train,test_labels[0:n_lab_t_sam,])
# print(train_labels.shape)
#
# test_data_fin = test_data[n_lab_t_sam:,:]
# print(test_data_fin.shape)
# test_labels_fin = test_labels[n_lab_t_sam:,]
# print(test_labels_fin.shape)

#random.seed(101)


batch_size = 8
num_classes = 2
epochs = 101

test_data = np.load('data_128by128_CNN_test_dermquest.npy')
test_labels = np.load('test_data_proj_fin_lab.npy')


n_lab_t_sam = 137
train_data = test_data[0:n_lab_t_sam, :]
print(train_data.shape)
train_labels = test_labels[0:n_lab_t_sam,]
print(train_labels.shape)

# test_data_fin = test_data[n_lab_t_sam:,:]
# print(test_data_fin.shape)
# test_labels_fin = test_labels[n_lab_t_sam:,]
# print(test_labels_fin.shape)

train_data = train_data.astype('float32')
train_data /= 255
print(train_data.shape[0], 'train samples')

X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.1, random_state=42)
model = create_Model(X_train[0].shape)

model.load_weights('src_mdl_wghts.h5')
# for layer in model.layers[:1]:
#     layer.trainable = False

adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
#sgd = keras.optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9)
model.compile(loss="binary_crossentropy", optimizer=adam, metrics=['accuracy'])
earlystop  = EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=0, mode='auto')
history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs, callbacks=[earlystop],
                    verbose=1,
                    validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=0)
pre = model.predict(X_test)
print(pre)
#model.save('CNN_mdl_src_data_sig_1class_bal_aug.h5')
print('Test loss:', score[0])
print('Test accuracy:', score[1])

pd.DataFrame(history.history).to_csv("history_SFT_NFT.csv")
#model.save('ML_project_CNN_src_adap_tst_fr1.h5')