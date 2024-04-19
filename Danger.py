import cv2
import os
import tensorflow as tf
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

image_directory="Data/"

no=os.listdir(image_directory+'no/')
yes=os.listdir(image_directory+'yes/')
dataset=[]
lebel=[]
Input_Size=64
path='no0.jpg'
print(path.split('.')[1])

for i,image_name in enumerate(no):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory+'no/'+image_name)
        image=Image.fromarray(image,'RGB')
        iamge=image.resize((Input_Size,Input_Size))
        dataset.append(np.array(iamge))
        lebel.append(0)

for i,image_name in enumerate(yes):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory+'yes/'+image_name)
        image=Image.fromarray(image,'RGB')
        iamge=image.resize((Input_Size,Input_Size))
        dataset.append(np.array(iamge))
        lebel.append(1)

dataset=np.array(dataset)
lebel=np.array(lebel)
print(dataset)
print(lebel)
x_train, x_test, y_train, y_test=train_test_split(dataset, lebel, test_size=0.2, random_state=0)

# print(x_train.shape)
# print(x_test.shape)
print("train your program")
x_train=tf.keras.utils.normalize(x_train, axis=1)
x_train=tf.keras.utils.normalize(x_train, axis=1)
print("test your code")
# x_test=tf.keras.utils.normalize(x_test, axis=1)
# x_test=tf.keras.utils.normalize(x_test, axis=1)
#
# x_test=tf.keras.utils.normalize(x_test, axis=1)
# x_test=tf.keras.utils.normalize(x_test, axis=1)
#
# x_test=tf.keras.utils.normalize(x_test, axis=1)
# x_test=tf.keras.utils.normalize(x_test, axis=1)
#
# x_test=tf.keras.utils.normalize(x_test, axis=1)
# x_test=tf.keras.utils.normalize(x_test, axis=1)
#
# x_test=tf.keras.utils.normalize(x_test, axis=1)
# x_test=tf.keras.utils.normalize(x_test, axis=1)
#
# x_test=tf.keras.utils.normalize(x_test, axis=1)
# x_test=tf.keras.utils.normalize(x_test, axis=1)



y_train=tf.keras.utils.to_categorical(y_train, num_classes=0)
y_test=tf.keras.utils.to_categorical(y_test, num_classes=0)



model=Sequential()
model.add(Conv2D(32,(3,3), input_shape=(Input_Size, Input_Size, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(2))
model.add(Dropout(0.5))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam', metrics='accuracy')

model.fit(x_train, y_train, batch_size=16, verbose=1, epochs=10,validation_data=(x_test,y_test),shuffle=False)
model.save('damage.h5')
print("done")
