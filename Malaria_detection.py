import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.image import imread
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.layers import Dense,Dropout,Conv2D,MaxPool2D,Flatten
from keras._tf_keras.keras.callbacks import EarlyStopping
from keras._tf_keras.keras.models import  Sequential
from sklearn.metrics import classification_report

img_Dir = "E:\ProfessionalCourses\cell_images"
print(os.listdir(img_Dir))
train_path = img_Dir + '\\train\\'
test_path = img_Dir + '\\test\\'
img = imread(train_path + '\\parasitized\\' + 'C100P61ThinF_IMG_20150918_144104_cell_162.png')
print(img.shape)

"""
# Calculating the mean dimension of the images so that they can be uniform
for image_file in os.listdir(train_path + '\\parasitized\\'):
    Dimension1 = []
    Dimension2 = []
    imga = imread(train_path + '\\parasitized\\' + image_file)
    D1, D2, Colors = imga.shape
    Dimension1.append(D1)
    Dimension2.append(D2)

#Calculating the mean Dimension for the images
print(np.mean(Dimension1))
print(np.mean(Dimension2))

# After Calculating the mean size the Image dimensions are standardized for a particular Dataset
image_shape = (127, 124, 3)  # (height,width,Color)
"""
image_shape = (127, 124, 3)

#The images for training the dataset maybe limited or maybe too uniform to solve this problem,
#We randomize the images we have, i.e, rotae them stretch them change their pixel sizes etc to have
#Flexibility in the images we use to train

image_gen = ImageDataGenerator(
    rotation_range=30,
    height_shift_range=0.2,
    width_shift_range=0.2,
    shear_range=0.2,
    fill_mode="nearest",
    zoom_range=0.2,
    horizontal_flip=True
)

#The code given below are Pretty much self-Explanatory
print("The images and the classes as perceived by the Tensorflow class : ")
print(image_gen.flow_from_directory(train_path))
print("The actual image")
plt.imshow(img)
plt.show()
print("After random image generator")
plt.imshow(image_gen.random_transform(img))
plt.show()

#Creating our test data
Image_train = image_gen.flow_from_directory(train_path,
                                            shuffle=True,
                                            target_size=(127,124),
                                            color_mode='rgb',
                                            batch_size= 8
                                            )

print(Image_train.class_indices) #The label used to identify our images

Image_test = image_gen.flow_from_directory(test_path,
                                            shuffle=False,
                                            target_size=(127,124),
                                            color_mode='rgb',
                                            batch_size= 8
                                            )
model = Sequential()
model.add(Conv2D(filters=256,kernel_size=(3,3),padding='same',input_shape = (127, 124, 3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=128,kernel_size=(3,3),padding='same',input_shape = (127, 124, 3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same',input_shape = (127, 124, 3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=32,kernel_size=(3,3),padding='same',input_shape = (127, 124, 3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              metrics =['accuracy'],
              optimizer='adam')

early_stopping = EarlyStopping(monitor='val_loss',patience=2)
model.fit(Image_train,epochs=20,validation_data=Image_test,callbacks=early_stopping)
model.save('Malaria_model.keras')
metrics= pd.DataFrame(model.history.history)
metrics[['loss','val_loss']].plot()
plt.show()
metrics[['accuracy','val_accuracy']].plot()
plt.show()

my_Image = Image_test[0]

print("The Prediction of the Animal that is displayed")
test = model.predict(my_Image.reshape(127, 124, 3))
print(np.argmax(test,axis=1))

predictions = np.argmax(model.predict(Image_test), axis=1)
print(classification_report(Image_test.flatten(), predictions))

