import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.datasets import mnist
from keras.layers import Flatten, Conv2D, Dense, MaxPool2D
from keras.models import Sequential
from keras.utils import to_categorical

(X_train,Y_train),(X_test,Y_test) = mnist.load_data()
print(X_test.shape)
Y_test = to_categorical(Y_test)
Y_train = to_categorical(Y_train)
X_test = X_test/255
X_train =X_train/255
X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)


model= Sequential()
model.add(Conv2D(filters=32,padding='same',kernel_size=(3,3),strides=(1,1),input_shape=(28,28,1),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(10,activation='softplus'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss',patience=1)
model.fit(X_train,Y_train,epochs=8,validation_data=(X_test,Y_test),callbacks=[early_stop])
loss = pd.DataFrame(model.history.history)
loss[['loss','val_loss']].plot()
plt.show()
my_number = X_test[0]

plt.imshow(my_number.reshape(28,28))
plt.show()

test = model.predict(my_number.reshape(1,28,28,1))
print(np.argmax(test,axis=1))

model.save('testone.keras')

