import sys
import os
import numpy as np
import tensorflow as tf
import pandas as pd

sf = pd.read_csv('/home/rants/PycharmProjects/kbs-project/source/fer2013.csv')

X_train, y_train, X_test, y_test = [], [], [], []

for index, row in sf.iterrows():
    value = row['pixels'].split(" ")
    try:
        if 'Training' in row['Usage']:
            X_train.append(np.array(value, 'float32'))
            y_train.append(row['emotion'])
        elif 'PublicTest' in row['Usage']:
            X_test.append(np.array(value, 'float32'))
            y_test.append(row['emotion'])
    except:
        print(f"error occured at index: {index} and row: {row}")

X_train = np.array(X_train, 'float32')
y_train = np.array(y_train, 'float32')
X_test = np.array(X_test, 'float32')
y_test = np.array(y_test, 'float32')

#normalising my data
X_train -= np.mean(X_train, axis=0)
X_train /= np.std(X_train, axis=0)

X_test -= np.mean(X_test, axis=0)
X_test /= np.std(X_test, axis=0)

num_features = 64
num_labels = 7
batch_size = 30
width, height = 48, 48

X_train = X_train.reshape(X_train.shape[0], width, height, 1)
X_test = X_test.reshape(X_test.shape[0], width, height, 1)

#Developing the CNN
test_model = tf.keras.Sequential()

#first layer
test_model.add(tf.keras.layers.Conv2D(num_features, kernel_size=(3,3), activation='relu', input_shape=(X_train.shape[1:])))
test_model.add(tf.keras.layers.Conv2D(num_features, kernel_size=(3,3), activation='relu'))
test_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
test_model.add(tf.keras.layers.Dropout(0.5))

#Second Layer
test_model.add(tf.keras.layers.Conv2D(num_features, kernel_size=(3,3), activation='relu'))
test_model.add(tf.keras.layers.Conv2D(num_features, kernel_size=(3,3), activation='relu'))
test_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
test_model.add(tf.keras.layers.Dropout(0.5))

#Third Layer
test_model.add(tf.keras.layers.Conv2D(2*num_features, kernel_size=(3, 3), activation='relu'))
test_model.add(tf.keras.layers.Conv2D(2*num_features, kernel_size=(3, 3), activation='relu'))
test_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2, 2)))

test_model.add(tf.keras.layers.Flatten())

test_model.add(tf.keras.layers.Dense(2*2*2*2*num_features, activation='relu'))
test_model.add(tf.keras.layers.Dropout(0.2))
test_model.add(tf.keras.layers.Dense(2*2*2*2*num_features, activation='relu'))
test_model.add(tf.keras.layers.Dropout(0.2))

test_model.add(tf.keras.layers.Dense(num_labels, activation='softmax'))

# test_model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
test_model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=.0001), loss='sparse_categorical_crossentropy', metrics  = ['accuracy'])

test_model.fit(X_train, y_train, batch_size= batch_size,
               epochs=30,
               verbose=1,
               validation_data=(X_test, y_test),
               shuffle=True,
               )

#Saving my model to a JSON File
fer_json = test_model.to_json()
with open("../json_files/fer.json", "w") as json_file:
    json_file.write(fer_json)

test_model.save_weights("fer.h5")








