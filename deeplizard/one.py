# -*- coding: utf-8 -*-

import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

train_labels=[]
train_samples=[]

for i in range(50):
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(1)
    
    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(0)
    
for i in range(1000):
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(0)
    
    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(1)
    
    
for i in train_samples:
    print(i)
    
    
for i in train_labels:
    print(i)
 
    
train_labels = np.array(train_labels)
train_samples = np.array(train_samples)

train_labels,train_samples =  shuffle(train_labels,train_samples)

scaler  = MinMaxScaler(feature_range= (0,1))
scaler_trains_samples = scaler.fit_transform(train_samples.reshape(-1,1))


import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation,Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy

##categorical_crossentropy:  For regression models, the commonly used loss function used is mean squared error 
# function while for classification models predicting the probability, the loss function most commonly used is cross entropy. 

models = Sequential([
        Dense(units=16, input_shape=(1, ), activation='relu'),
         Dense(units=32 , activation='relu'),
         Dense(units=2, activation='softmax'),
         
        ])
    
models.summary()


models.compile(optimizer = Adam(),
             loss = 'sparse_categorical_crossentropy',
             metrics = ['accuracy'])

models.fit(scaler_trains_samples, train_labels, validation_split=0.1 ,batch_size=10, epochs=30, shuffle=True, verbose=2)


test_labels=[]
test_samples=[]

for i in range(10):
    random_younger = randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(1)
    
    random_older = randint(65,100)
    test_samples.append(random_older)
    test_labels.append(0)
    
for i in range(200):
    random_younger = randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(0)
    
    random_older = randint(65,100)
    test_samples.append(random_older)
    test_labels.append(1)
    
test_labels = np.array(test_labels)
test_samples = np.array(test_samples)

test_labels,test_samples =  shuffle(test_labels,test_samples)

scaler_test_samples = scaler.fit_transform(test_samples.reshape(-1,1))

pred = models.predict(scaler_test_samples, batch_size=10, verbose=0)

rounded_pred  = np.argmax(pred, axis=-1)

    
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_labels, rounded_pred)

import matplotlib.pyplot as plt
import itertools


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
cm_plot_labels =["no_side_effect", "have_side_effect"]

plot_confusion_matrix(cm, cm_plot_labels)

import os.path

if os.path.isfile("model.h5") is False:
    models.save("model.h5")
    
    
from tensorflow.keras.models import load_model
new_model = load_model("model.h5")
new_model.summary()
new_model.get_weights()
new_model.optimizer


##to save only architecture of model:use to_json
json_string = models.to_json()

from tensorflow.keras.models import model_from_json
model_arch = model_from_json(json_string)
model_arch.summary()


#save in yaml
yaml_string = models.to_yaml()

from tensorflow.keras.models import model_from_yaml
model_arch = model_from_yaml(json_string)
model_arch.summary()


## save only weights
if os.path.isfile("model_weights.h5") is False:
    models.save("model_weights.h5")
    
    
model2 = Sequential([
        Dense(units=16, input_shape=(1, ), activation='relu'),
         Dense(units=32 , activation='relu'),
         Dense(units=2, activation='softmax'),
         
        ])
    
models.load_weights('model_weights.h5')

models.get_weights()