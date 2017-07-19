
# coding: utf-8

# In[1]:



# In[2]:

from keras.applications.inception_v3 import InceptionV3
from keras.models import Sequential, Model
from keras.layers import Input
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from sklearn.metrics import fbeta_score
import numpy as np


# In[3]:

import os
import sys
base_module_path = os.path.abspath(os.path.join('..'))
if base_module_path not in sys.path:
    sys.path.append(base_module_path)
import ama as a
TrainBatch = a.trainbatch.TrainBatch


# In[4]:

cv_version = str(1)


# In[5]:

path= '../data/sample/'
batch_size = 128
img_size = (128,128)


# In[6]:

inc = InceptionV3(weights='imagenet', include_top=False, input_tensor=Input(shape=(3,)+img_size))


# In[7]:

x = inc.output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
preds = Dense(17, activation='sigmoid')(x)

model = Model(input=inc.input, output=preds)
for layer in model.layers:
    layer.trainable = False
model.layers[-1].trainable = True


# In[8]:

valgen = TrainBatch(path+'val-jpg'+cv_version+'/', path+'train_v2.csv', batch_size=batch_size, img_size=img_size)


# In[9]:

from keras.preprocessing.image import ImageDataGenerator

gen = ImageDataGenerator(rotation_range=5, horizontal_flip=True, vertical_flip=True)

traingen = TrainBatch(path+'train-jpg'+cv_version+'/', path+'train_v2.csv', batch_size=batch_size, img_size=img_size,
                      imagegen=gen)


# In[10]:

from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from ama.persistenthistory import PersistentHistory

earlystop = EarlyStopping(patience=4)
history = PersistentHistory()


# In[ ]:

#filepath = '../data/weights/resnet'+cv_version+'_best_lr1.hk'
#saver = ModelCheckpoint(filepath, verbose=1, save_best_only=True, save_weights_only=True)
model.compile(optimizer=Adam(lr=0.01),loss='binary_crossentropy', metrics=['accuracy'])
model.fit_generator(traingen, samples_per_epoch=traingen.nb_sample, nb_epoch=5,
                        validation_data=valgen, nb_val_samples=valgen.nb_sample,
                        callbacks=[history, earlystop])


# In[ ]:

for layer in model.layers:
    layer.trainable=True
model.layers[0].trainable=False


# In[ ]:

#filepath = '../data/weights/resnet'+cv_version+'_best_lr1.hk'
#saver = ModelCheckpoint(filepath, verbose=1, save_best_only=True, save_weights_only=True)
model.compile(optimizer=Adam(lr=0.001),loss='binary_crossentropy', metrics=['accuracy'])
model.fit_generator(traingen, samples_per_epoch=traingen.nb_sample, nb_epoch=20,
                        validation_data=valgen, nb_val_samples=valgen.nb_sample,
                        callbacks=[history, earlystop])


# In[ ]:



