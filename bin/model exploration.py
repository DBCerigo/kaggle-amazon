
# coding: utf-8

# In[1]:


# In[2]:

import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.optimizers import Adam, RMSprop
from sklearn.metrics import fbeta_score


# In[3]:

import os
import sys
base_module_path = os.path.abspath(os.path.join('..'))
if base_module_path not in sys.path:
    sys.path.append(base_module_path)
import ama as a


# In[4]:

Vgg = a.vgg.Vgg
TrainBatch = a.trainbatch.TrainBatch


# In[5]:

path= '../data/'
cv_version = 1
cv_version = str(cv_version)
batch_size = 128
img_size = (64,64)
#class_weights = [189, 3.2, 1.7, 2.3, 4.8, 463, 34, 1, 80, 59, 25, 66, 560, 155, 1, 2.2, 90]


# In[6]:

vgg = Vgg(input_shape=(3,)+img_size)


# In[7]:

valgen = TrainBatch(path+'val-jpg'+cv_version+'/', path+'train_v2.csv', batch_size=batch_size, img_size=img_size)


# In[8]:

from keras.preprocessing.image import ImageDataGenerator

gen = ImageDataGenerator(rotation_range=5, horizontal_flip=True, vertical_flip=True)

traingen = TrainBatch(path+'train-jpg'+cv_version+'/', path+'train_v2.csv', batch_size=batch_size, img_size=img_size,
                      imagegen=gen)


# In[9]:

from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from ama.persistenthistory import PersistentHistory

earlystop = EarlyStopping(patience=4)
history = PersistentHistory()


# In[10]:

filepath = '../data/weights/atc'+cv_version+'_v2_best_lr1.hk'
saver = ModelCheckpoint(filepath, verbose=1, save_best_only=True, save_weights_only=True)
vgg.model.compile(optimizer=Adam(lr=0.001),loss='binary_crossentropy', metrics=['accuracy'])
vgg.model.fit_generator(traingen, samples_per_epoch=traingen.nb_sample, nb_epoch=30,
                        validation_data=valgen, nb_val_samples=valgen.nb_sample,
                        callbacks=[history, saver, earlystop])


# In[11]:

vgg.model.load_weights(filepath)


# In[12]:

filepath = '../data/weights/atc'+cv_version+'_v2_best_lr2.hk'
saver = ModelCheckpoint(filepath, verbose=1, save_best_only=True, save_weights_only=True)
vgg.model.compile(optimizer=Adam(lr=0.0001),loss='binary_crossentropy', metrics=['accuracy'])
vgg.model.fit_generator(traingen, samples_per_epoch=traingen.nb_sample, nb_epoch=10,
                        validation_data=valgen, nb_val_samples=valgen.nb_sample,
                        callbacks=[history, saver, earlystop])


# In[13]:

vgg.model.load_weights(filepath)


# In[14]:

filepath = '../data/weights/atc'+cv_version+'_v2_best_lr3.hk'
saver = ModelCheckpoint(filepath, verbose=1, save_best_only=True, save_weights_only=True)
vgg.model.compile(optimizer=Adam(lr=0.00001),loss='binary_crossentropy', metrics=['accuracy'])
vgg.model.fit_generator(traingen, samples_per_epoch=traingen.nb_sample, nb_epoch=10,
                        validation_data=valgen, nb_val_samples=valgen.nb_sample,
                        callbacks=[history, saver, earlystop])


# In[15]:

vgg.model.load_weights(filepath)
vgg.model.save_weights('../data/weights/atc_'+cv_version+'_v2.hk')


# In[16]:

vgg.model.load_weights('../data/weights/atc_'+cv_version+'_v2.hk')


# In[17]:

valgen = TrainBatch(path+'val-jpg'+cv_version+'/', path+'train_v2.csv', batch_size=batch_size, img_size=img_size)
testgen = TrainBatch(path+'test-jpg/', None, batch_size=batch_size, img_size=img_size, train=False)


# In[18]:

val_preds = vgg.model.predict_generator(valgen, valgen.nb_sample)
val_labels = valgen.labels
val_fns = valgen.filenames
labelorder = valgen.labelorder


# In[19]:

import csv

def write_val_csv(preds, labels, filenames, labelorder, fp):
    with open(fp, 'w') as csvf:
        writer = csv.writer(csvf)
        pred_headers = [l+'_pred' for l in labelorder]
        true_headers = [l+'_true' for l in labelorder]
        writer.writerow(['image_name']+pred_headers+true_headers)
        for fn, pred, true in zip(filenames, preds, labels):
            writer.writerow([fn]+list(pred)+list(true))
            
def onehot_to_string(vector, thresholds, labelorder):
    s = ''
    for v, th, name in zip(vector, thresholds, labelorder):
        if v>th:
            s += ' '+name
    return s[1:]

def write_presubmission(preds, filenames, labelorder, fp):
    """Writes a file with a header for each label and the sigmoid return 
    that the model game. Not actually a submission file, as we need to 
    pick appropriate thresholds."""
    with open(fp, 'w') as csvf:
        writer = csv.writer(csvf)
        writer.writerow(['image_name']+labelorder)
        for fn, pred in zip(filenames, preds):
            writer.writerow([fn]+list(pred))
                                  
def write_submission(preds, filenames, labelorder, thresholds, fp):
    """Writes the actual submission file given an array of threholds,
    each corresponding to each class in labelorder"""
    with open(fp, 'w') as csvf:
        writer = csv.writer(csvf)
        writer.writerow(['image_name', 'tags'])
        for fn, pred in zip(filenames, preds):
            writer.writerow([fn, onehot_to_string(pred, thresholds, labelorder)])                            


# In[20]:

write_val_csv(val_preds, val_labels, val_fns, labelorder, '../submissions/atc_'+cv_version+'_v2_val.csv')


# In[24]:

test_preds = vgg.model.predict_generator(testgen, testgen.nb_sample)


# In[25]:

test_fns = testgen.filenames
labelorder = testgen.labelorder


# In[26]:

write_presubmission(test_preds, test_fns, labelorder, '../submissions/atc_'+cv_version+'_v2_test_presub.csv')


# In[27]:

test_preds[0]


# In[28]:

thresholds = [0.165 for _ in range(17)]


# In[29]:

write_submission(test_preds, test_fns, labelorder, thresholds, '../submissions/atc_'+cv_version+'_v2_test_sub_thresh0-165.csv')


