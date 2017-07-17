import os.path
import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

class TrainBatch():
    """Returns a generator to give to train_generator or predict_generator. NOTE
    that currently shuffling doesn't work, hence it being left at False"""
    def __init__(self, path, label_path, img_size=(128,128), batch_size=128, 
            imagegen=ImageDataGenerator(), train=True):
        self.path = path
        self.label_path = label_path
        self.batch_size = batch_size
        self.train = train
        self.labelorder =  ["agriculture",
            "artisinal_mine",
            "bare_ground",
            "blooming",
            "blow_down",
            "clear",
            "cloudy",
            "conventional_mine",
            "cultivation",
            "habitation",
            "haze",
            "partly_cloudy",
            "primary",
            "road",
            "selective_logging",
            "slash_burn",
            "water"
        ]
        self.create(img_size, imagegen)

    def __next__(self):
        return self.next()
    
    def next(self):
        if self.train:
            return (self.image_gen.next(), self.label_gen.next())
        else:
            return self.image_gen.next()
        
    def create(self, img_size, imagegen):
        img_batches = self.create_image_gen(img_size, imagegen)
        filenames = [os.path.splitext(os.path.basename(fn))[0] 
                    for fn in self.image_gen.filenames]
        self.filenames = filenames
        if self.train:
            labels = self.get_labels(filenames)
            self.label_gen = self.create_label_gen(labels)
            self.labels = labels
        
    def create_image_gen(self, img_size, imagegen):
        batches = self.get_batches(self.path, gen=imagegen, 
            batch_size=self.batch_size, target_size=img_size)
        self.image_gen = batches
        self.nb_sample = batches.nb_sample
        
    def get_batches(self, dirname, gen, shuffle=False,
                    batch_size=128, class_mode=None, target_size=(256,256)):
        return gen.flow_from_directory(dirname, target_size=target_size,
                class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)
        
    def get_labels(self, filenames):
        tags = pd.read_csv(self.label_path)
        df = pd.DataFrame()
        df['image_name'] = filenames
        df = df.merge(tags, how='left', on='image_name')
        labels = np.stack([self.onehot_labels(v) for v in df.tags],axis=0)
        return labels

    def onehot_labels(self, strings):
        labels = self.labelorder
        strings = strings.split(' ')
        vector = np.zeros(len(labels))
        for s in strings:
            try:
                idx = labels.index(s)
                vector += to_categorical([idx], len(labels)).squeeze()
            except ValueError:
                raise Exception('Unrecognised label '+s)
        return vector
        
    def create_label_gen(self, labels):
        batch_size = self.batch_size
        counter = 0
        while True:
			upper = min((counter+1)*batch_size, len(labels))
			yield labels[counter*batch_size:upper,:]
			if upper == len(labels):
				counter = 0
			else: counter+=1
